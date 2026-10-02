"""
stats_analysis.py
------------------
Utilidades de análisis estadístico para el dashboard de gasto oncológico
INEN — SIS/FISSAL:

  - linear_trend()        -> tendencia lineal simple (mínimos cuadrados),
                              usada para la tendencia ANUAL.
  - segmented_its()        -> regresión segmentada de INTERRUPCIONES
                              MÚLTIPLES (series temporales interrumpidas,
                              ITS) sobre la serie MENSUAL, con un punto de
                              corte por cada norma/paquete normativo en
                              data/norms.csv. Errores robustos a
                              heterocedasticidad y autocorrelación
                              (Newey-West/HAC).
  - mann_kendall_trend()   -> test de tendencia robusto a autocorrelación
                              (Hamed & Rao 1998, respaldo Yue-Wang 2004) +
                              pendiente de Sen, aplicado a la serie mensual
                              o anual seleccionada.
  - abc_classification()   -> clasificación ABC (Pareto 80/15/5) de
                              productos/ítems por gasto acumulado.
  - high_cost_medications()-> evaluación de medicamentos de alto costo
                              según el umbral DIGEMID de 9 UIT.
  - smooth_series()        -> suavizado LOWESS de la serie mensual.

Por qué series temporales interrumpidas de interrupciones múltiples
---------------------------------------------------------------------
Entre 2019 y 2025 se promulgaron al menos 9 disposiciones normativas
oncológicas. Modelar cada una por separado con Chow/ITS de un solo quiebre
ignora que los efectos son acumulativos y que varias normas caen muy cerca
en el tiempo (lo que además rompe la identificabilidad estadística si se
tratan como quiebres independientes). `segmented_its()` ajusta un único
modelo con un término de nivel y un término de pendiente por cada punto de
corte, acumulando pendientes segmento a segmento — el mismo enfoque
metodológico usado en los artículos de esta serie de investigación.

Por qué Newey-West y no errores OLS clásicos
-----------------------------------------------
El gasto mensual de una institución de salud está autocorrelacionado
(ejecución presupuestal, compras trimestrales, etc.). Los errores estándar
OLS clásicos subestiman la incertidumbre en ese escenario; Newey-West la
corrige explícitamente sin asumir independencia de los residuos.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

MK_AVAILABLE = True
"""Antes indicaba si pymannkendall estaba instalado. Ya no depende de esa
librería: Mann-Kendall (clásico, Hamed-Rao y Yue-Wang) y la pendiente de
Sen están implementados directamente en este archivo con numpy/scipy
(fórmulas de Hamed & Rao 1998 y Yue & Wang 2004, verificadas línea por
línea contra pymannkendall antes de reemplazarlo). Se mantiene esta
constante por compatibilidad con app.py, que la usa solo para decidir si
mostrar la nota informativa sobre qué variante se está usando."""

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
except ImportError:  # pragma: no cover
    _lowess = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from statsmodels.tools.sm_exceptions import HessianInversionWarning, IterationLimitWarning, SingularMatrixWarning
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    warnings.filterwarnings("ignore", category=SingularMatrixWarning)
except ImportError:
    pass


@dataclass
class TrendResult:
    slope: float
    intercept: float
    r2: float
    p_value: float
    x: np.ndarray
    y_pred: np.ndarray


@dataclass
class SegmentResult:
    """Resultado de UN punto de corte dentro de un modelo ITS de
    interrupciones múltiples."""
    break_id: str
    label: str
    date: pd.Timestamp
    level_change: float
    level_p: float
    level_significant: bool
    slope_change: float
    slope_p: float
    slope_significant: bool
    slope_before: float
    slope_after: float
    n_post: int


@dataclass
class ITSResult:
    n_obs: int
    r2: float
    pre_slope: float
    pre_slope_p: float
    segments: list[SegmentResult]
    fitted: pd.Series
    observed: pd.Series


@dataclass
class DiDResult:
    """Resultado de una regresión de diferencia en diferencias (DiD) sobre
    log(gasto), comparando un grupo TRATAMIENTO (p. ej. gasto oncológico)
    contra un grupo CONTROL (el comparador no oncológico) alrededor de un
    punto de corte normativo."""
    break_date: pd.Timestamp
    n_obs: int
    did_coef: float          # coeficiente de tratamiento×post, en log-puntos
    did_se: float
    did_p: float
    pct_effect: float        # (exp(did_coef)-1)*100, efecto % aproximado
    pretrend_coef: float      # coeficiente de tratamiento×tendencia (chequeo de tendencias paralelas)
    pretrend_p: float
    parallel_trends_ok: bool  # True si pretrend_p >= alpha (no se rechaza el supuesto)
    cell_means: dict          # {"treat_pre","treat_post","control_pre","control_post"} en soles/mes
    n_pre: int
    n_post: int


def diff_in_diff(
    treat: pd.Series, control: pd.Series, break_date: pd.Timestamp, alpha: float = 0.05,
    hac_lags: int = 3,
) -> DiDResult | None:
    """DiD de dos grupos (tratamiento vs. control) y dos períodos (antes/
    después de `break_date`), sobre log(gasto+1) para que el coeficiente se
    interprete como efecto porcentual y no quede dominado por la diferencia
    de escala entre ambos grupos. Incluye tratamiento×tendencia para poder
    chequear el supuesto de tendencias paralelas antes del quiebre — con
    solo 2 grupos no puede probarse formalmente, pero un coeficiente no
    significativo en ese término es la evidencia disponible más cercana.

    `treat` y `control` deben ser series mensuales (pd.Series con índice de
    fecha) ya alineadas al mismo rango; se recorta automáticamente a la
    intersección de ambos índices."""
    idx = treat.index.intersection(control.index)
    if len(idx) < 12 or break_date <= idx.min() or break_date > idx.max():
        return None
    treat, control = treat.reindex(idx), control.reindex(idx)

    panel = pd.concat([
        pd.DataFrame({"tiempo": idx, "y": treat.values, "treatment": 1}),
        pd.DataFrame({"tiempo": idx, "y": control.values, "treatment": 0}),
    ], ignore_index=True)
    panel["log_y"] = np.log(panel["y"].clip(lower=1))
    panel["t"] = _month_index_dates(panel["tiempo"], idx.min())
    panel["post"] = (panel["tiempo"] >= break_date).astype(int)
    panel["treat_x_post"] = panel["treatment"] * panel["post"]
    panel["treat_x_t"] = panel["treatment"] * panel["t"]

    X = sm.add_constant(panel[["t", "treatment", "post", "treat_x_t", "treat_x_post"]])
    model = sm.OLS(panel["log_y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    did_coef = model.params["treat_x_post"]
    did_p = model.pvalues["treat_x_post"]
    did_se = model.bse["treat_x_post"]
    pretrend_coef = model.params["treat_x_t"]
    pretrend_p = model.pvalues["treat_x_t"]

    pre_mask, post_mask = idx < break_date, idx >= break_date
    cell_means = {
        "treat_pre": float(treat[pre_mask].mean()) if pre_mask.any() else float("nan"),
        "treat_post": float(treat[post_mask].mean()) if post_mask.any() else float("nan"),
        "control_pre": float(control[pre_mask].mean()) if pre_mask.any() else float("nan"),
        "control_post": float(control[post_mask].mean()) if post_mask.any() else float("nan"),
    }

    return DiDResult(
        break_date=break_date, n_obs=len(panel), did_coef=float(did_coef), did_se=float(did_se),
        did_p=float(did_p), pct_effect=float((np.exp(did_coef) - 1) * 100),
        pretrend_coef=float(pretrend_coef), pretrend_p=float(pretrend_p),
        parallel_trends_ok=bool(pretrend_p >= alpha),
        cell_means=cell_means, n_pre=int(pre_mask.sum()), n_post=int(post_mask.sum()),
    )


def _month_index_dates(dates: pd.Series, ref: pd.Timestamp) -> np.ndarray:
    return np.array([(d.year - ref.year) * 12 + (d.month - ref.month) for d in dates], dtype=float)


@dataclass
class MannKendallResult:
    trend: str  # "creciente" | "decreciente" | "sin tendencia"
    h: bool
    p_value: float
    z: float
    sen_slope: float
    variant_used: str


@dataclass
class ABCResult:
    table: pd.DataFrame  # ranked, with %, % acumulado, Zona_ABC
    zone_summary: pd.DataFrame


@dataclass
class AutoBreakResult:
    """Resultado de la detección AUTOMÁTICA de un punto de quiebre
    estructural (dos tramos, sin fecha impuesta) — equivalente mensual
    de `detect_breakpoint` en el dashboard de casos."""
    date: pd.Timestamp
    slope_before: float
    slope_after: float
    intercept_before: float
    intercept_after: float
    f_stat: float
    p_value: float
    significant: bool
    x_before: np.ndarray
    y_pred_before: np.ndarray
    x_after: np.ndarray
    y_pred_after: np.ndarray


@dataclass
class ArbitraryBreakResult:
    """Test de Chow anclado a una fecha ELEGIDA por el usuario (una ley,
    la apertura de un centro, etc.), con ventana de implementación
    opcional a excluir justo después del evento."""
    event_date: pd.Timestamp
    implementation_lag_months: int
    slope_before: float
    slope_after: float
    f_stat: float
    p_value: float
    significant: bool
    n_before: int
    n_after: int
    x_before: np.ndarray
    y_pred_before: np.ndarray
    x_after: np.ndarray
    y_pred_after: np.ndarray


@dataclass
class ProjectionResult:
    idx_hist: pd.DatetimeIndex
    values_hist: np.ndarray
    idx_future: pd.DatetimeIndex
    conservative: np.ndarray
    recommended: np.ndarray
    recommended_lower: np.ndarray
    recommended_upper: np.ndarray
    optimistic: np.ndarray
    method_note: str


def _month_index(idx: pd.DatetimeIndex) -> np.ndarray:
    """Convierte un índice de fechas mensuales a un contador entero
    continuo (0, 1, 2, ...), para poder usar las mismas rutinas de
    regresión que en el dashboard de casos (que usa años enteros)."""
    return np.array([(d.year - idx[0].year) * 12 + (d.month - idx[0].month) for d in idx])


def detect_breakpoint_auto(monthly: pd.Series, min_segment: int = 6) -> AutoBreakResult | None:
    """Busca el mes que mejor divide la serie en dos tramos (mínima
    suma de cuadrados residual combinada) y aplica un test de Chow
    contra el modelo único para toda la serie. Requiere al menos
    2*min_segment observaciones."""
    y_all = monthly.values.astype(float)
    x_all = _month_index(monthly.index).astype(float)
    n = len(x_all)
    if n < 2 * min_segment:
        return None

    c_pooled = np.polyfit(x_all, y_all, 1)
    rss_pooled = np.sum((y_all - np.polyval(c_pooled, x_all)) ** 2)

    best = None
    for i in range(min_segment, n - min_segment + 1):
        x1, y1 = x_all[:i], y_all[:i]
        x2, y2 = x_all[i:], y_all[i:]
        c1 = np.polyfit(x1, y1, 1)
        c2 = np.polyfit(x2, y2, 1)
        rss = np.sum((y1 - np.polyval(c1, x1)) ** 2) + np.sum((y2 - np.polyval(c2, x2)) ** 2)
        if best is None or rss < best[0]:
            best = (rss, i, c1, c2, x1, y1, x2, y2)
    rss_split, split_idx, c1, c2, x1, y1, x2, y2 = best

    k, df1 = 2, 2
    df2 = n - 2 * k
    if df2 <= 0 or rss_split <= 0:
        f_stat, p_value = float("nan"), float("nan")
    else:
        f_stat = ((rss_pooled - rss_split) / df1) / (rss_split / df2)
        p_value = float(stats.f.sf(f_stat, df1, df2))

    return AutoBreakResult(
        date=monthly.index[split_idx], slope_before=c1[0], slope_after=c2[0],
        intercept_before=c1[1], intercept_after=c2[1], f_stat=f_stat, p_value=p_value,
        significant=bool(p_value < 0.05) if p_value == p_value else False,
        x_before=monthly.index[:split_idx], y_pred_before=np.polyval(c1, x1),
        x_after=monthly.index[split_idx:], y_pred_after=np.polyval(c2, x2),
    )


def chow_test_event(
    monthly: pd.Series, event_date: pd.Timestamp, implementation_lag_months: int = 0,
    min_segment: int = 4,
) -> ArbitraryBreakResult | None:
    """Test de Chow anclado a `event_date`. `implementation_lag_months`
    excluye una ventana justo después del evento (el efecto aún no se
    consolida), igual que `chow_test_arbitrary_break` en el dashboard
    de casos, pero en meses en vez de años."""
    x_all = _month_index(monthly.index).astype(float)
    y_all = monthly.values.astype(float)
    dates = monthly.index

    after_start = event_date + pd.DateOffset(months=implementation_lag_months)
    before_mask = dates < event_date
    after_mask = dates >= after_start

    x1, y1 = x_all[before_mask], y_all[before_mask]
    x2, y2 = x_all[after_mask], y_all[after_mask]
    if len(x1) < min_segment or len(x2) < min_segment:
        return None

    c1 = np.polyfit(x1, y1, 1)
    c2 = np.polyfit(x2, y2, 1)
    rss1 = np.sum((y1 - np.polyval(c1, x1)) ** 2)
    rss2 = np.sum((y2 - np.polyval(c2, x2)) ** 2)
    rss_split = rss1 + rss2

    x_pooled = np.concatenate([x1, x2])
    y_pooled = np.concatenate([y1, y2])
    c_pooled = np.polyfit(x_pooled, y_pooled, 1)
    rss_pooled = np.sum((y_pooled - np.polyval(c_pooled, x_pooled)) ** 2)

    n = len(x_pooled)
    k, df1 = 2, 2
    df2 = n - 2 * k
    if df2 <= 0 or rss_split <= 0:
        f_stat, p_value = float("nan"), float("nan")
    else:
        f_stat = ((rss_pooled - rss_split) / df1) / (rss_split / df2)
        p_value = float(stats.f.sf(f_stat, df1, df2))

    return ArbitraryBreakResult(
        event_date=event_date, implementation_lag_months=implementation_lag_months,
        slope_before=c1[0], slope_after=c2[0], f_stat=f_stat, p_value=p_value,
        significant=bool(p_value < 0.05) if p_value == p_value else False,
        n_before=len(x1), n_after=len(x2),
        x_before=dates[before_mask], y_pred_before=np.polyval(c1, x1),
        x_after=dates[after_mask], y_pred_after=np.polyval(c2, x2),
    )


def project_series(
    monthly: pd.Series, horizon: int = 6, n_boot: int = 300, ci: float = 0.90,
    min_obs: int = 12, random_state: int = 42,
) -> ProjectionResult | None:
    """Proyección a `horizon` MESES, con el mismo esquema de 3
    escenarios que el dashboard de casos (conservador / recomendado
    Holt amortiguado / lineal-Sen), adaptado a periodicidad mensual."""
    try:
        from statsmodels.tsa.holtwinters import Holt
    except ImportError:
        return None
    y = monthly.values.astype(float)
    if len(y) < min_obs:
        return None
    idx_future = pd.date_range(monthly.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")

    conservative = np.full(horizon, float(np.mean(y[-3:])))

    model = Holt(y, damped_trend=True, initialization_method="estimated")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(optimized=True)
        recommended = np.asarray(fit.forecast(horizon))

    resid = np.asarray(fit.resid)
    resid = resid[~np.isnan(resid)]
    rng = np.random.default_rng(random_state)
    boot = np.empty((n_boot, horizon))
    if len(resid) >= 3:
        for b in range(n_boot):
            sim_resid = rng.choice(resid, size=len(y), replace=True)
            y_sim = np.asarray(fit.fittedvalues) + sim_resid
            try:
                m_b = Holt(y_sim, damped_trend=True, initialization_method="estimated")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    boot[b] = m_b.fit(optimized=True).forecast(horizon)
            except Exception:
                boot[b] = recommended
        a = 1 - ci
        lower = np.nanpercentile(boot, 100 * a / 2, axis=0)
        upper = np.nanpercentile(boot, 100 * (1 - a / 2), axis=0)
    else:
        lower, upper = recommended.copy(), recommended.copy()

    mkr = mann_kendall_trend(y)
    if mkr is not None:
        t_last = len(y) - 1
        t_future = np.arange(len(y), len(y) + horizon)
        last_fit = y[-1]  # ancla en el último valor observado
        optimistic = last_fit + mkr.sen_slope * (t_future - t_last)
    else:
        optimistic = recommended.copy()

    conservative, recommended = np.clip(conservative, 0, None), np.clip(recommended, 0, None)
    lower, upper = np.clip(lower, 0, None), np.clip(upper, 0, None)
    optimistic = np.clip(optimistic, 0, None)

    note = (
        "Método recomendado: suavizado exponencial de Holt con tendencia amortiguada (damped trend), que "
        "reduce gradualmente la pendiente proyectada para no sobreestimar a mediano plazo. El escenario "
        "'lineal (Sen)' extrapola la pendiente de Sen sin amortiguar — no siempre es el más alto. Compara "
        "los tres para dimensionar la incertidumbre."
    )
    return ProjectionResult(
        idx_hist=monthly.index, values_hist=y, idx_future=idx_future,
        conservative=conservative, recommended=recommended,
        recommended_lower=lower, recommended_upper=upper, optimistic=optimistic, method_note=note,
    )


def ranking_table(
    df: pd.DataFrame, group_col: str, period_lo: pd.Timestamp, period_hi: pd.Timestamp,
    value_col: str = "total_neto",
) -> pd.DataFrame:
    """Ranking de `group_col` (diagnóstico / IAFAS / tipo de consumo) por
    gasto acumulado en [period_lo, period_hi], para el gráfico de barras
    horizontal estilo GLOBOCAN del dashboard de casos."""
    d = df[(df["tiempo"] >= period_lo) & (df["tiempo"] <= period_hi)]
    g = d.groupby(group_col, as_index=False)[value_col].sum().sort_values(value_col, ascending=False)
    g["ranking"] = range(1, len(g) + 1)
    return g


def cagr(first_value: float, last_value: float, n_periods: float) -> float | None:
    """Crecimiento anual compuesto. n_periods en AÑOS (usar meses/12)."""
    if first_value is None or last_value is None or first_value <= 0 or n_periods <= 0:
        return None
    return (last_value / first_value) ** (1 / n_periods) - 1



def alpha_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""


def significance_label(p: float, alpha: float = 0.05) -> str:
    if p < alpha:
        return "Significativo"
    if p < 0.10:
        return "Marginal"
    return "No significativo"


# ---------------------------------------------------------------------------
# Tendencia lineal simple (uso: tendencia ANUAL)
# ---------------------------------------------------------------------------

def linear_trend(x: np.ndarray, y: np.ndarray) -> TrendResult | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return TrendResult(
        slope=model.params[1], intercept=model.params[0], r2=model.rsquared,
        p_value=model.pvalues[1], x=x, y_pred=model.predict(X),
    )


# ---------------------------------------------------------------------------
# ITS de interrupciones múltiples (uso: serie MENSUAL)
# ---------------------------------------------------------------------------

def segmented_its(
    monthly: pd.Series,
    breaks: dict[str, pd.Timestamp],
    labels: dict[str, str] | None = None,
    hac_lags: int = 3,
    alpha: float = 0.05,
    unit_divisor: float = 1e6,
) -> ITSResult | None:
    """Ajusta Yt = b0 + b1*t + sum_k(delta_k*D_k + gamma_k*S_k), donde D_k
    es 1 desde la fecha de corte k en adelante y S_k es un contador que
    empieza en 1 ese mismo mes. `monthly` debe venir con índice de fecha
    mensual continuo (usar data_processing.monthly_series). Los puntos de
    corte posteriores al último mes de la serie, o anteriores al primero,
    se ignoran automáticamente (no aportan información)."""
    if monthly is None or len(monthly) < 12:
        return None
    labels = labels or {}
    s = pd.DataFrame({"y": monthly.values / unit_divisor}, index=monthly.index)
    s["t"] = range(1, len(s) + 1)

    active_breaks = {
        k: v for k, v in breaks.items() if monthly.index.min() < v <= monthly.index.max()
    }
    if not active_breaks:
        # Sin puntos de corte dentro del rango: solo tendencia simple.
        X = sm.add_constant(s[["t"]])
        model = sm.OLS(s["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        fitted = pd.Series(model.predict(X).values * unit_divisor, index=s.index)
        return ITSResult(
            n_obs=len(s), r2=model.rsquared, pre_slope=model.params["t"],
            pre_slope_p=model.pvalues["t"], segments=[], fitted=fitted, observed=monthly,
        )

    for k, bdate in active_breaks.items():
        s[f"D_{k}"] = (s.index >= bdate).astype(int)
        months_since = [(d.year - bdate.year) * 12 + (d.month - bdate.month) + 1 for d in s.index]
        s[f"S_{k}"] = np.where(s.index >= bdate, months_since, 0)

    xcols = ["t"] + [f"D_{k}" for k in active_breaks] + [f"S_{k}" for k in active_breaks]
    X = sm.add_constant(s[xcols])
    model = sm.OLS(s["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    fitted = pd.Series(model.predict(X).values * unit_divisor, index=s.index)

    segments = []
    cum_slope = model.params["t"]
    for k, bdate in active_breaks.items():
        level = model.params[f"D_{k}"]
        level_p = model.pvalues[f"D_{k}"]
        dslope = model.params[f"S_{k}"]
        dslope_p = model.pvalues[f"S_{k}"]
        slope_before = cum_slope
        cum_slope = cum_slope + dslope
        n_post = int(s[f"D_{k}"].sum())
        segments.append(SegmentResult(
            break_id=k, label=labels.get(k, k), date=bdate,
            level_change=level, level_p=level_p, level_significant=level_p < alpha,
            slope_change=dslope, slope_p=dslope_p, slope_significant=dslope_p < alpha,
            slope_before=slope_before, slope_after=cum_slope, n_post=n_post,
        ))

    return ITSResult(
        n_obs=len(s), r2=model.rsquared, pre_slope=model.params["t"],
        pre_slope_p=model.pvalues["t"], segments=segments, fitted=fitted, observed=monthly,
    )


# ---------------------------------------------------------------------------
# Mann-Kendall + pendiente de Sen
# ---------------------------------------------------------------------------
# Implementación propia (sin pymannkendall) del test de Mann-Kendall clásico
# y de las dos correcciones de varianza por autocorrelación más usadas en la
# literatura: Hamed & Rao (1998) y Yue & Wang (2004). Las fórmulas se
# tradujeron línea por línea del código fuente de pymannkendall 1.4.x y se
# verificaron numéricamente contra esa misma librería (Z, p-valor y
# pendiente de Sen coinciden hasta el 6º decimal) antes de reemplazarla —
# ver el historial de este archivo para el script de validación. Esto evita
# que la app dependa de que pymannkendall esté instalado en el entorno de
# quien la ejecute.

def _mk_score(x: np.ndarray, n: int) -> float:
    s = 0.0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))
    return s


def _mk_variance_s(x: np.ndarray, n: int) -> float:
    unique_x, counts = np.unique(x, return_counts=True)
    if n == len(unique_x):
        return (n * (n - 1) * (2 * n + 5)) / 18
    return (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18


def _mk_z_score(s: float, var_s: float) -> float:
    if var_s <= 0:
        return 0.0
    if s > 0:
        return (s - 1) / np.sqrt(var_s)
    if s < 0:
        return (s + 1) / np.sqrt(var_s)
    return 0.0


def _mk_p_and_trend(z: float, alpha: float = 0.05) -> tuple[float, bool, str]:
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    h = bool(abs(z) > stats.norm.ppf(1 - alpha / 2))
    if z < 0 and h:
        trend = "decreciente"
    elif z > 0 and h:
        trend = "creciente"
    else:
        trend = "sin tendencia"
    return p, h, trend


def _sen_slope(x: np.ndarray) -> float:
    n = len(x)
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    return float(np.median(slopes)) if slopes else 0.0


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    y = x - x.mean()
    n = len(x)
    d = n * np.ones(2 * n - 1)
    acov = (np.correlate(y, y, "full") / d)[n - 1:]
    return acov[:nlags + 1] / acov[0]


def _mk_classic(x: np.ndarray, alpha: float = 0.05) -> MannKendallResult:
    n = len(x)
    s = _mk_score(x, n)
    var_s = _mk_variance_s(x, n)
    z = _mk_z_score(s, var_s)
    p, h, trend = _mk_p_and_trend(z, alpha)
    return MannKendallResult(trend=trend, h=h, p_value=float(p), z=float(z),
                              sen_slope=_sen_slope(x), variant_used="Mann-Kendall clásico")


def _mk_hamed_rao(x: np.ndarray, alpha: float = 0.05) -> MannKendallResult | None:
    """Corrección de varianza de Hamed & Rao (1998): usa la autocorrelación
    de los RANGOS de la serie destendenciada (con la pendiente de Sen).
    Devuelve None si la corrección resulta numéricamente inestable
    (varianza corregida no positiva o no finita) para que el llamador
    pruebe Yue-Wang como respaldo."""
    n = len(x)
    s = _mk_score(x, n)
    var_s = _mk_variance_s(x, n)
    slope = _sen_slope(x)
    x_detrend = x - np.arange(1, n + 1) * slope
    ranks = stats.rankdata(x_detrend)
    acf_1 = _acf(ranks, nlags=n - 1)
    interval = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    sni = 0.0
    for i in range(1, n):
        if -interval <= acf_1[i] <= interval:
            continue
        sni += (n - i) * (n - i - 1) * (n - i - 2) * acf_1[i]
    n_ns = 1 + (2 / (n * (n - 1) * (n - 2))) * sni
    var_s_corr = var_s * n_ns
    if not np.isfinite(var_s_corr) or var_s_corr <= 0:
        return None
    z = _mk_z_score(s, var_s_corr)
    p, h, trend = _mk_p_and_trend(z, alpha)
    return MannKendallResult(trend=trend, h=h, p_value=float(p), z=float(z),
                              sen_slope=slope, variant_used="Hamed-Rao (1998)")


def _mk_yue_wang(x: np.ndarray, alpha: float = 0.05) -> MannKendallResult | None:
    """Corrección de varianza de Yue & Wang (2004): usa la autocorrelación
    de la serie destendenciada directamente (no de sus rangos)."""
    n = len(x)
    s = _mk_score(x, n)
    var_s = _mk_variance_s(x, n)
    slope = _sen_slope(x)
    x_detrend = x - np.arange(1, n + 1) * slope
    acf_1 = _acf(x_detrend, nlags=n - 1)
    idx = np.arange(1, n)
    sni = np.sum((1 - idx / n) * acf_1[idx])
    n_ns = 1 + 2 * sni
    var_s_corr = var_s * n_ns
    if not np.isfinite(var_s_corr) or var_s_corr <= 0:
        return None
    z = _mk_z_score(s, var_s_corr)
    p, h, trend = _mk_p_and_trend(z, alpha)
    return MannKendallResult(trend=trend, h=h, p_value=float(p), z=float(z),
                              sen_slope=slope, variant_used="Yue-Wang (2004)")


def mann_kendall_trend(values: np.ndarray, alpha: float = 0.05) -> MannKendallResult | None:
    """Prueba, en orden, Hamed-Rao (mejor corrección disponible) y
    Yue-Wang (respaldo) — ambas corrigen la varianza por autocorrelación,
    relevante en gasto mensual con ejecución presupuestal por trimestres.
    Si ambas resultan numéricamente inestables (series muy cortas o con
    autocorrelación extrema), cae al Mann-Kendall clásico, que siempre
    puede calcularse. Ninguna de las tres variantes depende de librerías
    externas; todas están implementadas arriba en este archivo."""
    if len(values) < 8:
        return None
    x = np.asarray(values, dtype=float)
    return _mk_hamed_rao(x, alpha) or _mk_yue_wang(x, alpha) or _mk_classic(x, alpha)


def smooth_series(y: np.ndarray, frac: float = 0.35) -> np.ndarray | None:
    if _lowess is None or len(y) < 6:
        return None
    x = np.arange(len(y))
    out = _lowess(y, x, frac=frac, return_sorted=False)
    return out


# ---------------------------------------------------------------------------
# Clasificación ABC (Pareto 80/15/5)
# ---------------------------------------------------------------------------

def abc_classification(
    df: pd.DataFrame, item_col: str = "DESC_CONSUMO", value_col: str = "total_neto",
    qty_col: str = "total_cantidad",
) -> ABCResult | None:
    if df.empty:
        return None
    g = df.groupby(item_col, as_index=False).agg(
        cantidad_total=(qty_col, "sum"), valor_total=(value_col, "sum")
    )
    g = g[g["valor_total"] > 0].sort_values("valor_total", ascending=False).reset_index(drop=True)
    if g.empty:
        return None
    total = g["valor_total"].sum()
    g["%_del_total"] = g["valor_total"] / total * 100
    g["%_acumulado"] = g["%_del_total"].cumsum()
    g["ranking"] = range(1, len(g) + 1)
    g["Zona_ABC"] = g["%_acumulado"].apply(lambda p: "A" if p <= 80 else ("B" if p <= 95 else "C"))

    n = len(g)
    zonas = g.groupby("Zona_ABC").agg(n_items=(item_col, "count"), valor=("valor_total", "sum"))
    zonas["%_items"] = zonas["n_items"] / n * 100
    zonas["%_valor"] = zonas["valor"] / total * 100
    zonas = zonas.reindex(["A", "B", "C"]).dropna(how="all")
    return ABCResult(table=g, zone_summary=zonas.reset_index())


# ---------------------------------------------------------------------------
# Medicamentos de alto costo (umbral DIGEMID: 9 UIT / paciente-año)
# ---------------------------------------------------------------------------

def high_cost_medications(product_df: pd.DataFrame) -> pd.DataFrame:
    """Costo unitario promedio (total_neto / total_cantidad) por producto y
    año, comparado contra el umbral de 9 UIT. NOTA: el umbral DIGEMID se
    define por paciente-año; esta base solo tiene unidades dispensadas
    (viales/tabletas), no pacientes, por lo que el resultado es una
    aproximación — ver nota en la pestaña correspondiente de la app."""
    d = product_df[product_df["tipo_consumo"] == "MEDICAMENTO"].copy()
    d = d[d["total_cantidad"] > 0]
    d["costo_unitario"] = d["total_neto"] / d["total_cantidad"]
    d["alto_costo_9UIT"] = d["costo_unitario"] > d["umbral_9UIT"]
    return d.sort_values("total_neto", ascending=False)
