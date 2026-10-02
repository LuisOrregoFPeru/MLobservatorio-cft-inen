"""
trend_compare.py
----------------
¿Dos series tienen la misma tendencia o tendencias distintas?

Pruebas (todas sobre períodos con cobertura completa y en común):
  1. Diferencia de pendientes: modelo conjunto con interacción tiempo × serie.
       - Conteos  -> regresión de Poisson (log-lineal): la interacción es la razón
         de tasas de crecimiento; errores estándar robustos (cluster por período).
       - Otras medidas -> regresión lineal (OLS): la interacción es la diferencia
         de pendientes (unidades por año), errores robustos por período.
     Opcional: intervalo y p-valor por bootstrap de bloques móviles (robusto a
     autocorrelación).
  2. Proporción de A en A+B: regresión binomial en el tiempo (¿cambia el peso relativo?).
  3. Mann-Kendall / Sen de cada serie y de la proporción.
  4. Co-movimiento: correlación de Spearman entre primeras diferencias.

Un p-valor alto NO demuestra que las tendencias sean iguales: solo que los datos no
permiten distinguirlas. Por eso se reporta el intervalo de la razón de crecimiento anual.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from stats_analysis import mann_kendall_trend


@dataclass
class TrendCompare:
    model: str                       # 'poisson' | 'ols'
    n: int
    t_years: np.ndarray
    idx: pd.DatetimeIndex
    slope_a: float                   # log-pendiente (poisson) o pendiente por año (ols)
    slope_b: float
    se_a: float
    se_b: float
    diff: float                      # slope_b - slope_a
    diff_se: float
    diff_p: float
    diff_ci: tuple
    ratio: float | None              # exp(diff): razón de crecimiento anual de crecimiento B/A
    ratio_ci: tuple | None
    growth_a: float | None           # %/año (poisson)
    growth_b: float | None
    growth_a_ci: tuple | None
    growth_b_ci: tuple | None
    fitted_a: np.ndarray
    fitted_b: np.ndarray
    boot: dict | None = None
    mk_a: object = None
    mk_b: object = None
    share: dict | None = None
    comove: dict | None = None
    notes: list = field(default_factory=list)


def _design(t: np.ndarray, g: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"const": 1.0, "g": g, "t": t, "tg": t * g})


def _fit(y, t, g, pid, model, robust=True):
    X = _design(t, g)
    if model == "poisson":
        m = sm.GLM(y, X, family=sm.families.Poisson())
    else:
        m = sm.OLS(y, X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if robust:
            return m.fit(cov_type="cluster", cov_kwds={"groups": pid}, use_t=True) if model == "ols" \
                else m.fit(cov_type="cluster", cov_kwds={"groups": pid})
        return m.fit()


def _block_bootstrap(a: np.ndarray, b: np.ndarray, t: np.ndarray, model: str, n_boot: int, seed: int = 7) -> dict:
    n = len(a)
    L = max(2, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / L))
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        ix = np.concatenate([(np.arange(s, s + L) % n) for s in starts])[:n]
        y = np.concatenate([a[ix], b[ix]])
        tt = np.concatenate([t[ix], t[ix]])
        g = np.concatenate([np.zeros(n), np.ones(n)])
        try:
            r = _fit(y, tt, g, None, model, robust=False)
            diffs.append(float(r.params["tg"]))
        except Exception:  # noqa: BLE001
            continue
    d = np.array(diffs)
    d = d[np.isfinite(d)]
    if len(d) < 50:
        return {}
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"lo": lo, "hi": hi, "p": min(1.0, p), "L": L, "n_boot": len(d)}


def compare_trends(a: pd.Series, b: pd.Series, model: str = "auto", additive: bool = True,
                   boot: int = 0) -> TrendCompare | None:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 6:
        return None
    idx = j.index
    t = ((idx - idx[0]).days / 365.25).to_numpy(dtype=float)
    ya, yb = j["a"].to_numpy(float), j["b"].to_numpy(float)
    notes = []
    counts = additive and (ya >= 0).all() and (yb >= 0).all()
    if model == "auto":
        model = "poisson" if counts else "ols"
    if model == "poisson" and not counts:
        notes.append("Poisson requiere conteos no negativos; se usó regresión lineal.")
        model = "ols"
    y = np.concatenate([ya, yb])
    tt = np.concatenate([t, t])
    g = np.concatenate([np.zeros(len(t)), np.ones(len(t))])
    pid = np.concatenate([np.arange(len(t)), np.arange(len(t))])
    res = _fit(y, tt, g, pid, model)
    V = res.cov_params()
    p_ = res.params
    sa, sb = float(p_["t"]), float(p_["t"] + p_["tg"])
    se_a = float(np.sqrt(V.loc["t", "t"]))
    se_b = float(np.sqrt(V.loc["t", "t"] + V.loc["tg", "tg"] + 2 * V.loc["t", "tg"]))
    d, d_se = float(p_["tg"]), float(np.sqrt(V.loc["tg", "tg"]))
    d_p = float(res.pvalues["tg"])
    crit = stats.t.ppf(0.975, max(len(t) - 2, 1)) if model == "ols" else 1.96
    d_ci = (d - crit * d_se, d + crit * d_se)
    if model == "poisson":
        ratio, ratio_ci = float(np.exp(d)), (float(np.exp(d_ci[0])), float(np.exp(d_ci[1])))
        ga, gb = float(np.exp(sa) - 1) * 100, float(np.exp(sb) - 1) * 100
        ga_ci = tuple((np.exp(sa + np.array([-1, 1]) * 1.96 * se_a) - 1) * 100)
        gb_ci = tuple((np.exp(sb + np.array([-1, 1]) * 1.96 * se_b) - 1) * 100)
        fa = np.exp(p_["const"] + sa * t)
        fb = np.exp(p_["const"] + p_["g"] + sb * t)
    else:
        ratio = ratio_ci = ga = gb = ga_ci = gb_ci = None
        fa = p_["const"] + sa * t
        fb = p_["const"] + p_["g"] + sb * t
    out = TrendCompare(model=model, n=len(t), t_years=t, idx=idx, slope_a=sa, slope_b=sb, se_a=se_a, se_b=se_b,
                       diff=d, diff_se=d_se, diff_p=d_p, diff_ci=d_ci, ratio=ratio, ratio_ci=ratio_ci,
                       growth_a=ga, growth_b=gb, growth_a_ci=ga_ci, growth_b_ci=gb_ci,
                       fitted_a=np.asarray(fa), fitted_b=np.asarray(fb), notes=notes)
    if boot and boot >= 100:
        out.boot = _block_bootstrap(ya, yb, t, model, boot)
    out.mk_a = mann_kendall_trend(ya) if len(ya) >= 8 else None
    out.mk_b = mann_kendall_trend(yb) if len(yb) >= 8 else None
    # proporción de A en A+B
    if counts:
        tot = ya + yb
        ok = tot > 0
        if ok.sum() >= 8:
            sh = ya[ok] / tot[ok]
            X = sm.add_constant(t[ok])
            L = max(1, int(4 * (ok.sum() / 100) ** (2 / 9)))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rb = sm.GLM(np.column_stack([ya[ok], yb[ok]]), X, family=sm.families.Binomial()).fit(
                        cov_type="HAC", cov_kwds={"maxlags": L})
                bt, bse = float(rb.params[1]), float(rb.bse[1])
                fit_sh = rb.predict(X)
                out.share = {"or_year": float(np.exp(bt)), "or_ci": (float(np.exp(bt - 1.96 * bse)), float(np.exp(bt + 1.96 * bse))),
                             "p": float(rb.pvalues[1]), "start": float(fit_sh[0]), "end": float(fit_sh[-1]),
                             "obs_start": float(sh[: max(3, len(sh) // 4)].mean()), "obs_end": float(sh[-max(3, len(sh) // 4):].mean()),
                             "mk": mann_kendall_trend(sh) if len(sh) >= 8 else None, "idx": idx[ok], "share": sh, "fit": np.asarray(fit_sh)}
            except Exception:  # noqa: BLE001
                out.share = None
    # co-movimiento
    da, db = np.diff(ya), np.diff(yb)
    if len(da) >= 8 and np.std(da) > 0 and np.std(db) > 0:
        r, p = stats.spearmanr(da, db)
        out.comove = {"rho": float(r), "p": float(p), "n": len(da)}
    return out


def verdict(tc: TrendCompare, name_a: str, name_b: str, alpha: float = 0.05) -> tuple[str, str]:
    """(nivel, texto) — nivel: 'diff' | 'same' | 'weak'."""
    if tc.model == "poisson":
        head = (f"**{name_a}**: {tc.growth_a:+.1f}%/año · **{name_b}**: {tc.growth_b:+.1f}%/año · razón de crecimiento anual "
                f"({name_b}/{name_a}) = **{tc.ratio:.2f}** (IC95% {tc.ratio_ci[0]:.2f}–{tc.ratio_ci[1]:.2f}), p={tc.diff_p:.3f}.")
    else:
        head = (f"Pendiente **{name_a}**: {tc.slope_a:+.2f}/año · **{name_b}**: {tc.slope_b:+.2f}/año · diferencia = "
                f"**{tc.diff:+.2f}** (IC95% {tc.diff_ci[0]:+.2f} a {tc.diff_ci[1]:+.2f}), p={tc.diff_p:.3f}.")
    if tc.diff_p < alpha:
        return "diff", head + f" Las tendencias son **estadísticamente DIFERENTES** (α={alpha})."
    if tc.n < 24:
        return "weak", head + " No se detecta diferencia, pero con pocas observaciones la prueba tiene poca potencia."
    return "same", head + (" **No hay evidencia de diferencia**; esto no prueba igualdad: revisa si el intervalo excluye "
                           "diferencias que consideres relevantes.")
