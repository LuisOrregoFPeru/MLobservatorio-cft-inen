"""
Observatorio del Comité Farmacoterapéutico (CFT) — INEN
=======================================================
Crecimiento de los requerimientos (evaluaciones) y aprobaciones del CFT desde
2021, con subanálisis de medicamentos de alto costo (listado no PNUME).
Arquitectura y herramientas estadísticas tomadas del Observatorio del
Financiamiento Oncológico INEN (series mensuales, eventos normativos,
Mann-Kendall, Chow, proyección, comparación con series externas).
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as sps

import data_processing as dp
from trend_compare import compare_trends, verdict
from stats_analysis import (
    cagr, chow_test_event, detect_breakpoint_auto, linear_trend,
    mann_kendall_trend, project_series, smooth_series,
)

# Compatibilidad: width="stretch" solo existe en Streamlit >= 1.50; antes se usaba use_container_width.
def _st_version() -> tuple:
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001
        return (0, 0)


STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Observatorio del CFT — INEN", page_icon="💊", layout="wide",
                   initial_sidebar_state="expanded")

PRIMARY, SECONDARY, ACCENT = "#2C5F8A", "#D9542F", "#1F4E78"
PALETTES = {
    "INEN (azul/naranja)": [PRIMARY, SECONDARY, "#2F9E44", "#8850C4", "#C9A13B", "#5C6AC4", "#7D7D7D", "#4AA8C9",
                             "#B5495B", "#3B8686", "#A67C52", "#6C8EAD"],
    "Daltonismo (Okabe-Ito)": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"],
    "Vivos (Bold)": px.colors.qualitative.Bold,
    "Suaves (Set2)": px.colors.qualitative.Set2,
}
st.markdown(f"""
<style>
 .main .block-container {{ padding-top: 1.2rem; max-width: 1400px; }}
 h1,h2,h3 {{ color:{ACCENT}; }}
 .app-header {{ background: linear-gradient(90deg,{ACCENT} 0%,{PRIMARY} 100%); padding:1.1rem 1.6rem;
    border-radius:10px; color:white; margin-bottom:1rem; display:flex; align-items:center; gap:1rem; }}
 .app-header .ico {{ font-size:2.6rem; }}
 .app-header h1 {{ color:white; margin:0; font-size:1.55rem; font-weight:800; text-transform:uppercase; line-height:1.15; }}
 .app-header p {{ color:#e8eef5; margin:.2rem 0 0 0; font-size:.93rem; }}
 div[data-testid="stMetric"] {{ background:#f7f8fb; border:1px solid #e2e6ee; border-radius:10px; padding:.6rem .9rem; }}
 .interp-box {{ background:#f4f7fb; border-left:4px solid {PRIMARY}; border-radius:6px; padding:.7rem 1rem;
    margin:.6rem 0 1rem 0; font-size:.92rem; color:#333; }}
 .warn-box {{ background:#fff8ec; border-left:4px solid #E8A33D; border-radius:6px; padding:.6rem 1rem;
    margin:.5rem 0 1rem 0; font-size:.9rem; color:#5a4200; }}
 .mini-title {{ font-size:.95rem; font-weight:600; color:{ACCENT}; margin-bottom:.2rem; }}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Estado de datos
# ---------------------------------------------------------------------------
ss = st.session_state
if "alias" not in ss:
    ss["alias"] = dp.load_alias()
if "hc" not in ss:
    ss["hc"] = dp.load_hc()
if "cft" not in ss:
    ss["cft"] = dp.reapply_keys(dp.load_cft(), ss["alias"])
if "events_custom" not in ss:
    ss["events_custom"] = pd.DataFrame({"fecha": pd.Series(dtype="datetime64[ns]"), "etiqueta": pd.Series(dtype="str")})

CFT: pd.DataFrame = ss["cft"]
HC: pd.DataFrame = ss["hc"]
ALIAS: pd.DataFrame = ss["alias"]
NAMES = dp.key_names(ALIAS)
EVENTS = dp.load_events()
DATA_SIG = f"{len(CFT)}_{CFT['fecha'].max():%Y%m%d}"

MIN_D, MAX_D = CFT["fecha"].min().date(), CFT["fecha"].max().date()
ALL_MONTHS = pd.date_range(CFT["fecha"].min().to_period("M").to_timestamp(),
                           CFT["fecha"].max().to_period("M").to_timestamp(), freq="MS")
DEFAULT_INC = dp.default_incomplete_months(CFT)


def fmt_int(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"


def measure_unit(m: str) -> str:
    return {"pct_aprob": "% de aprobación"}.get(m, "N.º")


# ---------------------------------------------------------------------------
# Panel de control
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Panel de control")

with st.sidebar.expander("📐 Serie y medida", expanded=True):
    freq_label = st.selectbox("Granularidad", list(dp.FREQ_LABELS), index=0, key="freq_label")
    FREQ = dp.FREQ_LABELS[freq_label]
    measure_label = st.selectbox("Medida", list(dp.MEASURES), index=0, key="measure_label")
    MEASURE = dp.MEASURES[measure_label]
    chart_type = st.radio("Tipo de gráfico", ["Línea", "Barras", "Área"], horizontal=True, key="chart_type")
    transform = st.selectbox("Transformación", ["Valor", "Acumulado", "Índice (base 100)", "Variación interanual (%)"],
                             key="transform",
                             help="Acumulado solo tiene sentido para medidas aditivas. Índice: primer período válido = 100.")
    ma_win = st.slider("Media móvil (períodos; 0 = sin)", 0, 12, 3, key="ma_win")
    lowess_on = st.checkbox("Suavizado LOWESS", value=False, key="lowess_on")
    lowess_frac = st.slider("Ancho LOWESS", 0.15, 0.9, 0.35, 0.05, key="lowess_frac") if lowess_on else 0.35
    ytd_k = st.slider("Mes de corte para comparaciones anuales (ene–…)", 1, 12, 8, key="ytd_k",
                      help="Compara enero hasta este mes en todos los años, para no comparar años incompletos con años completos.")
    YRS_AVAIL = sorted(int(y) for y in CFT["fecha"].dt.year.unique())
    growth_basis = st.selectbox("Base para medir el crecimiento (indicadores superiores)",
                                ["Año completo (total del año)", "Enero–mes de corte", "Rango de meses"], key="growth_basis",
                                help="«Año completo» compara totales anuales y estima el año incompleto según la estacionalidad de los años completos. "
                                     "Elige otro rango si un pico estacional cae fuera de la ventana.")
    g1, g2 = st.columns(2)
    yr_a = g1.selectbox("Año inicial", YRS_AVAIL, index=0, key="yr_a")
    yr_b = g2.selectbox("Año final", YRS_AVAIL, index=len(YRS_AVAIL) - 1, key="yr_b")
    if growth_basis == "Rango de meses":
        m_range = st.select_slider("Meses (desde–hasta)", options=list(range(1, 13)), value=(1, 12),
                                   format_func=lambda m: dp.MONTHS_ES[m - 1], key="m_range")
    else:
        m_range = (1, ytd_k)

with st.sidebar.expander("🧩 Desglose y diseño", expanded=True):
    breakdown = st.selectbox("Desglosar por", [
        "Sin desglose", "Decisión (aprobado/denegado)", "Grupo de área", "Área solicitante",
        "Alto costo vs resto", "Oncológico (según CFT, 2025+)", "Principio activo (Top N)",
        "Fármaco de alto costo (Top N)"], key="breakdown")
    top_n = st.slider("Top N (áreas / principios activos / fármacos)", 3, 15, 6, key="top_n")
    layout_mode = st.radio("Diseño", ["Un solo gráfico", "Gráficos separados"], horizontal=True, key="layout_mode")
    n_cols = st.slider("Columnas (gráficos separados)", 1, 3, 2, key="n_cols") if layout_mode == "Gráficos separados" else 2
    shared_y = st.checkbox("Mismo eje Y en todos los paneles", value=False, key="shared_y") if layout_mode == "Gráficos separados" else False
    stacked = st.checkbox("Apilar (barras/áreas)", value=False, key="stacked")
    show_total = st.checkbox("Añadir línea de total (un solo gráfico)", value=False, key="show_total")

cmp_box = st.sidebar.expander("🧪 Comparar tendencias (2 series)", expanded=True)

with st.sidebar.expander("🔎 Filtros", expanded=False):
    d_range = st.date_input("Rango de fechas", value=(MIN_D, MAX_D), min_value=MIN_D, max_value=MAX_D, key="d_range")
    dec_sel = st.multiselect("Decisión", ["APROBADO", "DENEGADO"], default=["APROBADO", "DENEGADO"], key="dec_sel")
    grp_sel = st.multiselect("Grupo de área", dp.GRUPOS_AREA, default=dp.GRUPOS_AREA, key="grp_sel")
    areas_all = sorted(CFT["area_canon"].unique())
    area_sel = st.multiselect("Área solicitante (vacío = todas)", areas_all, default=[], key="area_sel")
    hc_mode = st.radio("Medicamentos de alto costo", ["Todos", "Solo alto costo (listado)", "Excluir alto costo"], key="hc_mode")
    txt_q = st.text_input("Buscar en DCI (contiene)", "", key="txt_q")

with st.sidebar.expander("🎨 Estilo", expanded=False):
    palette_name = st.selectbox("Paleta", list(PALETTES), key="palette")
    PAL = PALETTES[palette_name]
    height = st.slider("Alto del gráfico (px)", 320, 900, 460, 20, key="height")
    markers = st.checkbox("Mostrar marcadores", value=True, key="markers")
    log_y = st.checkbox("Escala logarítmica en Y", value=False, key="log_y")
    labels_on = st.checkbox("Etiquetas de valor (barras)", value=False, key="labels_on")

with st.sidebar.expander("📌 Eventos / puntos de corte", expanded=False):
    ev_on = st.checkbox("Mostrar eventos en los gráficos", value=True, key="ev_on")
    ev_opts = [f"{r.letra} — {r.etiqueta}" for r in EVENTS.itertuples()]
    ev_sel = st.multiselect("Eventos normativos precargados", ev_opts, default=ev_opts, key="ev_sel",
                            help="Vigencia efectiva de las normas del Observatorio del Financiamiento Oncológico.")
    st.caption("Agrega tus propios eventos (fecha y etiqueta):")
    ss["events_custom"] = st.data_editor(
        ss["events_custom"], num_rows="dynamic", key="ev_editor", hide_index=True,
        column_config={"fecha": st.column_config.DateColumn("Fecha"), "etiqueta": st.column_config.TextColumn("Etiqueta")})

with st.sidebar.expander("🕳️ Cobertura de datos", expanded=False):
    inc_labels_all = [f"{t:%Y-%m}" for t in ALL_MONTHS]
    inc_default = [f"{t:%Y-%m}" for t in DEFAULT_INC if f"{t:%Y-%m}" in inc_labels_all]
    inc_sel = st.multiselect("Meses sin cobertura completa", inc_labels_all, default=inc_default, key=f"inc_{DATA_SIG}",
                             help="Meses sin registros o con registro parcial (hoja actualizada solo hasta cierta fecha).")
    gap_mode = st.radio("Períodos incompletos", ["Marcar (banda gris)", "Ocultar", "Mostrar normal"], key="gap_mode")
INC_MONTHS = [pd.Timestamp(f"{s}-01") for s in inc_sel]

# ---------------------------------------------------------------------------
# Filtrado global
# ---------------------------------------------------------------------------
lo_d, hi_d = (d_range if isinstance(d_range, (list, tuple)) and len(d_range) == 2 else (MIN_D, MAX_D))
D0 = CFT[(CFT["fecha"].dt.date >= lo_d) & (CFT["fecha"].dt.date <= hi_d)]
D0 = D0[D0["decision"].isin(dec_sel) & D0["grupo_area"].isin(grp_sel)]
if area_sel:
    D0 = D0[D0["area_canon"].isin(area_sel)]
if txt_q.strip():
    D0 = D0[D0["dci"].str.contains(txt_q.strip(), case=False, na=False)]
DF_BASE = D0                      # sin filtro de alto costo (lo usa la pestaña de alto costo)
if hc_mode == "Solo alto costo (listado)":
    DF = D0[D0["alto_costo_clave"].notna()]
elif hc_mode == "Excluir alto costo":
    DF = D0[D0["alto_costo_clave"].isna()]
else:
    DF = D0

# Eventos activos
ev_rows = [(r.letra, r.fecha, r.etiqueta) for r in EVENTS.itertuples() if f"{r.letra} — {r.etiqueta}" in ev_sel]
cust = ss["events_custom"].dropna(subset=["fecha"]) if len(ss["events_custom"]) else ss["events_custom"]
for i, r in enumerate(cust.itertuples()):
    ev_rows.append((f"X{i + 1}", pd.Timestamp(r.fecha), str(r.etiqueta) if isinstance(r.etiqueta, str) else "Evento"))
ACTIVE_EVENTS = ev_rows if ev_on else []

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown(f"""<div class="app-header"><div class="ico">💊</div><div>
<h1>Observatorio del Comité Farmacoterapéutico — INEN</h1>
<p><i>INEN Pharmacy &amp; Therapeutics Committee Observatory</i></p>
<p>Requerimientos (evaluaciones) y aprobaciones del CFT · {CFT['fecha'].min():%b-%Y} a {CFT['fecha'].max():%b-%Y} ·
{len(CFT):,} evaluaciones · {HC['clave'].nunique()} principios activos en el listado no PNUME de alto costo</p></div></div>""",
            unsafe_allow_html=True)
st.caption(f"**Selección actual:** {len(DF):,} de {len(CFT):,} evaluaciones · {freq_label.lower()} · {measure_label} · "
           f"{lo_d:%d-%m-%Y} a {hi_d:%d-%m-%Y}")
if DF.empty:
    st.warning("Los filtros no dejan ninguna evaluación. Ajusta el panel lateral.")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers de series y gráficos
# ---------------------------------------------------------------------------
def make_groups(df: pd.DataFrame, choice: str, n: int) -> tuple[pd.DataFrame, str | None, list | None]:
    """Devuelve (df con columna _grp, nombre_col, lista ordenada de grupos)."""
    d = df.copy()
    if choice == "Sin desglose":
        return d, None, None
    if choice.startswith("Decisión"):
        d["_grp"] = d["decision"]
        return d, "_grp", [g for g in ["APROBADO", "DENEGADO"] if g in set(d["_grp"])]
    if choice == "Grupo de área":
        d["_grp"] = d["grupo_area"]
        return d, "_grp", [g for g in dp.GRUPOS_AREA if g in set(d["_grp"])]
    if choice == "Alto costo vs resto":
        d["_grp"] = np.where(d["alto_costo_clave"].notna(), "Alto costo (listado)", "Resto")
        return d, "_grp", ["Alto costo (listado)", "Resto"]
    if choice.startswith("Oncológico"):
        d["_grp"] = d["oncologico_reportado"].fillna("Sin dato (hasta 2024)")
        order = ["ONCOLÓGICO", "NO ONCOLÓGICO", "Sin dato (hasta 2024)"]
        return d, "_grp", [g for g in order if g in set(d["_grp"])]
    if choice == "Área solicitante":
        col = "area_canon"
    elif choice.startswith("Principio activo"):
        col = "dci_base"
    else:  # fármaco de alto costo
        d = d[d["alto_costo_clave"].notna()].copy()
        d["_lab"] = d["alto_costo_clave"].map(lambda k: NAMES.get(k, k.title()))
        col = "_lab"
    vc = d[col].value_counts()
    top = list(vc.head(n).index)
    lab = (lambda s: s.title()) if col == "dci_base" else (lambda s: s)
    d["_grp"] = np.where(d[col].isin(top), d[col].map(lab), "Otros")
    order = [lab(t) for t in top] + (["Otros"] if (d["_grp"] == "Otros").any() else [])
    return d, "_grp", order


def apply_transform(w: pd.DataFrame, kind: str, ppy: int, additive: bool) -> tuple[pd.DataFrame, str | None]:
    if kind == "Valor":
        return w, None
    if kind == "Acumulado":
        if not additive:
            return w, "El acumulado solo aplica a medidas aditivas (conteos); se muestra el valor original."
        return w.fillna(0).cumsum(), None
    if kind == "Índice (base 100)":
        out = w.copy()
        for c in out.columns:
            s = out[c]
            base = s[(s.notna()) & (s > 0)]
            out[c] = s / base.iloc[0] * 100 if len(base) else np.nan
        return out, None
    if kind == "Variación interanual (%)":
        prev = w.shift(ppy)
        out = (w / prev.replace(0, np.nan) - 1) * 100
        return out.replace([np.inf, -np.inf], np.nan), None
    return w, None


def hide_incomplete(w: pd.DataFrame, flags: pd.Series, mode: str, freq: str) -> pd.DataFrame:
    out = w.copy()
    if mode == "Ocultar":
        out.loc[flags.values, :] = np.nan
    elif mode.startswith("Marcar") and freq == "M":
        # en meses sin cobertura, un 0 no es un valor real
        mask = flags.values[:, None] & (out.values == 0)
        out = out.mask(mask)
    return out


def _axis_ids(idx: int) -> tuple[str, str]:
    return ("x" if idx == 1 else f"x{idx}"), ("y" if idx == 1 else f"y{idx}")


def decorate(fig: go.Figure, x0, x1, freq: str, flags: pd.Series | None, mode: str, n_axes: int = 1) -> None:
    """Bandas de cobertura incompleta y líneas de eventos en cada eje."""
    step = {"M": pd.DateOffset(months=1), "Q": pd.DateOffset(months=3), "S": pd.DateOffset(months=6),
            "A": pd.DateOffset(years=1)}[freq]
    for ai in range(1, n_axes + 1):
        xa, ya = _axis_ids(ai)
        if flags is not None and mode.startswith("Marcar"):
            for t in flags.index[flags.values]:
                fig.add_shape(type="rect", xref=xa, yref=f"{ya} domain", x0=t, x1=t + step, y0=0, y1=1,
                              fillcolor="rgba(120,120,120,0.16)", line_width=0, layer="below")
        for letra, fecha, etiqueta in ACTIVE_EVENTS:
            if x0 <= fecha <= x1 + step:
                fig.add_shape(type="line", xref=xa, yref=f"{ya} domain", x0=fecha, x1=fecha, y0=0, y1=1,
                              line=dict(color="#666", width=1, dash="dot"))
                fig.add_annotation(xref=xa, yref=f"{ya} domain", x=fecha, y=0.99, text=f"<b>{letra}</b>",
                                   showarrow=False, yanchor="top", xanchor="left", xshift=2, font=dict(size=10, color="#444"))


def add_series_traces(fig, x, name, y, color, ctype, ma_y, lowess_y, row=None, col=None, legend=True,
                      stack=False, ma_name="", show_markers=True, text_labels=False, dash=None, width=None):
    kw = dict(row=row, col=col) if row else {}
    has_ma = ma_y is not None
    if ctype == "Barras":
        fig.add_trace(go.Bar(x=x, y=y, name=name, marker_color=color, showlegend=legend, legendgroup=name,
                             opacity=0.55 if has_ma else 0.9,
                             text=[f"{v:,.0f}" if pd.notna(v) else "" for v in y] if text_labels else None,
                             textposition="outside" if (text_labels and not stack) else "auto"), **kw)
    elif ctype == "Área":
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=1.5),
                                 stackgroup="one" if stack else None, fill="tonexty" if stack else "tozeroy",
                                 fillcolor=color.replace(")", ",0.35)").replace("rgb", "rgba") if color.startswith("rgb") else None,
                                 showlegend=legend, legendgroup=name, connectgaps=False), **kw)
    else:
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers" if show_markers else "lines",
                                 line=dict(color=color, width=width or (1.4 if has_ma else 2.4), dash=dash),
                                 marker=dict(size=5), opacity=0.5 if has_ma else 1, showlegend=legend,
                                 legendgroup=name, connectgaps=False), **kw)
    if has_ma:
        fig.add_trace(go.Scatter(x=x, y=ma_y, name=f"{name} ({ma_name})", mode="lines",
                                 line=dict(color=color, width=3.2), showlegend=legend,
                                 legendgroup=name, connectgaps=False), **kw)
    if lowess_y is not None:
        fig.add_trace(go.Scatter(x=x, y=lowess_y, name=f"{name} (LOWESS)", mode="lines",
                                 line=dict(color=color, width=2.4, dash="dash"), showlegend=legend,
                                 legendgroup=name), **kw)


def lowess_col(s: pd.Series, frac: float) -> pd.Series | None:
    v = s.dropna()
    if len(v) < 8:
        return None
    sm = smooth_series(v.values.astype(float), frac=frac)
    if sm is None:
        return None
    return pd.Series(sm, index=v.index).reindex(s.index)


def style_fig(fig, h, ytitle, log=False, title=None):
    fig.update_layout(template="plotly_white", height=h, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50 if title else 30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), yaxis_title=ytitle)
    if title:
        fig.update_layout(title=title)
    if log:
        fig.update_yaxes(type="log")
    fig.update_xaxes(showgrid=False)


def box(txt: str, cls: str = "interp-box") -> None:
    """Caja de texto con negritas **así** convertidas a HTML."""
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


def note(txt: str) -> None:
    """Interpretación breve que acompaña a cada gráfico o tabla."""
    box("<b>Cómo leerlo.</b> " + txt)


def _pct(a, b) -> str:
    try:
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a == 0:
            return "n/d"
    except TypeError:
        return "n/d"
    return f"{(b / a - 1) * 100:+.0f}%"


def half_change(v: pd.Series) -> str:
    """Compara la media de la primera y de la segunda mitad de una serie."""
    v = v.dropna()
    if len(v) < 6:
        return ""
    h = len(v) // 2
    a, b = v.iloc[:h].mean(), v.iloc[h:].mean()
    return f" La media de la segunda mitad del período ({b:,.1f}) es {_pct(a, b)} respecto de la primera ({a:,.1f})."


def render_events_legend():
    if ACTIVE_EVENTS:
        st.caption("**Eventos:** " + " · ".join(f"**{l}** {e} ({f:%b-%Y})" for l, f, e in ACTIVE_EVENTS))
    st.caption("Sombreado gris = período sin cobertura completa de datos (ver 'Cobertura de datos').")


def series_kpis(w_total: pd.Series) -> dict:
    v = w_total.dropna()
    out = {"pico": (None, None), "mk": None}
    if len(v):
        out["pico"] = (v.idxmax(), v.max())
    if len(v) >= 8:
        out["mk"] = mann_kendall_trend(v.values.astype(float))
    return out


def growth_line(color_pos="#2F9E44"):
    return color_pos


def build_wide(df: pd.DataFrame, freq: str, measure: str, group_choice: str, n: int, lo=None, hi=None):
    d, gcol, groups = make_groups(df, group_choice, n)
    if d.empty:
        return pd.DataFrame(), None, None, d
    w = dp.aggregate(d, freq, measure, gcol, lo=lo, hi=hi, groups=groups)
    return w, gcol, groups, d


# ---------------------------------------------------------------------------
# Series principales y KPIs
# ---------------------------------------------------------------------------
ADD = MEASURE in dp.ADDITIVE
PPY = dp.PERIODS_PER_YEAR[FREQ]
W, GCOL, GROUPS, DG = build_wide(DF, FREQ, MEASURE, breakdown, top_n, lo=DF["fecha"].min(), hi=DF["fecha"].max())
WT_raw = dp.aggregate(DF, FREQ, MEASURE)
FLAGS = dp.incomplete_flags(WT_raw.index, FREQ, INC_MONTHS)
if not W.empty and len(W.index) != len(WT_raw.index):
    W = W.reindex(WT_raw.index)

# --- Panel: comparación de tendencias ------------------------------------------------
groups_cmp = [c for c in W.columns if c != "Total"] if not W.empty else []
CMP_ON, CMP = False, {}
with cmp_box:
    if len(groups_cmp) < 2:
        st.caption("Elige un desglose con al menos 2 series (p. ej. «Alto costo vs resto» o «Decisión») para comparar sus tendencias.")
    else:
        CMP_ON = st.checkbox("Activar comparación", value=True, key="cmp_on")
        ca = st.selectbox("Serie A", groups_cmp, index=0, key=f"cmp_a_{breakdown}_{top_n}")
        cb = st.selectbox("Serie B", groups_cmp, index=1, key=f"cmp_b_{breakdown}_{top_n}")
        cmp_model = st.radio("Modelo", ["Automático", "Poisson (conteos)", "Lineal (OLS)"], key="cmp_model",
                             help="Automático: Poisson si la medida es un conteo, lineal en los demás casos.")
        cmp_alpha = st.selectbox("Nivel de significancia α", [0.05, 0.10, 0.01], key="cmp_alpha")
        cmp_scale = st.radio("Escala del gráfico", ["Original", "Índice base 100"], key="cmp_scale",
                             help="Índice: 100 = promedio de los primeros 12 períodos válidos de cada serie.")
        cmp_boot = st.checkbox("Intervalo bootstrap por bloques (más lento)", value=False, key="cmp_boot")
        CMP = dict(a=ca, b=cb, model=cmp_model, alpha=cmp_alpha, scale=cmp_scale, boot=cmp_boot)

YT = dp.ytd_table(DF, ytd_k, MEASURE, INC_MONTHS)
LAST_D = CFT["fecha"].max()
PIV = dp.monthly_pivot(DF, MEASURE) if ADD else None
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Evaluaciones (filtro)", f"{len(DF):,}", f"{(DF['decision'] == 'APROBADO').sum():,} aprobadas", delta_color="off")
k2.metric("% aprobación", f"{(DF['decision'] == 'APROBADO').mean() * 100:.1f}%")
mes_txt = dp.MONTHS_ES[ytd_k - 1]
ya_, yb_ = int(yr_a), int(yr_b)
kpi_warn = []
is_pct = MEASURE == "pct_aprob"


def _delta_txt(va, vb):
    if is_pct:
        return f"{va:.1f}% → {vb:.1f}%"
    pct = (vb / va - 1) * 100 if va and va > 0 else float("nan")
    return f"{pct:+.1f}% · {va:,.0f} → {vb:,.0f}"


if ya_ == yb_:
    k3.metric("Comparación de años", "Elige años distintos")
    k4.metric("—", "n/d")
    k5.metric("TCAC", "n/d")
elif growth_basis.startswith("Año completo"):
    ann = dp.aggregate(DF, "A", MEASURE)["Total"]
    ann.index = ann.index.year
    va, vb = float(ann.get(ya_, np.nan)), float(ann.get(yb_, np.nan))
    inc_years = {t.year for t in INC_MONTHS}
    for y_ in (ya_, yb_):
        if y_ in inc_years:
            kpi_warn.append(y_)
    diff = vb - va
    k3.metric(f"{'Valor' if is_pct else 'Total'} {ya_}→{yb_} (observado){' ⚠' if kpi_warn else ''}",
              f"{diff:+.1f} p.p." if is_pct else f"{diff:+,.0f}", _delta_txt(va, vb), delta_color="off",
              help="Total registrado en cada año. Un año con cobertura incompleta subestima su total: ver el indicador de al lado.")
    tc_val = cagr(va, vb, abs(yb_ - ya_)) if (ADD and va > 0 and vb > 0) else None
    if ADD:
        ea = dp.estimate_year(PIV, ya_, INC_MONTHS, LAST_D)
        eb = dp.estimate_year(PIV, yb_, INC_MONTHS, LAST_D)
        if ea and eb and ea["est"] > 0:
            pm = (eb["est"] / ea["est"] - 1) * 100
            plo, phi = (eb["lo"] / ea["hi"] - 1) * 100, (eb["hi"] / ea["lo"] - 1) * 100
            k4.metric(f"Año completo estimado {ya_}→{yb_}", f"{eb['est'] - ea['est']:+,.0f}",
                      f"{pm:+.0f}% (rango {plo:+.0f}% a {phi:+.0f}%) · {ea['est']:,.0f} → {eb['est']:,.0f}", delta_color="off",
                      help="Total del año = lo observado en los meses con cobertura completa ÷ fracción del año que esos meses suelen "
                           "representar en los años completos (media, y mín./máx. para el rango).")
            tc_val = cagr(ea["est"], eb["est"], abs(yb_ - ya_))
        else:
            k4.metric(f"Año completo estimado {ya_}→{yb_}", "n/d", "requiere ≥ 2 años completos", delta_color="off")
    else:
        k4.metric("Año completo estimado", "n/d", "solo para conteos", delta_color="off")
    k5.metric("TCAC (año completo, est.)", f"{tc_val * 100:+.1f}%/año" if tc_val is not None else "n/d")
else:
    m_from, m_to = m_range
    RT = dp.range_table(DF, m_from, m_to, MEASURE, INC_MONTHS).set_index("anio")
    va, vb = float(RT.loc[ya_, "valor"]), float(RT.loc[yb_, "valor"])
    kpi_warn = [y_ for y_ in (ya_, yb_) if bool(RT.loc[y_, "incompleto"])]
    lab = f"{dp.MONTHS_ES[m_from - 1]}–{dp.MONTHS_ES[m_to - 1]}"
    diff = vb - va
    k3.metric(f"{lab} {ya_}→{yb_}{' ⚠' if kpi_warn else ''}", f"{diff:+.1f} p.p." if is_pct else f"{diff:+,.0f}", _delta_txt(va, vb), delta_color="off")
    if ADD:
        comp = dp.complete_years(PIV, INC_MONTHS, LAST_D)
        if len(comp) >= 2:
            sh = dp.seasonal_shares(PIV, comp)[list(range(m_from, m_to + 1))].sum(axis=1) * 100
            k4.metric("Peso de este rango en el año", f"{sh.mean():.0f}%", f"rango {sh.min():.0f}–{sh.max():.0f}% (años completos)", delta_color="off",
                      help="Qué parte del total anual suele caer en estos meses. Si es baja, la comparación omite gran parte del año.")
        else:
            k4.metric("Peso de este rango en el año", "n/d", "requiere ≥ 2 años completos", delta_color="off")
    else:
        k4.metric("Peso de este rango", "n/d", "solo para conteos", delta_color="off")
    tc_val = cagr(va, vb, abs(yb_ - ya_)) if (ADD and va > 0 and vb > 0) else None
    k5.metric(f"TCAC ({lab})", f"{tc_val * 100:+.1f}%/año" if tc_val is not None else "n/d")
SK = series_kpis(WT_raw["Total"].mask(FLAGS.values) if gap_mode != "Mostrar normal" else WT_raw["Total"])
mk = SK["mk"]
k6.metric("Tendencia (Mann-Kendall)", mk.trend if mk else "n/d", f"p={mk.p_value:.3f}" if mk else "muestra < 8", delta_color="off")
if kpi_warn:
    st.caption("⚠ Cobertura incompleta en " + " y ".join(str(y) for y in kpi_warn)
               + ": el valor observado de ese año subestima su total (2021 sin diciembre; 2022 hasta el 10-sep; 2026 hasta el 2-sep).")

tabs = st.tabs(["📈 Serie temporal", "📅 Comparación anual", "💊 Alto costo", "✂️ Punto de corte", "🌿 Tendencia",
                "🔮 Proyección", "🔬 Qué explica el crecimiento", "🔀 Serie externa", "📋 Tabla y descarga", "🗂️ Datos"])
(t_series, t_year, t_hc, t_break, t_trend, t_proj, t_rank, t_ext, t_table, t_data) = tabs

# ============================== 1. SERIE TEMPORAL ============================
with t_series:
    st.subheader(f"{measure_label} — {freq_label.lower()}")
    if W.empty:
        st.info("Sin datos para este desglose.")
    else:
        Wd = hide_incomplete(W, FLAGS, gap_mode, FREQ)
        if show_total and breakdown != "Sin desglose" and ADD:
            Wd = Wd.copy()
            Wd["Total"] = hide_incomplete(WT_raw, FLAGS, gap_mode, FREQ)["Total"]
        Wt, warn = apply_transform(Wd, transform, PPY, ADD)
        if warn:
            st.info(warn)
        ma_eff = 0 if FREQ == "A" else ma_win      # una media móvil sobre 6 años no aporta
        if FREQ == "A" and ma_win >= 2:
            st.caption("La media móvil no se aplica a la granularidad anual.")
        Wma = Wt.rolling(ma_eff, min_periods=ma_eff).mean() if ma_eff >= 2 else None
        ytitle = {"Valor": measure_unit(MEASURE), "Acumulado": "Acumulado", "Índice (base 100)": "Índice (base 100)",
                  "Variación interanual (%)": "% vs mismo período del año previo"}[transform]
        cols = list(Wt.columns)
        ma_name = f"MM{ma_win}"
        if layout_mode == "Un solo gráfico" or len(cols) == 1:
            fig = go.Figure()
            if chart_type == "Barras":
                fig.update_layout(barmode="stack" if (stacked and ADD and transform in ("Valor", "Acumulado")) else "group")
            for i, c in enumerate(cols):
                is_tot = (c == "Total" and breakdown != "Sin desglose")
                color = "#222" if is_tot else PAL[i % len(PAL)]
                add_series_traces(fig, Wt.index, c, Wt[c], color, "Línea" if is_tot else chart_type,
                                  None if is_tot else (Wma[c] if Wma is not None else None),
                                  lowess_col(Wt[c], lowess_frac) if lowess_on and not is_tot else None,
                                  stack=stacked and ADD, ma_name=ma_name, show_markers=markers,
                                  text_labels=labels_on, dash="dash" if is_tot else None, width=2.2 if is_tot else None)
            decorate(fig, W.index.min(), W.index.max(), FREQ, FLAGS, gap_mode)
            style_fig(fig, height, ytitle, log_y)
        else:
            rows = -(-len(cols) // n_cols)
            fig = make_subplots(rows=rows, cols=n_cols, subplot_titles=cols, shared_yaxes=shared_y,
                                vertical_spacing=0.16 / max(rows, 1) * 2, horizontal_spacing=0.06)
            for i, c in enumerate(cols):
                r, cc = divmod(i, n_cols)
                add_series_traces(fig, Wt.index, c, Wt[c], PAL[i % len(PAL)], chart_type,
                                  Wma[c] if Wma is not None else None,
                                  lowess_col(Wt[c], lowess_frac) if lowess_on else None,
                                  row=r + 1, col=cc + 1, legend=False, ma_name=ma_name, show_markers=markers,
                                  text_labels=labels_on)
            decorate(fig, W.index.min(), W.index.max(), FREQ, FLAGS, gap_mode, n_axes=len(cols))
            style_fig(fig, max(height, 260 * rows), ytitle, log_y)
            fig.update_layout(showlegend=False)
        st.plotly_chart(fig, **STRETCH)
        ss["main_fig"] = fig
        render_events_legend()
        # lectura automática
        tot = WT_raw["Total"].mask(FLAGS.values) if gap_mode != "Mostrar normal" else WT_raw["Total"]
        v = tot.dropna()
        if len(v) >= 2:
            txt = (f"En **{freq_label.lower()}** el valor medio es **{v.mean():,.1f}** por período (mín. {v.min():,.0f}, máx. "
                   f"{v.max():,.0f} en {v.idxmax():%b-%Y}).")
            if mk:
                txt += f" Mann-Kendall: tendencia **{mk.trend}** (p={mk.p_value:.3f}, pendiente de Sen {mk.sen_slope:+.2f} por período)."
            txt += half_change(v)
            if breakdown != "Sin desglose" and ADD and not W.empty:
                tot_g = W.sum().sort_values(ascending=False)
                if tot_g.sum() > 0:
                    txt += f" Según el desglose, el mayor peso es **{tot_g.index[0]}** ({tot_g.iloc[0] / tot_g.sum() * 100:.0f}% del total)."
            txt += " Las bandas grises son períodos sin cobertura completa y las letras marcan eventos normativos."
            note(txt)

def render_trend_compare(Wsrc: pd.DataFrame, freq_text: str, tag: str = "serie") -> None:
    st.markdown("---")
    st.subheader("🧪 ¿Las tendencias son iguales o diferentes?")
    if not CMP_ON:
        st.caption("Actívalo en el panel lateral («Comparar tendencias») con un desglose de al menos 2 series.")
        return
    a_name, b_name = CMP["a"], CMP["b"]
    if a_name == b_name:
        st.warning("Elige dos series distintas (A y B) en el panel lateral.")
        return
    if a_name not in Wsrc.columns or b_name not in Wsrc.columns:
        st.info("Las series A y B elegidas en el panel no existen con este desglose o granularidad.")
        return
    Wc = Wsrc[[a_name, b_name]]                     # ya sin períodos de cobertura incompleta
    st.caption(f"Granularidad: {freq_text.lower()}. Las series y el modelo se eligen en el panel lateral («Comparar tendencias»).")
    mdl = {"Automático": "auto", "Poisson (conteos)": "poisson", "Lineal (OLS)": "ols"}[CMP["model"]]
    tc = compare_trends(Wc[a_name], Wc[b_name], model=mdl, additive=ADD, boot=500 if CMP["boot"] else 0)
    if tc is None:
        st.info("Se necesitan al menos 6 períodos con datos en ambas series. Usa una granularidad más fina o amplía el rango de fechas.")
        return
    level, text = verdict(tc, a_name, b_name, CMP["alpha"])
    box(text, "interp-box" if level != "weak" else "warn-box")
    for n_ in tc.notes:
        st.caption(n_)

    m1, m2, m3, m4 = st.columns(4)
    if tc.model == "poisson":
        m1.metric(f"Crecimiento anual — {a_name}", f"{tc.growth_a:+.1f}%", f"IC95% {tc.growth_a_ci[0]:+.0f} a {tc.growth_a_ci[1]:+.0f}", delta_color="off")
        m2.metric(f"Crecimiento anual — {b_name}", f"{tc.growth_b:+.1f}%", f"IC95% {tc.growth_b_ci[0]:+.0f} a {tc.growth_b_ci[1]:+.0f}", delta_color="off")
        m3.metric("Razón de crecimiento anual B/A", f"{tc.ratio:.2f}", f"IC95% {tc.ratio_ci[0]:.2f}–{tc.ratio_ci[1]:.2f}", delta_color="off")
    else:
        m1.metric(f"Pendiente/año — {a_name}", f"{tc.slope_a:+.2f}")
        m2.metric(f"Pendiente/año — {b_name}", f"{tc.slope_b:+.2f}")
        m3.metric("Diferencia (B−A)", f"{tc.diff:+.2f}", f"IC95% {tc.diff_ci[0]:+.2f} a {tc.diff_ci[1]:+.2f}", delta_color="off")
    m4.metric("p-valor (pendientes distintas)", f"{tc.diff_p:.3f}", "diferentes" if tc.diff_p < CMP["alpha"] else "sin diferencia detectable", delta_color="off")

    # gráfico: observado + ajuste del modelo conjunto
    idx = tc.idx
    obs_a, obs_b = Wc.loc[idx, a_name].to_numpy(float), Wc.loc[idx, b_name].to_numpy(float)
    fa, fb = tc.fitted_a, tc.fitted_b
    if CMP["scale"].startswith("Índice"):
        ba, bb = np.nanmean(obs_a[:12]), np.nanmean(obs_b[:12])
        ba, bb = (ba if ba > 0 else 1.0), (bb if bb > 0 else 1.0)
        obs_a, fa, obs_b, fb = obs_a / ba * 100, fa / ba * 100, obs_b / bb * 100, fb / bb * 100
        ytl = "Índice (base 100)"
    else:
        ytl = measure_unit(MEASURE)
    fig = go.Figure()
    for nm, ob, ft, col in ((a_name, obs_a, fa, PAL[0]), (b_name, obs_b, fb, PAL[1])):
        fig.add_trace(go.Scatter(x=idx, y=ob, name=nm, mode="lines+markers" if markers else "lines", opacity=0.55,
                                 line=dict(color=col, width=1.6)))
        fig.add_trace(go.Scatter(x=idx, y=ft, name=f"{nm} (tendencia ajustada)", mode="lines", line=dict(color=col, width=3.6, dash="dash")))
    style_fig(fig, 420, ytl, log_y)
    st.plotly_chart(fig, key=f"cmp_fit_{tag}", **STRETCH)
    note("Los puntos claros son lo observado y las líneas punteadas gruesas la tendencia estimada de cada serie. Si las punteadas son casi "
         "paralelas, ambas crecen a un ritmo parecido; si se abren o se cruzan, una crece más rápido. "
         + ("Aquí las pendientes difieren de forma estadísticamente significativa." if tc.diff_p < CMP["alpha"]
            else "Aquí la diferencia de pendientes no alcanza significancia estadística."))

    # peso relativo de A en A+B
    if tc.share:
        sh = tc.share
        cA, cB = st.columns([3, 2])
        with cA:
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=sh["idx"], y=sh["share"] * 100, name="Observado", mode="lines+markers", opacity=0.55,
                                    line=dict(color="#7D7D7D", width=1.6)))
            f2.add_trace(go.Scatter(x=sh["idx"], y=sh["fit"] * 100, name="Modelo binomial", mode="lines", line=dict(color=PAL[0], width=3.4)))
            style_fig(f2, 300, f"% de «{a_name}» dentro de A+B")
            st.plotly_chart(f2, key=f"cmp_share_{tag}", **STRETCH)
        with cB:
            sig = sh["p"] < CMP["alpha"]
            st.markdown(f"**Peso relativo de {a_name} en (A+B)**")
            st.markdown(f"Según el modelo, pasa de **{sh['start'] * 100:.0f}%** a **{sh['end'] * 100:.0f}%** "
                        f"(odds ×{sh['or_year']:.2f}/año, IC95% {sh['or_ci'][0]:.2f}–{sh['or_ci'][1]:.2f}, p={sh['p']:.3f}). "
                        + ("El peso relativo **cambia** de forma significativa." if sig else "No hay cambio significativo del peso relativo."))
            note("Responde a otra pregunta: aunque ambas series crezcan, ¿una gana participación frente a la otra? "
                 "La curva muestra la proporción de A dentro de A+B; si sube, A gana peso relativo.")

    with st.expander("Detalles estadísticos"):
        rows = []
        for nm, mk in ((a_name, tc.mk_a), (b_name, tc.mk_b)):
            rows.append({"Serie": nm, "Mann-Kendall": mk.trend if mk else "n<8", "p (MK)": mk.p_value if mk else np.nan,
                         "Pendiente de Sen (por período)": mk.sen_slope if mk else np.nan})
        if tc.share and tc.share.get("mk"):
            mk = tc.share["mk"]
            rows.append({"Serie": f"Proporción de {a_name} en A+B", "Mann-Kendall": mk.trend, "p (MK)": mk.p_value,
                         "Pendiente de Sen (por período)": mk.sen_slope})
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH, column_config={
            "p (MK)": st.column_config.NumberColumn(format="%.4f"), "Pendiente de Sen (por período)": st.column_config.NumberColumn(format="%.4f")})
        note("Mann-Kendall indica si cada serie sube o baja de forma sostenida (p < 0,05 = tendencia clara); la pendiente de Sen es el cambio típico por período. "
             "Sirve para confirmar el sentido de cada serie, no para compararlas entre sí: eso lo hace la prueba de pendientes de arriba.")
        if tc.comove:
            st.markdown(f"**Co-movimiento** (Spearman entre primeras diferencias): ρ = {tc.comove['rho']:+.2f}, p = {tc.comove['p']:.3f}, n = {tc.comove['n']}.")
        if tc.boot:
            b = tc.boot
            lo, hi = (np.exp(b["lo"]), np.exp(b["hi"])) if tc.model == "poisson" else (b["lo"], b["hi"])
            st.markdown(f"**Bootstrap de bloques** (bloques de {b['L']} períodos, {b['n_boot']} réplicas): "
                        f"{'razón de crecimiento anual' if tc.model == 'poisson' else 'diferencia'} IC95% {lo:.2f}–{hi:.2f}, p≈{b['p']:.3f}.")
        st.caption(f"Modelo: {'Poisson log-lineal' if tc.model == 'poisson' else 'lineal (OLS)'} conjunto con interacción tiempo × serie; "
                   f"errores estándar robustos por período; n = {tc.n} períodos con cobertura completa en ambas series "
                   f"({tc.idx.min():%b-%Y} a {tc.idx.max():%b-%Y}). Un p-valor alto no demuestra igualdad; mira el intervalo.")


with t_series:
    render_trend_compare(W.mask(np.repeat(FLAGS.values[:, None], W.shape[1], axis=1)) if not W.empty else W, freq_label, "serie")

# ============================== 2. COMPARACIÓN ANUAL =========================
with t_year:
    st.subheader("Comparación entre años")
    pc1, pc2 = st.columns([2, 2])
    per_choice = pc1.selectbox("Período del año a comparar", ["Ene–mes de corte (según panel)", "1.er semestre (ene–jun)",
                                                              "2.º semestre (jul–dic)", "Año completo (ene–dic)", "Rango personalizado"],
                               key="yr_period")
    if per_choice.startswith("Ene–mes"):
        mf, mt = 1, ytd_k
    elif per_choice.startswith("1.er"):
        mf, mt = 1, 6
    elif per_choice.startswith("2.º"):
        mf, mt = 7, 12
    elif per_choice.startswith("Año completo"):
        mf, mt = 1, 12
    else:
        mf, mt = pc2.select_slider("Meses (desde–hasta)", options=list(range(1, 13)), value=(7, 12),
                                   format_func=lambda m: dp.MONTHS_ES[m - 1], key="yr_range")
    plabel = f"{dp.MONTHS_ES[mf - 1]}–{dp.MONTHS_ES[mt - 1]}"
    RT_tab = dp.range_table(DF, mf, mt, MEASURE, INC_MONTHS)
    bad_years = [int(a) for a in RT_tab.loc[RT_tab["incompleto"], "anio"]]
    st.caption(f"Período comparado: **{plabel}**. " + (f"Sin cobertura completa en: {', '.join(map(str, bad_years))} "
               "(su valor observado subestima; el rombo muestra el estimado según la estacionalidad de los años completos)." if bad_years
               else "Todos los años tienen cobertura completa en este período."))
    view = st.radio("Vista", ["Barras del período por año", "Semestre 1 vs Semestre 2", "Líneas por año (mes a mes)", "Acumulado por año",
                              "Mapa de calor año × mes", "Perfil estacional (% del año por mes)"], horizontal=True, key="year_view")
    if view.startswith("Barras"):
        show_full = st.checkbox("Mostrar también el total del año (con los datos disponibles)", value=False, key="show_full")
        yt = RT_tab.copy()
        yt["var"] = yt["valor"].pct_change() * 100
        base_val = yt.loc[~yt["incompleto"], "valor"].iloc[0] if (~yt["incompleto"]).any() else np.nan
        fig = go.Figure()
        if show_full:
            full = dp.aggregate(DF, "A", MEASURE)["Total"]
            fig.add_bar(x=[str(t.year) for t in full.index], y=full.values, name="Año (datos disponibles)",
                        marker_color="#B8C7D9", text=[f"{x:,.0f}" for x in full.values], textposition="outside")
        fig.add_bar(x=yt["anio"].astype(str), y=yt["valor"], name=plabel,
                    marker_color=[PAL[1] if b else PAL[0] for b in yt["incompleto"]],
                    text=[f"{x:,.0f}" if pd.notna(x) else "" for x in yt["valor"]], textposition="outside")
        fig.update_layout(barmode="group")
        est_tbl = {}
        if ADD:
            for y_ in yt["anio"]:
                e_ = dp.estimate_range(PIV, int(y_), list(range(mf, mt + 1)), INC_MONTHS, LAST_D)
                if e_:
                    est_tbl[int(y_)] = e_
            if est_tbl:
                ys_ = [str(k) for k in est_tbl]
                fig.add_trace(go.Scatter(x=ys_, y=[v["est"] for v in est_tbl.values()], mode="markers", name=f"{plabel} estimado (rango)",
                                         marker=dict(symbol="diamond", size=11, color="#222"),
                                         error_y=dict(type="data", symmetric=False, array=[v["hi"] - v["est"] for v in est_tbl.values()],
                                                      arrayminus=[v["est"] - v["lo"] for v in est_tbl.values()], color="#222", thickness=1.6)))
        style_fig(fig, height, measure_unit(MEASURE))
        st.plotly_chart(fig, **STRETCH)
        st.caption(f"Naranja = año cuyo período {plabel} no tiene cobertura completa. Rombo = valor estimado del período con la estacionalidad de los "
                   "años completos (barra = rango mín.–máx.); en años completos coincide con lo observado.")
        okv = yt[~yt["incompleto"]]
        if len(okv) >= 2 and pd.notna(okv["valor"].iloc[0]) and okv["valor"].iloc[0] != 0:
            a0, a1 = okv.iloc[0], okv.iloc[-1]
            t_ = (f"En **{plabel}**, entre {int(a0.anio)} y {int(a1.anio)} (años con cobertura completa) el valor pasó de {a0.valor:,.1f} a "
                  f"{a1.valor:,.1f} ({_pct(a0.valor, a1.valor)}).")
        else:
            t_ = f"En **{plabel}** casi ningún año tiene cobertura completa: guíate por los valores estimados (rombos)."
        inc_est = [(y, e) for y, e in est_tbl.items() if not e["complete"]]
        if inc_est:
            t_ += " Para los años incompletos, el estimado del período es: " + "; ".join(
                f"{y}: {e['est']:,.0f} (rango {e['lo']:,.0f}–{e['hi']:,.0f})" for y, e in inc_est) + "."
        note(t_)
        tbl = yt.rename(columns={"anio": "Año", "valor": plabel, "var": "Var. vs año previo (%)", "incompleto": "Incompleto"})
        tbl[f"Var. vs 1.er año comparable (%)"] = (yt["valor"] / base_val - 1) * 100 if base_val and base_val > 0 else np.nan
        if est_tbl:
            tbl["Período estimado"] = [est_tbl[a]["est"] if a in est_tbl else np.nan for a in yt["anio"]]
            tbl["Est. mín."] = [est_tbl[a]["lo"] if a in est_tbl else np.nan for a in yt["anio"]]
            tbl["Est. máx."] = [est_tbl[a]["hi"] if a in est_tbl else np.nan for a in yt["anio"]]
            base_est = pd.Series([est_tbl[a]["est"] if a in est_tbl else v for a, v in zip(yt["anio"], yt["valor"])])
            tbl["Var. estimada vs año previo (%)"] = base_est.pct_change().values * 100
        st.dataframe(tbl, **STRETCH, hide_index=True, column_config={
            "Var. estimada vs año previo (%)": st.column_config.NumberColumn(format="%+.1f"),
            plabel: st.column_config.NumberColumn(format="%.1f"),
            "Var. vs año previo (%)": st.column_config.NumberColumn(format="%+.1f"),
            "Var. vs 1.er año comparable (%)": st.column_config.NumberColumn(format="%+.1f"),
            "Período estimado": st.column_config.NumberColumn(format="%.0f"),
            "Est. mín.": st.column_config.NumberColumn(format="%.0f"), "Est. máx.": st.column_config.NumberColumn(format="%.0f")})
        note("Cada fila es un año. «Var. vs año previo» usa lo observado, por lo que se distorsiona cuando un año tiene datos incompletos; "
             "«Var. estimada» sustituye esos años por su estimado y es la comparación más justa.")
    elif view.startswith("Semestre"):
        S1, S2 = list(range(1, 7)), list(range(7, 13))
        r1 = dp.range_table(DF, 1, 6, MEASURE, INC_MONTHS).set_index("anio")
        r2 = dp.range_table(DF, 7, 12, MEASURE, INC_MONTHS).set_index("anio")
        yrs = list(r1.index)
        fig = go.Figure()
        for nm, rr, col, off in (("1.er semestre (ene–jun)", r1, PAL[0], -0.2), ("2.º semestre (jul–dic)", r2, PAL[1], 0.2)):
            fig.add_bar(x=[y + off for y in yrs], y=rr["valor"].values, width=0.38, name=nm, marker_color=col,
                        marker_pattern_shape=["/" if rr.loc[y, "incompleto"] else "" for y in yrs],
                        text=[f"{v:,.0f}" if pd.notna(v) else "" for v in rr["valor"].values], textposition="outside")
        est1, est2 = {}, {}
        if ADD:
            for y in yrs:
                for months, dct in ((S1, est1), (S2, est2)):
                    e_ = dp.estimate_range(PIV, int(y), months, INC_MONTHS, LAST_D)
                    if e_ and not e_["complete"]:
                        dct[int(y)] = e_
            pts = [(y - 0.2, e_) for y, e_ in est1.items()] + [(y + 0.2, e_) for y, e_ in est2.items()]
            if pts:
                fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1]["est"] for p in pts], mode="markers", name="Estimado (rango)",
                                         marker=dict(symbol="diamond", size=11, color="#222"),
                                         error_y=dict(type="data", symmetric=False, array=[p[1]["hi"] - p[1]["est"] for p in pts],
                                                      arrayminus=[p[1]["est"] - p[1]["lo"] for p in pts], color="#222", thickness=1.6)))
        style_fig(fig, height, measure_unit(MEASURE))
        fig.update_xaxes(tickvals=yrs, ticktext=[str(y) for y in yrs])
        st.plotly_chart(fig, **STRETCH)
        st.caption("Barras rayadas = semestre sin cobertura completa (el observado subestima). Rombo = estimado según la estacionalidad de los años completos.")
        both = [y for y in yrs if not r1.loc[y, "incompleto"] and not r2.loc[y, "incompleto"]]
        if both and ADD:
            n2 = sum(r2.loc[y, "valor"] > r1.loc[y, "valor"] for y in both)
            shr = np.mean([r2.loc[y, "valor"] / max(r1.loc[y, "valor"] + r2.loc[y, "valor"], 1) for y in both]) * 100
            note(f"En **{n2} de {len(both)}** años completos ({', '.join(map(str, both))}) el 2.º semestre supera al 1.º; en promedio concentra "
                 f"**{shr:.0f}%** del año. {'El segundo semestre pesa más de manera consistente, por lo que conviene compararlo por separado.' if n2 == len(both) else 'La diferencia entre semestres no es constante, así que no conviene asumir que el primero siempre es bajo.'}")
        else:
            note("No hay años con ambos semestres completos para comparar; usa los estimados (rombos).")
        tb = pd.DataFrame({"Año": yrs, "1.er semestre": r1["valor"].values, "2.º semestre": r2["valor"].values})
        tb["2.º sem. estimado"] = [est2[y]["est"] if y in est2 else (r2.loc[y, "valor"] if not r2.loc[y, "incompleto"] else np.nan) for y in yrs]
        tb["2.º sem. mín."] = [est2[y]["lo"] if y in est2 else np.nan for y in yrs]
        tb["2.º sem. máx."] = [est2[y]["hi"] if y in est2 else np.nan for y in yrs]
        tb["Var. 1.er sem. vs año previo (%)"] = tb["1.er semestre"].pct_change() * 100
        tb["Var. 2.º sem. est. vs año previo (%)"] = tb["2.º sem. estimado"].pct_change() * 100
        tb["Cobertura incompleta"] = [", ".join(n for n, rr in (("S1", r1), ("S2", r2)) if rr.loc[y, "incompleto"]) for y in yrs]
        st.dataframe(tb, hide_index=True, **STRETCH, column_config={c: st.column_config.NumberColumn(format="%.0f") for c in
                     ("1.er semestre", "2.º semestre", "2.º sem. estimado", "2.º sem. mín.", "2.º sem. máx.")} | {
                     c: st.column_config.NumberColumn(format="%+.1f") for c in ("Var. 1.er sem. vs año previo (%)", "Var. 2.º sem. est. vs año previo (%)")})
        note("La tabla muestra ambos semestres por año. El 2.º semestre «estimado» completa los años sin cobertura total; las variaciones usan ese "
             "estimado para no confundir datos faltantes con una caída real.")
    elif view.startswith("Perfil"):
        if not ADD:
            st.info("El perfil estacional solo aplica a medidas aditivas (conteos).")
        else:
            comp = dp.complete_years(PIV, INC_MONTHS, LAST_D)
            if len(comp) < 2:
                st.info("Se necesitan al menos 2 años con cobertura completa.")
            else:
                sh = dp.seasonal_shares(PIV, comp) * 100
                mean_, lo_, hi_ = sh.mean(), sh.min(), sh.max()
                fig = go.Figure(go.Bar(x=dp.MONTHS_ES, y=mean_.values, marker_color=[PAL[1] if v == mean_.max() else PAL[0] for v in mean_.values],
                                       error_y=dict(type="data", symmetric=False, array=(hi_ - mean_).values, arrayminus=(mean_ - lo_).values),
                                       text=[f"{v:.0f}%" for v in mean_.values], textposition="outside"))
                style_fig(fig, height, "% del total anual")
                fig.update_layout(hovermode="closest")
                st.plotly_chart(fig, **STRETCH)
                top = mean_.sort_values(ascending=False).head(3)
                ytd_sh = sh.loc[:, 1:ytd_k].sum(axis=1)
                st.markdown(f'<div class="interp-box">Años completos usados: <b>{", ".join(map(str, comp))}</b>. Los meses con más solicitudes son '
                            + ", ".join(f"<b>{dp.MONTHS_ES[m - 1]}</b> ({v:.0f}%)" for m, v in top.items())
                            + f". Enero–{mes_txt} concentra en promedio <b>{ytd_sh.mean():.0f}%</b> del total anual "
                            f"(rango {ytd_sh.min():.0f}–{ytd_sh.max():.0f}%): por eso conviene comparar años completos "
                            "o estimar el cierre del año en curso.</div>", unsafe_allow_html=True)
    else:
        Wm = dp.aggregate(DF, "M", MEASURE)["Total"]
        fl = dp.incomplete_flags(Wm.index, "M", INC_MONTHS)
        Wm = Wm.mask(fl.values & (Wm.values == 0))
        if gap_mode == "Ocultar":
            Wm = Wm.mask(fl.values)
        piv = pd.DataFrame({"anio": Wm.index.year, "mes": Wm.index.month, "v": Wm.values}).pivot(index="anio", columns="mes", values="v")
        piv = piv.reindex(columns=range(1, 13))
        pflag = pd.DataFrame({"anio": Wm.index.year, "mes": Wm.index.month, "f": fl.values}).pivot(index="anio", columns="mes", values="f")
        pflag = pflag.reindex(columns=range(1, 13)).fillna(False).astype(bool)
        if view.startswith("Líneas") or view.startswith("Acumulado"):
            if view.startswith("Acumulado"):
                if not ADD:
                    st.info("El acumulado solo aplica a medidas aditivas.")
                else:
                    piv = piv.cumsum(axis=1).where(piv.notna())
            fig = go.Figure()
            for i, y in enumerate(piv.index):
                fig.add_trace(go.Scatter(x=[dp.MONTHS_ES[m - 1] for m in piv.columns], y=piv.loc[y].values, name=str(y),
                                         mode="lines+markers" if markers else "lines", connectgaps=False,
                                         line=dict(color=PAL[i % len(PAL)], width=2.6)))
            style_fig(fig, height, measure_unit(MEASURE) + (" (acumulado)" if view.startswith("Acumulado") else ""), log_y)
            fig.update_xaxes(categoryorder="array", categoryarray=dp.MONTHS_ES)
            st.plotly_chart(fig, **STRETCH)
            if view.startswith("Acumulado") and ADD:
                k_ = 0
                for m_ in range(1, 13):
                    if piv[m_].notna().all():
                        k_ = m_
                    else:
                        break
                if k_:
                    vals = piv[k_].sort_values()
                    note(f"Cada línea suma las solicitudes acumuladas de un año. Al cierre de **{dp.MONTHS_ES[k_ - 1]}** (último mes con datos en todos los años) "
                         f"el acumulado va de {vals.iloc[0]:,.0f} ({int(vals.index[0])}) a {vals.iloc[-1]:,.0f} ({int(vals.index[-1])}). "
                         "Una línea que se separa hacia arriba indica un año con más ritmo de solicitudes.")
                else:
                    note("Cada línea suma las solicitudes acumuladas de un año; no hay un mes con datos en todos los años para compararlos.")
            elif view.startswith("Líneas"):
                pk = piv.idxmax(axis=1).dropna().astype(int)
                if len(pk):
                    mo = pk.mode().iloc[0]
                    note(f"Cada línea es un año. El mes de mayor volumen es **{dp.MONTHS_ES[mo - 1]}** en {int((pk == mo).sum())} de {len(pk)} años: "
                         "picos que se repiten en el mismo mes indican estacionalidad (p. ej., un ciclo anual de programación), no un evento aislado.")
        else:
            fig = go.Figure(go.Heatmap(z=piv.values, x=dp.MONTHS_ES, y=piv.index.astype(str), colorscale="Blues",
                                       text=np.where(np.isnan(piv.values), "", np.char.add(np.round(piv.values, 0 if ADD else 1).astype(str), np.where(pflag.values, "*", ""))),
                                       texttemplate="%{text}", hoverongaps=False, colorbar=dict(title=measure_unit(MEASURE))))
            style_fig(fig, max(320, 60 * len(piv) + 120), "Año")
            fig.update_yaxes(autorange="reversed", type="category")
            st.plotly_chart(fig, **STRETCH)
            st.caption("Celdas vacías = mes sin registros. * = mes con cobertura parcial (dato incompleto).")
            if np.isfinite(np.nanmax(piv.values)):
                r_, c_ = np.unravel_index(np.nanargmax(piv.values), piv.shape)
                note(f"Cada celda es un mes; más oscuro = más solicitudes. El máximo es **{np.nanmax(piv.values):,.0f}** en **{dp.MONTHS_ES[c_]}-{int(piv.index[r_])}**. "
                     "Las columnas oscuras que se repiten año tras año señalan meses de alta demanda estacional.")

# ============================== 3. ALTO COSTO ================================
with t_hc:
    st.subheader("Subanálisis de medicamentos de alto costo (listado no PNUME)")
    st.caption("Cruce por principio activo (todas las concentraciones y formas). Los filtros del panel lateral aplican, salvo "
               "«Medicamentos de alto costo», que aquí se controla con la selección de fármacos.")
    hc_price = HC.groupby("clave")["precio_ref"].apply(lambda s: s[s > 1].max() if (s > 1).any() else np.nan)
    hc_prog = HC.groupby("clave")["programacion"].first()
    prog_label = lambda p: "Estrategia oncológica no PNUME" if "ESTRATEGIA" in str(p).upper() else "Oncológicos no PNUME (alto costo/nuevo)"
    keys_all = list(hc_price.sort_values(ascending=False, na_position="last").index)
    DH = DF_BASE[DF_BASE["alto_costo_clave"].isin(keys_all)]
    eval_by_key = DH.groupby("alto_costo_clave").size().reindex(keys_all, fill_value=0)
    first_year = dp.hc_first_year(DF_BASE)
    y_first_series = int(CFT["fecha"].dt.year.min())

    c1, c2, c3 = st.columns([2, 1, 1])
    presets = ["Top N por evaluaciones", "Todos", "Estrategia oncológica no PNUME", "Oncológicos no PNUME – alto costo/nuevo",
               "Inmunoterapia (anti-PD-1/PD-L1)", "Anticuerpos monoclonales (-mab)", "Inhibidores de quinasas (-nib)",
               "Precio ref. ≥ S/ 5 000", "Evaluados por 1.ª vez después del primer año", "Personalizado"]
    preset = c1.selectbox("Conjunto de fármacos", presets, key="hc_preset")
    topn_hc = c2.slider("N (para Top N)", 2, 20, 6, key="hc_topn")
    hc_freq_label = c3.selectbox("Granularidad", list(dp.FREQ_LABELS), index=3, key="hc_freq")
    HFREQ = dp.FREQ_LABELS[hc_freq_label]

    def preset_keys(p: str) -> list[str]:
        if p == "Top N por evaluaciones":
            return list(eval_by_key.sort_values(ascending=False).head(topn_hc).index)
        if p == "Todos":
            return keys_all
        if p.startswith("Estrategia"):
            return [k for k in keys_all if "ESTRATEGIA" in str(hc_prog.get(k, "")).upper()]
        if p.startswith("Oncológicos no PNUME"):
            return [k for k in keys_all if "ESTRATEGIA" not in str(hc_prog.get(k, "")).upper()]
        if p.startswith("Inmunoterapia"):
            return [k for k in keys_all if k in ("PEMBROLIZUMAB", "NIVOLUMAB", "ATEZOLIZUMAB", "AVELUMAB")]
        if p.startswith("Anticuerpos"):
            return [k for k in keys_all if any(w.endswith("MAB") for w in k.split())]
        if p.startswith("Inhibidores"):
            return [k for k in keys_all if any(w.endswith("NIB") for w in k.split())]
        if p.startswith("Precio"):
            return [k for k in keys_all if pd.notna(hc_price.get(k)) and hc_price.get(k) >= 5000]
        if p.startswith("Evaluados por 1.ª"):
            return [k for k in keys_all if k in first_year.index and first_year[k] > y_first_series]
        return list(eval_by_key.sort_values(ascending=False).head(6).index)

    sel_keys = st.multiselect("Fármacos (editable)", keys_all, default=preset_keys(preset),
                              format_func=lambda k: NAMES.get(k, k.title()), key=f"hc_sel_{preset}_{topn_hc}")
    d1, d2, d3, d4 = st.columns(4)
    hc_layout = d1.selectbox("Diseño", ["Líneas (mismo gráfico)", "Barras agrupadas (mismo gráfico)",
                                        "Barras apiladas (mismo gráfico)", "Paneles separados", "Mapa de calor"], key="hc_layout")
    hc_unit = d2.selectbox("Unidad de análisis", ["Por fármaco", "Conjunto seleccionado (suma)", "Conjunto vs resto de evaluaciones"], key="hc_unit")
    hc_metric = d3.selectbox("Métrica", ["Evaluaciones", "Aprobadas", "Denegadas"], key="hc_metric")
    hc_share = d4.checkbox("Como % del total de evaluaciones del período", value=False, key="hc_share")
    hc_cols = st.slider("Columnas (paneles)", 1, 4, 3, key="hc_cols") if hc_layout == "Paneles separados" else 3
    hc_sharey = st.checkbox("Mismo eje Y en paneles", value=False, key="hc_sharey") if hc_layout == "Paneles separados" else False

    if not sel_keys:
        st.info("Selecciona al menos un fármaco.")
    else:
        HM = {"Evaluaciones": "evaluaciones", "Aprobadas": "aprobadas", "Denegadas": "denegadas"}[hc_metric]
        dsel = DF_BASE[DF_BASE["alto_costo_clave"].isin(sel_keys)].copy()
        lo_h, hi_h = DF_BASE["fecha"].min(), DF_BASE["fecha"].max()
        tot_all = dp.aggregate(DF_BASE, HFREQ, HM, lo=lo_h, hi=hi_h)["Total"]
        if hc_unit == "Por fármaco":
            dsel["_g"] = dsel["alto_costo_clave"].map(lambda k: NAMES.get(k, k.title()))
            order = [NAMES.get(k, k.title()) for k in sel_keys]
            Wh = dp.aggregate(dsel, HFREQ, HM, "_g", lo=lo_h, hi=hi_h, groups=order)
        elif hc_unit.startswith("Conjunto seleccionado"):
            Wh = dp.aggregate(dsel, HFREQ, HM, lo=lo_h, hi=hi_h).rename(columns={"Total": "Alto costo seleccionado"})
        else:
            sel_w = dp.aggregate(dsel, HFREQ, HM, lo=lo_h, hi=hi_h)["Total"]
            Wh = pd.DataFrame({"Alto costo seleccionado": sel_w, "Resto de evaluaciones": (tot_all - sel_w).clip(lower=0)})
        if hc_share:
            Wh = Wh.div(tot_all.replace(0, np.nan), axis=0) * 100
        hflags = dp.incomplete_flags(Wh.index, HFREQ, INC_MONTHS)
        Wh = hide_incomplete(Wh, hflags, gap_mode, HFREQ)
        yt_h = "% del total de evaluaciones" if hc_share else f"N.º de {hc_metric.lower()}"
        hcols = list(Wh.columns)
        if hc_layout == "Mapa de calor":
            fig = go.Figure(go.Heatmap(z=Wh.T.values, x=[f"{t:%Y}" if HFREQ == "A" else f"{t:%b-%y}" for t in Wh.index], y=hcols,
                                       colorscale="Blues", hoverongaps=False,
                                       text=np.where(np.isnan(Wh.T.values), "", np.round(Wh.T.values, 1 if hc_share else 0).astype(int if not hc_share else float).astype(str)),
                                       texttemplate="%{text}", colorbar=dict(title=yt_h)))
            style_fig(fig, max(340, 34 * len(hcols) + 140), "")
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(hovermode="closest")
        elif hc_layout == "Paneles separados":
            rows = -(-len(hcols) // hc_cols)
            fig = make_subplots(rows=rows, cols=hc_cols, subplot_titles=hcols, shared_yaxes=hc_sharey,
                                vertical_spacing=0.14 / max(rows, 1) * 2, horizontal_spacing=0.05)
            for i, c in enumerate(hcols):
                r, cc = divmod(i, hc_cols)
                if HFREQ == "A":
                    fig.add_trace(go.Bar(x=Wh.index, y=Wh[c], marker_color=PAL[i % len(PAL)], name=c, showlegend=False,
                                         text=[f"{v:,.0f}" if pd.notna(v) else "" for v in Wh[c]] if labels_on else None), row=r + 1, col=cc + 1)
                else:
                    fig.add_trace(go.Scatter(x=Wh.index, y=Wh[c], mode="lines+markers" if markers else "lines",
                                             line=dict(color=PAL[i % len(PAL)], width=2.2), name=c, showlegend=False,
                                             connectgaps=False), row=r + 1, col=cc + 1)
            decorate(fig, Wh.index.min(), Wh.index.max(), HFREQ, hflags, gap_mode, n_axes=len(hcols))
            style_fig(fig, max(height, 250 * rows), yt_h, log_y)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickformat="%Y" if HFREQ == "A" else "%b-%y")
        else:
            fig = go.Figure()
            for i, c in enumerate(hcols):
                if hc_layout.startswith("Líneas"):
                    fig.add_trace(go.Scatter(x=Wh.index, y=Wh[c], name=c, mode="lines+markers" if markers else "lines",
                                             line=dict(color=PAL[i % len(PAL)], width=2.4), connectgaps=False))
                else:
                    fig.add_trace(go.Bar(x=Wh.index, y=Wh[c], name=c, marker_color=PAL[i % len(PAL)],
                                         text=[f"{v:,.0f}" if pd.notna(v) else "" for v in Wh[c]] if labels_on else None))
            if hc_layout.startswith("Barras"):
                fig.update_layout(barmode="stack" if "apiladas" in hc_layout else "group")
            decorate(fig, Wh.index.min(), Wh.index.max(), HFREQ, hflags, gap_mode)
            style_fig(fig, height, yt_h, log_y)
            fig.update_xaxes(tickformat="%Y" if HFREQ == "A" else "%b-%y")
        st.plotly_chart(fig, **STRETCH)
        ss["hc_fig"] = fig
        render_events_legend()
        tot_c = Wh.sum().sort_values(ascending=False)
        ser_h = Wh.sum(axis=1, min_count=1).dropna()
        txt_hc = f"Se muestran **{len(hcols)}** serie(s). "
        if len(tot_c) > 1 and tot_c.iloc[0] > 0 and not hc_share:
            txt_hc += f"La de mayor volumen es **{tot_c.index[0]}** ({tot_c.iloc[0]:,.0f} en total, {tot_c.iloc[0] / tot_c.sum() * 100:.0f}% del conjunto). "
        if len(ser_h) >= 2:
            txt_hc += (f"Sumando el conjunto, pasa de {ser_h.iloc[0]:,.1f} a {ser_h.iloc[-1]:,.1f} entre el primer y el último período con cobertura completa "
                     f"({_pct(ser_h.iloc[0], ser_h.iloc[-1])}).")
        note(txt_hc + " Un fármaco que sube de forma sostenida gana peso en las solicitudes; picos aislados suelen ser renovaciones o casos puntuales.")

        # ---- tabla por fármaco -------------------------------------------------
        st.markdown("#### Detalle por fármaco")
        rows_t = []
        years = sorted(DF_BASE["fecha"].dt.year.unique())
        for k in sel_keys:
            g = DF_BASE[DF_BASE["alto_costo_clave"] == k]
            rec = {"Fármaco": NAMES.get(k, k.title()), "Presentaciones en listado": int((HC["clave"] == k).sum()),
                   "Precio ref. máx. (S/)": hc_price.get(k), "Programación": prog_label(hc_prog.get(k, "")),
                   "1.ª evaluación (año)": int(g["fecha"].dt.year.min()) if len(g) else None,
                   "Evaluaciones": len(g), "Aprobadas": int((g["decision"] == "APROBADO").sum()),
                   "Denegadas": int((g["decision"] == "DENEGADO").sum())}
            rec["% aprob."] = rec["Aprobadas"] / rec["Evaluaciones"] * 100 if rec["Evaluaciones"] else np.nan
            for y in years:
                rec[str(y)] = int((g["fecha"].dt.year == y).sum())
            rows_t.append(rec)
        tdf = pd.DataFrame(rows_t)
        st.dataframe(tdf, **STRETCH, hide_index=True, column_config={
            "Precio ref. máx. (S/)": st.column_config.NumberColumn(format="%.2f"),
            "% aprob.": st.column_config.NumberColumn(format="%.0f%%")})
        st.caption("Precio ref. = adjudicación referencial del listado (valores ≤ 1 = precio aún no determinado, se omiten). "
                   "«1.ª evaluación» es la primera dentro de la serie cargada: para el primer año no implica que sea la primera vez que el CFT lo evaluó.")
        if len(tdf):
            top_r = tdf.sort_values("Evaluaciones", ascending=False).iloc[0]
            tot_e = int(tdf["Evaluaciones"].sum())
            ap_e = tdf["Aprobadas"].sum() / tot_e * 100 if tot_e else float("nan")
            note(f"La tabla resume {len(tdf)} fármaco(s): {tot_e:,} evaluaciones en total, {ap_e:.0f}% aprobadas. El más solicitado es **{top_r['Fármaco']}** "
                 f"({int(top_r['Evaluaciones'])}). Las columnas por año muestran cómo evolucionó cada fármaco y «1.ª evaluación» cuándo aparece en la serie.")

        cA, cB = st.columns(2)
        with cA:
            st.markdown('<div class="mini-title">Concentración de evaluaciones (Pareto)</div>', unsafe_allow_html=True)
            par = tdf.sort_values("Evaluaciones", ascending=False)
            par = par[par["Evaluaciones"] > 0]
            if len(par):
                par["acum"] = par["Evaluaciones"].cumsum() / par["Evaluaciones"].sum() * 100
                fp = make_subplots(specs=[[{"secondary_y": True}]])
                fp.add_bar(x=par["Fármaco"], y=par["Evaluaciones"], marker_color=PAL[0], name="Evaluaciones")
                fp.add_scatter(x=par["Fármaco"], y=par["acum"], mode="lines+markers", line=dict(color=PAL[1]), name="% acumulado", secondary_y=True)
                fp.update_yaxes(title_text="Evaluaciones", secondary_y=False)
                fp.update_yaxes(title_text="% acumulado", range=[0, 105], tickvals=[0, 25, 50, 75, 100], showgrid=False, secondary_y=True)
                style_fig(fp, 380, "Evaluaciones")
                fp.update_layout(hovermode="x unified")
                st.plotly_chart(fp, **STRETCH)
                st.caption("Es concentración de **solicitudes**, no de gasto (el listado no trae cantidades consumidas).")
                k80 = int((par["acum"] < 80).sum()) + 1
                note(f"**{min(k80, len(par))} de {len(par)}** fármacos ({min(k80, len(par)) / len(par) * 100:.0f}%) concentran el 80% de las evaluaciones "
                     f"(encabezan: {', '.join(par['Fármaco'].head(min(k80, 3)))}). Mientras más rápido sube la curva, más concentrada está la demanda en pocos fármacos.")
        with cB:
            st.markdown('<div class="mini-title">Precio referencial vs. número de evaluaciones</div>', unsafe_allow_html=True)
            sc = tdf.dropna(subset=["Precio ref. máx. (S/)"])
            sc = sc[sc["Evaluaciones"] > 0]
            if len(sc):
                fs = px.scatter(sc, x="Precio ref. máx. (S/)", y="Evaluaciones", text="Fármaco", color="Programación",
                                log_x=True, color_discrete_sequence=PAL)
                fs.update_traces(textposition="top center", marker=dict(size=11), cliponaxis=False)
                style_fig(fs, 400, "Evaluaciones")
                fs.update_layout(hovermode="closest", legend=dict(orientation="h", y=-0.35, x=0, title_text=""))
                st.plotly_chart(fs, **STRETCH)
                if len(sc) >= 5:
                    rho_, p_ = sps.spearmanr(sc["Precio ref. máx. (S/)"], sc["Evaluaciones"])
                    note(f"Correlación de Spearman entre precio y número de evaluaciones: ρ = {rho_:+.2f} (p = {p_:.3f}). "
                         + ("No hay una relación clara: los fármacos caros no se evalúan necesariamente más ni menos veces." if p_ >= 0.05
                            else f"Los fármacos más caros tienden a evaluarse {'más' if rho_ > 0 else 'menos'} veces."))
                else:
                    note("Cada punto es un fármaco: a la derecha, mayor precio referencial; arriba, más evaluaciones. Con pocos fármacos no se calcula correlación.")
        no_eval = [NAMES.get(k, k.title()) for k in keys_all if eval_by_key.get(k, 0) == 0]
        if no_eval:
            st.markdown(f'<div class="warn-box">Del listado, <b>{len(no_eval)}</b> principio(s) activo(s) sin evaluación CFT con los filtros actuales: '
                        f'{", ".join(no_eval)}.</div>', unsafe_allow_html=True)


def monthly_valid(df: pd.DataFrame, measure: str) -> pd.Series:
    """Serie mensual total, sin los meses sin cobertura completa."""
    wm = dp.aggregate(df, "M", measure)["Total"]
    fl = dp.incomplete_flags(wm.index, "M", INC_MONTHS)
    return wm.mask(fl.values).dropna()


# ============================== 4. PUNTO DE CORTE ============================
with t_break:
    st.subheader("Punto de corte: ¿cambió el ritmo de solicitudes?")
    st.caption("Test de Chow sobre la serie **mensual** (dos rectas: antes y después del evento). Se excluyen los meses sin cobertura completa.")
    ser = monthly_valid(DF, MEASURE)
    if len(ser) < 12:
        st.info("La serie mensual válida tiene menos de 12 observaciones con los filtros actuales.")
    else:
        opts = [f"{l} — {e} ({f:%b-%Y})" for l, f, e in ACTIVE_EVENTS] + ["Fecha personalizada"]
        b1, b2, b3 = st.columns([2, 1, 1])
        ev_choice = b1.selectbox("Evento", opts, index=min(2, len(opts) - 1), key="brk_ev")
        if ev_choice == "Fecha personalizada":
            ev_date = pd.Timestamp(b1.date_input("Fecha del evento", value=pd.Timestamp("2023-01-01"), key="brk_date")).to_period("M").to_timestamp()
        else:
            ev_date = pd.Timestamp([f for l, f, e in ACTIVE_EVENTS][opts.index(ev_choice)]).to_period("M").to_timestamp()
        lag = b2.slider("Rezago de implementación (meses)", 0, 6, 0, key="brk_lag")
        minseg = b3.slider("Mín. observaciones por tramo", 4, 12, 6, key="brk_min")
        ev = chow_test_event(ser, ev_date, implementation_lag_months=lag, min_segment=minseg)
        auto = detect_breakpoint_auto(ser, min_segment=6)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode="lines+markers" if markers else "lines", name="Observado",
                                 line=dict(color="#9DC3E6", width=1.6)))
        if ev:
            fig.add_trace(go.Scatter(x=ev.x_before, y=ev.y_pred_before, mode="lines", name=f"Tendencia antes ({ev.slope_before:+.2f}/mes)",
                                     line=dict(color=PAL[0], width=3.2)))
            fig.add_trace(go.Scatter(x=ev.x_after, y=ev.y_pred_after, mode="lines", name=f"Tendencia después ({ev.slope_after:+.2f}/mes)",
                                     line=dict(color=PAL[1], width=3.2)))
            fig.add_shape(type="line", xref="x", yref="paper", x0=ev_date, x1=ev_date, y0=0, y1=1, line=dict(color="#333", dash="dash"))
            if lag:
                fig.add_shape(type="rect", xref="x", yref="paper", x0=ev_date, x1=ev_date + pd.DateOffset(months=lag), y0=0, y1=1,
                              fillcolor="rgba(200,200,0,0.15)", line_width=0)
        if auto:
            fig.add_shape(type="line", xref="x", yref="paper", x0=auto.date, x1=auto.date, y0=0, y1=1, line=dict(color="#2F9E44", dash="dot"))
            fig.add_annotation(x=auto.date, yref="paper", y=1, text="quiebre automático", showarrow=False, yanchor="bottom",
                               font=dict(size=10, color="#2F9E44"))
        style_fig(fig, height, measure_unit(MEASURE), log_y)
        st.plotly_chart(fig, **STRETCH)
        note("La línea clara es la serie mensual (sin meses incompletos). La recta azul es la tendencia antes del evento (línea negra discontinua) y la naranja "
             "la tendencia después: si la naranja sube más que la azul, el ritmo de solicitudes se aceleró tras el evento. La línea verde marca el mes en que la "
             "serie cambia más de ritmo según los datos, sin fecha previa.")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Evento elegido (Chow)**")
            if ev:
                st.dataframe(pd.DataFrame({
                    "Indicador": ["Pendiente antes (por mes)", "Pendiente después (por mes)", "Cambio de pendiente", "F", "p-valor", "n antes / después"],
                    "Valor": [f"{ev.slope_before:+.3f}", f"{ev.slope_after:+.3f}", f"{ev.slope_after - ev.slope_before:+.3f}",
                              f"{ev.f_stat:.2f}", f"{ev.p_value:.4f}", f"{ev.n_before} / {ev.n_after}"]}), hide_index=True, **STRETCH)
                note(f'El cambio de pendiente {"**es**" if ev.significant else "**no es**"} estadísticamente '
                     f'significativo (p={ev.p_value:.3f}): antes del evento la serie variaba {ev.slope_before:+.2f} por mes y después {ev.slope_after:+.2f}. '
                     'Es una asociación temporal, no prueba causal: coinciden otros factores (nuevos fármacos en el mercado, guías, cambios de casuística).')
            else:
                st.info("Tramos demasiado cortos para estimar el test con esta fecha (baja el mínimo por tramo o cambia el evento).")
        with m2:
            st.markdown("**Quiebre estructural automático**")
            if auto:
                st.dataframe(pd.DataFrame({
                    "Indicador": ["Fecha del quiebre", "Pendiente antes", "Pendiente después", "F", "p-valor"],
                    "Valor": [f"{auto.date:%b-%Y}", f"{auto.slope_before:+.3f}", f"{auto.slope_after:+.3f}", f"{auto.f_stat:.2f}", f"{auto.p_value:.4f}"]}),
                    hide_index=True, **STRETCH)
                note(f"Sin fijar una fecha, los datos sitúan el mayor cambio de ritmo en **{auto.date:%b-%Y}**: la pendiente pasa de {auto.slope_before:+.2f} a "
                     f"{auto.slope_after:+.2f} por mes (p={auto.p_value:.3f}). Como se buscó el mejor corte, el p-valor es optimista: úsalo como pista, no como prueba.")
            pre, post = ser[ser.index < ev_date], ser[ser.index >= ev_date + pd.DateOffset(months=lag)]
            if len(pre) >= 4 and len(post) >= 4:
                u = sps.mannwhitneyu(pre.values, post.values, alternative="two-sided")
                note(f"El nivel medio mensual pasa de {pre.mean():.1f} antes del evento a {post.mean():.1f} después ({_pct(pre.mean(), post.mean())}; "
                     f"Mann-Whitney p={u.pvalue:.3f}): compara el nivel promedio, aunque no la pendiente.")

# ============================== 5. TENDENCIA =================================
with t_trend:
    st.subheader("Tendencia (Mann-Kendall, Sen y regresión lineal)")
    r1, r2 = st.columns(2)
    tr_freq_label = r1.selectbox("Granularidad", ["Mensual", "Trimestral"], index=1, key="tr_freq")
    alpha = r2.selectbox("Nivel de significancia", [0.05, 0.10, 0.01], key="tr_alpha")
    TF = dp.FREQ_LABELS[tr_freq_label]
    Wtr, _, _, _ = build_wide(DF, TF, MEASURE, breakdown, top_n, lo=DF["fecha"].min(), hi=DF["fecha"].max())
    if Wtr.empty:
        st.info("Sin datos.")
    else:
        fl_t = dp.incomplete_flags(Wtr.index, TF, INC_MONTHS)
        Wtr = Wtr.mask(np.repeat(fl_t.values[:, None], Wtr.shape[1], axis=1))
        Wtr_tot = dp.aggregate(DF, TF, MEASURE)["Total"].mask(fl_t.values)
        cols_tr = ["Total"] + [c for c in Wtr.columns if c != "Total"] if breakdown != "Sin desglose" else ["Total"]
        src = pd.concat([Wtr_tot.rename("Total"), Wtr.drop(columns=["Total"], errors="ignore")], axis=1) if breakdown != "Sin desglose" else Wtr_tot.to_frame("Total")
        recs = []
        for c in cols_tr:
            v = src[c].dropna()
            mkr = mann_kendall_trend(v.values.astype(float), alpha=alpha) if len(v) >= 8 else None
            lt = linear_trend(np.arange(len(v)), v.values) if len(v) >= 3 else None
            recs.append({"Serie": c, "n": len(v), "Tendencia MK": mkr.trend if mkr else "n<8", "p (MK)": mkr.p_value if mkr else np.nan,
                         "Z": mkr.z if mkr else np.nan, "Pendiente de Sen (por período)": mkr.sen_slope if mkr else np.nan,
                         "Variante": mkr.variant_used if mkr else "", "Pend. lineal": lt.slope if lt else np.nan,
                         "R²": lt.r2 if lt else np.nan, "p (lineal)": lt.p_value if lt else np.nan})
        st.dataframe(pd.DataFrame(recs), **STRETCH, hide_index=True, column_config={
            "p (MK)": st.column_config.NumberColumn(format="%.4f"), "Z": st.column_config.NumberColumn(format="%.2f"),
            "Pendiente de Sen (por período)": st.column_config.NumberColumn(format="%.3f"),
            "Pend. lineal": st.column_config.NumberColumn(format="%.3f"), "R²": st.column_config.NumberColumn(format="%.2f"),
            "p (lineal)": st.column_config.NumberColumn(format="%.4f")})
        r0 = recs[0]
        unit_p = "mes" if TF == "M" else "trimestre"
        ppy_t = 12 if TF == "M" else 4
        n_up = sum(1 for r in recs if r["Tendencia MK"] == "creciente")
        t_tr = (f"Para **{r0['Serie']}**: tendencia **{r0['Tendencia MK']}**" +
                (f" (p={r0['p (MK)']:.3f}); el cambio típico (pendiente de Sen) es {r0['Pendiente de Sen (por período)']:+.2f} por {unit_p}, "
                 f"≈ {r0['Pendiente de Sen (por período)'] * ppy_t:+.1f} por año." if pd.notna(r0["p (MK)"]) else " (muestra insuficiente)."))
        if len(recs) > 1:
            t_tr += f" De {len(recs)} series, {n_up} muestran tendencia creciente."
        note(t_tr + f" Con p < {alpha} la tendencia es estadísticamente clara; «sin tendencia» significa que los datos no permiten afirmar que la serie suba o baje de forma sostenida.")
        v = Wtr_tot.dropna()
        if len(v) >= 3:
            lt = linear_trend(np.arange(len(v)), v.values)
            fig = go.Figure()
            fig.add_scatter(x=Wtr_tot.index, y=Wtr_tot.values, mode="lines+markers", name="Total", connectgaps=False,
                            line=dict(color=PAL[0], width=2.4))
            fig.add_scatter(x=v.index, y=lt.y_pred, mode="lines", name=f"Lineal (R²={lt.r2:.2f})", line=dict(color=PAL[1], dash="dash", width=2.4))
            style_fig(fig, 380, measure_unit(MEASURE))
            st.plotly_chart(fig, **STRETCH)
            note(f"La línea discontinua es la recta de tendencia: sube {lt.slope:+.2f} por {unit_p} y explica {lt.r2 * 100:.0f}% de la variación (R²). "
                 "Un R² bajo indica que la serie fluctúa mucho alrededor de la tendencia (típico cuando hay picos estacionales); no invalida que exista tendencia.")
        st.caption("Mann-Kendall con corrección por autocorrelación (Hamed-Rao / Yue-Wang). Períodos sin cobertura completa excluidos.")
        render_trend_compare(Wtr, tr_freq_label, "tend")

# ============================== 6. PROYECCIÓN ================================
with t_proj:
    st.subheader("Proyección (mensual)")
    ser_all = dp.aggregate(DF, "M", MEASURE)["Total"]
    fl = dp.incomplete_flags(ser_all.index, "M", INC_MONTHS)
    ok = (~fl.values).astype(int)
    # último tramo contiguo de meses válidos (se descartan meses incompletos al final)
    end = len(ok)
    while end > 0 and ok[end - 1] == 0:
        end -= 1
    start = end
    while start > 0 and ok[start - 1] == 1:
        start -= 1
    blk = ser_all.iloc[start:end]
    p1, p2 = st.columns(2)
    horizon = p1.slider("Horizonte (meses)", 3, 12, 6, key="pj_h")
    ci = p2.slider("Nivel del intervalo (bootstrap)", 0.5, 0.95, 0.90, 0.05, key="pj_ci")
    if len(blk) < 12:
        st.info("El último tramo continuo de meses con cobertura completa tiene menos de 12 observaciones.")
    else:
        pr = project_series(blk, horizon=horizon, ci=ci)
        if pr is None:
            st.info("No fue posible ajustar el modelo.")
        else:
            st.caption(f"Se usa el último tramo continuo con cobertura completa: {blk.index[0]:%b-%Y} a {blk.index[-1]:%b-%Y} ({len(blk)} meses).")
            fig = go.Figure()
            fig.add_scatter(x=pr.idx_hist, y=pr.values_hist, name="Histórico", mode="lines+markers" if markers else "lines", line=dict(color=PAL[0], width=2.4))
            fig.add_scatter(x=pr.idx_future, y=pr.recommended_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=pr.idx_future, y=pr.recommended_lower, mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor="rgba(217,84,47,0.18)", name=f"IC {ci:.0%} (recomendado)", hoverinfo="skip")
            fig.add_scatter(x=pr.idx_future, y=pr.recommended, name="Recomendado (Holt amortiguado)", line=dict(color=PAL[1], width=3))
            fig.add_scatter(x=pr.idx_future, y=pr.conservative, name="Conservador (media 3 m)", line=dict(color="#7D7D7D", dash="dash"))
            fig.add_scatter(x=pr.idx_future, y=pr.optimistic, name="Lineal (Sen)", line=dict(color="#2F9E44", dash="dot"))
            style_fig(fig, height, measure_unit(MEASURE))
            st.plotly_chart(fig, **STRETCH)
            base_h = blk.iloc[-horizon:].mean()
            note(f"En los próximos {horizon} meses el escenario recomendado promedia **{pr.recommended.mean():,.1f}** por mes (intervalo {pr.recommended_lower.mean():,.1f}–"
                 f"{pr.recommended_upper.mean():,.1f}), frente a {base_h:,.1f} en los últimos {horizon} meses observados ({_pct(base_h, pr.recommended.mean())}). "
                 "Los escenarios conservador y lineal delimitan un rango plausible: cuanto más se separan, mayor la incertidumbre. Es una extrapolación estadística, no un pronóstico de gestión.")
            if ADD:
                st.dataframe(pd.DataFrame({"Escenario": ["Conservador", "Recomendado", "Lineal (Sen)"],
                                           f"Total próximos {horizon} meses": [pr.conservative.sum(), pr.recommended.sum(), pr.optimistic.sum()],
                                           f"Últimos {horizon} meses observados": [blk.iloc[-horizon:].sum()] * 3}),
                             hide_index=True, **STRETCH, column_config={
                                 f"Total próximos {horizon} meses": st.column_config.NumberColumn(format="%.0f"),
                                 f"Últimos {horizon} meses observados": st.column_config.NumberColumn(format="%.0f")})
                note("La tabla suma cada escenario sobre los próximos meses y lo compara con el mismo número de meses observados, útil para dimensionar la demanda esperada.")
            last_v, prev_m = blk.iloc[-1], blk.iloc[-13:-1].mean() if len(blk) > 13 else blk.iloc[:-1].mean()
            if ADD and prev_m > 0 and last_v > 1.5 * prev_m:
                st.markdown(f'<div class="warn-box">El último mes observado ({last_v:,.0f}) es un pico frente al promedio previo ({prev_m:,.1f}). '
                            'El escenario «Lineal (Sen)» se ancla en ese último valor y puede sobreestimar; prefiere el recomendado o el conservador.</div>',
                            unsafe_allow_html=True)
            st.caption(pr.method_note)

# ============================== 7. QUÉ EXPLICA EL CRECIMIENTO ================
with t_rank:
    st.subheader("¿Qué explica el crecimiento?")
    st.caption("Compara dos períodos y reparte el cambio en el número de solicitudes entre fármacos, áreas o grupos: quién aporta más al aumento y si "
               "el crecimiento viene de más solicitudes de los mismos fármacos o de fármacos nuevos.")
    years_all = sorted(int(y) for y in DF["fecha"].dt.year.unique())
    q1, q2, q3, q4 = st.columns([1.4, 1, 1.3, 1])
    rk_by = q1.selectbox("Repartir el cambio por", ["Principio activo (DCI)", "Área solicitante", "Grupo de área", "Fármaco de alto costo"], index=2, key="rk_by")
    rk_metric = q2.selectbox("Métrica", ["Evaluaciones", "Aprobadas", "Denegadas"], key="rk_metric")
    rk_months = q3.selectbox("Meses a considerar", ["Año completo (ene–dic)", "Ene–mes de corte (según panel)", "1.er semestre (ene–jun)",
                                                    "2.º semestre (jul–dic)"], index=1, key="rk_months")
    rk_n = q4.slider("Mostrar top N", 5, 25, 10, key="rk_n")
    mf_, mt_ = (1, 12) if rk_months.startswith("Año") else (1, ytd_k) if rk_months.startswith("Ene") else (1, 6) if rk_months.startswith("1.er") else (7, 12)
    r_a = st.select_slider("Período A (año o rango de años)", options=years_all, value=(years_all[0], years_all[0]), key="rk_a")
    r_b = st.select_slider("Período B (año o rango de años)", options=years_all, value=(years_all[-1], years_all[-1]), key="rk_b")
    lab_a = f"{r_a[0]}" if r_a[0] == r_a[1] else f"{r_a[0]}–{r_a[1]}"
    lab_b = f"{r_b[0]}" if r_b[0] == r_b[1] else f"{r_b[0]}–{r_b[1]}"
    na_, nb_ = r_a[1] - r_a[0] + 1, r_b[1] - r_b[0] + 1
    per_year = na_ != nb_
    if r_a[1] >= r_b[0]:
        st.warning("Los períodos A y B se solapan; para un cambio interpretable elige períodos distintos.")
    rt_flags = dp.range_table(DF, mf_, mt_, "evaluaciones", INC_MONTHS).set_index("anio")["incompleto"]
    bad_rk = sorted({y for rng in (r_a, r_b) for y in range(rng[0], rng[1] + 1) if bool(rt_flags.get(y, False))})
    if bad_rk:
        st.markdown(f'<div class="warn-box">Sin cobertura completa en estos meses: <b>{", ".join(map(str, bad_rk))}</b>. Su total está subestimado y el cambio '
                    'observado también; elige otros meses (p. ej., ene–ago) o interpreta con cautela.</div>', unsafe_allow_html=True)

    def take(rng):
        d = DF[(DF["fecha"].dt.year >= rng[0]) & (DF["fecha"].dt.year <= rng[1]) & (DF["fecha"].dt.month >= mf_) & (DF["fecha"].dt.month <= mt_)]
        if rk_metric == "Aprobadas":
            d = d[d["decision"] == "APROBADO"]
        elif rk_metric == "Denegadas":
            d = d[d["decision"] == "DENEGADO"]
        if rk_by.startswith("Principio"):
            lab = d["dci_base"].str.title()
        elif rk_by.startswith("Área"):
            lab = d["area_canon"]
        elif rk_by.startswith("Grupo"):
            lab = d["grupo_area"]
        else:
            d = d[d["alto_costo_clave"].notna()]
            lab = d["alto_costo_clave"].map(lambda k: NAMES.get(k, k.title()))
        return lab.value_counts().astype(float)

    ca, cb = take(r_a), take(r_b)
    if per_year:
        ca, cb = ca / na_, cb / nb_
    unit_rk = "por año" if per_year else "en el período"
    idx_ = ca.index.union(cb.index)
    tb = pd.DataFrame({"A": ca.reindex(idx_).fillna(0.0), "B": cb.reindex(idx_).fillna(0.0)})
    tb["Δ"] = tb["B"] - tb["A"]
    ta_, tb_, td_ = tb["A"].sum(), tb["B"].sum(), tb["Δ"].sum()
    if len(tb) == 0 or (ta_ == 0 and tb_ == 0):
        st.info("No hay datos en los períodos elegidos con los filtros actuales.")
    else:
        tb["% del cambio neto"] = tb["Δ"] / td_ * 100 if td_ != 0 else np.nan
        top = tb.reindex(tb["Δ"].abs().sort_values(ascending=False).head(rk_n).index).sort_values("Δ", ascending=False)
        rest = tb.drop(index=top.index)
        names = [f"Total {lab_a}"] + list(top.index) + ([f"Otros ({len(rest)})"] if len(rest) else []) + [f"Total {lab_b}"]
        vals = [ta_] + list(top["Δ"]) + ([rest["Δ"].sum()] if len(rest) else []) + [tb_]
        meas = ["absolute"] + ["relative"] * (len(names) - 2) + ["total"]
        fw = go.Figure(go.Waterfall(orientation="h", y=names, x=vals, measure=meas, text=[f"{v:+,.1f}" if m == "relative" else f"{v:,.1f}" for v, m in zip(vals, meas)],
                                    textposition="outside", increasing=dict(marker=dict(color="#2F9E44")), decreasing=dict(marker=dict(color="#C94141")),
                                    totals=dict(marker=dict(color="#7D7D7D")), connector=dict(line=dict(color="#BBB", width=1))))
        style_fig(fw, max(380, 30 * len(names) + 110), "")
        fw.update_layout(hovermode="closest", xaxis_title=f"{rk_metric} ({unit_rk})", yaxis_title=None, showlegend=False)
        fw.update_yaxes(autorange="reversed")
        st.plotly_chart(fw, **STRETCH)
        t_ = (f"El total pasa de **{ta_:,.1f}** ({lab_a}) a **{tb_:,.1f}** ({lab_b}) {unit_rk}: **{tb_ - ta_:+,.1f}** ({_pct(ta_, tb_)}). ")
        pos = top[top["Δ"] > 0]
        if td_ > 0 and len(pos):
            t_ += (f"El mayor aporte al aumento es **{pos.index[0]}** ({pos['Δ'].iloc[0]:+,.1f}; {pos['Δ'].iloc[0] / td_ * 100:.0f}% del cambio neto). "
                   f"Los {min(3, len(pos))} mayores aportes suman {pos['Δ'].head(3).sum() / td_ * 100:.0f}% del cambio neto (puede pasar de 100% si otros bajan).")
        elif td_ < 0:
            t_ += "El total disminuye; las barras rojas son los grupos que más caen."
        note(t_ + " Cada barra verde suma al total y cada roja resta: así se ve qué concentra el crecimiento.")

        # descomposición: intensidad vs diversificación
        both = (tb["A"] > 0) & (tb["B"] > 0)
        new = (tb["A"] == 0) & (tb["B"] > 0)
        lost = (tb["A"] > 0) & (tb["B"] == 0)
        d_both, d_new, d_lost = float((tb["B"] - tb["A"])[both].sum()), float(tb["B"][new].sum()), float(-tb["A"][lost].sum())
        c1_, c2_, c3_ = st.columns(3)
        c1_.metric(f"Ya presentes en {lab_a} ({int(both.sum())})", f"{d_both:+,.1f}", "más (o menos) solicitudes de los mismos", delta_color="off")
        c2_.metric(f"Nuevos en {lab_b} ({int(new.sum())})", f"{d_new:+,.1f}", "aparecen solo en B", delta_color="off")
        c3_.metric(f"Dejan de aparecer ({int(lost.sum())})", f"{d_lost:+,.1f}", "estaban en A y no en B", delta_color="off")
        dom = "más solicitudes por los mismos grupos (crecimiento de intensidad)" if abs(d_both) >= max(abs(d_new), abs(d_lost)) else (
            "grupos que antes no aparecían (crecimiento por diversificación)" if abs(d_new) >= abs(d_lost) else "grupos que dejaron de aparecer")
        note(f"El cambio de {tb_ - ta_:+,.1f} se descompone en tres partes: {d_both:+,.1f} por grupos que ya estaban en {lab_a}, {d_new:+,.1f} por "
             f"grupos nuevos y {d_lost:+,.1f} por grupos que salen. Predomina {dom}.")

        show = tb.sort_values("Δ", ascending=False).reset_index(names=rk_by)
        st.dataframe(show, hide_index=True, **STRETCH, column_config={
            "A": st.column_config.NumberColumn(f"A ({lab_a})", format="%.1f"), "B": st.column_config.NumberColumn(f"B ({lab_b})", format="%.1f"),
            "Δ": st.column_config.NumberColumn(format="%+.1f"), "% del cambio neto": st.column_config.NumberColumn(format="%+.0f%%")})
        note(f"La tabla completa ordena los grupos por su aporte al cambio (Δ = B − A). «% del cambio neto» indica qué parte del cambio total explica cada uno; "
             f"los valores están {unit_rk}. Sirve para señalar qué fármacos o áreas revisar primero si se quiere entender o gestionar el aumento.")

    st.markdown("#### Principios activos evaluados por primera vez")
    nov = DF[DF["primera_vez_serie"] == 1].copy()
    nov["anio_ev"] = nov["fecha"].dt.year
    cnt = nov.groupby("anio_ev").size()
    fn = go.Figure(go.Bar(x=cnt.index.astype(str), y=cnt.values, marker_color=[PAL[1] if a == years_all[0] else PAL[0] for a in cnt.index],
                          text=cnt.values, textposition="outside"))
    style_fig(fn, 320, "N.º de principios activos")
    st.plotly_chart(fn, **STRETCH)
    rec_new = cnt[cnt.index != years_all[0]]
    note(f"Cada barra cuenta los principios activos que aparecen por primera vez en el CFT ese año; {years_all[0]} (naranja) es la línea base, no un ingreso real. "
         + (f"Entre {int(rec_new.index[0])} y {int(rec_new.index[-1])} las incorporaciones pasan de {int(rec_new.iloc[0])} a {int(rec_new.iloc[-1])} por año: "
            f"{'el CFT evalúa cada vez más moléculas nuevas, no solo repite las conocidas.' if rec_new.iloc[-1] > rec_new.iloc[0] else 'no hay un aumento claro de moléculas nuevas.'} " if len(rec_new) >= 2 else "")
         + "Aproximado (variantes de escritura del DCI).")
    ysel = st.selectbox("Ver el detalle de un año", [a for a in years_all if a != years_all[0]] or years_all, index=None, placeholder="Elige un año", key="nov_y")
    if ysel:
        dd = nov[nov["anio_ev"] == ysel][["fecha", "dci", "concentracion", "area_canon", "decision", "alto_costo_clave"]].copy()
        dd["alto_costo_clave"] = dd["alto_costo_clave"].map(lambda k: NAMES.get(k, k) if pd.notna(k) else "")
        dd = dd.rename(columns={"fecha": "Fecha", "dci": "DCI", "concentracion": "Concentración", "area_canon": "Área", "decision": "Decisión",
                                "alto_costo_clave": "Alto costo (listado)"})
        st.dataframe(dd, hide_index=True, **STRETCH, column_config={"Fecha": st.column_config.DateColumn(format="DD-MM-YYYY")})
        note(f"En {ysel} se evaluaron por primera vez {len(dd)} principios activos ({int((dd['Alto costo (listado)'] != '').sum())} del listado de alto costo). "
             "Revisa esta lista para ver qué áreas y qué tipo de fármacos están entrando al CFT.")

# ============================== 8. SERIE EXTERNA =============================
def parse_external(raw: pd.DataFrame, date_col: str, val_col: str, agg: str, freq: str) -> pd.Series:
    t = raw[date_col]
    if pd.api.types.is_datetime64_any_dtype(t):
        dt = pd.to_datetime(t)
    else:
        num = pd.to_numeric(t, errors="coerce")
        if num.notna().all() and num.between(1900, 2100).all():
            dt = pd.to_datetime(num.astype(int).astype(str), format="%Y")
        else:
            ts = t.astype(str)
            dt = pd.to_datetime(ts, format="ISO8601", errors="coerce")   # ISO primero (evita inversión día/mes)
            if dt.isna().any():
                dt = dt.fillna(pd.to_datetime(ts, dayfirst=True, errors="coerce"))
    d = pd.DataFrame({"t": dt, "v": pd.to_numeric(raw[val_col], errors="coerce")}).dropna(subset=["t"])
    d = d[d["t"] >= pd.Timestamp("1990-01-01")]
    d["p"] = dp.period_start(d["t"], freq)
    return d.groupby("p")["v"].agg("sum" if agg == "Suma" else "mean")


with t_ext:
    st.subheader("Comparar con una serie externa")
    st.caption("Sube una serie con columna de fecha y una o más columnas numéricas (p. ej., el gasto oncológico SIS/FISSAL mensual del "
               "Observatorio del Financiamiento Oncológico). Se alinea a la granularidad elegida en el panel lateral.")
    up = st.file_uploader("Archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="ext_up")
    if up is not None:
        try:
            raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {e}")
            raw = None
        if raw is not None and len(raw.columns) >= 2:
            e1, e2, e3, e4 = st.columns(4)
            dcol = e1.selectbox("Columna de fecha", list(raw.columns), key="ext_dcol")
            vcol = e2.selectbox("Columna de valor", [c for c in raw.columns if c != dcol], key="ext_vcol")
            agg = e3.radio("Agregar por", ["Suma", "Promedio"], horizontal=True, key="ext_agg")
            elabel = e4.text_input("Etiqueta", "Serie externa", key="ext_lbl")
            try:
                ext = parse_external(raw, dcol, vcol, agg, FREQ)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error al procesar la serie: {e}")
                ext = pd.Series(dtype=float)
            cft_s = dp.aggregate(DF, FREQ, MEASURE)["Total"]
            cft_s = cft_s.mask(dp.incomplete_flags(cft_s.index, FREQ, INC_MONTHS).values)
            j = pd.concat([cft_s.rename("CFT"), ext.rename("EXT")], axis=1, join="inner").dropna()
            if len(j) < 6:
                st.warning(f"Solo {len(j)} períodos en común con datos; se requieren al menos 6. Revisa fechas y granularidad.")
            else:
                st.success(f"{len(j)} períodos en común: {j.index.min():%b-%Y} a {j.index.max():%b-%Y}.")
                norm_mode = st.radio("Comparación visual", ["Índice base 100", "Z-score", "Valores originales (doble eje)"], horizontal=True, key="ext_norm")
                if norm_mode.startswith("Valores"):
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_scatter(x=j.index, y=j["CFT"], name=f"CFT — {measure_label}", line=dict(color=PAL[0], width=2.6), secondary_y=False)
                    fig.add_scatter(x=j.index, y=j["EXT"], name=elabel, line=dict(color=PAL[1], width=2.6), secondary_y=True)
                    fig.update_yaxes(title_text=measure_unit(MEASURE), secondary_y=False)
                    fig.update_yaxes(title_text=elabel, secondary_y=True)
                else:
                    jn = j / j.iloc[0] * 100 if norm_mode.startswith("Índice") else (j - j.mean()) / j.std(ddof=0)
                    fig = go.Figure()
                    fig.add_scatter(x=jn.index, y=jn["CFT"], name=f"CFT — {measure_label}", line=dict(color=PAL[0], width=2.6))
                    fig.add_scatter(x=jn.index, y=jn["EXT"], name=elabel, line=dict(color=PAL[1], width=2.6))
                style_fig(fig, height, "" if norm_mode.startswith("Valores") else norm_mode)
                st.plotly_chart(fig, **STRETCH)
                note("Ambas series se muestran en una escala comparable (" + ("índice base 100 en el primer período" if norm_mode.startswith("Índice") else
                     "z-score" if norm_mode.startswith("Z") else "doble eje, cada una con su propia escala") +
                     "). Si suben y bajan juntas hay una asociación visual; la prueba de abajo la cuantifica.")
                use_diff = st.checkbox("Correlacionar primeras diferencias (recomendado: evita correlación espuria entre series con tendencia)",
                                       value=True, key="ext_diff")
                jj = j.diff().dropna() if use_diff else j
                pr_, pp_ = sps.pearsonr(jj["CFT"], jj["EXT"])
                sr_, sp_ = sps.spearmanr(jj["CFT"], jj["EXT"])
                m1, m2, m3 = st.columns(3)
                m1.metric("Pearson r", f"{pr_:+.2f}", f"p={pp_:.3f}", delta_color="off")
                m2.metric("Spearman ρ", f"{sr_:+.2f}", f"p={sp_:.3f}", delta_color="off")
                m3.metric("n", f"{len(jj)}")
                fuerza = "muy débil" if abs(pr_) < 0.2 else "débil" if abs(pr_) < 0.4 else "moderada" if abs(pr_) < 0.6 else "fuerte"
                note(f"Correlación **{fuerza}** y {'positiva' if pr_ > 0 else 'negativa'} (r = {pr_:+.2f}, p = {pp_:.3f}): "
                     + ("es estadísticamente distinguible de cero. " if pp_ < 0.05 else "no se distingue de cero con estos datos. ")
                     + ("Se calculó sobre primeras diferencias, es decir, si los cambios de un período a otro coinciden. " if use_diff else
                        "Se calculó sobre los niveles: con series que crecen, esto puede dar correlaciones espurias. ")
                     + "Correlación no implica causalidad.")
                maxlag = int(min(12 if FREQ == "M" else 4, len(jj) // 3))
                cc = []
                for k in range(-maxlag, maxlag + 1):
                    a = jj["CFT"]
                    b = jj["EXT"].shift(-k)
                    m = pd.concat([a, b], axis=1).dropna()
                    cc.append((k, sps.pearsonr(m.iloc[:, 0], m.iloc[:, 1])[0] if len(m) >= 6 else np.nan))
                ccd = pd.DataFrame(cc, columns=["rezago", "r"])
                best = ccd.loc[ccd["r"].abs().idxmax()] if ccd["r"].notna().any() else None
                fc = go.Figure(go.Bar(x=ccd["rezago"], y=ccd["r"], marker_color=[PAL[1] if (best is not None and k == best["rezago"]) else PAL[0] for k in ccd["rezago"]]))
                style_fig(fc, 320, "Correlación cruzada (r)")
                fc.update_xaxes(title="Rezago (períodos): k>0 → la serie externa va DESPUÉS del CFT; k<0 → la externa se ADELANTA")
                fc.update_layout(hovermode="closest")
                st.plotly_chart(fc, **STRETCH)
                if best is not None:
                    st.markdown(f'<div class="interp-box">Mayor |r| = <b>{best["r"]:+.2f}</b> con rezago <b>{int(best["rezago"])}</b> período(s). '
                                f'Con pocas observaciones y muchos rezagos evaluados, este máximo puede ser azar; trátalo como exploratorio.</div>',
                                unsafe_allow_html=True)
    else:
        st.info("Sube un archivo para comparar. Formatos de fecha aceptados: 2021-01-31, 31/01/2021, enero 2021 o solo el año (2021).")

# ============================== 9. TABLA Y DESCARGA ==========================
with t_table:
    st.subheader("Tabla y descarga")
    tw = hide_incomplete(W, FLAGS, gap_mode, FREQ) if not W.empty else WT_raw
    out_w = tw.copy()
    out_w.insert(0, "Incompleto", FLAGS.values)
    out_w.index = out_w.index.strftime("%Y-%m-%d")
    st.markdown("**Serie agregada (según panel lateral)**")
    st.dataframe(out_w, **STRETCH)
    note(f"Serie agregada por {freq_label.lower()}: {len(out_w)} períodos, de los cuales {int(out_w['Incompleto'].sum())} están marcados como incompletos "
         "(sin cobertura completa de datos). Es la misma serie que alimenta el gráfico principal y el resto de la app, lista para descargar.")
    st.markdown("**Evaluaciones filtradas**")
    show_cols = ["fecha", "n_evaluacion", "dci", "concentracion", "forma_farmaceutica", "area_canon", "grupo_area", "decision",
                 "oncologico_reportado", "alto_costo_clave"]
    st.dataframe(DF[show_cols], **STRETCH, hide_index=True, column_config={
        "fecha": st.column_config.DateColumn("Fecha", format="DD-MM-YYYY")})
    note(f"Lista las {len(DF):,} evaluaciones que cumplen los filtros del panel, una por fila. Sirve para auditar cualquier cifra de los gráficos "
         "(por ejemplo, ver qué fármacos componen un pico).")

    def to_excel() -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            out_w.to_excel(xw, sheet_name="Serie")
            YT.to_excel(xw, sheet_name=f"Ene-{mes_txt} por año", index=False)
            DF.to_excel(xw, sheet_name="Evaluaciones", index=False)
            HC.to_excel(xw, sheet_name="Listado alto costo", index=False)
            pd.DataFrame({"Parámetro": ["Granularidad", "Medida", "Desglose", "Rango", "Decisión", "Alto costo", "Meses incompletos"],
                          "Valor": [freq_label, measure_label, breakdown, f"{lo_d} a {hi_d}", ", ".join(dec_sel), hc_mode, ", ".join(inc_sel)]}
                         ).to_excel(xw, sheet_name="Filtros", index=False)
        return buf.getvalue()

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.download_button("⬇️ Excel completo", to_excel(), "CFT_analisis.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **STRETCH)
    dcol2.download_button("⬇️ Serie (CSV)", out_w.to_csv().encode("utf-8-sig"), "CFT_serie.csv", "text/csv", **STRETCH)
    dcol3.download_button("⬇️ Evaluaciones (CSV)", DF.to_csv(index=False).encode("utf-8-sig"), "CFT_evaluaciones_filtradas.csv",
                          "text/csv", **STRETCH)
    if ss.get("main_fig") is not None:
        dcol4.download_button("⬇️ Gráfico principal (HTML)", ss["main_fig"].to_html(include_plotlyjs="cdn").encode("utf-8"),
                              "CFT_grafico.html", "text/html", **STRETCH)

# ============================== 10. DATOS ====================================
with t_data:
    st.subheader("Datos, calidad y actualización")
    resumen = CFT.groupby(CFT["fecha"].dt.year).agg(evaluaciones=("dci", "size"), aprobadas=("decision", lambda s: int((s == "APROBADO").sum())),
                                                    denegadas=("decision", lambda s: int((s == "DENEGADO").sum())),
                                                    primera=("fecha", "min"), ultima=("fecha", "max"),
                                                    fechas_corregidas=("fecha_corregida", "sum")).reset_index(names="año")
    st.dataframe(resumen, **STRETCH, hide_index=True, column_config={
        "primera": st.column_config.DateColumn("Primera fecha", format="DD-MM-YYYY"),
        "ultima": st.column_config.DateColumn("Última fecha", format="DD-MM-YYYY")})
    partial_y = [int(r.año) for r in resumen.itertuples() if (r.ultima.month < 12 and r.año <= CFT["fecha"].dt.year.max())]
    note(f"Resume por año cuántas evaluaciones hay, hasta qué fecha llegan los datos y cuántas fechas se corrigieron. "
         + (f"Los años cuya última fecha es anterior a diciembre ({', '.join(map(str, partial_y))}) tienen cobertura incompleta: sus totales anuales subestiman el año real. " if partial_y else "")
         + "Los datos vienen de una hoja por año; conviene revisar esta tabla cada vez que se cargue información nueva.")
    st.caption(f"Meses marcados como sin cobertura completa: {', '.join(inc_sel) or 'ninguno'}.")

    st.markdown("#### Incorporar datos nuevos")
    u1, u2 = st.columns(2)
    with u1:
        up_cft = st.file_uploader("Excel de aprobaciones del CFT (una hoja por año)", type=["xlsx"], key="up_cft")
        if up_cft is not None:
            try:
                new_cft, rep = dp.parse_cft_excel(up_cft, ALIAS)
                st.success(f"{len(new_cft):,} evaluaciones ({new_cft['fecha'].min():%d-%m-%Y} a {new_cft['fecha'].max():%d-%m-%Y}). "
                           f"Duplicados entre hojas descartados: {rep['duplicados_descartados']} · fechas con año corregido: {rep['fechas_corregidas']}.")
                if st.button("Usar esta base de aprobaciones", key="use_cft"):
                    ss["cft"] = new_cft
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo procesar: {e}")
    with u2:
        up_hc = st.file_uploader("Listado no PNUME de alto costo (Excel)", type=["xlsx", "xls"], key="up_hc")
        if up_hc is not None:
            try:
                new_hc, new_alias = dp.parse_hc_excel(up_hc, ALIAS)
                added = len(new_alias) - len(ALIAS)
                st.success(f"{len(new_hc)} ítems · {new_hc['clave'].nunique()} principios activos"
                           + (f" · {added} alias nuevos generados automáticamente (revísalos abajo)." if added else "."))
                if st.button("Usar este listado", key="use_hc"):
                    ss["hc"], ss["alias"] = new_hc, new_alias
                    ss["cft"] = dp.reapply_keys(ss["cft"], new_alias)
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo procesar: {e}")

    with st.expander("Alias de principios activos (cómo se vincula el listado con los DCI del CFT)"):
        st.caption("`patrones` = textos (separados por |, sin tildes, en mayúsculas) que identifican al fármaco dentro del DCI del CFT.")
        edited = st.data_editor(ALIAS, num_rows="dynamic", hide_index=True, key="alias_ed", **STRETCH)
        if st.button("Aplicar alias", key="apply_alias"):
            ss["alias"] = edited.fillna("")
            ss["cft"] = dp.reapply_keys(ss["cft"], ss["alias"])
            st.rerun()
    c1_, c2_, c3_ = st.columns(3)
    c1_.download_button("⬇️ Base CFT consolidada (CSV)", CFT.to_csv(index=False).encode("utf-8-sig"), "cft_evaluaciones.csv", "text/csv", **STRETCH)
    c2_.download_button("⬇️ Listado alto costo (CSV)", HC.to_csv(index=False).encode("utf-8-sig"), "alto_costo.csv", "text/csv", **STRETCH)
    if c3_.button("↩️ Restaurar datos incluidos", **STRETCH):
        for k in ("cft", "hc", "alias"):
            ss.pop(k, None)
        st.rerun()

    with st.expander("Notas metodológicas"):
        st.markdown("""
- **Unidad de análisis:** una fila = una solicitud evaluada por el CFT (DCI, concentración y forma). No equivale a pacientes ni a gasto.
- **Duplicados entre hojas:** filas con igual N.° de evaluación, DCI y fecha en una hoja anterior se descartan (la hoja 2022 repite filas de 2021).
- **Fechas con año distinto** al de su hoja se asumen error de digitación y se corrigen (`fecha_corregida` = 1).
- **Cobertura:** meses sin registros o de hojas actualizadas solo hasta cierta fecha se marcan como incompletos; por eso las comparaciones anuales usan enero–mes de corte.
- **Clasificación por área:** «Oncología» agrupa Oncología Médica y Pediátrica (la columna oncológico/no oncológico existe solo desde 2025).
- **Alto costo:** cruce por principio activo con la tabla de alias; no distingue presentaciones. El listado es una foto a una fecha, no una serie histórica.
- **Principios activos distintos / 1.ª vez:** aproximados (variantes de escritura normalizadas de forma básica).
- **Pruebas estadísticas:** Mann-Kendall con corrección por autocorrelación, Chow para cambio de pendiente, Holt amortiguado con bootstrap para proyección.
  Resultados exploratorios; asociación temporal no implica causalidad.""")

st.markdown("---")
st.caption("Fuente: listados de medicamentos evaluados por el Comité Farmacoterapéutico y listado no PNUME de alto costo — INEN. "
           "Elaboración propia. Estructura y métodos adaptados del Observatorio del Financiamiento Oncológico INEN.")
