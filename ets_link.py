"""
Vínculo entre informes de Evaluación de Tecnologías Sanitarias (ETS) del INS y
las evaluaciones del Comité Farmacoterapéutico (CFT) del INEN.

- Cada informe ETS se lleva a una o más «moléculas» con un léxico de patrones.
  La primera molécula de la intervención es la «principal»; las demás, «combinación».
  Solo se lee el tramo de la intervención (antes de «para / en / como / vs»), para
  no tomar comparadores ni tratamientos previos descritos en la población.
- Las evaluaciones del CFT se llevan a la misma molécula con el mismo léxico.
- Estudio de eventos: evaluaciones del CFT por mes antes y después de la publicación
  (mes de publicación excluido), con exposición en meses de cobertura completa.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

DATA_DIR = Path(__file__).parent / "data"

# clave, nombre a mostrar, patrón (texto en minúsculas y sin tildes)
LEXICON: list[tuple[str, str, str]] = [
    ("TRASTUZUMAB_DXD", "Trastuzumab deruxtecan", r"trastuzumab\s+deruxtecan"),
    ("TRASTUZUMAB_DM1", "Trastuzumab emtansina", r"trastuzumab\s+emtansina|\bt\s?dm\s?1\b"),
    ("TRASTUZUMAB", "Trastuzumab", r"(?<!ado )trastuzumab(?!\s+(deruxtecan|emtansina))"),
    ("PEMBROLIZUMAB", "Pembrolizumab", r"pembrolizumab"),
    ("NIVOLUMAB", "Nivolumab", r"nivolumab"),
    ("IPILIMUMAB", "Ipilimumab", r"ipilimumab"),
    ("DURVALUMAB", "Durvalumab", r"durvalumab"),
    ("ATEZOLIZUMAB", "Atezolizumab", r"atezolizumab"),
    ("AVELUMAB", "Avelumab", r"avelumab"),
    ("CETUXIMAB", "Cetuximab", r"cetuximab"),
    ("BEVACIZUMAB", "Bevacizumab", r"bevacizumab"),
    ("PERTUZUMAB", "Pertuzumab", r"pertuzumab"),
    ("AMIVANTAMAB", "Amivantamab", r"amivanta[nm]ab"),
    ("DARATUMUMAB", "Daratumumab", r"daratumumab"),
    ("BLINATUMOMAB", "Blinatumomab", r"blinatum[ou]mab"),
    ("RAVULIZUMAB", "Ravulizumab", r"ravulizumab"),
    ("SECUKINUMAB", "Secukinumab", r"secukinumab"),
    ("PALIVIZUMAB", "Palivizumab", r"palivizumab"),
    ("TOCILIZUMAB", "Tocilizumab", r"tocilizumab"),
    ("ETANERCEPT", "Etanercept", r"etanercept"),
    ("RUXOLITINIB", "Ruxolitinib", r"ruxolitinib"),
    ("PONATINIB", "Ponatinib", r"ponatinib"),
    ("IBRUTINIB", "Ibrutinib", r"ibrutinib"),
    ("ACALABRUTINIB", "Acalabrutinib", r"acalabrutinib"),
    ("ENCORAFENIB", "Encorafenib", r"encorafenib"),
    ("BINIMETINIB", "Binimetinib", r"binimetinib"),
    ("VEMURAFENIB", "Vemurafenib", r"vemurafenib"),
    ("CABOZANTINIB", "Cabozantinib", r"cabozantinib"),
    ("SORAFENIB", "Sorafenib", r"sorafenib"),
    ("AXITINIB", "Axitinib", r"axitinib"),
    ("OSIMERTINIB", "Osimertinib", r"osimertinib"),
    ("TOFACITINIB", "Tofacitinib", r"tofacitinib"),
    ("ABEMACICLIB", "Abemaciclib", r"abemaciclib"),
    ("RIBOCICLIB", "Ribociclib", r"ribociclib"),
    ("PALBOCICLIB", "Palbociclib", r"palbociclib"),
    ("OLAPARIB", "Olaparib", r"olaparib"),
    ("TALAZOPARIB", "Talazoparib", r"talazoparib"),
    ("FULVESTRANT", "Fulvestrant", r"fulvestrant"),
    ("ENZALUTAMIDA", "Enzalutamida", r"enzalutamida"),
    ("APALUTAMIDA", "Apalutamida", r"apalutamida"),
    ("LENALIDOMIDA", "Lenalidomida", r"lenalidomida"),
    ("LUSPATERCEPT", "Luspatercept", r"luspatercept"),
    ("TIOTEPA", "Tiotepa", r"tiotepa"),
    ("TRIOXIDO_ARSENICO", "Trióxido de arsénico", r"trioxido\s+de\s+arsenico"),
    ("PEGASPARGASA", "Pegaspargasa (L-asparaginasa pegilada)", r"pegaspargasa|asparaginasa\b.*\bpegilad"),
    ("POSACONAZOL", "Posaconazol", r"posaconazol"),
    ("ANFOTERICINA_LIP", "Anfotericina B liposomal", r"[ae]n?fotericina\s+b\s+liposomal"),
    ("INMUNOGLOBULINA", "Inmunoglobulina humana", r"inmunoglobulina|\bivig\b"),
    ("RISDIPLAM", "Risdiplam", r"risdiplam"),
    ("IVACAFTOR", "Ivacaftor", r"ivacaftor"),
    ("DIMETILFUMARATO", "Dimetilfumarato", r"dimetilfumarato"),
    ("TECOVIRIMAT", "Tecovirimat", r"tecovirimat"),
    ("DOLUTEGRAVIR", "Dolutegravir", r"dolutegravir"),
    ("TEDUGLUTIDA", "Teduglutida", r"teduglutida"),
    ("DESVENLAFAXINA", "Desvenlafaxina", r"desvenlafaxina"),
    ("DULOXETINA", "Duloxetina", r"duloxetina"),
    ("ESCITALOPRAM", "Escitalopram", r"escitalopram"),
    ("ARIPIPRAZOL", "Aripiprazol", r"aripiprazol"),
    ("METILFENIDATO", "Metilfenidato", r"metilfenidato"),
    ("NALTREXONA", "Naltrexona", r"naltrexona"),
    ("DONEPEZILO", "Donepezilo", r"donepezilo"),
    ("MEMANTINA", "Memantina", r"memantina"),
    ("OXICODONA", "Oxicodona", r"oxicodona"),
    ("ACIDO_URSODESOXICOLICO", "Ácido ursodesoxicólico", r"ursode(s)?oxicolico"),
    ("ACEITE_IODADO", "Aceite iodado (Lipiodol)", r"aceite\s+iodado|lipiodol"),
    ("PERTECNETATO", "Pertecnetato de sodio Tc99m", r"pertecnetato"),
    ("TROLAMINA", "Trolamina", r"trolamina"),
    ("SITAGLIPTINA", "Sitagliptina", r"sitagliptina"),
    ("BEDAQUILINA", "Bedaquilina", r"bedaquilina"),
    ("DELAMANID", "Delamanid", r"delamanid"),
    ("RIFAPENTINA", "Rifapentina", r"rifapentina"),
    ("MEFLOQUINA", "Mefloquina", r"mefloquina"),
    ("LOPERAMIDA", "Loperamida", r"loperamida"),
    ("COLESTIRAMINA", "Colestiramina", r"colestiramina"),
    ("ATROPINA", "Atropina", r"atropina"),
    ("HIERRO_POLIMALTOSADO", "Hierro polimaltosado", r"hierro\s+polimaltosado"),
]
_LEX = [(k, n, re.compile(p)) for k, n, p in LEXICON]
NAMES = {k: n for k, n, _ in LEXICON}

DICT_ORDER = ["A favor", "Mixta", "En contra", "Sin recomendación formal"]
DICT_COLORS = {"A favor": "#2F9E44", "Mixta": "#C9A13B", "En contra": "#D9542F", "Sin recomendación formal": "#7D7D7D",
               "Sin ETS previa": "#C9D3DF"}
ONCO_AREAS = ["Oncología", "Oncohematología", "Hematología / TPH"]


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+ ]", " ", s)).strip()


_LEAD = re.compile(r"^(eficacia y seguridad (de|del)\s+(la |el |los |las )?(uso de |dispositivo )?|uso de )")
_CUT = re.compile(r"\s(para|en|como|vs|versus|comparad[oa])\s")


def intervention_text(tecnologia: str) -> str:
    t = _LEAD.sub("", _norm(tecnologia))
    m = _CUT.search(" " + t + " ")
    return t[: max(m.start() - 1, 0)] if m else t


def molecules_in(text: str, intervention_only: bool = True) -> list[str]:
    t = intervention_text(text) if intervention_only else _norm(text)
    hits = []
    for k, _, rx in _LEX:
        m = rx.search(t)
        if m:
            hits.append((m.start(), k))
    return [k for _, k in sorted(hits)]


def cft_key(dci) -> str | None:
    t = _norm(dci)
    best = None
    for k, _, rx in _LEX:
        m = rx.search(t)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), k)
    return best[1] if best else None


# ---------------------------------------------------------------------------
def load_ets(path_or_buf=DATA_DIR / "ets_informes.csv") -> pd.DataFrame:
    e = pd.read_csv(path_or_buf)
    need = {"codigo", "tecnologia", "fecha_pub", "dictamen_grupo"}
    miss = need - set(e.columns)
    if miss:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(miss))}")
    if "id" not in e.columns:
        e["id"] = range(1, len(e) + 1)
    e["fecha_pub"] = pd.to_datetime(e["fecha_pub"], errors="coerce")
    e = e.dropna(subset=["fecha_pub"]).copy()
    for c, d in {"tipo": "ETS", "area": "Sin área", "tipo_tec": "Medicamento", "solicitante": "No consignado",
                 "dictamen": "", "certeza": "", "pdf": ""}.items():
        if c not in e.columns:
            e[c] = d
        e[c] = e[c].fillna(d)
    e["dict_grp"] = e["dictamen_grupo"].where(e["dictamen_grupo"].isin(DICT_ORDER), "Sin recomendación formal")
    e["solic_inen"] = e["solicitante"].str.contains(r"\bINEN\b", case=False, na=False)
    return e


def link_table(ets: pd.DataFrame) -> pd.DataFrame:
    """Una fila por informe × molécula de la intervención."""
    rows = []
    for r in ets.itertuples():
        for i, k in enumerate(molecules_in(r.tecnologia)):
            rows.append({"id": r.id, "clave": k, "molecula": NAMES[k], "rol": "Principal" if i == 0 else "Combinación"})
    return pd.DataFrame(rows, columns=["id", "clave", "molecula", "rol"])


def tag_cft(cft: pd.DataFrame) -> pd.DataFrame:
    out = cft.copy()
    out["ets_clave"] = out["dci"].map(cft_key)
    return out


def valid_months(cft: pd.DataFrame, inc_months: list[pd.Timestamp]) -> pd.DatetimeIndex:
    lo = cft["fecha"].min().to_period("M").to_timestamp()
    hi = cft["fecha"].max().to_period("M").to_timestamp()
    allm = pd.date_range(lo, hi, freq="MS")
    inc = {pd.Timestamp(t).to_period("M").to_timestamp() for t in inc_months}
    return allm[~allm.isin(list(inc))]


def _mdiff(a: pd.Series | pd.Timestamp, b: pd.Timestamp):
    pa, pb = pd.Timestamp(b).to_period("M"), None
    if isinstance(a, pd.Series):
        p = a.dt.to_period("M")
        return (p.dt.year - pa.year) * 12 + (p.dt.month - pa.month)
    pb = pd.Timestamp(a).to_period("M")
    return (pb.year - pa.year) * 12 + (pb.month - pa.month)


def event_pairs(links: pd.DataFrame, ets: pd.DataFrame, cft: pd.DataFrame, window: int,
                months_ok: pd.DatetimeIndex) -> pd.DataFrame:
    """Evaluaciones del CFT antes y después de cada publicación (informe × molécula)."""
    if links.empty:
        return pd.DataFrame()
    E = links.merge(ets[["id", "codigo", "tipo", "tecnologia", "fecha_pub", "dictamen", "dict_grp", "area",
                         "solicitante", "solic_inen", "certeza", "pdf"]], on="id")
    ok_set = set(months_ok)
    ctrl = cft[~cft["ets_clave"].isin(links["clave"].unique())]
    cm = ctrl["fecha"].dt.to_period("M").dt.to_timestamp()
    ctrl = ctrl[cm.isin(list(ok_set))]
    recs = []
    for r in E.itertuples():
        g = cft[cft["ets_clave"] == r.clave]
        pm = pd.Timestamp(r.fecha_pub).to_period("M").to_timestamp()
        pre_m = [pm - pd.DateOffset(months=i) for i in range(1, window + 1)]
        post_m = [pm + pd.DateOffset(months=i) for i in range(1, window + 1)]
        exp_pre = sum(m in ok_set for m in pre_m)
        exp_post = sum(m in ok_set for m in post_m)
        gm = g["fecha"].dt.to_period("M").dt.to_timestamp()
        gv = g[gm.isin(list(ok_set))]
        rel = _mdiff(gv["fecha"], r.fecha_pub)
        pre = gv[(rel >= -window) & (rel <= -1)]
        post = gv[(rel >= 1) & (rel <= window)]
        crel = _mdiff(ctrl["fecha"], r.fecha_pub)
        after_all = g[g["fecha"] >= r.fecha_pub]
        before_all = g[g["fecha"] < r.fecha_pub]
        recs.append({
            "id": r.id, "codigo": r.codigo, "tipo": r.tipo, "fecha_pub": r.fecha_pub, "molecula": r.molecula,
            "clave": r.clave, "rol": r.rol, "dictamen": r.dictamen, "dict_grp": r.dict_grp, "area_ets": r.area,
            "solicitante": r.solicitante, "solic_inen": r.solic_inen, "certeza": r.certeza, "tecnologia": r.tecnologia,
            "pdf": r.pdf,
            "cft_total": len(g), "cft_antes": len(before_all), "cft_despues": len(after_all),
            "den_antes": int((before_all["decision"] == "DENEGADO").sum()),
            "den_despues": int((after_all["decision"] == "DENEGADO").sum()),
            "primera_cft": g["fecha"].min() if len(g) else pd.NaT,
            "ultima_cft": g["fecha"].max() if len(g) else pd.NaT,
            "n_pre": len(pre), "n_post": len(post), "exp_pre": exp_pre, "exp_post": exp_post,
            "apr_pre": int((pre["decision"] == "APROBADO").sum()), "apr_post": int((post["decision"] == "APROBADO").sum()),
            "ctrl_pre": int(((crel >= -window) & (crel <= -1)).sum()), "ctrl_post": int(((crel >= 1) & (crel <= window)).sum()),
        })
    P = pd.DataFrame(recs)
    P["tasa_pre"] = P["n_pre"] / P["exp_pre"].replace(0, np.nan) * 12
    P["tasa_post"] = P["n_post"] / P["exp_post"].replace(0, np.nan) * 12
    P["cft_previo"] = P["cft_antes"] > 0
    lag = (P["fecha_pub"] - P["primera_cft"]).dt.days / 30.4375
    P["meses_desde_1a_cft"] = lag.where(P["cft_previo"])
    return P


def event_time(P: pd.DataFrame, cft: pd.DataFrame, window: int, months_ok: pd.DatetimeIndex) -> pd.DataFrame:
    """Evaluaciones por par y mes relativo a la publicación, solo meses observados."""
    ok_set = set(months_ok)
    out = []
    for r in P.itertuples():
        pm = pd.Timestamp(r.fecha_pub).to_period("M").to_timestamp()
        g = cft[cft["ets_clave"] == r.clave]
        rel = _mdiff(g["fecha"], r.fecha_pub)
        cnt = rel.value_counts()
        for k in range(-window, window + 1):
            m = pm + pd.DateOffset(months=k)
            if m in ok_set:
                out.append({"par": f"{r.codigo}|{r.clave}", "dict_grp": r.dict_grp, "rel": k, "n": int(cnt.get(k, 0))})
    return pd.DataFrame(out, columns=["par", "dict_grp", "rel", "n"])


def rate_ratio(n_post: int, t_post: float, n_pre: int, t_pre: float) -> dict:
    """Razón de tasas post/pre (Poisson) con IC 95% exacto condicional (binomial)."""
    if t_post <= 0 or t_pre <= 0:
        return {"rr": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    N = n_post + n_pre
    rr = (n_post / t_post) / (n_pre / t_pre) if n_pre > 0 else np.inf
    if N == 0:
        return {"rr": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    p0 = t_post / (t_post + t_pre)
    bt = sps.binomtest(n_post, N, p0)
    ci = bt.proportion_ci(0.95, method="exact")
    conv = lambda q: (q / (1 - q)) * (t_pre / t_post) if q < 1 else np.inf  # noqa: E731
    return {"rr": rr, "lo": conv(ci.low), "hi": conv(ci.high), "p": bt.pvalue}


def prior_ets_status(cft: pd.DataFrame, links: pd.DataFrame, ets: pd.DataFrame) -> pd.Series:
    """Para cada evaluación del CFT: dictamen de la ETS más reciente publicada antes (misma molécula)."""
    L = links.merge(ets[["id", "fecha_pub", "dict_grp"]], on="id").sort_values("fecha_pub")
    res = pd.Series("Sin ETS previa", index=cft.index, dtype="object")
    for k, g in L.groupby("clave"):
        idx = cft.index[cft["ets_clave"] == k]
        if len(idx) == 0:
            continue
        f = cft.loc[idx, "fecha"]
        pos = np.searchsorted(g["fecha_pub"].values, f.values, side="right") - 1
        lab = np.where(pos >= 0, g["dict_grp"].values[np.clip(pos, 0, None)], "Sin ETS previa")
        res.loc[idx] = lab
    return res


def first_per_molecule(P: pd.DataFrame) -> pd.DataFrame:
    """Primera publicación de cada molécula; si hay varias ese día con dictámenes distintos, «Mixta»."""
    if P.empty:
        return P
    P = P.sort_values(["fecha_pub", "id"])
    first_date = P.groupby("clave")["fecha_pub"].transform("min")
    F = P[P["fecha_pub"] == first_date]
    mix = F.groupby("clave")["dict_grp"].nunique()
    F = F.drop_duplicates("clave").copy()
    F.loc[F["clave"].map(mix) > 1, "dict_grp"] = "Mixta"
    return F


def relative_rr(a_post: int, a_pre: int, c_post: int, c_pre: int) -> dict:
    """Razón de razones (expuestos vs control en las mismas ventanas), IC 95% log-normal."""
    if min(a_post, a_pre, c_post, c_pre) <= 0:
        return {"rrr": np.nan, "lo": np.nan, "hi": np.nan}
    r = (a_post / a_pre) / (c_post / c_pre)
    se = np.sqrt(1 / a_post + 1 / a_pre + 1 / c_post + 1 / c_pre)
    return {"rrr": r, "lo": r * np.exp(-1.96 * se), "hi": r * np.exp(1.96 * se)}
