"""
data_processing.py
------------------
Carga, limpieza y agregación de las evaluaciones del Comité Farmacoterapéutico
(CFT) del INEN y del listado no PNUME de alto costo.

Reglas de limpieza (idénticas a las usadas en el análisis original):
  * Una hoja por año (2021, 2022, ...). El encabezado se localiza buscando la
    fila cuya 2.ª columna dice "AÑO".
  * Las filas de una hoja que ya existen en una hoja anterior (mismo N° de
    evaluación, p. ej. 078-CFT-2021 repetido en la hoja 2022) se descartan.
  * Si la fecha de una fila tiene un año distinto al de su hoja, se asume error
    de digitación y se corrige el año (columna `fecha_corregida` = 1).
  * Se cuenta por fila (una fila = una solicitud evaluada), no por N°.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
GRUPOS_AREA = [
    "Oncología (médica/pediátrica)", "Hematología/Trasplante", "Infectología",
    "Med. nuclear / Radiodiagnóstico", "Soporte nutricional", "Otras áreas",
]

# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def norm_txt(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", s)).strip()


def dci_base(d: str) -> str:
    """Normalización básica del DCI para contar principios activos distintos."""
    d = re.sub(r"RRUBICIN", "RUBICIN", d)
    d = d.replace("CLADRIBINE", "CLADRIBINA")
    d = re.sub(r"^(CLORHIDRATO|CLOHIDRATO|SULFATO|ACIDO) DE ", "", d)
    return d


_AREA_RULES = [
    ("ONCOLOGIA MEDICA", "Oncología médica"), ("ONCOLOGIA PEDIATRICA", "Oncología pediátrica"),
    ("RADIODIAG", "Radiodiagnóstico"), ("CIRUGIA UROLOGICA", "Cirugía urológica"),
    ("CIRUJIA UROLOGICA", "Cirugía urológica"), ("MEDICINA NUCLEAR", "Medicina nuclear"),
    ("INFECTOLOGIA", "Infectología"), ("SOPORTE NUTRICIONAL", "Soporte nutricional"),
    ("TERAPIA METABOLICA", "Terapia metabólica"), ("MEDICINA CRITICA", "Medicina crítica"),
    ("SALUD MENTAL", "Salud mental"), ("ANESTESIOLOGIA", "Anestesiología"),
    ("GASTRO", "Gastroenterología"), ("MAMAS", "Mamas y tejidos blandos"),
    ("HEMATOLOGIA", "Hematología"), ("OFTAMOLOGIA", "Oftalmología"), ("NEUMOLOGIA", "Neumología"),
    ("NUEMOLOGIA", "Neumología"), ("NEFROLOGIA", "Nefrología"), ("CARDIOLOGIA", "Cardiología"),
    ("TRASPLANTE", "Trasplante de progenitores"), ("MEDICINA DEL DOLOR", "Medicina del dolor"),
    ("BANCO DE SANGRE", "Banco de sangre"),
]


def canon_area(a) -> str:
    n = norm_txt(a)
    for k, v in _AREA_RULES:
        if k in n:
            return v
    if re.search(r"\bUCI\b", n):
        return "UCI / Medicina crítica"
    return n.title() if n else "Sin dato"


def grupo_area(a) -> str:
    n = norm_txt(a)
    if any(k in n for k in ("ONCOLOGIA", "ONCOLOGICA", "ONCOLOGICO")):
        return GRUPOS_AREA[0]
    if "HEMATOLOGIA" in n or "TRASPLANTE" in n:
        return GRUPOS_AREA[1]
    if "INFECTOLOGIA" in n:
        return GRUPOS_AREA[2]
    if "MEDICINA NUCLEAR" in n or "RADIODIAG" in n:
        return GRUPOS_AREA[3]
    if "NUTRICIONAL" in n:
        return GRUPOS_AREA[4]
    return GRUPOS_AREA[5]


def _nkey(neval) -> tuple | None:
    """'092-CFT-2021' y '92-CFT-2021' -> (92, 2021)."""
    m = re.match(r"\s*0*(\d+)\s*-\s*CFT\s*-\s*(\d{4})", str(neval).upper())
    return (int(m.group(1)), int(m.group(2))) if m else None


# ---------------------------------------------------------------------------
# Alto costo: alias, listado y asignación de claves
# ---------------------------------------------------------------------------

def load_alias(path: Path = DATA_DIR / "alias_alto_costo.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def load_hc(path: Path = DATA_DIR / "alto_costo.csv") -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"siga": str, "sismed": str})
    d["precio_ref"] = pd.to_numeric(d["precio_ref"], errors="coerce")
    return d


def load_events(path: Path = DATA_DIR / "eventos.csv") -> pd.DataFrame:
    d = pd.read_csv(path)
    d["fecha"] = pd.to_datetime(d["fecha"], format="ISO8601")
    return d


def assign_keys(dcin: pd.Series, alias: pd.DataFrame) -> pd.Series:
    pats = [(r.clave, [p for p in r.patrones.split("|") if p]) for r in alias.itertuples()]

    def f(d):
        for k, ps in pats:
            if any(p in d for p in ps):
                return k
        return None
    return dcin.map(f)


def key_names(alias: pd.DataFrame) -> dict[str, str]:
    return {r.clave: (r.nombre or r.clave.title()) for r in alias.itertuples()}


def parse_hc_excel(file, alias: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee el listado no PNUME de alto costo (cualquier hoja con columnas
    DESCRIPCIÓN y PRECIO). Devuelve (listado, alias actualizado): los ítems
    cuyo principio activo no está en la tabla de alias reciben uno automático
    (primera palabra de la descripción)."""
    raw = pd.read_excel(file)
    cols = {c: norm_txt(c) for c in raw.columns}

    def pick(*keys):
        for c, n in cols.items():
            if any(k in n for k in keys):
                return c
        return None
    c_desc, c_price = pick("DESCRIPCION"), pick("PRECIO")
    if c_desc is None or c_price is None:
        raise ValueError("El listado debe tener columnas 'DESCRIPCIÓN' y 'PRECIO ...'.")
    c_n, c_siga, c_sis = pick("N"), pick("CODIGO SIGA"), pick("CODIGO SISMED")
    c_prog, c_obs = pick("PROGRAMACION"), pick("OBS")
    out = pd.DataFrame({
        "n": raw[c_n] if c_n in raw else range(1, len(raw) + 1),
        "siga": raw[c_siga].astype(str) if c_siga else "",
        "sismed": raw[c_sis].astype(str) if c_sis else "",
        "descripcion": raw[c_desc].astype(str).str.strip(),
        "programacion": raw[c_prog].astype(str).str.replace(r"\s+", " ", regex=True).str.strip() if c_prog else "",
        "precio_ref": pd.to_numeric(raw[c_price], errors="coerce"),
        "obs": raw[c_obs] if c_obs else "",
    })
    out = out[out["descripcion"].str.len() > 2].copy()
    alias = alias.copy()
    new = []
    keys = []
    for desc in out["descripcion"]:
        d = norm_txt(desc)
        hit = None
        for r in alias.itertuples():
            if any(p and p in d for p in r.patrones.split("|")):
                hit = r.clave
                break
        if hit is None:
            hit = d.split(" ")[0]
            new.append({"clave": hit, "nombre": hit.title(), "patrones": hit})
            alias = pd.concat([alias, pd.DataFrame([new[-1]])], ignore_index=True)
        keys.append(hit)
    out["clave"] = keys
    return out.reset_index(drop=True), alias


# ---------------------------------------------------------------------------
# CFT: lectura del Excel de aprobaciones
# ---------------------------------------------------------------------------

def _std_columns(cols: list[str]) -> dict[str, str]:
    m = {}
    for c in cols:
        n = norm_txt(c)
        if n.startswith("N DE EVALUACION"):
            m[c] = "n_evaluacion"
        elif n.startswith("FECHA DE EVALUACION"):
            m[c] = "fecha"
        elif n.startswith("DENOMINACION COMUN"):
            m[c] = "dci"
        elif n.startswith("CONCENTRACION"):
            m[c] = "concentracion"
        elif n in ("F F", "FF"):
            m[c] = "forma_farmaceutica"
        elif n.startswith("AREA"):
            m[c] = "area_solicitante"
        elif n.startswith("DECISION"):
            m[c] = "decision"
        elif n.startswith("TRATAMIENTO"):
            m[c] = "oncologico_reportado"
    return m


def parse_cft_excel(file, alias: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Consolida todas las hojas-año del Excel de aprobaciones del CFT.
    Devuelve (base, reporte) donde `reporte` resume duplicados y correcciones."""
    xl = pd.ExcelFile(file)
    sheets = sorted([s for s in xl.sheet_names if re.fullmatch(r"\d{4}", str(s).strip())])
    if not sheets:
        raise ValueError("No se encontraron hojas con nombre de año (2021, 2022, ...).")
    frames, seen, rep = [], set(), {"duplicados_descartados": 0, "fechas_corregidas": 0, "sin_fecha": 0, "hojas": sheets}
    for s in sheets:
        raw = xl.parse(s, header=None)
        hits = raw.index[raw.iloc[:, 1].astype(str).str.strip().str.upper() == "AÑO"]
        if len(hits) == 0:
            continue
        hi = hits[0]
        cols = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.iloc[hi]]
        d = raw.iloc[hi + 1:].copy()
        d.columns = cols
        d = d.rename(columns=_std_columns(cols))
        for need in ("n_evaluacion", "fecha", "dci", "decision"):
            if need not in d:
                raise ValueError(f"Hoja {s}: falta la columna '{need}'.")
        for opt in ("concentracion", "forma_farmaceutica", "area_solicitante", "oncologico_reportado"):
            if opt not in d:
                d[opt] = np.nan
        d = d[d["dci"].notna() & (d["dci"].astype(str).str.strip() != "")].copy()
        d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
        rep["sin_fecha"] += int(d["fecha"].isna().sum())
        d = d[d["fecha"].notna()].copy()
        year = int(s)
        d["_k"] = [(_nkey(n), norm_txt(x), t.normalize()) for n, x, t in zip(d["n_evaluacion"], d["dci"], d["fecha"])]
        dup = d["_k"].map(lambda k: k[0] is not None and k in seen)
        rep["duplicados_descartados"] += int(dup.sum())
        d = d[~dup].copy()
        seen.update(k for k in d["_k"] if k[0] is not None)
        bad = d["fecha"].dt.year != year
        d["fecha_corregida"] = bad.astype(int)
        rep["fechas_corregidas"] += int(bad.sum())
        if bad.any():
            d.loc[bad, "fecha"] = d.loc[bad, "fecha"].map(lambda t: t.replace(year=year, day=min(t.day, 28)) if (t.month == 2 and t.day > 28) else t.replace(year=year))
        d["anio"] = year
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return _finalize(df, alias), rep


def _finalize(df: pd.DataFrame, alias: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], format="ISO8601") if df["fecha"].dtype == object else pd.to_datetime(df["fecha"])
    df["n_evaluacion"] = df["n_evaluacion"].astype(str).str.strip()
    df["dci"] = df["dci"].astype(str).str.strip()
    dec = df["decision"].astype(str).str.upper().str.strip()
    df["decision"] = np.where(dec.str.startswith("APROBAD"), "APROBADO", np.where(dec.str.startswith("DENEG"), "DENEGADO", dec))
    df["area_solicitante"] = df["area_solicitante"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["area_canon"] = df["area_solicitante"].map(canon_area)
    df["grupo_area"] = df["area_solicitante"].map(grupo_area)
    dcin = df["dci"].map(norm_txt)
    df["alto_costo_clave"] = assign_keys(dcin, alias)
    df["dci_base"] = dcin.map(dci_base)
    df = df.sort_values(["fecha", "n_evaluacion"], kind="stable").reset_index(drop=True)
    df["primera_vez_serie"] = (~df["dci_base"].duplicated()).astype(int)
    if "fecha_corregida" not in df:
        df["fecha_corregida"] = 0
    keep = ["fecha", "anio", "n_evaluacion", "dci", "concentracion", "forma_farmaceutica", "area_solicitante",
            "area_canon", "grupo_area", "decision", "oncologico_reportado", "alto_costo_clave", "dci_base",
            "primera_vez_serie", "fecha_corregida"]
    return df[keep]


def load_cft(path: Path = DATA_DIR / "cft_evaluaciones.csv") -> pd.DataFrame:
    d = pd.read_csv(path)
    d["fecha"] = pd.to_datetime(d["fecha"], format="ISO8601")
    return d


def reapply_keys(df: pd.DataFrame, alias: pd.DataFrame) -> pd.DataFrame:
    """Recalcula la clave de alto costo si cambia la tabla de alias."""
    df = df.copy()
    df["alto_costo_clave"] = assign_keys(df["dci_base"], alias)
    return df


# ---------------------------------------------------------------------------
# Cobertura y agregación temporal
# ---------------------------------------------------------------------------

FREQ_LABELS = {"Mensual": "M", "Trimestral": "Q", "Semestral": "S", "Anual": "A"}
PERIODS_PER_YEAR = {"M": 12, "Q": 4, "S": 2, "A": 1}
MEASURES = {
    "Evaluaciones (requerimientos)": "evaluaciones",
    "Aprobadas": "aprobadas",
    "Denegadas": "denegadas",
    "% de aprobación": "pct_aprob",
    "Principios activos distintos (aprox.)": "dci_distintos",
    "Principios activos evaluados por 1.ª vez (aprox.)": "dci_primera_vez",
}
ADDITIVE = {"evaluaciones", "aprobadas", "denegadas", "dci_primera_vez"}


def default_incomplete_months(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Meses sin cobertura completa: (a) meses sin ningún registro entre el primer
    y el último mes con datos; (b) el último mes del período si su última fecha
    cae antes del día 25; (c) para cada hoja-año que termina antes de diciembre
    (y no es el año en curso), los meses restantes del año."""
    m = df["fecha"].dt.to_period("M").dt.to_timestamp()
    last = df["fecha"].max()
    last_m = last.to_period("M").to_timestamp()
    full = pd.date_range(m.min(), last_m, freq="MS")
    inc = set(full.difference(pd.DatetimeIndex(m.unique())))
    if last.day < 25:
        inc.add(last_m)
    for y, g in df.groupby("anio"):
        gl = g["fecha"].max()
        if gl.year != y or gl.month >= 12 or y >= last.year:
            continue
        for mm in range(gl.month + 1, 13):
            inc.add(pd.Timestamp(year=int(y), month=mm, day=1))
        if gl.day < 25:
            inc.add(gl.to_period("M").to_timestamp())
    return sorted(inc)


def period_start(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "M":
        return dates.dt.to_period("M").dt.to_timestamp()
    if freq == "Q":
        return dates.dt.to_period("Q").dt.to_timestamp()
    if freq == "S":
        m = np.where(dates.dt.month <= 6, 1, 7)
        return pd.to_datetime(pd.DataFrame({"year": dates.dt.year, "month": m, "day": 1}))
    return pd.to_datetime(pd.DataFrame({"year": dates.dt.year, "month": 1, "day": 1}))


def period_index(lo: pd.Timestamp, hi: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    lo_p = period_start(pd.Series([lo]), freq).iloc[0]
    hi_p = period_start(pd.Series([hi]), freq).iloc[0]
    step = {"M": "MS", "Q": "QS", "S": "6MS", "A": "YS"}[freq]
    return pd.date_range(lo_p, hi_p, freq=step)


def _measure_series(g: pd.DataFrame, measure: str) -> pd.Series:
    if measure == "evaluaciones":
        return g.groupby("_p").size()
    if measure == "aprobadas":
        return g[g["decision"] == "APROBADO"].groupby("_p").size()
    if measure == "denegadas":
        return g[g["decision"] == "DENEGADO"].groupby("_p").size()
    if measure == "dci_primera_vez":
        return g.groupby("_p")["primera_vez_serie"].sum()
    if measure == "dci_distintos":
        return g.groupby("_p")["dci_base"].nunique()
    raise ValueError(measure)


def aggregate(df: pd.DataFrame, freq: str, measure: str, group_col: str | None = None,
              lo: pd.Timestamp | None = None, hi: pd.Timestamp | None = None,
              groups: list | None = None) -> pd.DataFrame:
    """Serie agregada por período. Devuelve DataFrame ancho (índice = inicio del
    período; columnas = grupos o 'Total'). Períodos sin filas -> 0 (o NaN para
    % de aprobación)."""
    lo = lo or df["fecha"].min()
    hi = hi or df["fecha"].max()
    idx = period_index(lo, hi, freq)
    d = df.copy()
    d["_p"] = period_start(d["fecha"], freq)
    if group_col is None:
        d["_g"] = "Total"
    else:
        d["_g"] = d[group_col].fillna("Sin dato").astype(str)
    cols = groups if groups is not None else sorted(d["_g"].unique())
    if not cols:
        cols = ["Total"]
    out = {}
    for c in cols:
        g = d[d["_g"] == c]
        if measure == "pct_aprob":
            a = g[g["decision"] == "APROBADO"].groupby("_p").size().reindex(idx, fill_value=0)
            t = g.groupby("_p").size().reindex(idx, fill_value=0)
            out[c] = (a / t.replace(0, np.nan) * 100)
        else:
            s = _measure_series(g, measure) if len(g) else pd.Series(dtype=float)
            out[c] = s.reindex(idx, fill_value=0).astype(float)
    res = pd.DataFrame(out, index=idx)
    res.index.name = "periodo"
    return res


def incomplete_flags(idx: pd.DatetimeIndex, freq: str, inc_months: list[pd.Timestamp]) -> pd.Series:
    """True si el período contiene algún mes sin cobertura completa."""
    inc = pd.Series(pd.DatetimeIndex(inc_months))
    if inc.empty:
        return pd.Series(False, index=idx)
    ps = set(period_start(inc, freq))
    return pd.Series([t in ps for t in idx], index=idx)


def ytd_table(df: pd.DataFrame, month_end: int, measure: str, inc_months: list[pd.Timestamp]) -> pd.DataFrame:
    """Valor acumulado de enero a `month_end` por año (comparable entre años)."""
    d = df[df["fecha"].dt.month <= month_end].copy()
    d["_p"] = d["fecha"].dt.year
    years = list(range(int(df["fecha"].dt.year.min()), int(df["fecha"].dt.year.max()) + 1))
    if measure == "pct_aprob":
        a = d[d["decision"] == "APROBADO"].groupby("_p").size().reindex(years, fill_value=0)
        t = d.groupby("_p").size().reindex(years, fill_value=0)
        val = a / t.replace(0, np.nan) * 100
    else:
        val = _measure_series(d, measure).reindex(years, fill_value=0).astype(float)
    inc = pd.Series(pd.DatetimeIndex(inc_months)) if inc_months else pd.Series([], dtype="datetime64[ns]")
    flags = []
    for y in years:
        bad = any((t.year == y and t.month <= month_end) for t in inc)
        # año en curso: el corte supera la última fecha disponible
        if pd.Timestamp(year=y, month=month_end, day=1) > df["fecha"].max().to_period("M").to_timestamp():
            bad = True
        flags.append(bad)
    return pd.DataFrame({"anio": years, "valor": val.values, "incompleto": flags})


def hc_first_year(df: pd.DataFrame) -> pd.Series:
    d = df[df["alto_costo_clave"].notna()]
    return d.groupby("alto_costo_clave")["fecha"].min().dt.year


# ---------------------------------------------------------------------------
# Comparaciones entre años: rangos de meses, estacionalidad y estimación de año completo
# ---------------------------------------------------------------------------

def range_table(df: pd.DataFrame, m_from: int, m_to: int, measure: str, inc_months: list[pd.Timestamp]) -> pd.DataFrame:
    """Valor por año restringido a los meses [m_from, m_to] (generaliza ytd_table)."""
    d = df[(df["fecha"].dt.month >= m_from) & (df["fecha"].dt.month <= m_to)].copy()
    d["_p"] = d["fecha"].dt.year
    years = list(range(int(df["fecha"].dt.year.min()), int(df["fecha"].dt.year.max()) + 1))
    if measure == "pct_aprob":
        a = d[d["decision"] == "APROBADO"].groupby("_p").size().reindex(years, fill_value=0)
        t = d.groupby("_p").size().reindex(years, fill_value=0)
        val = a / t.replace(0, np.nan) * 100
    else:
        val = _measure_series(d, measure).reindex(years, fill_value=0).astype(float)
    incs = {(t.year, t.month) for t in inc_months}
    last = df["fecha"].max()
    flags = []
    for y in years:
        bad = any((y, m) in incs or (y, m) > (last.year, last.month) for m in range(m_from, m_to + 1))
        flags.append(bad)
    return pd.DataFrame({"anio": years, "valor": val.values, "incompleto": flags})


def monthly_pivot(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Tabla año × mes (1..12) de una medida aditiva."""
    w = aggregate(df, "M", measure)["Total"]
    p = pd.DataFrame({"y": w.index.year, "m": w.index.month, "v": w.values}).pivot(index="y", columns="m", values="v")
    return p.reindex(columns=range(1, 13))


def covered_months(y: int, inc_months: list[pd.Timestamp], last_date: pd.Timestamp) -> list[int]:
    incs = {(t.year, t.month) for t in inc_months}
    return [m for m in range(1, 13) if (y, m) not in incs and (y, m) <= (last_date.year, last_date.month)]


def complete_years(piv: pd.DataFrame, inc_months: list[pd.Timestamp], last_date: pd.Timestamp) -> list[int]:
    return [int(y) for y in piv.index if len(covered_months(int(y), inc_months, last_date)) == 12]


def seasonal_shares(piv: pd.DataFrame, comp_years: list[int]) -> pd.DataFrame:
    """% del total anual que cae en cada mes, por año completo."""
    sub = piv.loc[comp_years].fillna(0)
    return sub.div(sub.sum(axis=1).replace(0, np.nan), axis=0)


def estimate_year(piv: pd.DataFrame, y: int, inc_months: list[pd.Timestamp], last_date: pd.Timestamp) -> dict | None:
    """Total anual estimado: lo observado en los meses con cobertura completa dividido por la
    fracción del año que esos meses suelen representar en los años completos (media, mín. y máx.)."""
    if y not in piv.index:
        return None
    cov = covered_months(int(y), inc_months, last_date)
    obs_cov = float(piv.loc[y, cov].fillna(0).sum())
    obs_all = float(piv.loc[y].fillna(0).sum())
    if len(cov) == 12:
        return {"obs": obs_all, "est": obs_cov, "lo": obs_cov, "hi": obs_cov, "complete": True, "cov": cov}
    comp = [c for c in complete_years(piv, inc_months, last_date) if c != y]
    if len(comp) < 2 or not cov:
        return None
    sh = seasonal_shares(piv, comp)[cov].sum(axis=1)
    if (sh <= 0).any():
        return None
    return {"obs": obs_all, "est": obs_cov / sh.mean(), "lo": obs_cov / sh.max(), "hi": obs_cov / sh.min(),
            "complete": False, "cov": cov, "n_comp": len(comp)}


def estimate_range(piv: pd.DataFrame, y: int, months: list[int], inc_months: list[pd.Timestamp],
                   last_date: pd.Timestamp) -> dict | None:
    """Total estimado de un conjunto de meses de un año (p. ej. jul–dic) cuando algunos meses no tienen
    cobertura completa: observado en los meses cubiertos ÷ fracción que esos meses suelen representar
    dentro del rango, según los años completos (media, mín., máx.)."""
    if y not in piv.index:
        return None
    months = list(months)
    cov = [m for m in covered_months(int(y), inc_months, last_date) if m in months]
    obs_all = float(piv.loc[y, months].fillna(0).sum())
    obs_cov = float(piv.loc[y, cov].fillna(0).sum())
    if len(cov) == len(months):
        return {"obs": obs_all, "est": obs_cov, "lo": obs_cov, "hi": obs_cov, "complete": True, "cov": cov}
    comp = [c for c in complete_years(piv, inc_months, last_date) if c != y]
    if not cov or len(comp) < 2:
        return None
    sh = seasonal_shares(piv, comp)
    ratio = sh[cov].sum(axis=1) / sh[months].sum(axis=1)
    if (ratio <= 0).any() or ratio.isna().any():
        return None
    return {"obs": obs_all, "est": obs_cov / ratio.mean(), "lo": obs_cov / ratio.max(), "hi": obs_cov / ratio.min(),
            "complete": False, "cov": cov}
