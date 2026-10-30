"""
qa_engine.py
------------
«Pregunta a los datos» para el Observatorio del CFT.

Modo básico (por reglas): interpreta preguntas en español sobre las evaluaciones
del CFT (años, decisión, áreas, principios activos, alto costo, % de aprobación,
primera vez, evolución temporal, rankings) y responde con agregaciones de pandas.

Modo IA (opcional): envía a un modelo de Claude solo tablas resumidas (sin N.º de
evaluación ni otros identificadores de registro) para redactar la respuesta.
Todas las cifras provienen de las agregaciones calculadas aquí.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px

from data_processing import MONTHS_ES, norm_txt

EXAMPLES = [
    "¿Cuántas evaluaciones hubo por año?",
    "Top 10 principios activos evaluados en 2025",
    "Evolución anual del % de aprobación",
    "¿Qué áreas solicitan más evaluaciones?",
    "Medicamentos de alto costo evaluados en 2026",
    "Denegadas por grupo de área",
    "Principios activos evaluados por primera vez en 2025",
    "Evolución de pembrolizumab y nivolumab",
    "Alto costo con mayor precio de referencia",
]

AI_MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]

# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------


@dataclass
class Answer:
    ok: bool
    title: str
    text: str
    table: pd.DataFrame | None = None
    fig: object | None = None
    understood: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _has(q: str, *pats: str) -> bool:
    return any(re.search(p, q) for p in pats)


def _fmt(x) -> str:
    return f"{x:,.0f}".replace(",", " ")


# ---------------------------------------------------------------------------
# Interpretación
# ---------------------------------------------------------------------------


def _match_dcis(q: str, df: pd.DataFrame, names: dict[str, str], alias: pd.DataFrame | None) -> list[str]:
    """Principios activos mencionados en la pregunta (por dci_base o por patrones de alias)."""
    found: list[str] = []
    bases = sorted({b for b in df["dci_base"].dropna().astype(str).unique() if len(b) >= 5}, key=len, reverse=True)
    covered = q
    for b in bases:
        if re.search(rf"\b{re.escape(b)}\b", covered):
            found.append(b)
            covered = covered.replace(b, " ")
    if alias is not None and len(alias):
        for r in alias.itertuples():
            for p in [p for p in str(r.patrones).split("|") if len(p) >= 5]:
                if re.search(rf"\b{re.escape(p)}", covered):
                    hits = df.loc[df["alto_costo_clave"] == r.clave, "dci_base"].dropna().unique().tolist()
                    found.extend(h for h in hits if h not in found)
                    covered = covered.replace(p, " ")
    return found


def _match_areas(q: str, df: pd.DataFrame) -> list[str]:
    out, covered = [], q
    for a in sorted(df["area_canon"].dropna().unique(), key=len, reverse=True):
        na = norm_txt(a)
        if len(na) >= 5 and re.search(rf"\b{re.escape(na)}\b", covered):
            out.append(a)
            covered = covered.replace(na, " ")
    return out


def _parse(question: str, df: pd.DataFrame, names: dict[str, str], alias: pd.DataFrame | None) -> dict:
    q = " " + norm_txt(question) + " "
    p: dict = {"q": q}

    # Años
    yrs = sorted({int(y) for y in re.findall(r"\b(20[12]\d)\b", q)})
    if len(yrs) == 2 and _has(q, r"\bENTRE\b", r"\bDESDE\b", r"\bHASTA\b", r"\bAL\b", r"\bA\b"):
        p["years"] = list(range(yrs[0], yrs[1] + 1))
        p["years_range"] = True
    elif len(yrs) == 1 and _has(q, r"\bDESDE\b"):
        p["years"] = list(range(yrs[0], int(df["anio"].max()) + 1))
        p["years_range"] = True
    else:
        p["years"] = yrs
        p["years_range"] = False

    # Métrica
    if "%" in question or _has(q, r"PORCENTAJE", r"TASA DE APROB", r"TASA DE DENEG", r"TASA DE RECHAZ", r"PROPORCION"):
        p["metric"] = "pct"
    elif _has(q, r"PRIMERA VEZ", r"\bNUEVOS?\b", r"\bNUEVAS?\b", r"INCORPORAD"):
        p["metric"] = "first"
    elif _has(q, r"DISTINTOS", r"DIFERENTES", r"CUANTOS PRINCIPIOS", r"CUANTOS MEDICAMENTOS", r"CUANTOS FARMACOS"):
        p["metric"] = "distinct"
    else:
        p["metric"] = "count"
    p["price"] = _has(q, r"PRECIO", r"MAS CAROS?", r"COSTO REFERENCIAL", r"VALOR REFERENCIAL")

    # Decisión
    if p["metric"] != "pct":
        if _has(q, r"DENEGAD", r"RECHAZAD", r"NO APROBAD"):
            p["decision"] = "DENEGADO"
        elif _has(q, r"\bAPROBAD", r"\bAPROBACIONES\b"):
            p["decision"] = "APROBADO"

    # Alto costo / oncológico
    if _has(q, r"ALTO COSTO", r"NO PNUME"):
        p["hc"] = "vs" if _has(q, r"\bVS\b", r"VERSUS", r"RESTO", r"FRENTE") else "solo"
    elif p["price"]:
        p["hc"] = "solo"
    if _has(q, r"ONCOLOGIC\w* Y NO ONCOLOGIC", r"NO ONCOLOGIC\w* Y ONCOLOGIC", r"POR TIPO DE TRATAMIENTO"):
        p["onco_split"] = True
    elif _has(q, r"NO ONCOLOGIC"):
        p["onco"] = "NO ONCOLÓGICO"
    elif _has(q, r"\bONCOLOGICOS?\b", r"\bONCOLOGICAS?\b"):
        p["onco"] = "ONCOLÓGICO"

    # Entidades
    p["dcis"] = _match_dcis(q, df, names, alias)
    p["areas"] = _match_areas(q, df)
    if not p["areas"] and _has(q, r"\bONCOLOGIA\b") and not _has(q, r"POR AREA", r"QUE AREAS?"):
        p["grupo"] = "Oncología (médica/pediátrica)"

    # Top N
    m = re.search(r"\bTOP\s*(\d{1,2})\b", q) or re.search(r"\b(\d{1,2})\s+(?:PRINCIPALES|PRIMER[OA]S|MAS)\b", q)
    p["top"] = int(m.group(1)) if m else None

    # Dimensión categórica
    cat = None
    if _has(q, r"GRUPOS? DE AREA", r"POR GRUPO"):
        cat = "grupo_area"
    elif _has(q, r"\bAREAS?\b", r"SERVICIOS?", r"SOLICITANTE", r"DEPARTAMENTOS?", r"ESPECIALIDAD"):
        cat = "area_canon"
    elif _has(q, r"POR DECISION", r"SEGUN DECISION", r"APROBADAS Y DENEGADAS", r"APROBADOS Y DENEGADOS", r"POR RESULTADO"):
        cat = "decision"
    elif _has(q, r"FORMAS? FARMACEUTICA"):
        cat = "forma_farmaceutica"
    elif p.get("hc") == "vs":
        cat = "_hc"
    elif p.get("onco_split"):
        cat = "oncologico_reportado"
    elif _has(q, r"MEDICAMENTOS?", r"PRINCIPIOS? ACTIVOS?", r"FARMACOS?", r"\bDCI\b", r"MOLECULAS?", r"\bCUALES\b",
              r"\bQUE\b.*\bSE EVALU") and p["metric"] != "distinct" and len(p["dcis"]) != 1:
        cat = "dci_base"
    if p["top"] and cat is None and len(p["dcis"]) != 1:
        cat = "dci_base"
    if p["metric"] == "first" and cat is None:
        cat = "dci_base" if p["years"] and len(p["years"]) == 1 else None
    # Las áreas mencionadas como filtro no deben convertirse en desglose por área
    if cat == "area_canon" and p["areas"] and not _has(q, r"POR AREA", r"QUE AREAS?", r"\bAREAS\b", r"SEGUN AREA"):
        cat = None
    p["cat"] = cat

    # Dimensión temporal
    if _has(q, r"MENSUAL", r"POR MES", r"CADA MES", r"MES A MES"):
        p["time"] = "M"
    elif _has(q, r"TRIMESTR"):
        p["time"] = "Q"
    elif _has(q, r"EVOLUCION", r"TENDENCIA", r"POR ANO", r"ANUAL", r"CADA ANO", r"CRECI", r"EN EL TIEMPO", r"HISTORI",
              r"A LO LARGO", r"AUMENT", r"DISMINU", r"CAMBI", r"SERIE"):
        p["time"] = "A"
    elif len(p["dcis"]) >= 1 and cat is None:
        p["time"] = "A"
    elif len(p["years"]) >= 2 and not p["years_range"]:
        p["time"] = "A"
    else:
        p["time"] = None
    # ¿Hay alguna señal reconocible en la pregunta?
    p["signal"] = bool(p["years"] or p.get("decision") or p.get("hc") or p.get("onco") or p.get("onco_split")
                       or p["dcis"] or p["areas"] or p.get("grupo") or cat or p["time"] or p["price"]
                       or p["metric"] != "count"
                       or _has(q, r"CUANT", r"TOTAL", r"RESUMEN", r"EVALUAC", r"REQUERIMIENT", r"SOLICITUD", r"APROBAC"))
    return p


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------


def _label_cat(cat: str) -> str:
    return {"grupo_area": "Grupo de área", "area_canon": "Área solicitante", "decision": "Decisión",
            "forma_farmaceutica": "Forma farmacéutica", "_hc": "Alto costo", "dci_base": "Principio activo",
            "oncologico_reportado": "Tipo de tratamiento"}.get(cat, cat)


def _time_col(d: pd.DataFrame, t: str) -> pd.Series:
    if t == "M":
        return d["fecha"].dt.to_period("M").astype(str)
    if t == "Q":
        return d["fecha"].dt.to_period("Q").astype(str)
    return d["anio"].astype(int)


def _incomplete_years(inc_months: list[pd.Timestamp], last: pd.Timestamp) -> set[int]:
    ys = {t.year for t in inc_months}
    if last.month < 12:
        ys.add(last.year)
    return ys


def _metric_table(d: pd.DataFrame, dims: list[str], metric: str) -> pd.DataFrame:
    if metric == "pct":
        g = d.groupby(dims, dropna=False)
        t = pd.DataFrame({"Evaluaciones": g.size(), "Aprobadas": g["decision"].apply(lambda s: (s == "APROBADO").sum()),
                          "Denegadas": g["decision"].apply(lambda s: (s == "DENEGADO").sum())})
        t["% aprobación"] = (t["Aprobadas"] / t["Evaluaciones"] * 100).round(1)
        return t
    if metric == "first":
        return d.groupby(dims, dropna=False)["primera_vez_serie"].sum().to_frame("Evaluados por 1.ª vez")
    if metric == "distinct":
        return d.groupby(dims, dropna=False)["dci_base"].nunique().to_frame("Principios activos distintos")
    g = d.groupby(dims, dropna=False)
    t = pd.DataFrame({"Evaluaciones": g.size(), "Aprobadas": g["decision"].apply(lambda s: (s == "APROBADO").sum())})
    return t


def answer(question: str, df: pd.DataFrame, hc: pd.DataFrame, names: dict[str, str],
           alias: pd.DataFrame | None = None, inc_months: list[pd.Timestamp] | None = None,
           palette: list[str] | None = None) -> Answer:
    inc_months = inc_months or []
    pal = palette or px.colors.qualitative.Bold
    if not question or not question.strip():
        return Answer(False, "", "Escribe una pregunta o elige un ejemplo.")
    p = _parse(question, df, names, alias)
    if not p["signal"]:
        return Answer(False, "No entendí la pregunta",
                      "No reconocí años, medidas, áreas ni principios activos en la pregunta. Prueba con uno de los "
                      "ejemplos o menciona qué contar (evaluaciones, aprobadas, % de aprobación, primera vez) y cómo "
                      "desglosarlo (por año, por área, por principio activo). Con el modo IA activado se pueden "
                      "responder preguntas más abiertas.")
    und: list[str] = []
    warns: list[str] = []
    d = df.copy()
    d["_hc"] = np.where(d["alto_costo_clave"].notna(), "Alto costo (listado)", "Resto")

    # ---- filtros
    if p["years"]:
        d = d[d["anio"].isin(p["years"])]
        und.append("Años: " + (f"{p['years'][0]}–{p['years'][-1]}" if p["years_range"] else ", ".join(map(str, p["years"]))))
    if p.get("decision"):
        d = d[d["decision"] == p["decision"]]
        und.append(f"Decisión: {p['decision'].lower()}")
    if p.get("hc") == "solo":
        d = d[d["alto_costo_clave"].notna()]
        und.append("Solo alto costo (listado no PNUME)")
    if p.get("onco"):
        d = d[d["oncologico_reportado"] == p["onco"]]
        und.append(f"Tratamiento: {p['onco'].lower()}")
        warns.append("El CFT registra si el tratamiento es oncológico solo desde 2025; antes de ese año no hay dato.")
    if p["areas"]:
        d = d[d["area_canon"].isin(p["areas"])]
        und.append("Área: " + ", ".join(p["areas"]))
    elif p.get("grupo"):
        d = d[d["grupo_area"] == p["grupo"]]
        und.append(f"Grupo: {p['grupo']}")
    if p["dcis"]:
        d = d[d["dci_base"].isin(p["dcis"])]
        und.append("Principio activo: " + ", ".join(x.title() for x in p["dcis"]))
    metric_lbl = {"pct": "% de aprobación", "first": "evaluados por 1.ª vez", "distinct": "principios activos distintos",
                  "count": "evaluaciones"}[p["metric"]]
    if p.get("onco_split"):
        d = d[d["oncologico_reportado"].notna()]
        warns.append("El CFT registra si el tratamiento es oncológico solo desde 2025; antes de ese año no hay dato.")
    if not p["price"]:
        und.append(f"Medida: {metric_lbl}")

    last = df["fecha"].max()
    inc_y = _incomplete_years(inc_months, last)

    # ---- precio (alto costo)
    if p["price"]:
        h = hc.groupby("clave", as_index=False).agg(precio_ref=("precio_ref", "max"), descripcion=("descripcion", "first"))
        ev = d[d["alto_costo_clave"].notna()].groupby("alto_costo_clave").agg(
            Evaluaciones=("dci", "size"), Aprobadas=("decision", lambda s: (s == "APROBADO").sum()),
            Primera=("fecha", "min"))
        t = h.merge(ev, left_on="clave", right_index=True, how="left").fillna({"Evaluaciones": 0, "Aprobadas": 0})
        t["Principio activo"] = t["clave"].map(lambda k: names.get(k, str(k).title()))
        t["Primera evaluación"] = pd.to_datetime(t["Primera"]).dt.strftime("%m-%Y").fillna("—")
        sort_by = "Evaluaciones" if _has(p["q"], r"MAS EVALUAD", r"EVALUACIONES") and not _has(p["q"], r"PRECIO") else "precio_ref"
        t = t.sort_values(sort_by, ascending=False)
        n = p["top"] or 10
        t = t.head(n)[["Principio activo", "descripcion", "precio_ref", "Evaluaciones", "Aprobadas", "Primera evaluación"]]
        t = t.rename(columns={"descripcion": "Presentación", "precio_ref": "Precio ref. (S/)"})
        t[["Evaluaciones", "Aprobadas"]] = t[["Evaluaciones", "Aprobadas"]].astype(int)
        top = t.iloc[0]
        txt = (f"El ítem de mayor precio de referencia es **{top['Principio activo']}** (S/ {_fmt(top['Precio ref. (S/)'])}), "
               f"con {int(top['Evaluaciones'])} evaluaciones en la selección. Se muestran {len(t)} ítems del listado no PNUME.")
        fig = px.bar(t.iloc[::-1], x="Precio ref. (S/)", y="Principio activo", orientation="h", text="Evaluaciones",
                     color_discrete_sequence=[pal[0]])
        fig.update_traces(texttemplate="%{text} eval.", textposition="outside")
        return Answer(True, f"Alto costo: precio de referencia (top {len(t)})", txt, t.reset_index(drop=True), fig, und, warns)

    if d.empty:
        return Answer(False, "Sin resultados", "No hay evaluaciones que cumplan lo que entendí de la pregunta.",
                      None, None, und, warns)

    cat, tm = p["cat"], p["time"]

    # ---- sin dimensiones: resumen
    if cat is None and tm is None:
        n, ap = len(d), int((d["decision"] == "APROBADO").sum())
        dn = int((d["decision"] == "DENEGADO").sum())
        nd = d["dci_base"].nunique()
        nf = int(d["primera_vez_serie"].sum())
        t = pd.DataFrame({"Indicador": ["Evaluaciones", "Aprobadas", "Denegadas", "% aprobación",
                                        "Principios activos distintos", "Evaluados por 1.ª vez", "Desde", "Hasta"],
                          "Valor": [_fmt(n), _fmt(ap), _fmt(dn), f"{ap / n * 100:.1f}%", _fmt(nd), _fmt(nf),
                                    f"{d['fecha'].min():%d-%m-%Y}", f"{d['fecha'].max():%d-%m-%Y}"]})
        main = {"pct": f"El % de aprobación es **{ap / n * 100:.1f}%** ({_fmt(ap)} de {_fmt(n)} evaluaciones).",
                "first": f"Se evaluaron **{_fmt(nf)}** principios activos por primera vez en la serie.",
                "distinct": f"Hubo **{_fmt(nd)}** principios activos distintos.",
                "count": f"Hubo **{_fmt(n)}** evaluaciones ({_fmt(ap)} aprobadas, {_fmt(dn)} denegadas; "
                         f"{ap / n * 100:.1f}% de aprobación) sobre {_fmt(nd)} principios activos distintos."}[p["metric"]]
        yrs_in = set(d["anio"].unique())
        if yrs_in & inc_y:
            warns.append("Incluye años con cobertura incompleta: " + ", ".join(str(y) for y in sorted(yrs_in & inc_y)) + ".")
        return Answer(True, "Resumen", main, t, None, und, warns)

    dims = ([ "_t"] if tm else []) + ([cat] if cat else [])
    if tm:
        d["_t"] = _time_col(d, tm)
    if cat:
        d[cat] = d[cat].fillna("Sin dato").astype(str)
        if cat == "dci_base":
            d[cat] = d[cat].str.title()
    tlabel = {"A": "Año", "Q": "Trimestre", "M": "Mes"}.get(tm, "")

    # ---- principios activos evaluados por primera vez: listado
    if p["metric"] == "first" and cat == "dci_base" and not tm:
        f = d[d["primera_vez_serie"] == 1].sort_values("fecha")
        t = pd.DataFrame({"Fecha": f["fecha"].dt.strftime("%d-%m-%Y"), "Principio activo": f["dci_base"],
                          "Área solicitante": f["area_canon"], "Decisión": f["decision"],
                          "Alto costo": np.where(f["alto_costo_clave"].notna(), "Sí", "")}).reset_index(drop=True)
        if t.empty:
            return Answer(False, "Sin resultados", "No hubo principios activos evaluados por primera vez en esa selección.",
                          None, None, und, warns)
        nhc = int((t["Alto costo"] == "Sí").sum())
        ga = f["grupo_area"].value_counts()
        txt = (f"Se evaluaron **{len(t)}** principios activos por primera vez en la serie del CFT"
               f"{f' ({nhc} del listado de alto costo)' if nhc else ''}. El grupo que más aportó fue "
               f"{ga.index[0]} ({ga.iloc[0]}).")
        warns.append("«Primera vez» se cuenta desde el inicio de la serie (2021); en los primeros meses casi todo es «primera vez».")
        fig = px.histogram(f.assign(Mes=f["fecha"].dt.to_period("M").astype(str)), x="Mes", color="grupo_area",
                           color_discrete_sequence=pal, labels={"grupo_area": "Grupo de área"})
        return Answer(True, "Principios activos evaluados por primera vez", txt, t, fig, und, warns)

    # ---- solo categoría: ranking
    if cat and not tm:
        t = _metric_table(d, [cat], p["metric"])
        vcol = t.columns[-1] if p["metric"] == "pct" else t.columns[0]
        if p["metric"] == "pct":
            t = t[t["Evaluaciones"] >= 3] if (t["Evaluaciones"] >= 3).sum() >= 3 else t
        t = t.sort_values([vcol, t.columns[0]], ascending=False)
        if p["metric"] != "pct" and "Evaluaciones" in t:
            t["% del total"] = (t["Evaluaciones"] / t["Evaluaciones"].sum() * 100).round(1)
        n_all = len(t)
        n = p["top"] or (10 if n_all > 12 else n_all)
        t = t.head(n)
        t.index.name = _label_cat(cat)
        t = t.reset_index()
        lead = t.iloc[0]
        vfmt = (lambda v: f"{v:.1f}%") if p["metric"] == "pct" else _fmt
        txt = (f"**{lead.iloc[0]}** encabeza con {vfmt(lead[vcol])}"
               f"{' de aprobación' if p['metric'] == 'pct' else ' ' + metric_lbl}. "
               f"Se muestran {len(t)} de {n_all} categorías.")
        if p["metric"] == "pct":
            txt += " Para el % de aprobación se excluyen categorías con menos de 3 evaluaciones."
        if p["metric"] != "pct" and len(t) >= 3 and "% del total" in t:
            txt += f" Las 3 primeras concentran el {t['% del total'].head(3).sum():.0f}% de la selección."
        fig = px.bar(t.iloc[::-1], x=vcol, y=t.columns[0], orientation="h", text=vcol, color_discrete_sequence=[pal[0]])
        fig.update_traces(texttemplate="%{text:.1f}" if p["metric"] == "pct" else "%{text:,.0f}", textposition="outside")
        return Answer(True, f"{metric_lbl.capitalize()} por {_label_cat(cat).lower()}", txt, t, fig, und, warns)

    # ---- serie temporal (con o sin categoría)
    if tm and not cat and not p["dcis"]:
        t = _metric_table(d, ["_t"], p["metric"])
        if tm == "A":
            full = range(int(d["anio"].min()), int(d["anio"].max()) + 1)
            t = t.reindex(list(full)).fillna(0) if p["metric"] != "pct" else t.reindex(list(full))
        vcol = t.columns[-1] if p["metric"] == "pct" else t.columns[0]
        t.index.name = tlabel
        t = t.reset_index()
        if tm == "A":
            t["Cobertura"] = ["incompleta" if int(y) in inc_y else "completa" for y in t[tlabel]]
        txt = _series_text(t, tlabel, vcol, p["metric"], tm, inc_y)
        fig = px.bar(t, x=tlabel, y=vcol, text=vcol, color_discrete_sequence=[pal[0]]) if tm == "A" else \
            px.line(t, x=tlabel, y=vcol, markers=True, color_discrete_sequence=[pal[0]])
        if tm == "A":
            fig.update_traces(texttemplate="%{text:.1f}" if p["metric"] == "pct" else "%{text:,.0f}", textposition="outside")
            fig.update_xaxes(type="category")
        return Answer(True, f"{metric_lbl.capitalize()} por {tlabel.lower()}", txt, t, fig, und, warns + _inc_warn(t, tlabel, inc_y, tm))

    # Principios activos concretos -> serie por fármaco
    if p["dcis"] and not cat:
        cat = "dci_base"
        d[cat] = d[cat].astype(str).str.title()
        if not tm:
            tm, tlabel = "A", "Año"
            d["_t"] = _time_col(d, tm)
    if not tm:
        tm, tlabel = "A", "Año"
        d["_t"] = _time_col(d, tm)
    keep = None
    if cat != "decision" and d[cat].nunique() > (p["top"] or 6):
        n = p["top"] or 6
        keep = d[cat].value_counts().head(n).index.tolist()
        d = d[d[cat].isin(keep)]
    t = _metric_table(d, ["_t", cat], p["metric"])
    vcol = t.columns[-1] if p["metric"] == "pct" else t.columns[0]
    wide = t[vcol].unstack(cat)
    if tm == "A":
        wide = wide.reindex(range(int(df["anio"].min() if not p["years"] else min(p["years"])),
                                  int(df["anio"].max() if not p["years"] else max(p["years"])) + 1))
    if p["metric"] != "pct":
        wide = wide.fillna(0).astype(int)
    order = wide.sum().sort_values(ascending=False).index
    wide = wide[order]
    wide.index.name = tlabel
    wide.columns.name = None
    long = wide.reset_index().melt(id_vars=tlabel, var_name=_label_cat(cat), value_name=vcol)
    fig = px.line(long, x=tlabel, y=vcol, color=_label_cat(cat), markers=True, color_discrete_sequence=pal)
    if tm == "A":
        fig.update_xaxes(type="category")
    tot = wide.sum() if p["metric"] != "pct" else None
    if tot is not None:
        lead = tot.index[0]
        txt = (f"**{lead}** suma {_fmt(tot.iloc[0])} {metric_lbl} en el período mostrado."
               + (f" Le sigue **{tot.index[1]}** con {_fmt(tot.iloc[1])}." if len(tot) > 1 else ""))
        if p["dcis"] and len(p["dcis"]) >= 1:
            dd = d.sort_values("fecha")
            txt += (f" Primera evaluación: {dd['fecha'].iloc[0]:%d-%m-%Y}; última: {dd['fecha'].iloc[-1]:%d-%m-%Y}. "
                    f"Áreas solicitantes principales: {', '.join(dd['area_canon'].value_counts().head(3).index)}.")
    else:
        txt = "Evolución del % de aprobación por categoría (solo años con evaluaciones)."
    if keep is not None:
        txt += f" Se muestran las {len(keep)} categorías con más evaluaciones."
    out = wide.reset_index()
    out[tlabel] = out[tlabel].astype(str)
    return Answer(True, f"{metric_lbl.capitalize()} por {tlabel.lower()} y {_label_cat(cat).lower()}", txt, out, fig, und,
                  warns + _inc_warn(out, tlabel, inc_y, tm))


def _inc_warn(t: pd.DataFrame, tlabel: str, inc_y: set[int], tm: str) -> list[str]:
    if tm != "A":
        return []
    ys = sorted({int(y) for y in t[tlabel] if int(y) in inc_y})
    if not ys:
        return []
    return ["Años con cobertura incompleta (su total está subestimado): " + ", ".join(map(str, ys))
            + ". Para comparar años usa la pestaña «Comparación anual» (enero–mes de corte)."]


def _series_text(t: pd.DataFrame, tlabel: str, vcol: str, metric: str, tm: str, inc_y: set[int]) -> str:
    s = t.set_index(tlabel)[vcol].dropna()
    if s.empty:
        return "Sin datos."
    vf = (lambda v: f"{v:.1f}%") if metric == "pct" else _fmt
    imax = s.idxmax()
    txt = f"El valor más alto es **{vf(s.max())}** en {imax}."
    if tm == "A":
        comp = s[[int(y) not in inc_y for y in s.index]]
        if len(comp) >= 2:
            a, b = comp.index[0], comp.index[-1]
            if metric == "pct":
                txt += f" Entre {a} y {b} (años completos) pasó de {vf(comp.iloc[0])} a {vf(comp.iloc[-1])} " \
                       f"({comp.iloc[-1] - comp.iloc[0]:+.1f} p.p.)."
            elif comp.iloc[0] > 0:
                ch = (comp.iloc[-1] / comp.iloc[0] - 1) * 100
                txt += f" Entre {a} y {b} (años completos) pasó de {vf(comp.iloc[0])} a {vf(comp.iloc[-1])} ({ch:+.0f}%)."
    else:
        txt += f" Promedio por período: {vf(s.mean())}."
    return txt


# ---------------------------------------------------------------------------
# Modo IA
# ---------------------------------------------------------------------------

AI_SYSTEM = (
    "Eres un analista del Observatorio del Comité Farmacoterapéutico (CFT) del INEN, Perú. Respondes en español, "
    "en prosa breve y clara (máximo 180 palabras), sin guiones largos y usando dos puntos solo si es imprescindible. "
    "Usa EXCLUSIVAMENTE las cifras de las tablas que se te entregan; no inventes ni estimes valores que no estén en ellas. "
    "Si la pregunta no puede responderse con esas tablas, dilo y sugiere qué pestaña del observatorio usar. "
    "Recuerda que una fila es una solicitud evaluada por el CFT (no pacientes ni gasto) y que hay años con cobertura "
    "incompleta (dic-2021, sep-dic 2022 y el año en curso), por lo que sus totales están subestimados."
)


def ai_context(df: pd.DataFrame, hc: pd.DataFrame, names: dict[str, str], inc_months: list[pd.Timestamp],
               rule_answer: Answer | None) -> str:
    """Tablas resumidas (sin identificadores de registro) que se envían al modelo."""
    parts = []
    last = df["fecha"].max()
    parts.append(f"Período: {df['fecha'].min():%Y-%m-%d} a {last:%Y-%m-%d}. Evaluaciones: {len(df)}.")
    parts.append("Meses con cobertura incompleta: " + ", ".join(f"{t:%Y-%m}" for t in inc_months))
    y = pd.crosstab(df["anio"], df["decision"])
    y["total"] = y.sum(axis=1)
    y["primera_vez"] = df.groupby("anio")["primera_vez_serie"].sum()
    y["dci_distintos"] = df.groupby("anio")["dci_base"].nunique()
    parts.append("## Evaluaciones por año y decisión\n" + y.to_csv())
    parts.append("## Grupo de área por año\n" + pd.crosstab(df["grupo_area"], df["anio"]).to_csv())
    top_a = df["area_canon"].value_counts().head(15).index
    parts.append("## Área solicitante (top 15) por año\n" + pd.crosstab(df.loc[df["area_canon"].isin(top_a), "area_canon"],
                                                                         df["anio"]).to_csv())
    top_d = df["dci_base"].value_counts().head(40).index
    parts.append("## Principio activo (top 40) por año\n" + pd.crosstab(df.loc[df["dci_base"].isin(top_d), "dci_base"],
                                                                          df["anio"]).to_csv())
    h = df[df["alto_costo_clave"].notna()]
    hcs = hc.groupby("clave")["precio_ref"].max().to_frame("precio_ref_soles")
    hcs["nombre"] = [names.get(k, k) for k in hcs.index]
    hcs = hcs.join(pd.crosstab(h["alto_costo_clave"], h["anio"]), how="left").fillna(0)
    parts.append("## Listado no PNUME de alto costo: precio de referencia y evaluaciones por año\n" + hcs.to_csv())
    if df["oncologico_reportado"].notna().any():
        parts.append("## Tipo de tratamiento (registrado desde 2025) por año\n"
                     + pd.crosstab(df["oncologico_reportado"].fillna("sin dato"), df["anio"]).to_csv())
    if rule_answer is not None and rule_answer.ok and rule_answer.table is not None:
        parts.append(f"## Resultado del modo por reglas ({rule_answer.title}; filtros entendidos: "
                     f"{'; '.join(rule_answer.understood)})\n" + rule_answer.table.head(60).to_csv(index=False))
    return "\n\n".join(parts)


def get_api_key(secrets) -> str | None:
    try:
        k = secrets.get("ANTHROPIC_API_KEY")
        if k:
            return str(k)
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def ask_ai(question: str, context: str, api_key: str, model: str, timeout: int = 60) -> str:
    import requests

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        data=json.dumps({
            "model": model, "max_tokens": 700, "system": AI_SYSTEM,
            "messages": [{"role": "user", "content": f"Tablas del observatorio:\n\n{context}\n\nPregunta: {question}"}],
        }),
        timeout=timeout,
    )
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except Exception:  # noqa: BLE001
            msg = r.text
        raise RuntimeError(f"Error {r.status_code}: {msg[:300]}")
    js = r.json()
    return "".join(b.get("text", "") for b in js.get("content", []) if b.get("type") == "text").strip()
