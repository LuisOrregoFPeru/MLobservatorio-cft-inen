# Observatorio del Comité Farmacoterapéutico (CFT) — INEN

App Streamlit para analizar cómo han crecido los **requerimientos (evaluaciones) y aprobaciones** del CFT desde 2021,
con subanálisis de los **medicamentos de alto costo** del listado no PNUME. Reutiliza la arquitectura y los métodos
del *Observatorio del Financiamiento Oncológico INEN* (eventos normativos, Mann-Kendall, Chow, proyección, serie externa).

## Ejecutar
```bash
pip install -r requirements.txt
streamlit run app.py
```
Despliegue en Streamlit Community Cloud: subir **todos** los archivos del repositorio (incluida `data/`) y usar `app.py` como módulo principal.

## 💬 Pregunta a los datos (panel lateral)
Escribe una pregunta en español o toca un ejemplo. La respuesta aparece bajo la cabecera con una frase, gráfico, tabla
descargable, los filtros que se entendieron (chips) y avisos de cobertura incompleta.
- **Modo básico por reglas** (`qa_engine.py`): reconoce años o rangos («entre 2023 y 2025», «desde 2024»), decisión,
  alto costo, tratamiento oncológico (2025+), áreas, grupos, principios activos por nombre, Top N, medidas
  (evaluaciones, % de aprobación, primera vez, principios activos distintos, precio de referencia) y desgloses por año,
  trimestre, mes, área, grupo, decisión o principio activo.
- **Aplicar los filtros del panel**: activado responde sobre la selección actual; desactivado, sobre toda la base.
- **Modo IA (opcional)**: redacta la respuesta con Claude a partir de tablas resumidas (año × decisión, áreas, top 40
  principios activos, alto costo con precio, resultado por reglas). No se envían N.º de evaluación ni registros
  individuales. Requiere `ANTHROPIC_API_KEY` en `.streamlit/secrets.toml` / secrets de Streamlit Cloud, o pegar la clave
  en la sesión.

## Pestañas
| Pestaña | Qué hace |
|---|---|
| 📈 Serie temporal | **Comparación de tendencias entre 2 series (¿iguales o diferentes?)**; Mensual/trimestral/semestral/anual; 6 medidas; línea/barras/área; acumulado, índice 100, var. interanual; media móvil y LOWESS; desglose por decisión, área, alto costo, principio activo, etc.; **un gráfico o paneles separados**; eventos y bandas de cobertura |
| 📅 Comparación anual | Ene–mes de corte por año, total anual con estimación de año completo y perfil estacional, líneas por año, acumulado, mapa de calor año × mes |
| 💊 Alto costo | Selección de fármacos (Top N, clases, precio, personalizado); líneas, barras, apilado, **paneles separados** o mapa de calor; por fármaco / suma / vs resto; Pareto y precio vs evaluaciones |
| ✂️ Punto de corte | Chow sobre un evento (con rezago) y quiebre automático |
| 🌿 Tendencia | Mann-Kendall (Hamed-Rao/Yue-Wang), Sen y regresión lineal por serie; **comparación de tendencias entre 2 series** (igual que en Serie temporal) |
| 🔮 Proyección | Holt amortiguado con bootstrap, 3 escenarios |
| 🔬 Qué explica el crecimiento | Compara dos períodos y reparte el cambio entre fármacos, áreas o grupos (cascada); separa crecimiento por intensidad, diversificación y salidas; principios activos evaluados por 1.ª vez |
| 🧾 ETS (INS) | Cruza los informes ETS/ETS-EMC del INS con las evaluaciones del CFT por molécula: % de solicitudes con ETS previa por año, línea de tiempo por molécula, antes/después de la publicación (RR con IC exacto y RR relativo frente al resto del CFT), decisión del CFT según el dictamen previo y ETS solicitadas por el CFT INEN |
| 🔀 Serie externa | Sube otra serie (p. ej. gasto SIS/FISSAL) y compara: correlación y correlación cruzada con rezago |
| 📋 Tabla y descarga | Excel, CSV y HTML del gráfico |
| 🗂️ Datos | Carga de Excel nuevo del CFT y del listado de alto costo; alias; notas metodológicas |

## Datos incluidos (`data/`)
- `cft_evaluaciones.csv` — 775 evaluaciones 2021–sep 2026, ya consolidadas (sin duplicados entre hojas, fechas corregidas).
- `alto_costo.csv` — listado no PNUME de alto costo al 22/09/2026 (42 ítems, 39 principios activos).
- `alias_alto_costo.csv` — cómo se vincula cada principio activo del listado con los DCI del CFT.
- `ets_informes.csv` — 153 informes ETS del INS (2018–2026) con dictamen agrupado y solicitante; se vinculan con `ets_link.py`.
- `eventos.csv` — eventos normativos precargados (del Observatorio del Financiamiento Oncológico).

## Actualizar
Pestaña **🗂️ Datos** → subir el Excel de aprobaciones (una hoja por año) → «Usar esta base». Para que el cambio sea permanente,
reemplaza `data/cft_evaluaciones.csv` con la descarga «Base CFT consolidada».

## Notas
- Cobertura incompleta (dic-2021, sep–dic 2022, sep-2026) marcada en gris; las comparaciones anuales usan enero–mes de corte.
- Una fila = una solicitud evaluada; no equivale a pacientes ni a gasto.
- Probado con Python 3.12, pandas 3.0, Streamlit 1.64, plotly 7.

## Interpretación
Cada gráfico y tabla incluye un recuadro «Cómo leerlo» con una lectura automática breve de lo que se está mostrando.
