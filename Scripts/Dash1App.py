from dash import Dash, html, dcc, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import argparse
from dash import Dash, html
from IPython.display import display

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=63641)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

print(f"Iniciando Host:{args.host}, Port:{args.port}...")



# =========================
# CONFIG
# =========================
DATA_FILE = Path(r"Temp/clinical_analytics.csv.gz").resolve()
ADMIT_SOURCE_STRATEGY = "mode"  # "mode" (completar con más frecuente) o "drop" (eliminar nulos)

# Paleta de colores menos saturada (evita el azul monolítico)
COLOR_SEQ = px.colors.qualitative.Set2  # Alternativas: Pastel1, Set3, Prism

# =========================
# CARGA & LIMPIEZA
# =========================
def load_raw_df(path: Path) -> pd.DataFrame:
    # Carga robusta (autodetecta separador; compresión inferida si fuese .gz)
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        compression="infer",
        na_values=["", "NA", "NaN", "null", "None", " "]
    )

def clean_df(df: pd.DataFrame):
    """
    Limpieza:
    - Diagnosis Primary -> "None" si vacío.
    - Encounter Status  -> completar con la moda.
    - Number of Records -> eliminar (siempre 1).
    - Fechas: Event Datetime = Discharge Datetime new (fallback Check-In Time).
      Event Date = solo fecha (YYYY-MM-DD) para filtros/series.
    - Wait Time Min -> numérico y >= 0.
    - Encounter Number -> string.
    - Admit Source -> según estrategia; si no existe, "Unknown".
    Devuelve: (df_limpio, nombre_columna_fecha)
    """
    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", pd.NA, regex=True).copy()

    # Diagnosis Primary
    if "Diagnosis Primary" in df.columns:
        df["Diagnosis Primary"] = df["Diagnosis Primary"].fillna("None")
    else:
        df["Diagnosis Primary"] = "None"

    # Encounter Status -> moda
    if "Encounter Status" in df.columns:
        mode_vals = df["Encounter Status"].dropna()
        fill_val = mode_vals.mode().iloc[0] if not mode_vals.empty else "Unknown"
        df["Encounter Status"] = df["Encounter Status"].fillna(fill_val)
    else:
        df["Encounter Status"] = "Unknown"

    # Number of Records -> fuera
    if "Number of Records" in df.columns:
        df = df.drop(columns=["Number of Records"])

    # Fechas
    discharge_col = "Discharge Datetime new"
    checkin_col   = "Check-In Time"

    # columnas de fecha/hora
    discharge_dt = None
    if discharge_col in df.columns:
        discharge_dt = pd.to_datetime(df[discharge_col], errors="coerce")

    checkin_dt = None
    if checkin_col in df.columns:
        # Formato: 2014-01-02 11:24:00 PM  ->  %Y-%m-%d %I:%M:%S %p
        chk_try = pd.to_datetime(df[checkin_col], format="%Y-%m-%d %I:%M:%S %p", errors="coerce")

        if chk_try.isna().mean() > 0.5:
            chk_try = pd.to_datetime(df[checkin_col], errors="coerce")
        checkin_dt = chk_try

    if discharge_dt is not None or checkin_dt is not None:
        if discharge_dt is None:
            event_dt = checkin_dt
        elif checkin_dt is None:
            event_dt = discharge_dt
        else:
            event_dt = discharge_dt.fillna(checkin_dt)

        df["Event Datetime"] = pd.to_datetime(event_dt, errors="coerce")
        df["Event Date"]     = df["Event Datetime"].dt.date
        date_col = "Event Date"
    else:
        date_col = None

    # Wait Time Min
    if "Wait Time Min" in df.columns:
        df["Wait Time Min"] = pd.to_numeric(df["Wait Time Min"], errors="coerce")
        df.loc[df["Wait Time Min"] < 0, "Wait Time Min"] = 0

    # Encounter Number
    if "Encounter Number" in df.columns:
        df["Encounter Number"] = df["Encounter Number"].astype(str)

    # Admit Source
    if "Admit Source" in df.columns:
        if ADMIT_SOURCE_STRATEGY == "mode":
            mv = df["Admit Source"].dropna()
            fill_val = mv.mode().iloc[0] if not mv.empty else "Unknown"
            df["Admit Source"] = df["Admit Source"].fillna(fill_val)
        elif ADMIT_SOURCE_STRATEGY == "drop":
            df = df[df["Admit Source"].notna()].copy()
    else:
        df["Admit Source"] = "Unknown"

    return df, date_col

# Verifica existencia (sin fallbacks)
if not DATA_FILE.exists():
    print(f"[ERROR] No se encontró el archivo: {DATA_FILE}", file=sys.stderr)
    print("Colócalo en esa ruta o actualiza DATA_FILE en app.py", file=sys.stderr)
    sys.exit(1)

raw = load_raw_df(DATA_FILE)
df, date_col = clean_df(raw)
print(f"[INFO] Cargadas {len(df):,} filas de {DATA_FILE.name}".replace(",", "."))
if date_col and df[date_col].notna().any():
    display(f"[INFO] Rango {date_col}: {df[date_col].min()} -: {df[date_col].max()}")

# Opciones para filtros
dept_options   = [{"label": v, "value": v} for v in sorted(df["Department"].dropna().unique())] if "Department" in df.columns else []
status_options = [{"label": v, "value": v} for v in sorted(df["Encounter Status"].dropna().unique())] if "Encounter Status" in df.columns else []
diag_options   = [{"label": v, "value": v} for v in sorted(df["Diagnosis Primary"].dropna().unique())] if "Diagnosis Primary" in df.columns else []
admit_options  = [{"label": v, "value": v} for v in sorted(df["Admit Source"].dropna().unique())] if "Admit Source" in df.columns else []

# Grupo para boxplot (solo si existen)
box_group_candidates = [c for c in ["Admit Source", "Clinic Name", "Department"] if c in df.columns]
box_group_default = box_group_candidates[0] if box_group_candidates else None

# Candidatas para colorear histogramas (categorías)
color_by_candidates = [c for c in ["Admit Source", "Department", "Encounter Status", "Clinic Name", "Diagnosis Primary"] if c in df.columns]
color_by_default = color_by_candidates[0] if color_by_candidates else None

# DatePickerRange (si hay fecha)
date_picker_props = {}
if date_col and df[date_col].notna().any():
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if len(dates):
        date_picker_props = dict(
            min_date_allowed = dates.min().date(),
            max_date_allowed = dates.max().date(),
            start_date       = dates.min().date(),
            end_date         = dates.max().date(),
        )

# =========================
# APP (Bootstrap)
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
app.title = "Dashboard 1 - Trabajo Final de Vizualización de Avanzada de Datos — Clinical Analytics"

def kpi_card(title, id_):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H2(id=id_, className="mb-0")
        ]), className="h-100"
    )

app.layout = dbc.Container(fluid=True, children=[
    # Navbar con botón de ayuda destacado
    dbc.NavbarSimple(
        brand="Trabajo 3 - Clinical Analytics",
        brand_style={
        "textTransform": "uppercase",
        "fontWeight": "700",
        "letterSpacing": "0.06em"
        },
        color="primary",
        dark=True,
        className="mb-3",
        children=[
            dbc.Button(
                [html.Span("GUÍA DE USUARIO", className="fw-bold"), html.Span(className="pulse-dot")],
                id="btn-help",
                color="warning",  # llama más la atención que 'light'
                size="sm",
                className="ms-auto"
            ),
        ],
    ),

    # Tooltip para reforzar la acción
    dbc.Tooltip("Haz clic para ver la guía de uso", target="btn-help", placement="bottom"),


# Panel lateral con la guía
dbc.Offcanvas(
    children=html.Div([
        html.P("Este dashboard resume atenciones clínicas (encounters) y tiempos de espera.", className="mb-2"),
        html.Ul([
            html.Li("KPI # Atenciones: cuenta encounters únicos tras aplicar filtros."),
            html.Li("KPI Espera: promedio del Wait Time por encounter y luego promedio global."),
            html.Li("Filtros (izquierda): Fecha, Department, Status, Diagnosis, Admit Source."),
            html.Li("Serie temporal: atenciones únicas por día."),
            html.Li("Histograma: distribución de Wait Time; usa 'Color por' para segmentar."),
            html.Li("Caja y bigotes: Wait Time por la categoría elegida."),
            html.Li("Tip: si el rango de fechas queda vacío, NO se filtra por fecha."),
            html.Li("Superpuesto (overlay): todas las categorías en el mismo histograma, una encima de otra con transparencia. Útil para ver la forma de cada distribución."),
            html.Li("Apilado (stack): las barras se suman verticalmente. Muestra totales y la composición por categoría en cada bin."),
            html.Li("Agrupado (group): barras lado a lado por categoría en cada bin. Ideal para comparar categorías directamente"),

        ], className="mb-0"),
    ]),
    id="help-canvas",
    title="Guía rápida",
    is_open=False,
    placement="end",
),

# KPIs
dbc.Row([
    dbc.Col(kpi_card("# Atenciones", "kpi-count"), md=4),
    dbc.Col(kpi_card("Tiempo promedio de espera (min)", "kpi-wait-mean"), md=4),
    dbc.Col(kpi_card("% Diagnóstico 'None'", "kpi-diag-none"), md=4),
], className="g-3"),

# Filtros + serie temporal
dbc.Row([
    dbc.Col([
        dbc.Card(dbc.CardBody([
            html.H5("Filtros", className="mb-3"),

            html.Label("Fecha"),
            dcc.DatePickerRange(
                id="date-range",
                disabled=(not bool(date_picker_props)),
                **date_picker_props
            ),
            html.Br(), html.Br(),
            html.Div([
                dbc.Button("Todo el rango", id="btn-reset-dates", size="sm",
                            className="me-2", n_clicks=0,
                            disabled=(not bool(date_picker_props))),
                dbc.Button("Sin fecha", id="btn-clear-dates", size="sm",
                            color="secondary", outline=True, n_clicks=0,
                            disabled=(not bool(date_picker_props))),
            ], className="mb-3"),

            html.Label("Department"),
            dcc.Dropdown(id="f-dept", options=dept_options, multi=True),

            html.Br(),
            html.Label("Encounter Status"),
            dcc.Dropdown(id="f-status", options=status_options, multi=True),

            html.Br(),
            html.Label("Diagnosis Primary"),
            dcc.Dropdown(id="f-diag", options=diag_options, multi=True),

            html.Br(),
            html.Label("Admit Source"),
            dcc.Dropdown(id="f-admit", options=admit_options, multi=True),

            html.Hr(),
            html.Label("Histograma: color por"),
            dcc.Dropdown(
                id="hist-color-by",
                options=[{"label": "Sin color (un solo tono)", "value": "__none__"}] +
                        [{"label": c, "value": c} for c in color_by_candidates],
                value="__none__" if not color_by_default else color_by_default,
                clearable=False
            ),
            dbc.RadioItems(
                id="hist-mode",
                options=[
                    {"label": "Superpuesto", "value": "overlay"},
                    {"label": "Apilado", "value": "stack"},
                    {"label": "Agrupado", "value": "group"},
                ],
                value="overlay",
                inline=True,
                className="mt-2"
            ),
        ])),
    ], md=3),

    dbc.Col([
        dbc.Card(dbc.CardBody([
            html.H5("Atenciones a lo largo del tiempo", className="mb-3"),
            dcc.Graph(id="ts-encounters")
        ])),
    ], md=9),
], className="g-3"),

# Histograma
dbc.Row([
    dbc.Col([
        dbc.Card(dbc.CardBody([
            html.H5("Distribución de Wait Time (min)", className="mb-3"),
            dcc.Graph(id="hist-wait")
        ])),
    ], md=12),
], className="g-3"),

# Boxplot
dbc.Row([
    dbc.Col([
        dbc.Card(dbc.CardBody([
            html.H5("Caja y bigotes de Wait Time por categoría", className="mb-3"),
            html.Div([
                html.Label("Agrupar por"),
                dcc.Dropdown(
                    id="box-group",
                    options=[{"label": c, "value": c} for c in box_group_candidates],
                    value=box_group_default,
                    clearable=False,
                    disabled=(box_group_default is None)
                ),
            ], className="mb-3"),
            dcc.Graph(id="box-wait")
        ])),
    ], md=12),
], className="g-3"),

html.Div(f"Archivo: {DATA_FILE.name}", className="mt-3 text-muted"),
])

# =========================
# FILTROS
# =========================
def apply_filters(dff, start_date, end_date, dept_vals, status_vals, diag_vals, admit_vals):
    # Fechas
    if date_col and start_date and end_date:
        mask = (pd.to_datetime(dff[date_col], errors="coerce") >= pd.to_datetime(start_date)) & \
               (pd.to_datetime(dff[date_col], errors="coerce") <= pd.to_datetime(end_date))
        dff = dff.loc[mask]

    if dept_vals and "Department" in dff.columns:
        dff = dff[dff["Department"].isin(dept_vals)]
    if status_vals and "Encounter Status" in dff.columns:
        dff = dff[dff["Encounter Status"].isin(status_vals)]
    if diag_vals and "Diagnosis Primary" in dff.columns:
        dff = dff[dff["Diagnosis Primary"].isin(diag_vals)]
    if admit_vals and "Admit Source" in dff.columns:
        dff = dff[dff["Admit Source"].isin(admit_vals)]

    return dff

# =========================
# CONTROL PARA UNA PEQUEÑA GUÍA DE USUARIO LA ABRE O LA CIERRA
# =========================

@app.callback(
    Output("help-canvas", "is_open"),
    Input("btn-help", "n_clicks"),
    State("help-canvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_help(n, is_open):
    return not is_open


# =========================
# CONTROL DE FECHAS (Reset / Clear)
# =========================
@app.callback(
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("btn-reset-dates", "n_clicks"),
    Input("btn-clear-dates", "n_clicks"),
    State("date-range", "min_date_allowed"),
    State("date-range", "max_date_allowed"),
    prevent_initial_call=True,
)
def control_dates(n_reset, n_clear, min_allowed, max_allowed):
    if not ctx.triggered_id:
        return no_update, no_update
    if ctx.triggered_id == "btn-reset-dates":
        return min_allowed, max_allowed
    if ctx.triggered_id == "btn-clear-dates":
        return None, None
    return no_update, no_update

# =========================
# CALLBACK PRINCIPAL (KPIs, serie, histograma)
# =========================
@app.callback(
    Output("ts-encounters", "figure"),
    Output("hist-wait", "figure"),
    Output("kpi-count", "children"),
    Output("kpi-wait-mean", "children"),
    Output("kpi-diag-none", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("f-dept", "value"),
    Input("f-status", "value"),
    Input("f-diag", "value"),
    Input("f-admit", "value"),
    Input("hist-color-by", "value"),
    Input("hist-mode", "value"),
)
def update_views(start_date, end_date, dept_vals, status_vals, diag_vals, admit_vals, hist_color_by, hist_mode):
    dff = apply_filters(df.copy(), start_date, end_date, dept_vals, status_vals, diag_vals, admit_vals)

    # Atenciones (encounters únicos)
    if "Encounter Number" in dff.columns:
        dff_enc = dff.dropna(subset=["Encounter Number"]).drop_duplicates(subset=["Encounter Number"])
    else:
        dff_enc = dff

    if dff_enc.empty:
        fig_ts = go.Figure();  fig_ts.update_layout(title_text="Sin datos para los filtros seleccionados")
        fig_hi = go.Figure();  fig_hi.update_layout(title_text="Wait Time no disponible")
        return fig_ts, fig_hi, "0", "—", "—"

    # KPI: # Atenciones
    k_count = f"{len(dff_enc):,}".replace(",", ".")

    # KPI: Tiempo promedio de espera (promedio por atención y luego global)
    if "Wait Time Min" in dff.columns:
        w_enc = (dff.assign(**{"Wait Time Min": pd.to_numeric(dff["Wait Time Min"], errors="coerce").clip(lower=0)})
                   .dropna(subset=["Encounter Number"])
                   .groupby("Encounter Number", as_index=False)["Wait Time Min"].mean()["Wait Time Min"])
        k_wait = f"{w_enc.mean():,.2f}".replace(",", ".") if len(w_enc) else "—"
    else:
        k_wait = "—"

    # KPI: % Diagnosis 'None'
    k_none = (f"{(dff['Diagnosis Primary'].eq('None').mean()*100):.1f}%"
              if "Diagnosis Primary" in dff.columns and len(dff) else "—")

   # Serie temporal: área apilada por clínica (encounters únicos por día y clínica)
    if date_col and dff_enc[date_col].notna().any():
        if "Clinic Name" in dff_enc.columns:
            ts = dff_enc.copy()
            ts["__date__"] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=["__date__"])
            # Contar encounters únicos por día y clínica
            grp = (ts.dropna(subset=["Clinic Name"])
                     .groupby([ts["__date__"].dt.to_period("D"), "Clinic Name"])
                     .agg(encounters=("Encounter Number", pd.Series.nunique) if "Encounter Number" in ts.columns else ("Clinic Name", "size"))
                     .reset_index())
            grp["__date__"] = grp["__date__"].dt.to_timestamp()

            if grp.empty:
                fig_ts = go.Figure(); fig_ts.update_layout(title_text="Sin datos para los filtros seleccionados", template="plotly_white")
            else:
                # Limitar a TOP N clínicas (evitamos leyendas infinitas)
                TOP_N = 10
                top_clinics = (grp.groupby("Clinic Name")["encounters"].sum()
                                  .sort_values(ascending=False)
                                  .head(TOP_N).index.tolist())
                grp["Clinic Name"] = grp["Clinic Name"].where(grp["Clinic Name"].isin(top_clinics), other="Otros")

                # Reagrupar después de reasignar "Otros"
                grp = (grp.groupby(["__date__", "Clinic Name"], as_index=False)["encounters"].sum())

                fig_ts = px.area(
                    grp, x="__date__", y="encounters", color="Clinic Name",
                    color_discrete_sequence=COLOR_SEQ,
                    title="Atenciones por día — área apilada por clínica (encounters únicos)"
                )
                fig_ts.update_layout(
                    template="plotly_white",
                    xaxis_title="Fecha", yaxis_title="# Atenciones",
                    legend_title_text="Clínica",
                    hovermode="x unified"
                )
        else:
            ts = dff_enc.copy()
            ts["__date__"] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=["__date__"])
            ts = (ts.groupby(ts["__date__"].dt.to_period("D"))
                    .agg(encounters=("Encounter Number", pd.Series.nunique) if "Encounter Number" in ts.columns else ("__date__", "size"))
                    .rename("encounters")
                    .to_timestamp())
            fig_ts = px.line(ts, x=ts.index, y="encounters", title="Atenciones por día (encounters únicos)")
            fig_ts.update_traces(line_color=COLOR_SEQ[0])
            fig_ts.update_layout(xaxis_title="Fecha", yaxis_title="# Atenciones", template="plotly_white")
    else:
        fig_ts = go.Figure(); fig_ts.update_layout(title_text="Sin columna de fecha válida", template="plotly_white")

    # Histograma de espera 
    if "Wait Time Min" in dff.columns:
        h = pd.to_numeric(dff["Wait Time Min"], errors="coerce").clip(lower=0)
        dff_h = dff.assign(**{"Wait Time Min": h})
        dff_h = dff_h.dropna(subset=["Wait Time Min"])

        if dff_h.empty:
            fig_hi = go.Figure(); fig_hi.update_layout(title_text="Wait Time no disponible", template="plotly_white")
        else:
            if hist_color_by and hist_color_by != "__none__" and hist_color_by in dff_h.columns:
                # Limita categorías a top 12 por frecuencia, porque no queremos saturar
                top = (dff_h[hist_color_by].value_counts(dropna=True)
                       .head(min(12, len(COLOR_SEQ)))).index.tolist()
                dff_h = dff_h[dff_h[hist_color_by].isin(top)]

                fig_hi = px.histogram(
                    dff_h, x="Wait Time Min", color=hist_color_by, nbins=50,
                    color_discrete_sequence=COLOR_SEQ,
                    title=f"Wait Time (min) — coloreado por {hist_color_by}"
                )
                # Modo de barras 
                fig_hi.update_layout(barmode=hist_mode)
                if hist_mode == "overlay":
                    fig_hi.update_traces(opacity=0.70)
            else:
                # Un solo color, pero usando un tono de la paleta (no azul chillón por defecto)
                fig_hi = px.histogram(dff_h, x="Wait Time Min", nbins=50,
                                      color_discrete_sequence=[COLOR_SEQ[2]],
                                      title="Wait Time (min)")
            fig_hi.update_layout(template="plotly_white", xaxis_title="Wait Time (min)", yaxis_title="Frecuencia")
    else:
        fig_hi = go.Figure(); fig_hi.update_layout(title_text="Wait Time no disponible", template="plotly_white")

    return fig_ts, fig_hi, k_count, k_wait, k_none

# =========================
# CALLBACK BOX PLOT
# =========================
@app.callback(
    Output("box-wait", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("f-dept", "value"),
    Input("f-status", "value"),
    Input("f-diag", "value"),
    Input("f-admit", "value"),
    Input("box-group", "value"),
)
def update_box(start_date, end_date, dept_vals, status_vals, diag_vals, admit_vals, group_col):
    if group_col is None or group_col not in df.columns:
        fig = go.Figure(); fig.update_layout(title_text="No hay columna válida para agrupar", template="plotly_white")
        return fig

    dff = apply_filters(df.copy(), start_date, end_date, dept_vals, status_vals, diag_vals, admit_vals)

    if "Wait Time Min" not in dff.columns:
        fig = go.Figure(); fig.update_layout(title_text="Wait Time Min no disponible", template="plotly_white")
        return fig

    dff["Wait Time Min"] = pd.to_numeric(dff["Wait Time Min"], errors="coerce").clip(lower=0)
    dff = dff.dropna(subset=[group_col, "Wait Time Min"])

    if dff.empty:
        fig = go.Figure(); fig.update_layout(title_text="Sin datos para los filtros seleccionados", template="plotly_white")
        return fig

    # Ordenar categorías por mediana (desc)
    med = dff.groupby(group_col)["Wait Time Min"].median().sort_values(ascending=False)

    # Top 30 para evitar labels infinitos
    categories = med.index.tolist()
    if len(categories) > 30:
        categories = categories[:30]
        dff = dff[dff[group_col].isin(categories)]

    fig = px.box(
        dff, x=group_col, y="Wait Time Min",
        points="outliers",
        category_orders={group_col: categories},
        color=group_col,  # colorea cada caja con paleta
        color_discrete_sequence=COLOR_SEQ,
        title=f"Wait Time (min) — Caja y bigotes por {group_col}"
    )
    fig.update_layout(template="plotly_white", xaxis_title=group_col, yaxis_title="Wait Time (min)", boxmode="group")
    return fig

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True, host=args.host, port=args.port)
