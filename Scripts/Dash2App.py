from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import calendar
import warnings
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=63642)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

print(f"Iniciando Host:{args.host}, Port:{args.port}...")

# -------------------------------
# 1. Cargar datos
# -------------------------------

# Suprimir warning de formato de fecha
warnings.filterwarnings("ignore", message="Could not infer format")
FILENAME  = Path("Temp/clinical_analytics.csv").resolve()
df = pd.read_csv("Temp/clinical_analytics.csv")
df["Appt Start Time"] = pd.to_datetime(df["Appt Start Time"], errors="coerce")
df["YearMonth"] = df["Appt Start Time"].dt.to_period("M")
df["MesNombre"] = df["Appt Start Time"].dt.month.apply(lambda m: calendar.month_name[m])

start_date_default = df["Appt Start Time"].min().date()
end_date_default = df["Appt Start Time"].max().date()

# -------------------------------
# 2. Inicializar app
# -------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Dashboard 2 - Trabajo Final de Vizualización de Avanzada de Datos — Clinical Analytics"
server = app.server

# -------------------------------
# 3. Layout
# -------------------------------
app.layout = dbc.Container([
    html.H2("Dashboard Clínico", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.H5("Filtros"),
            dbc.Select(id="filtro_clinic",
                       options=[{"label": c, "value": c} for c in sorted(df["Clinic Name"].dropna().unique())],
                       placeholder="Nombre de Clínica"),
            dbc.Select(id="filtro_department",
                       options=[{"label": d, "value": d} for d in sorted(df["Department"].dropna().unique())],
                       placeholder="Departamento", className="mt-2"),
            dbc.Select(id="filtro_admit",
                       options=[{"label": a, "value": a} for a in sorted(df["Admit Type"].dropna().unique())],
                       placeholder="Tipo de Admisión", className="mt-2"),
            dbc.Select(id="filtro_diagnostico",
                       options=[{"label": a, "value": a} for a in sorted(df["Diagnosis Primary"].dropna().unique())],
                       placeholder="Diagnóstico", className="mt-2"),
            dcc.DatePickerRange(
                id="filtro_fecha",
                start_date=start_date_default,
                end_date=end_date_default,
                display_format="YYYY-MM-DD",
                className="mt-2"
            ),
            dbc.Button("Restablecer filtros", id="reset_button", color="secondary",
                       className="mt-3 w-100")
        ], width=3),

        dbc.Col([
            # KPIs
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Tiempo de espera promedio"),
                                  dbc.CardBody(html.H4(id="kpi_espera"))])),
                dbc.Col(dbc.Card([dbc.CardHeader("Calificación de atención promedio"),
                                  dbc.CardBody(html.H4(id="kpi_care"))])),
                dbc.Col(dbc.Card([dbc.CardHeader("Número de registros"),
                                  dbc.CardBody(html.H4(id="kpi_registros"))]))
            ], className="mb-4"),

            # Gráficos 2x2
            dbc.Row([
                dbc.Col(dcc.Graph(id="pie_clinicas", config={"displayModeBar": True}), width=6),
                dbc.Col(dcc.Graph(id="pie_admit_type", config={"displayModeBar": True}), width=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="graf_pacientes_mes", config={"displayModeBar": True}), width=6),
                dbc.Col(dcc.Graph(id="graf_pacientes_dia", config={"displayModeBar": True}), width=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="graf_pacientes_hora", config={"displayModeBar": True}), width=6),
                dbc.Col(dcc.Graph(id="graf_care_box", config={"displayModeBar": True}), width=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="graf_series", config={"displayModeBar": True}), width=6),
                dbc.Col(dcc.Graph(id="graf_admit_source_barras", config={"displayModeBar": True}), width=6)
            ], className="mb-4"),

        ], width=9)
    ])
], fluid=True)

# -------------------------------
# 4. Reset de filtros
# -------------------------------
@app.callback(
    Output("filtro_clinic", "value"),
    Output("filtro_department", "value"),
    Output("filtro_admit", "value"),
    Output("filtro_diagnostico", "value"),
    Output("filtro_fecha", "start_date"),
    Output("filtro_fecha", "end_date"),
    Output("pie_clinicas", "clickData"),
    Output("pie_admit_type", "clickData"),
    Output("graf_pacientes_mes", "selectedData"),
    Output("graf_pacientes_dia", "selectedData"),
    Output("graf_care_box", "selectedData"),
    Output("graf_series", "selectedData"),
    Output("graf_admit_source_barras", "selectedData"),
    Output("graf_pacientes_hora", "selectedData"),
    Input("reset_button", "n_clicks"),
    prevent_initial_call=True
)
def reset_all(n):
    return (None, None, None, None,
            start_date_default, end_date_default,
            None, None, None, None, None, None, None, None)

# -------------------------------
# 5. Callback principal
# -------------------------------
@app.callback(
    Output("kpi_espera", "children"),
    Output("kpi_espera", "style"),
    Output("kpi_care", "children"),
    Output("kpi_care", "style"),
    Output("kpi_registros", "children"),
    Output("graf_pacientes_mes", "figure"),
    Output("graf_care_box", "figure"),
    Output("graf_series", "figure"),
    Output("graf_pacientes_dia", "figure"),
    Output("pie_clinicas", "figure"),
    Output("pie_admit_type", "figure"),
    Output("graf_admit_source_barras", "figure"),
    Output("graf_pacientes_hora", "figure"),
    Input("filtro_clinic", "value"),
    Input("filtro_department", "value"),
    Input("filtro_admit", "value"),
    Input("filtro_diagnostico", "value"),
    Input("filtro_fecha", "start_date"),
    Input("filtro_fecha", "end_date"),
    Input("pie_clinicas", "clickData"),
    Input("pie_admit_type", "clickData"),
    Input("graf_pacientes_mes", "selectedData"),
    Input("graf_pacientes_dia", "selectedData"),
    Input("graf_care_box", "selectedData"),
    Input("graf_series", "selectedData"),
    Input("graf_admit_source_barras", "selectedData"),
    Input("graf_pacientes_hora", "selectedData")
)
def actualizar_dashboard(clinic, department, admit, diagnostico, start_date, end_date,
                         clinica_click, admit_click,
                         sel_mes, sel_dia, sel_care, sel_series, sel_admit_source, sel_hora):
    dff = df.copy()

    # --- Filtros
    if clinic:
        dff = dff[dff["Clinic Name"] == clinic]
    if department:
        dff = dff[dff["Department"] == department]
    if admit:
        dff = dff[dff["Admit Type"] == admit]
    if diagnostico:
        dff = dff[dff["Diagnosis Primary"] == diagnostico]
    if clinica_click:
        dff = dff[dff["Clinic Name"] == clinica_click["points"][0]["label"]]
    if admit_click:
        dff = dff[dff["Admit Type"] == admit_click["points"][0]["label"]]
    if sel_mes:
        valores = [p["x"] for p in sel_mes["points"]]
        dff = dff[dff["MesNombre"].isin(valores)]
    if sel_dia:
        valores = [p["x"] for p in sel_dia["points"]]
        dff["DiaSemana"] = dff["Appt Start Time"].dt.weekday
        dff["NombreDia"] = dff["DiaSemana"].apply(lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        dff = dff[dff["NombreDia"].isin(valores)]
    if sel_care:
        valores = [p["x"] for p in sel_care["points"]]
        dff = dff[dff["Department"].isin(valores)]
    if sel_admit_source:
        valores = [p["x"] for p in sel_admit_source["points"]]
        dff = dff[dff["Admit Source"].isin(valores)]
    if start_date and end_date:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        dff = dff[(dff["Appt Start Time"] >= start) & (dff["Appt Start Time"] <= end)]

    # KPIs
    espera_prom = dff["Wait Time Min"].mean() if not dff.empty else 0
    care_prom = dff["Care Score"].mean() if not dff.empty else 0
    num_reg = len(dff)
    color_espera = {"color": "green"} if espera_prom < 30 else {"color": "red", "fontWeight": "bold"}
    color_care = {"color": "green"} if care_prom >= 4 else {"color": "orange", "fontWeight": "bold"}

    # Pacientes por mes
    df_mes = (
        dff.groupby(["YearMonth"])
           .size()
           .reset_index(name="Pacientes")
           .sort_values("YearMonth")
    )
    df_mes["YearMonthTs"] = df_mes["YearMonth"].dt.to_timestamp()
    df_mes["YearMonthStr"] = df_mes["YearMonth"].dt.strftime("%b %Y")

    fig_pacientes_mes = px.bar(
        df_mes,
        x="YearMonthTs",
        y="Pacientes",
        color="YearMonthStr",
        text=df_mes["Pacientes"].apply(lambda x: f"{x:,}".replace(",", " ")),
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Pacientes por Mes"
    )
    fig_pacientes_mes.update_traces(textposition="outside")
    fig_pacientes_mes.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=df_mes["YearMonthTs"],
            ticktext=df_mes["YearMonthStr"],
            tickangle=-45
        ),
        xaxis_title="Mes",
        yaxis_title="Pacientes",
        showlegend=False,
        dragmode="select"
    )

    # Boxplot Care Score
    fig_care_box = px.box(dff, x="Department", y="Care Score", color="Department",
                          title="Calificación de la atención por Departamento")
    fig_care_box.update_layout(dragmode="select",
                               xaxis_title="Nombre del Departamento",
                               yaxis_title="Calificación de la atención")

    # Serie diaria
    df_series = dff.groupby(dff["Appt Start Time"].dt.date).size().reset_index(name="Pacientes")
    fig_series = px.line(df_series, x="Appt Start Time", y="Pacientes", title="Pacientes por Fecha")
    fig_series.update_layout(dragmode="select",
                              xaxis_title="Hora de inicio de la cita",
                              yaxis_title="Pacientes")

    # Pacientes por día de la semana
    if not dff.empty:
        dff["DiaSemana"] = dff["Appt Start Time"].dt.weekday
        dff["NombreDia"] = dff["DiaSemana"].apply(lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        orden = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        df_dia = dff.groupby("NombreDia").size().reset_index(name="Pacientes")
        df_dia["NombreDia"] = pd.Categorical(df_dia["NombreDia"], categories=orden, ordered=True)
        df_dia = df_dia.sort_values("NombreDia")
        fig_dia = px.bar(df_dia, x="NombreDia", y="Pacientes",
                         text=df_dia["Pacientes"].apply(lambda x: f"{x:,}".replace(",", " ")),
                         color="NombreDia",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title="Pacientes por Día de la Semana")
        fig_dia.update_traces(textposition="outside")
        fig_dia.update_layout(xaxis_title="Día de la Semana", yaxis_title="Pacientes",
                              showlegend=False, dragmode="select")
    else:
        fig_dia = px.bar(title="Pacientes por Día de la Semana (sin datos)")

    # Pacientes por hora del día
    if not dff.empty:
        dff["Hora"] = dff["Appt Start Time"].dt.hour
        df_hora = dff.groupby("Hora").size().reset_index(name="Pacientes")
        df_hora["HoraStr"] = df_hora["Hora"].astype(str)
        fig_hora = px.bar(
            df_hora,
            x="Hora",
            y="Pacientes",
            color="HoraStr",
            text=df_hora["Pacientes"].apply(lambda x: f"{x:,}".replace(",", " ")),
            color_discrete_sequence=px.colors.qualitative.Set3,
            title="Pacientes por Hora del Día"
        )
        fig_hora.update_traces(textposition="outside")
        fig_hora.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=df_hora["Hora"],
                ticktext=df_hora["Hora"].astype(str),
                tickangle=0,
                title="Hora del Día"
            ),
            yaxis_title="Pacientes",
            showlegend=False,
            dragmode="select"
        )
    else:
        fig_hora = px.bar(title="Pacientes por Hora del Día (sin datos)")

    # Gráficos circulares
    fig_clinicas = px.pie(dff, names="Clinic Name", title="Distribución de Clínicas")
    fig_admit_type = px.pie(dff, names="Admit Type", title="Distribución de Tipo de Admisión")

    # Admit Source
    df_admit_source = (
        dff.groupby("Admit Source").size()
           .reset_index(name="Pacientes")
           .sort_values("Pacientes", ascending=False)
    )
    fig_admit_source_barras = px.bar(df_admit_source, x="Admit Source", y="Pacientes",
                                     text=df_admit_source["Pacientes"].apply(lambda x: f"{x:,}".replace(",", " ")),
                                     color="Admit Source",
                                     color_discrete_sequence=px.colors.qualitative.Set2,
                                     title="Distribución de Origen de Admisión")
    fig_admit_source_barras.update_traces(textposition="outside")
    fig_admit_source_barras.update_layout(xaxis_title="Origen de Admisión", yaxis_title="Pacientes",
                                          showlegend=False, dragmode="select")

    return (f"{espera_prom:.1f} min", color_espera,
            f"{care_prom:.2f}", color_care,
            f"{num_reg}",
            fig_pacientes_mes, fig_care_box, fig_series, fig_dia,
            fig_clinicas, fig_admit_type, fig_admit_source_barras, fig_hora)

# -------------------------------
# 6. Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host=args.host, port=args.port)
