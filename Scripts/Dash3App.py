print("Iniciando aplicación Dash3App.py...")
import argparse
import sys
import os
from dash import Dash, html

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Modules import CoreFXs as CoreFXs
import pandas as pd
import pandas
from pathlib import Path
from IPython.display import display
from enum import Enum
import subprocess

sys.stderr = sys.stdout
sys.stdout.reconfigure(encoding="utf-8")

pandas.set_option("display.max_rows", None)


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=63643)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

print(f"Iniciando Host:{args.host}, Port:{args.port}...")


def LoadDataset():
    DOWNLOAD_DIR = "Temp"
    DATA_FILENAME = Path(f"{DOWNLOAD_DIR}/clinical_analytics.csv").resolve()
    CoreFXs.ShowInfoMessage("Cargando datos en DataFrame")
    df = pd.read_csv(DATA_FILENAME)
    print(
        f"Archivo CSV cargado: {DATA_FILENAME} ({df.shape[0]} filas x {df.shape[1]} columnas)"
    )
    return df


def CleanDataset(df: pandas.DataFrame):
    CoreFXs.ShowInfoMessage("Limpiando y transformando datos")
    df.columns = [
        col.strip().title().replace(" ", "").replace("_", "").replace("-", "").strip()
        for col in df.columns
    ]
    df.AdmitSource = df.AdmitSource.astype(pandas.StringDtype()).str.strip()
    df.AdmitType = df.AdmitType.astype(pandas.StringDtype()).str.strip()
    df.ApptStartTime = pandas.to_datetime(
        df.ApptStartTime, format="%Y-%m-%d %I:%M:%S %p", errors="raise"
    )
    df.CareScore = df.CareScore.astype(pandas.Int64Dtype())
    df.CheckInTime = pandas.to_datetime(
        df.CheckInTime, format="%Y-%m-%d %I:%M:%S %p", errors="raise"
    )

    df = df.dropna(subset=["ApptStartTime", "CheckInTime"])

    df.ClinicName = df.ClinicName.astype(pandas.StringDtype()).str.strip()
    df.Department = df.Department.astype(pandas.StringDtype()).str.strip()
    df.DiagnosisPrimary = df.DiagnosisPrimary.astype
    df.DischargeDatetimeNew = pandas.to_datetime(
        df.DischargeDatetimeNew, format="%Y-%m-%d", errors="raise"
    )
    df.EncounterNumber = df.EncounterNumber.astype(pandas.StringDtype()).str.strip()
    df.EncounterStatus = df.EncounterStatus.astype(pandas.StringDtype()).str.strip()
    df.NumberOfRecords = df.NumberOfRecords.astype(pandas.Int64Dtype())

    df.WaitTimeMin = (df.ApptStartTime - df.CheckInTime).dt.total_seconds() / 60
    df.WaitTimeMin = df.WaitTimeMin.round().astype(pandas.Int64Dtype())
    df.WaitTimeMin = df.WaitTimeMin.astype(pandas.Int64Dtype())
    return df


def ShowDatasetInfo(df: pandas.DataFrame):
    # Mostrar información del DataFrame
    CoreFXs.ShowTableInfo(df, "DataFrame Original")
    CoreFXs.ShowTableHead(df, "DataFrame Original", 10)
    CoreFXs.ShowTableShape(df, "DataFrame Original")
    CoreFXs.ShowTableStats(df, "DataFrame Original")
    CoreFXs.ShowUniqueCounts(df, "DataFrame Original")


dataOriginal = LoadDataset()
dfMain = dataOriginal.copy()
dfMain = CleanDataset(dfMain)
# ShowDatasetInfo(dfMain)


###############3


import pandas as pandas
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px


# Valores para filtros (maneja ausencia de columnas)
departments = (
    sorted(dfMain["Department"].dropna().unique()) if "Department" in dfMain else []
)
admit_types = (
    sorted(dfMain["AdmitType"].dropna().unique()) if "AdmitType" in dfMain else []
)


app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])
app.title = "Dashboard 3 - Trabajo Final de Vizualización de Avanzada de Datos - Clinical Analytics — Atenciones Clínicas"

admitSources = sorted(dfMain.AdmitSource.dropna().astype(str).unique().tolist())
clinicNames = sorted(dfMain.ClinicName.dropna().astype(str).unique().tolist())
departments = sorted(dfMain.Department.dropna().astype(str).unique().tolist())


def KpiCard(title, id_):
    return dbc.Card(
        dbc.CardBody(
            [html.H6(title, className="text-muted"), html.H2(id=id_, className="mb-0")]
        ),
        className="h-100",
    )


app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Componente práctico 3 - Clinical Analytics - Atenciones Clínicas",
            brand_style={
                "textTransform": "uppercase",
                "fontWeight": "700",
                "letterSpacing": "0.06em",
            },
            color="primary",
            dark=False,
            className="mb-3",
            children=[],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Clínicas"),
                        dcc.Dropdown(
                            id="SelectClinicNames",
                            options=[{"label": s, "value": s} for s in clinicNames],
                            placeholder="Selecciona las clinicas",
                            multi=True,
                            clearable=True,
                        ),
                        html.Small(id="EchoClinicNames", className="text-muted"),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Orígenes de Admisión"),
                        dcc.Dropdown(
                            id="SelectAdmitSources",
                            options=[{"label": s, "value": s} for s in admitSources],
                            placeholder="Selecciona las fuentes de admisión",
                            multi=True,
                            clearable=True,
                        ),
                        html.Small(id="EchoAdmitSources", className="text-muted"),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Departamentos"),
                        dcc.Dropdown(
                            id="SelectDepartments",
                            options=[{"label": s, "value": s} for s in departments],
                            placeholder="Selecciona los departamentos",
                            multi=True,
                            clearable=True,
                        ),
                        html.Small(id="EchoDepartments", className="text-muted"),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Rango de Fechas en Check-In"),
                        dcc.DatePickerRange(
                            id="SelectDateRange",
                            min_date_allowed=dfMain["ApptStartTime"].min().date(),
                            max_date_allowed=dfMain["ApptStartTime"].max().date(),
                            start_date=dfMain["ApptStartTime"].min().date(),
                            end_date=dfMain["ApptStartTime"].max().date(),
                            display_format="YYYY-MM-DD",
                            minimum_nights=0,
                            clearable=False,
                        ),
                        html.Small(id="EchoDateRange", className="text-muted"),
                    ],
                    md=3,
                ),
            ],
            className="g-2 mt-3",
        ),
        dbc.Row(
            [
                dbc.Row(
                    [
                        dbc.Col(KpiCard("Promedio", "KpiMean"), md=2),
                        dbc.Col(KpiCard("Máximo", "KpiMax"), md=2),
                        dbc.Col(KpiCard("Mínimo", "KpiMin"), md=2),
                        dbc.Col(KpiCard("Moda", "KpiMode"), md=2),
                        dbc.Col(KpiCard("# Registros", "KpiCount"), md=4),
                    ],
                    className="g-3",
                ),
            ],
            className="g-2 mt-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [html.H6("Clínicas"), dcc.Graph(id="FigClinicNames")]
                        )
                    ),
                    md=12,
                ),
            ],
            className="g-3 mt-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("Orígenes de admisión"),
                                dcc.Graph(id="FigAdmitSources"),
                            ]
                        )
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [html.H6("Departmentos"), dcc.Graph(id="FigDepartments")]
                        )
                    ),
                    md=6,
                ),
            ],
            className="g-3 mt-2",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("EchoAdmitSources", "children"),
    Input("SelectAdmitSources", "value"),
)
def ShowAdmitSource(values):
    if not values:
        return "Nada seleccionado"
    if isinstance(values, list):
        return f"Seleccionados: {', '.join(values)}"
    return f"Seleccionado: {values}"


@app.callback(
    Output("EchoClinicNames", "children"),
    Input("SelectClinicNames", "value"),
)
def ShowClinicNames(values):
    if not values:
        return "Nada seleccionado"
    if isinstance(values, list):
        return f"Seleccionados: {', '.join(values)}"
    return f"Seleccionado: {values}"


@app.callback(
    Output("EchoDepartments", "children"),
    Input("SelectDepartments", "value"),
)
def ShowDepartments(values):
    if not values:
        return "Nada seleccionado"
    if isinstance(values, list):
        return f"Seleccionados: {', '.join(values)}"
    return f"Seleccionado: {values}"


@app.callback(
    Output("EchoDateRange", "children"),
    Input("SelectDateRange", "start_date"),
    Input("SelectDateRange", "end_date"),
)
def ShowDateRange(start_date, end_date):
    if not start_date or not end_date:
        return "Sin rango seleccionado"
    return f"Rango seleccionado: {start_date} → {end_date}"


@app.callback(
    Output("KpiMean", "children"),
    Output("KpiMax", "children"),
    Output("KpiMin", "children"),
    Output("KpiMode", "children"),
    Output("KpiCount", "children"),
    Output("FigAdmitSources", "figure"),
    Output("FigClinicNames", "figure"),
    Output("FigDepartments", "figure"),
    Input("SelectClinicNames", "value"),
    Input("SelectAdmitSources", "value"),
    Input("SelectDepartments", "value"),
    Input("SelectDateRange", "start_date"),
    Input("SelectDateRange", "end_date"),
)
def CalculateWaitTimeMinStats(sel_clinics, sel_admits, sel_depts, start_date, end_date):
    df = dfMain.copy()
    if sel_clinics:
        df = df[df["ClinicName"].astype(str).isin(sel_clinics)]
    if sel_admits:
        df = df[df["AdmitSource"].astype(str).isin(sel_admits)]
    if sel_depts:
        df = df[df["Department"].astype(str).isin(sel_depts)]
    if start_date:
        df = df[df["CheckInTime"] >= pandas.to_datetime(start_date)]
    if end_date:
        df = df[
            df["CheckInTime"]
            <= pandas.to_datetime(end_date)
            + pandas.Timedelta(days=1)
            - pandas.Timedelta(seconds=1)
        ]

    s = df["WaitTimeMin"]
    s = s[s >= 0]

    if s.empty:
        return "Sin datos para los filtros seleccionados."

    promedio = float(s.mean(skipna=True))
    maximo = int(s.max())
    minimo = int(s.min())
    total = int(s.count())

    moda_series = s.mode(dropna=True)
    if moda_series.empty:
        moda_txt = "—"
    else:
        moda_vals = sorted({int(v) for v in moda_series.tolist()})
        if len(moda_vals) > 5:
            moda_txt = ", ".join(map(str, moda_vals[:5])) + "…"
        else:
            moda_txt = ", ".join(map(str, moda_vals))

    def pie_count(df_src, col, titulo):
        if col not in df_src.columns or df_src.empty:
            return px.pie(title=f"Sin datos de {titulo}")

        s = df_src[col].fillna("Desconocido").astype(str).value_counts().reset_index()
        s.columns = [col, "Count"]
        MAX_CATEGORIES = 7
        if len(s) > MAX_CATEGORIES:
            s = s.head(MAX_CATEGORIES)
        fig = px.pie(s, names=col, values="Count", title=titulo, template="plotly_dark")
        fig.update_traces(textposition="inside") 
        fig.update_layout(
            plot_bgcolor="#222831", 
            paper_bgcolor="#393E46",  # gris más claro
        )
        return fig

    # --------- Figuras ---------
    fig_admit = pie_count(df, "AdmitSource", "Atenciones por orígenes de admisión")
    fig_clinic = pie_count(df, "ClinicName", "Atenciones por Clinic Name")
    fig_dept = pie_count(df, "Department", "Atenciones por Department")

    return (
        # f"Promedio: {promedio:.2f} minutos | Máximo: {maximo} minutos | Mínimo: {minimo} minutos | Moda: {moda_txt} minutos | Registros: {total}",
        f"{promedio:.2f} minutos",
        f"{maximo} minutos",
        f"{minimo} minutos",
        f"{moda_txt} minutos",
        f"{total}",
        fig_admit,
        fig_clinic,
        fig_dept,
    )


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False)


sys.exit(0)
