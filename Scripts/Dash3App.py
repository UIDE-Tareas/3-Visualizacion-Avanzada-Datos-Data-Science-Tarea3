import argparse
import sys
import os
from dash import Dash, html
import pandas as pd
import pandas
from pathlib import Path
from IPython.display import display
from enum import Enum
import subprocess


def RunCommand(commandList: list[str], printCommand: bool = True, printError:bool=True) -> subprocess.CompletedProcess:
    print("⏳", " ".join(commandList))
    stdOutput = None if printCommand else subprocess.DEVNULL
    errorOutput = None if printError else subprocess.PIPE
    result = subprocess.run(commandList,stdout=stdOutput, stderr=errorOutput, text=True)
    if result.returncode != 0 and printError:
        print(result.stderr) 
    return result

def ShowEnvironmentInfo():
    print("ℹ️  Environment Info:")
    print("Python Version:", sys.version)
    print("Platform:", sys.platform)
    print("Executable Path:", sys.executable)
    print("Current Working Directory:", os.getcwd())
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
    print("sys.prefix:", sys.prefix)
    print("sys.base_prefix:", sys.base_prefix)


LIBS = [
    "requests",
    "numpy",
    "pandas",
    "seaborn",
    "matplotlib",
    "ipython",
]

def InstallDeps(libs: list[str] = []):
    print("ℹ️ Installing deps.")
    RunCommand([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], printCommand=True) 
    if not libs or len(libs) == 0:
        libs = []
    RunCommand([sys.executable, "-m", "pip", "install",*LIBS, *libs], printCommand=True) 
    print("Deps installed.")

from IPython.display import display
import pandas
import requests
import zipfile
import gzip
import shutil
    
# Función para mostrar mensajes de información.
def ShowWarningMessage(message: str):
    print()
    print(f"⚠️ {message}".upper())

# Función para mostrar la información del DataFrame.
def ShowTableInfo(df:pandas.DataFrame, title):
    print(f"ℹ️ {title} ℹ️".upper())
    df.info()
    print()

# Función para mostrar las n primeras filas del DataFrame.
def ShowTableHead(df:pandas.DataFrame, title:str, headQty=10):
    print(f"ℹ️ {title.upper()}: Primeros {headQty} elementos.")
    print(df.head(headQty))
    print()

# Función para mostrar las n últimas filas del DataFrame.
def ShowTableTail(df:pandas.DataFrame,title:str ,tailQty=10):
    print(f"ℹ️ {title.upper()}: Últimos {tailQty} elementos.")
    print(df.tail(tailQty))
    print()

# Mostrar el tamaño del DataFrame
def ShowTableShape(df:pandas.DataFrame, title:str):
    print(f"ℹ️ {title.upper()} - Tamaño de los datos")
    print(f"{df.shape[0]} filas x {df.shape[1]} columnas")
    print()

# Función para mostrar la estadística descriptiva de todas las columnas del DataFrame, por tipo de dato.
def ShowTableStats(df: pandas.DataFrame, title:str = ""):
    print(f"ℹ️ Estadística descriptiva - {title}".upper())
    numeric_types = ['int64', 'float64', 'Int64', 'Float64']
    numeric_cols = df.select_dtypes(include=numeric_types)
    if not numeric_cols.empty:
        print("    🔢 Columnas numéricas".upper())
        numeric_desc = numeric_cols.describe().round(2).T  # Transpuesta para añadir columna
        numeric_desc["var"] = numeric_cols.var().round(2)  # Añadir varianza
        print(numeric_desc.T) 
    non_numeric_types = ['object', 'string', 'bool', 'category']
    non_numeric_cols = df.select_dtypes(include=non_numeric_types)
    if not non_numeric_cols.empty:
        print("    🔡 Columnas no numéricas".upper())
        non_numeric_desc = non_numeric_cols.describe()
        print(non_numeric_desc)
    datetime_cols = df.select_dtypes(include=['datetime'])
    if not datetime_cols.empty:
        print("    📅 Columnas fechas".upper())
        datetime_desc = datetime_cols.describe()
        print(datetime_desc)

# Función para mostrar los valores nulos o NaN de cada columna en un DataFrame
def ShowNanValues(df: pandas.DataFrame):
    print(f"ℹ️ Contador de valores Nulos".upper())
    nulls_count = df.isnull().sum()
    nulls_df = nulls_count.reset_index()
    nulls_df.columns = ['Columna', 'Cantidad_Nulos']
    print(nulls_df)
    print()

def ShowUniqueCounts(df: pandas.DataFrame, title: str):
    ShowInfoMessage(f"Valores únicos por columna", title)
    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ['Columna', 'CantidadUnicos']
    print(unique_counts)
    print()

# Función para descargar un archivo
def DownloadFile(uri: str, filename: str, overwrite: bool = False, timeout: int = 20):
    dest = Path(filename).resolve()
    if dest.exists() and dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        print(f"✅ Ya existe: \"{dest}\". No se descarga (use overwrite=True para forzar).")
        return
    if dest.parent and not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"ℹ️ Descargando \"{uri}\" → \"{dest}\"")
    try:
        with requests.get(uri, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:  # filtra keep-alive chunks
                        f.write(chunk)
            tmp.replace(dest)
        print(f"✅ Archivo \"{dest}\" descargado exitosamente.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar: {e}")

# Función para descomprimir un archivo zip
def UnzipFile(filename: str, outputDir: str):
    print(f'ℹ️ Descomprimiendo "{filename}" en "{outputDir}"')
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(outputDir)
        print(f"Descomprimido en: {os.path.abspath(outputDir)}")
    except Exception as e:
        print(f"Error: {e}")

def UngzipFile(filename: str, outputDir: str):
    try:
        # Obtener el nombre base del archivo sin .gz
        base_name = os.path.basename(filename).replace(".gz", "")
        output_path = os.path.join(outputDir, base_name)
        print(f'ℹ️ Descomprimiendo "{filename}" en "{output_path}"')
        # Abrir .gz y escribir el archivo descomprimido
        with gzip.open(filename, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ Descomprimido en: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Función para mostrar un texto con colores dependiendo del valor de la condición. Verde si es True, rojo si es False.
def PrintAssert(message: str, boolExpression: bool):
    VERDE = "\033[92m"
    ROJO = "\033[91m"
    RESET = "\033[0m"
    if boolExpression:
        print(f"{VERDE}✅ {message}{RESET}")
    else:
        print(f"{ROJO}🚫 {message}{RESET}")


class Color(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

def PrintColor(message: str, color: Color) -> str:
    """Devuelve un texto coloreado con ANSI (sin imprimir)."""
    RESET = Color.RESET.value
    return f"{color.value}{message}{RESET}"

import sys

def _supports_utf8() -> bool:
    enc = (sys.stdout.encoding or '').lower()
    return 'utf-8' in enc

def ShowMessage(message: str, title: str, icon: str, color: Color):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print()
    colored_title = PrintColor(title.upper() + ":", color)
    print(f"{icon}  {colored_title} {message}")

def ShowInfoMessage(message: str, title: str = "Info"):
    ShowMessage(message, title, "[i]", Color.CYAN)

def ShowSuccessMessage(message: str, title: str = "Success"):
    ShowMessage(message, title, "[OK]", Color.GREEN)

def ShowErrorMessage(message: str, title: str = "Error"):
    ShowMessage(message, title, "[X]", Color.RED)

def ShowWarningMessage(message: str, title: str = "Warning"):
    ShowMessage(message, title, "[!]", Color.YELLOW)


pandas.set_option("display.max_rows", None)
# pandas.set_option("display.max_columns", None)

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=63643)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

print(f"Iniciando Host:{args.host}, Port:{args.port}...")


def LoadDataset():
    DOWNLOAD_DIR = "Temp"
    DATA_FILENAME = Path(f"{DOWNLOAD_DIR}/clinical_analytics.csv").resolve()
    print("-----")
    ShowInfoMessage("Cargando datos en DataFrame")
    print("-----2")
    df = pd.read_csv(DATA_FILENAME)
    print(
        f"Archivo CSV cargado: {DATA_FILENAME} ({df.shape[0]} filas x {df.shape[1]} columnas)"
    )
    return df


def CleanDataset(df: pandas.DataFrame):
    ShowInfoMessage("Limpiando y transformando datos")
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
    ShowTableInfo(df, "DataFrame Original")
    ShowTableHead(df, "DataFrame Original", 10)
    ShowTableShape(df, "DataFrame Original")
    ShowTableStats(df, "DataFrame Original")
    ShowUniqueCounts(df, "DataFrame Original")


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
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H2(id=id_, className="mb-0")
        ]), className="h-100"
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
        fig = px.pie(s, names=col, values="Count", title=titulo)
        fig.update_traces(textposition="inside")  # etiquetas dentro
        return fig

    # --------- Figuras ---------
    fig_admit = pie_count(df, "AdmitSource", "Atenciones por orígenes de admisión")
    fig_clinic = pie_count(df, "ClinicName", "Atenciones por Clinic Name")
    fig_dept = pie_count(df, "Department", "Atenciones por Department")

    return (
        #f"Promedio: {promedio:.2f} minutos | Máximo: {maximo} minutos | Mínimo: {minimo} minutos | Moda: {moda_txt} minutos | Registros: {total}",
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
