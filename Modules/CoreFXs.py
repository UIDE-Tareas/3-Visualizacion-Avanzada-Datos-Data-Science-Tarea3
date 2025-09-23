import sys
import subprocess
import argparse
import os
from pathlib import Path

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
def ShowInfoMessage(message: str):
    display()
    display(f"ℹ️ {message}".upper())

# Función para mostrar la información del DataFrame.
def ShowTableInfo(df:pandas.DataFrame, title):
    display(f"ℹ️ {title} ℹ️".upper())
    df.info()
    display()

# Función para mostrar las n primeras filas del DataFrame.
def ShowTableHead(df:pandas.DataFrame, title:str, headQty=10):
    display(f"ℹ️ {title.upper()}: Primeros {headQty} elementos.")
    display(df.head(headQty))
    display()

# Función para mostrar las n últimas filas del DataFrame.
def ShowTableTail(df:pandas.DataFrame,title:str ,tailQty=10):
    display(f"ℹ️ {title.upper()}: Últimos {tailQty} elementos.")
    display(df.tail(tailQty))
    display()

# Mostrar el tamaño del DataFrame
def ShowTableShape(df:pandas.DataFrame, title:str):
    display(f"ℹ️ {title.upper()} - Tamaño de los datos")
    display(f"{df.shape[0]} filas x {df.shape[1]} columnas")
    display()

# Función para mostrar la estadística descriptiva de todas las columnas del DataFrame, por tipo de dato.
def ShowTableStats(df: pandas.DataFrame, title:str = ""):
    display(f"ℹ️ Estadística descriptiva - {title}".upper())
    numeric_types = ['int64', 'float64', 'Int64', 'Float64']
    numeric_cols = df.select_dtypes(include=numeric_types)
    if not numeric_cols.empty:
        display("    🔢 Columnas numéricas".upper())
        numeric_desc = numeric_cols.describe().round(2).T  # Transpuesta para añadir columna
        numeric_desc["var"] = numeric_cols.var().round(2)  # Añadir varianza
        display(numeric_desc.T) 
    non_numeric_types = ['object', 'string', 'bool', 'category']
    non_numeric_cols = df.select_dtypes(include=non_numeric_types)
    if not non_numeric_cols.empty:
        display("    🔡 Columnas no numéricas".upper())
        non_numeric_desc = non_numeric_cols.describe()
        display(non_numeric_desc)
    datetime_cols = df.select_dtypes(include=['datetime'])
    if not datetime_cols.empty:
        display("    📅 Columnas fechas".upper())
        datetime_desc = datetime_cols.describe()
        display(datetime_desc)

# Función para mostrar los valores nulos o NaN de cada columna en un DataFrame
def ShowNanValues(df: pandas.DataFrame):
    display(f"ℹ️ Contador de valores Nulos".upper())
    nulls_count = df.isnull().sum()
    nulls_df = nulls_count.reset_index()
    nulls_df.columns = ['Columna', 'Cantidad_Nulos']
    display(nulls_df)
    display()


# Función para descargar un archivo
def DownloadFile(uri: str, filename: str, overwrite: bool = False, timeout: int = 20):
    dest = Path(filename).resolve()
    if dest.exists() and dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        display(f"✅ Ya existe: \"{dest}\". No se descarga (use overwrite=True para forzar).")
        return
    if dest.parent and not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
    display(f"ℹ️ Descargando \"{uri}\" → \"{dest}\"")
    try:
        with requests.get(uri, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:  # filtra keep-alive chunks
                        f.write(chunk)
            tmp.replace(dest)
        display(f"✅ Archivo \"{dest}\" descargado exitosamente.")
    except requests.exceptions.RequestException as e:
        display(f"❌ Error al descargar: {e}")

# Función para descomprimir un archivo zip
def UnzipFile(filename: str, outputDir: str):
    display(f'ℹ️ Descomprimiendo "{filename}" en "{outputDir}"')
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(outputDir)
        display(f"Descomprimido en: {os.path.abspath(outputDir)}")
    except Exception as e:
        display(f"Error: {e}")

def UngzipFile(filename: str, outputDir: str):
    try:
        # Obtener el nombre base del archivo sin .gz
        base_name = os.path.basename(filename).replace(".gz", "")
        output_path = os.path.join(outputDir, base_name)
        display(f'ℹ️ Descomprimiendo "{filename}" en "{output_path}"')
        # Abrir .gz y escribir el archivo descomprimido
        with gzip.open(filename, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        display(f"✅ Descomprimido en: {os.path.abspath(output_path)}")
    except Exception as e:
        display(f"❌ Error: {e}")

# Función para mostrar un texto con colores dependiendo del valor de la condición. Verde si es True, rojo si es False.
def PrintAssert(message: str, boolExpression: bool):
    VERDE = "\033[92m"
    ROJO = "\033[91m"
    RESET = "\033[0m"
    if boolExpression:
        print(f"{VERDE}✅ {message}{RESET}")
    else:
        print(f"{ROJO}🚫 {message}{RESET}")