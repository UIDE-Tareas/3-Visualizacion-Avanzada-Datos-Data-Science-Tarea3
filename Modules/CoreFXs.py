from enum import Enum
import sys
import subprocess
import argparse
import os
from pathlib import Path


import pandas
import requests
import zipfile
import gzip
import shutil

# sys.stderr = sys.stdout
# sys.stdout.reconfigure(encoding='utf-8')

def RunCommand(
    commandList: list[str], printCommand: bool = True, printError: bool = True
) -> subprocess.CompletedProcess:
    print("⏳", " ".join(commandList))
    stdOutput = None if printCommand else subprocess.DEVNULL
    errorOutput = None if printError else subprocess.PIPE
    result = subprocess.run(
        commandList, stdout=stdOutput, stderr=errorOutput, text=True
    )
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

# Función para mostrar mensajes de información.
def ShowWarningMessage(message: str):
    print()
    print(f"⚠️ {message}".upper())


# Función para mostrar la información del DataFrame.
def ShowTableInfo(df: pandas.DataFrame, title):
    print(f"ℹ️ {title} ℹ️".upper())
    df.info()
    print()


# Función para mostrar las n primeras filas del DataFrame.
def ShowTableHead(df: pandas.DataFrame, title: str, headQty=10):
    print(f"ℹ️ {title.upper()}: Primeros {headQty} elementos.")
    print(df.head(headQty))
    print()


# Función para mostrar las n últimas filas del DataFrame.
def ShowTableTail(df: pandas.DataFrame, title: str, tailQty=10):
    print(f"ℹ️ {title.upper()}: Últimos {tailQty} elementos.")
    print(df.tail(tailQty))
    print()


# Mostrar el tamaño del DataFrame
def ShowTableShape(df: pandas.DataFrame, title: str):
    print(f"ℹ️ {title.upper()} - Tamaño de los datos")
    print(f"{df.shape[0]} filas x {df.shape[1]} columnas")
    print()


# Función para mostrar la estadística descriptiva de todas las columnas del DataFrame, por tipo de dato.
def ShowTableStats(df: pandas.DataFrame, title: str = ""):
    print(f"ℹ️ Estadística descriptiva - {title}".upper())
    numeric_types = ["int64", "float64", "Int64", "Float64"]
    numeric_cols = df.select_dtypes(include=numeric_types)
    if not numeric_cols.empty:
        print("    🔢 Columnas numéricas".upper())
        numeric_desc = (
            numeric_cols.describe().round(2).T
        )  # Transpuesta para añadir columna
        numeric_desc["var"] = numeric_cols.var().round(2)  # Añadir varianza
        print(numeric_desc.T)
    non_numeric_types = ["object", "string", "bool", "category"]
    non_numeric_cols = df.select_dtypes(include=non_numeric_types)
    if not non_numeric_cols.empty:
        print("    🔡 Columnas no numéricas".upper())
        non_numeric_desc = non_numeric_cols.describe()
        print(non_numeric_desc)
    datetime_cols = df.select_dtypes(include=["datetime"])
    if not datetime_cols.empty:
        print("    📅 Columnas fechas".upper())
        datetime_desc = datetime_cols.describe()
        print(datetime_desc)


# Función para mostrar los valores nulos o NaN de cada columna en un DataFrame
def ShowNanValues(df: pandas.DataFrame):
    print(f"ℹ️ Contador de valores Nulos".upper())
    nulls_count = df.isnull().sum()
    nulls_df = nulls_count.reset_index()
    nulls_df.columns = ["Columna", "Cantidad_Nulos"]
    print(nulls_df)
    print()


def ShowUniqueCounts(df: pandas.DataFrame, title: str):
    ShowInfoMessage(f"Valores únicos por columna", title)
    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ["Columna", "CantidadUnicos"]
    print(unique_counts)
    print()


# Función para descargar un archivo
def DownloadFile(uri: str, filename: str, overwrite: bool = False, timeout: int = 20):
    dest = Path(filename).resolve()
    if dest.exists() and dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        print(
            f'✅ Ya existe: "{dest}". No se descarga (use overwrite=True para forzar).'
        )
        return
    if dest.parent and not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'ℹ️ Descargando "{uri}" → "{dest}"')
    try:
        with requests.get(uri, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:  # filtra keep-alive chunks
                        f.write(chunk)
            tmp.replace(dest)
        print(f'✅ Archivo "{dest}" descargado exitosamente.')
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar: {e}")


# Función para descomprimir un archivo zip
def UnzipFile(filename: str, outputDir: str):
    print(f'ℹ️ Descomprimiendo "{filename}" en "{outputDir}"')
    try:
        with zipfile.ZipFile(filename, "r") as zip_ref:
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
    enc = (sys.stdout.encoding or "").lower()
    return "utf-8" in enc


def ShowMessage(message: str, title: str, icon: str, color: Color):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print()
    colored_title = PrintColor(icon + f"  " + title.upper() + ":", color)
    print(f"{colored_title} {message}")


def ShowInfoMessage(message: str, title: str = "Info"):
    ShowMessage(message, title, "ℹ️", Color.CYAN)


def ShowSuccessMessage(message: str, title: str = "Success"):
    ShowMessage(message, title, "✅", Color.GREEN)


def ShowErrorMessage(message: str, title: str = "Error"):
    ShowMessage(message, title, "❌", Color.RED)


def ShowWarningMessage(message: str, title: str = "Warning"):
    ShowMessage(message, title, "⚠️", Color.YELLOW)
