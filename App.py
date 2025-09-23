import sys
import subprocess
import argparse
import os
from pathlib import Path
import Modules.CoreFXs as CoreFXs

LIBS = [

    "plotly",
    "dash",
    "dash-bootstrap-components",
    "ipython",
]



CoreFXs.ShowEnvironmentInfo()
CoreFXs.InstallDeps()

import numpy as np
import pandas as pd
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from IPython.display import display

pandas.set_option("display.max_rows", None) 
pandas.set_option("display.max_columns", None)

import warnings
warnings.filterwarnings("ignore")

DOWNLOAD_DIR = "Temp"

DATA_FILE_URI = "https://github.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3/raw/refs/heads/main/Data/clinical_analytics.csv.gz"
DATA_FILENAME = f"{DOWNLOAD_DIR}/clinical_analytics.csv.gz"

CUSTOM_CSS_URI = "https://raw.githubusercontent.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3/refs/heads/main/Assets/custom.css"
CUSTOM_CSS_FILENAME = "Assets/custom.css"

CoreFXs.DownloadFile(DATA_FILE_URI, DATA_FILENAME)
CoreFXs.UngzipFile(DATA_FILENAME, "Temp")
CoreFXs.DownloadFile(CUSTOM_CSS_URI, CUSTOM_CSS_FILENAME)


sys.exit(0) 