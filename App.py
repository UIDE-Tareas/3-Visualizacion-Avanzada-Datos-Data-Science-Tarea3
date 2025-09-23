import sys
import subprocess
import argparse
import os
from pathlib import Path
import Modules.CoreFXs as CoreFXs
from IPython.display import display

# AGREGAR LAS REFERENCIAS EXTERNAS AQUÍ EN ESTA LISTA
LIBS = (

    "plotly",
    "dash",
    "dash-bootstrap-components",
    "ipython",
    "customtkinter"
)

CoreFXs.ShowEnvironmentInfo()
CoreFXs.InstallDeps(LIBS)

import numpy as np
import pandas as pd
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

CoreFXs.ShowWarningMessage("🚀⚠️ En Windows ya viene instalado tkinter. Probar con `python -m tkinter`")
CoreFXs.ShowWarningMessage("🚀⚠️ En Ubuntu instalar tkinter con `sudo apt install python3-tk`. Probar con `python3 -m tkinter`")
CoreFXs.ShowWarningMessage("🚀⚠️ En MacOS ya viene instalado tkinter. Probar con `python3 -m tkinter`")
pandas.set_option("display.max_rows", None) 
pandas.set_option("display.max_columns", None)

import warnings
warnings.filterwarnings("ignore")

DOWNLOAD_DIR = "Temp"

DATA_FILE_URI = "https://github.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3/raw/refs/heads/main/Data/clinical_analytics.csv.gz"
DATA_FILENAME = f"{DOWNLOAD_DIR}/clinical_analytics.csv.gz"

CUSTOM_CSS_URI = "https://raw.githubusercontent.com/UIDE-Tareas/3-Visualizacion-Avanzada-Datos-Data-Science-Tarea3/refs/heads/main/Assets/custom.css"
CUSTOM_CSS_FILENAME = "assets/custom.css"

CoreFXs.DownloadFile(DATA_FILE_URI, DATA_FILENAME, overwrite=True)
CoreFXs.UngzipFile(DATA_FILENAME, "Temp")

CoreFXs.DownloadFile(CUSTOM_CSS_URI, CUSTOM_CSS_FILENAME, overwrite=True)


import os
import sys
import signal
import platform
import subprocess
import threading
import webbrowser
from pathlib import Path
import customtkinter as ctk


HOST = "localhost"
APPS = [
    {"label": "Dashboard 1", "path": "Scripts/Dash1App.py", "port": 63001},
    {"label": "Dashboard 2", "path": "Scripts/Dash2App.py", "port": 63002},
    {"label": "Dashboard 3", "path": "Scripts/Dash3App.py", "port": 63003},
]


def creation_kwargs():
    if platform.system() == "Windows":
        # CREATE_NEW_PROCESS_GROUP
        return {"creationflags": 0x00000200, "preexec_fn": None}
    else:
        # Nuevo grupo de proceso para poder matar todo el árbol
        return {"creationflags": 0, "preexec_fn": os.setsid}


class AppRow:
    def __init__(self, master, label, script_path, port):
        self.master = master
        self.label = label
        self.script_path = Path(script_path).resolve()
        self.port = port
        self.proc: subprocess.Popen | None = None

        frame = ctk.CTkFrame(master)
        frame.pack(fill="x", padx=10, pady=6)

        ctk.CTkLabel(
            frame, text=f"{label} • {self.script_path.name} • :{port}"
        ).grid(row=0, column=0, padx=6, pady=6, sticky="w")

        self.btn_run = ctk.CTkButton(frame, text="▶ Ejecutar", width=120, command=self.launch)

        self.btn_open = ctk.CTkButton(
            frame, text="🌐 Navegar", width=170, state="disabled", command=self.open_browser
        )
        self.btn_stop = ctk.CTkButton(
            frame, text="■ Detener", width=150, fg_color="indianred3", hover_color="indianred4",
            state="disabled", command=self.stop_clicked
        )

        self.btn_run.grid(row=0, column=1, padx=6)
        self.btn_open.grid(row=0, column=2, padx=6)
        self.btn_stop.grid(row=0, column=3, padx=6)

    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def launch(self):
        if self.proc and self.proc.poll() is None:
            display(f"[{self.label}] Ya está en marcha en el puerto {self.port}.")
            return

        if not self.script_path.exists():
            display(f"[{self.label}] No se encontró: {self.script_path}")
            return

        python_exe = sys.executable
        args = [python_exe, str(self.script_path), "--host", HOST, "--port", str(self.port)]

        try:
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **creation_kwargs()
            )
        except Exception as e:
            display(f"[{self.label}] Error al iniciar: {e}")
            return
        self.btn_run.configure(state="disabled")
        self.btn_open.configure(state="normal")
        self.btn_stop.configure(state="normal")

        display(f"[{self.label}] Iniciado en el puerto {self.port}.")
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        try:
            if self.proc and self.proc.stdout:
                for line in self.proc.stdout:
                    print(f"[{self.label}] {line}", end="")
        except Exception:
            pass
        finally:
            self.master.after(0, self._on_exit)

    def _on_exit(self):
        self.proc = None
        # UI: reactivar Run, desactivar Navegador y Detener
        self.btn_run.configure(state="normal")
        self.btn_open.configure(state="disabled")
        self.btn_stop.configure(state="disabled")

    def open_browser(self):
        webbrowser.open(self.url())

    def stop_clicked(self):
        self.stop_if_running()

    def stop_if_running(self):
        if self.proc and self.proc.poll() is None:
            try:
                if platform.system() == "Windows":
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                else:
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                    except Exception:
                        self.proc.terminate()
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            finally:
                self.proc = None
                self.btn_run.configure(state="normal")
                self.btn_open.configure(state="disabled")
                self.btn_stop.configure(state="disabled")


class Launcher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Visualización Avanzada de Datos en Data Science - Componente práctico 3 - DashBoards")
        self.geometry("820x280")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        ctk.CTkLabel(
            self, text="DashBoards - Clinical Analytics",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(12, 4))

        self.rows = []
        for cfg in APPS:
            self.rows.append(AppRow(self, cfg["label"], cfg["path"], cfg["port"]))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        for row in self.rows:
            row.stop_if_running()
        self.destroy()


if __name__ == "__main__":
    app = Launcher()
    app.mainloop()
    sys.exit(0)