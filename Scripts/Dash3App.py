import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Modules import CoreFXs as CoreFXs
from dash import Dash, html
import pandas as pd 
import pandas
from pathlib import Path
from IPython.display import display


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=63643)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

print(f"Iniciando Host:{args.host}, Port:{args.port}...")

DOWNLOAD_DIR = "Temp"
DATA_FILENAME = Path(f"{DOWNLOAD_DIR}/clinical_analytics.csv").resolve()

CoreFXs.ShowInfoMessage("Cargando datos en DataFrame")
dfOriginal = pd.read_csv(DATA_FILENAME)
print(f"Archivo CSV cargado: {DATA_FILENAME} ({dfOriginal.shape[0]} filas x {dfOriginal.shape[1]} columnas)")

data = dfOriginal.copy()

# Eliminar los espacios y poner en mayúscula la primera letra en los nombres de las columnas
data.columns = [col.strip().title().replace(" ", "").replace("_", "").replace("-","") for col in data.columns] 
CoreFXs.ShowTableInfo(data, "DataFrame Original")

# Convertir los tipos de datos de las columnas según el diccionario de datos
data.AdmitSource = data.AdmitSource.astype(pandas.StringDtype()).str.strip()
data.AdmitType = data.AdmitType.astype(pandas.StringDtype()).str.strip()
data.ApptStartTime = pandas.to_datetime(data.ApptStartTime, format="%Y-%m-%d %I:%M:%S %p", errors="raise") 
data.CareScore = data.CareScore.astype(pandas.Int64Dtype())
data.CheckInTime =  pandas.to_datetime(data.CheckInTime, format="%Y-%m-%d %I:%M:%S %p", errors="raise") 
data.ClinicName = data.ClinicName.astype(pandas.StringDtype()).str.strip()
data.Department = data.Department.astype(pandas.StringDtype()).str.strip()
data.DiagnosisPrimary = data.DiagnosisPrimary.astype
data.DischargeDatetimeNew = pandas.to_datetime(data.DischargeDatetimeNew, format="%Y-%m-%d", errors="raise")
data.EncounterNumber = data.EncounterNumber.astype(pandas.StringDtype()).str.strip()
data.EncounterStatus = data.EncounterStatus.astype(pandas.StringDtype()).str.strip()
data.NumberOfRecords = data.NumberOfRecords.astype
data.WaitTimeMin = data.WaitTimeMin.astype(pandas.Int64Dtype())

CoreFXs.ShowUniqueCounts(data)


display(data.columns)
sys.exit(0)
data.Id = data.Id.astype(pandas.Int64Dtype())
data.CaseNumber = data.CaseNumber.astype(pandas.StringDtype()).str.strip()
data.Date = pandas.to_datetime(data.Date.astype(pandas.StringDtype()), format=f"%m/%d/%y %H:%M", errors="raise")
data.Block = data.Block.astype(pandas.StringDtype()).str.strip()
data.Iucr = data.Iucr.astype(pandas.StringDtype()).str.strip()
data.PrimaryType = data.PrimaryType.astype(pandas.StringDtype()).str.strip()
data.Description = data.Description.astype(pandas.StringDtype()).str.strip()
data.LocationDescription = data.LocationDescription.astype(pandas.StringDtype()).str.strip()
data.Arrest = data.Arrest.astype(bool)
data.Domestic = data.Domestic.astype(bool)
data.BeatNum = data.BeatNum.astype(pandas.StringDtype()).str.zfill(4)
data.District = data.District.astype(pandas.StringDtype())
data.Ward = data.Ward.astype(pandas.Int64Dtype())
data.CommunityArea = data.CommunityArea.astype(pandas.StringDtype())
data.FbiCode = data.FbiCode.astype(pandas.StringDtype()).str.strip()
data.XCoordinate = data.XCoordinate.astype(pandas.Float64Dtype())
data.YCoordinate = data.YCoordinate.astype(pandas.Float64Dtype())
data.Year = data.Year.astype(pandas.Int64Dtype())
data.UpdatedOn = pandas.to_datetime(data.UpdatedOn.astype(pandas.StringDtype()), format=f"%m/%d/%y %H:%M", errors="raise")
data.Latitude = data.Latitude.astype(pandas.Float64Dtype())
data.Longitude = data.Longitude.astype(pandas.Float64Dtype())
data.Location = data.Location.astype(pandas.StringDtype()).str.strip()

# Mostrar información del DataFrame
CoreFXs.ShowTableInfo(data, "DataFrame Original")
CoreFXs.ShowTableHead(data, "DataFrame Original", 10)
CoreFXs.ShowTableShape(data, "DataFrame Original")
CoreFXs.ShowTableStats(data, "DataFrame Original")

CoreFXs.ShowTableInfo(dfOriginal, "DataFrame Original")
sys.exit(0)




















app = Dash(__name__)
app.title = "Dashboard 3 - Trabajo Final de Vizualización de Avanzada de Datos — Clinical Analytics"
app.layout = html.Div([
    html.H1("Servidor dash de prueba"),
    html.P("Servidor Dash de prueba en marcha.")
])

if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False)