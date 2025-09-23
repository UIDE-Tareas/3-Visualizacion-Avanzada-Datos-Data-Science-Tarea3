import argparse
from dash import Dash, html
print("Iniciando Dash 1...")
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=63643)
parser.add_argument("--host", default="127.0.0.1")
args = parser.parse_args()

app = Dash(__name__)
app.title = "Dashboard 3 - Trabajo Final de Vizualización de Avanzada de Datos — Clinical Analytics"
app.layout = html.Div([
    html.H1("Servidor dash de prueba"),
    html.P("Servidor Dash de prueba en marcha.")
])

if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False)