import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from forecast_model import load_and_prepare_data, remove_outliers, prepare_features, train_and_select_model, forecast_steps
import os
import logging
import numpy as np

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dash-App initialisieren mit dunklem Thema
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Umweltmonitoring Daashboard"

# Dateipfade
DATA_FILE = "data.csv"
LOCATION_FILE = "locations.csv"
MODEL_DIR = "models"

# Daten laden
def load_data():
    try:
        df = load_and_prepare_data(DATA_FILE)
        logging.info(f"Daten geladen von {DATA_FILE}, Zeilen: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Fehler beim Laden von data.csv: {e}")
        return pd.DataFrame(columns=['createdAt', 'Temperature', 'PM2.5', 'Humidity', 'PM10', 'Pressure'])

def load_locations():
    try:
        df_loc = pd.read_csv(LOCATION_FILE)
        df_loc['timestamp'] = pd.to_datetime(df_loc['timestamp'], utc=True, errors='coerce')
        df_loc = df_loc.dropna(subset=['timestamp'])
        logging.info(f"Standorte geladen von {LOCATION_FILE}, Zeilen: {len(df_loc)}")
        return df_loc
    except Exception as e:
        logging.error(f"Fehler beim Laden von locations.csv: {e}")
        return pd.DataFrame(columns=['timestamp', 'lat', 'lon'])

# Initiale Daten laden
df_all = load_data()
df_locations = load_locations()

# Sensoren mit sicheren IDs
SENSORS = {
    'Temperature': 'Temperature',
    'PM2.5': 'PM2_5',
    'Humidity': 'Humidity',
    'PM10': 'PM10',
    'Pressure': 'Pressure'
}

# Dashboard-Layout
app.layout = html.Div(className='container mx-auto p-4 bg-gray-900 text-white', children=[
    # Titel
    html.H1("📊 Umweltmonitoring Dashboard", className='text-4xl font-bold mb-6 text-center text-blue-400'),
    
    # Kontrollpanel
    html.Div(className='flex flex-wrap -mx-2 mb-6', children=[
        html.Div(className='w-full md:w-1/3 px-2 mb-4', children=[
            html.Label("Sensor auswählen", className='text-lg font-semibold mb-2'),
            dcc.Dropdown(
                id="sensor-dropdown",
                options=[{"label": name, "value": name} for name in SENSORS.keys()],
                value="Temperature",
                className='mb-4 bg-gray-800 text-black border-gray-700 rounded'
            ),
            html.Label("Vorhersagehorizont", className='text-white font-semibold mb-2'),
            dcc.Dropdown(
                id="forecast-horizon",
                options=[
                    {"label": "1 Stunde", "value": 6},  # 6 * 10min
                    {"label": "12 Stunden", "value": 72},
                    {"label": "1 Tag", "value": 144},
                    {"label": "3 Tage", "value": 432},
                    {"label": "7 Tage", "value": 1008},
                    {"label": "21 Tage", "value": 30024}
                ],
                value=144,
                className='mb-4 bg-gray-800 text-black border-gray-700 rounded'
            ),
            html.Button("Modell trainieren", id="train-btn", className='bg-blue-100 hover:bg-blue-700 text-black font-bold py-2 px-4 rounded mb-2 w-full'),
            html.Button("Vorhersage erstellen", id="forecast-btn", className='bg-green-100 hover:bg-green-700 text-black font-bold py-2 px-4 rounded mb-2 w-full'),
            html.Button("Daten herunterladen", id="download-btn", className='bg-gray-100 hover:bg-gray-700 text-black font-bold py-2 px-4 rounded mb-2 w-full'),
            dcc.Download(id="download-data"),
            html.Div(id="status-message", className='mt-4 text-lg text-blue-300')
        ]),
        # Neueste Werte
        html.Div(className='w-full md:w-2/3 px-2 mb-4', children=[
            html.H3("Neueste Sensorwerte", className='text-xl font-semibold mb-2'),
            dash_table.DataTable(
                id='latest-values-table',
                columns=[
                    {"name": "Sensor", "id": "sensor"},
                    {"name": "Wert", "id": "value"},
                    {"name": "Zeitstempel", "id": "timestamp"}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'backgroundColor': '#1F2937',
                    'color': 'white',
                    'padding': '10px'
                },
                style_header={
                    'backgroundColor': '#374151',
                    'fontWeight': 'bold'
                }
            )
        ])
    ]),
    
    # Zeitreihen-Diagramme
    html.H2("Zeitreihen-Diagramme", className='text-2xl font-semibold mb-2 mt-4'),
    html.Div([
        dcc.Graph(id=f"{SENSORS[name]}-graph", style={'height': '400px'}) for name in SENSORS
    ]),
    
    # Karte
    html.H2("📍 Sensorstandorte", className='text-2xl font-semibold mb-2 mt-4'),
    dcc.Graph(id="map-graph", style={'height': '400px'}),
    
    # Speicher für Vorhersagen
    dcc.Store(id='forecast-store'),
    
    # Auto-Refresh alle 5 Minuten
    dcc.Interval(id='interval-component', interval=5*60*1000, n_intervals=0)
])

# Callback zum Aktualisieren der neuesten Werte
@app.callback(
    Output("latest-values-table", "data"),
    Input("interval-component", "n_intervals")
)
def update_latest_values(n_intervals):
    try:
        global df_all
        df_all = load_data()  # Daten aktualisieren
        latest = df_all.iloc[-1]
        latest_timestamp = df_all.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        table_data = [
            {
                "sensor": name,
                "value": f"{latest.get(name, float('nan')):.2f}" if pd.notna(latest.get(name)) else "N/A",
                "timestamp": latest_timestamp
            }
            for name in SENSORS
        ]
        return table_data
    except Exception as e:
        logging.error(f"Fehler beim Aktualisieren der neuesten Werte: {str(e)}")
        return [{"sensor": name, "value": "N/A", "timestamp": "N/A"} for name in SENSORS]

# Callback zum Aktualisieren der Diagramme und Vorhersagen
@app.callback(
    [
        *[Output(f"{SENSORS[name]}-graph", "figure") for name in SENSORS],
        Output("status-message", "children"),
        Output("forecast-store", "data")
    ],
    [
        Input("train-btn", "n_clicks"),
        Input("forecast-btn", "n_clicks"),
        Input("interval-component", "n_intervals")
    ],
    [
        State("sensor-dropdown", "value"),
        State("forecast-horizon", "value"),
        State("forecast-store", "data")
    ]
)
def update_dashboard(train_clicks, forecast_clicks, n_intervals, sensor, horizon, stored_forecast):
    global df_all
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Initialisieren der Ausgaben
    figures = {name: go.Figure() for name in SENSORS}
    status_message = ""
    forecast_data = stored_forecast or {}

    try:
        # Daten aktualisieren bei Intervall
        if triggered_id == "interval-component":
            df_all = load_data()
            status_message = "✅ Daten aktualisiert."

        # Prüfen, ob Daten verfügbar sind
        if df_all.empty:
            for name in figures:
                figures[name].update_layout(title=f"Keine Daten für {name}", template='plotly_dark')
            return list(figures.values()) + ["❌ Keine Daten geladen.", {}]

        # Zeitreihen-Diagramme aktualisieren
        for name in figures:
            figures[name].add_trace(go.Scatter(
                x=df_all.index,
                y=df_all[name],
                mode='lines',
                name='Historisch',
                line=dict(color='#3B82F6')
            ))
            figures[name].update_layout(
                title=f"{name} Zeitreihe",
                xaxis_title="Zeit",
                yaxis_title=name,
                template='plotly_dark',
                height=400
            )

        # Button-Aktionen verarbeiten
        if triggered_id == "train-btn" and train_clicks:
            model_path = f"{MODEL_DIR}/{sensor}_model.pkl"
            if os.path.exists(model_path):
                status_message = f"✅ Modell für {sensor} existiert bereits."
            else:
                df_clean = remove_outliers(df_all, sensor)
                df_feat = prepare_features(df_clean, sensor)
                train_and_select_model(df_feat, sensor)
                status_message = f"✅ Modell für {sensor} trainiert."
        elif triggered_id == "forecast-btn" and forecast_clicks:
            forecast_df = forecast_steps(df_all, sensor, steps=horizon)
            forecast_data[sensor] = forecast_df.to_dict('records')
            figures[sensor].add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df[f'forecast_{sensor}'],
                mode='lines',
                name='Vorhersage',
                line=dict(color='#10B981', dash='dash')
            ))
            std = np.std(forecast_df[f'forecast_{sensor}'])
            figures[sensor].add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df[f'forecast_{sensor}'] + std,
                mode='lines',
                line=dict(color='#10B981', width=0),
                showlegend=False
            ))
            figures[sensor].add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df[f'forecast_{sensor}'] - std,
                mode='lines',
                line=dict(color='#10B981', width=0),
                fill='tonexty',
                fillcolor='rgba(16, 185, 129, 0.1)',
                showlegend=False
            ))
            status_message = f"✅ Vorhersage für {sensor} erstellt (Std: {std:.2f})."

        # Gespeicherte Vorhersagen hinzufügen
        for name in figures:
            if name in forecast_data:
                forecast_df = pd.DataFrame(forecast_data[name])
                figures[name].add_trace(go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df[f'forecast_{name}'],
                    mode='lines',
                    name='Vorhersage',
                    line=dict(color='#10B981', dash='dash')
                ))
                std = np.std(forecast_df[f'forecast_{name}'])
                figures[name].add_trace(go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df[f'forecast_{name}'] + std,
                    mode='lines',
                    line=dict(color='#10B981', width=0),
                    showlegend=False
                ))
                figures[name].add_trace(go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df[f'forecast_{name}'] - std,
                    mode='lines',
                    line=dict(color='#10B981', width=0),
                    fill='tonexty',
                    fillcolor='rgba(16, 185, 129, 0.1)',
                    showlegend=False
                ))

    except Exception as e:
        logging.error(f"Dashboard-Aktualisierungsfehler: {str(e)}")
        status_message = f"❌ Fehler: {str(e)}"

    return list(figures.values()) + [status_message, forecast_data]

# Callback zum Aktualisieren der Karte
@app.callback(
    Output("map-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_map(n_intervals):
    try:
        df_loc = load_locations()
        if df_loc.empty:
            logging.warning("Keine Standortdaten verfügbar")
            return go.Figure().update_layout(title="Keine Standortdaten", template='plotly_dark')
        fig = px.scatter_mapbox(
            df_loc,
            lat="lat",
            lon="lon",
            hover_data={"timestamp": True},
            zoom=9,
            height=400
        )
        fig.update_layout(mapbox_style="open-street-map", template='plotly_dark')
        logging.info("Karte erfolgreich aktualisiert")
        return fig
    except Exception as e:
        logging.error(f"Fehler beim Aktualisieren der Karte: {str(e)}")
        return go.Figure().update_layout(title=f"Fehler: {str(e)}", template='plotly_dark')

# Callback für Daten-Download
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("forecast-store", "data"),
    State("sensor-dropdown", "value")
)
def download_data(n_clicks, forecast_data, sensor):
    if n_clicks is None:
        return None
    try:
        df = df_all.copy()
        combined = df[[sensor]].copy()
        combined['type'] = 'historical'
        if forecast_data and sensor in forecast_data:
            forecast_df = pd.DataFrame(forecast_data[sensor])
            forecast_df = forecast_df.rename(columns={f'forecast_{sensor}': sensor, 'timestamp': 'createdAt'})
            forecast_df['type'] = 'forecast'
            combined = pd.concat([combined, forecast_df[['createdAt', sensor, 'type']]])
        combined = combined.reset_index()
        csv_string = combined.to_csv(index=False)
        logging.info(f"Daten für {sensor} heruntergeladen")
        return dict(content=csv_string, filename=f"{sensor}_data.csv")
    except Exception as e:
        logging.error(f"Fehler beim Herunterladen der Daten: {str(e)}")
        return dict(content=str(e), filename="error.csv")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)