import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta, timezone
import logging
import os
import requests
import numpy as np
import joblib # Benötigt für das Speichern/Laden von Modellen

# Importiere db_manager und Funktionen aus forecast_model
import db_manager # db_manager.py muss im PYTHONPATH des Containers sein
# WICHTIG: Die Funktion in forecast_model.py muss load_and_prepare_data heißen und einen DataFrame akzeptieren.
from forecast_model import load_and_prepare_data, remove_outliers, prepare_features, train_and_select_model, forecast_steps

# --- Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SENSEBOX_ID = os.getenv("SENSEBOX_ID", "60d828c48855dd001cf91983") # SenseBox ID aus Umgebungsvariable
MODEL_DIR = "models" # Verzeichnis innerhalb des Containers, wo Modelle gespeichert werden

# --- Hilfsfunktionen ---
def get_sensor_location_and_name(sensebox_id):
    """
    Ruft den Standort und Namen der SenseBox von der openSenseMap API ab.
    Verwendet Fallback-Koordinaten und -Namen, wenn der API-Aufruf fehlschlägt oder Daten unvollständig sind.
    """
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Löst HTTPError für fehlerhafte Antworten aus (4xx oder 5xx)

        metadata = response.json()

        if isinstance(metadata, list) and metadata:
            metadata = metadata[0] # Nehmen Sie das erste Element, wenn es eine Liste ist
        elif not isinstance(metadata, dict):
            raise ValueError(f"Unerwartetes API-Antwortformat: {type(metadata)}")

        loc_data = metadata.get("loc", {})
        if isinstance(loc_data, list) and loc_data:
            loc_data = loc_data[0]
        elif not isinstance(loc_data, dict):
            loc_data = {}

        coordinates = loc_data.get("coordinates", [])

        if len(coordinates) < 2:
            raise ValueError("Koordinaten im API-Antwort für 'loc' nicht gefunden oder unvollständig.")

        lat = coordinates[1]
        lon = coordinates[0]
        name = metadata.get("name", "SenseBox")
        logging.info(f"Sensorstandort geladen: Lat {lat}, Lon {lon}, Name: {name}")
        return lat, lon, name
    except Exception as e:
        logging.error(f"Fehler beim Abrufen des Sensorstandorts: {e}. Verwende Standardstandort.")
        return 57.280353, 27.069191, "Standard SenseBox" # Fallback-Koordinaten

# Standort und Namen einmal beim Start abrufen
SENSEBOX_LAT, SENSEBOX_LON, SENSEBOX_NAME = get_sensor_location_and_name(SENSEBOX_ID)

# Globale Variable für alle Sensordaten (wird von DB geladen)
# Initialisiere als leeren DataFrame
df_all_sensors_data = pd.DataFrame() 

# Definiere die erwarteten Sensoren basierend auf der DB-Struktur (sensor_id)
# Diese Namen müssen den 'sensor_id'-Werten in Ihrer Datenbank entsprechen.
SENSORS = {
    'Temperature': 'Temperature',
    'PM2.5': 'PM2.5',
    'Humidity': 'Humidity',
    'PM10': 'PM10',
    'Pressure': 'Pressure'
}

# --- Dash App Setup ---
# Initialisiere Dash-App mit einem dunklen Thema von Dash Bootstrap Components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY]) # Verwendung des DARKLY Themas
app.title = "Umweltmonitoring Dashboard"
server = app.server # Exponiere Flask-Server für Produktionsbereitstellung

# --- App Layout ---
app.layout = html.Div(className='container mx-auto p-4 bg-gray-900 text-white', children=[
    # Titel des Dashboards
    html.H1(f"📊 Umweltmonitoring Dashboard ({SENSEBOX_NAME})", className='text-4xl font-bold mb-6 text-center text-blue-400'),
    
    # Auto-Refresh Interval Komponente (alle 15 Minuten)
    dcc.Interval(id='interval-component', interval=15*60*1000, n_intervals=0),
    
    # Kontrollpanel und Neueste Werte
    html.Div(className='flex flex-wrap -mx-2 mb-6', children=[
        # Kontrollpanel für Sensor und Vorhersage
        html.Div(className='w-full md:w-1/3 px-2 mb-4', children=[
            html.Label("Sensor auswählen", className='text-lg font-semibold mb-2'),
            dcc.Dropdown(
                id="sensor-dropdown",
                options=[{"label": name, "value": col_name} for name, col_name in SENSORS.items()],
                value="Temperature", # Standardauswahl
                className='mb-4 bg-gray-800 text-white border-gray-700 rounded-lg shadow-lg' # Abgerundete Ecken, Schatten
            ),
            html.Label("Vorhersagehorizont (10-Minuten-Intervalle)", className='text-lg font-semibold mb-2'),
            dcc.Dropdown(
                id="forecast-horizon",
                options=[
                    {"label": "1 Stunde (6 Intervalle)", "value": 6},
                    {"label": "12 Stunden (72 Intervalle)", "value": 72},
                    {"label": "1 Tag (144 Intervalle)", "value": 144},
                    {"label": "3 Tage (432 Intervalle)", "value": 432},
                    {"label": "7 Tage (1008 Intervalle)", "value": 1008}
                ],
                value=144, # Standardauswahl
                className='mb-4 bg-gray-800 text-white border-gray-700 rounded-lg shadow-lg' # Abgerundete Ecken, Schatten
            ),
            html.Button("Modell trainieren", id="train-btn", 
                        className='bg-gradient-to-r from-blue-600 to-blue-800 hover:from-blue-700 hover:to-blue-900 text-white font-bold py-2 px-4 rounded-lg shadow-lg transform transition duration-300 hover:scale-105 mb-2 w-full border border-blue-900'), # Verbesserter Button-Stil
            html.Button("Vorhersage erstellen", id="forecast-btn", 
                        className='bg-gradient-to-r from-green-600 to-green-800 hover:from-green-700 hover:to-green-900 text-white font-bold py-2 px-4 rounded-lg shadow-lg transform transition duration-300 hover:scale-105 mb-2 w-full border border-green-900'), # Verbesserter Button-Stil
            html.Button("Daten herunterladen", id="download-btn", 
                        className='bg-gradient-to-r from-gray-600 to-gray-800 hover:from-gray-700 hover:to-gray-900 text-white font-bold py-2 px-4 rounded-lg shadow-lg transform transition duration-300 hover:scale-105 mb-2 w-full border border-gray-900'), # Verbesserter Button-Stil
            dcc.Download(id="download-data"),
            html.Div(id="status-message", className='mt-4 text-lg text-blue-300') # Statusmeldungen für Training/Vorhersage
        ]),
        # Neueste Werte Tabelle
        html.Div(className='w-full md:w-2/3 px-2 mb-4', children=[
            html.H3("Neueste Sensorwerte", className='text-xl font-semibold mb-2 text-blue-300'),
            dash_table.DataTable(
                id='latest-values-table',
                columns=[
                    {"name": "Sensor", "id": "sensor"},
                    {"name": "Wert", "id": "value"},
                    {"name": "Zeitstempel", "id": "timestamp"}
                ],
                style_table={'overflowX': 'auto', 'backgroundColor': '#2D3748', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.4)'}, # Dunklerer Hintergrund, abgerundete Ecken, Schatten
                style_cell={
                    'textAlign': 'left',
                    'backgroundColor': '#2D3748', # Dunklerer Hintergrund für Zellen
                    'color': '#E2E8F0', # Hellerer Text für besseren Kontrast
                    'padding': '12px',
                    'borderBottom': '1px solid #4A5568' # Dezenterer Rahmen
                },
                style_header={
                    'backgroundColor': '#4A5568', # Noch dunklerer Hintergrund für Header
                    'fontWeight': 'bold',
                    'color': '#E2E8F0', # Hellerer Text
                    'borderBottom': '1px solid #4A5568', # Dezenterer Rahmen
                    'borderTopLeftRadius': '10px', # Abgerundete Ecken nur oben
                    'borderTopRightRadius': '10px'
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#1A202C'}, # Wechselnde Reihenfarben
                    {'if': {'column_id': 'sensor'}, 'fontWeight': 'bold'} # Sensorname fett
                ]
            )
        ])
    ]),
    
    # Zeitreihen-Diagramme für alle Sensoren
    html.H2("Zeitreihen-Diagramme", className='text-2xl font-semibold mb-2 mt-4 text-center text-blue-400'),
    html.Div(
        id='all-sensors-graphs-container', # Container für dynamisch generierte Graphen
        className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4', # Responsives Raster
        children=[
            # Platzhalter für Graphen, sie werden per Callback aktualisiert
            dcc.Graph(id=f"graph-{col_name}", style={'height': '400px', 'borderRadius': '10px', 'overflow': 'hidden', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.4)'}) for col_name in SENSORS.values() # Abgerundete Ecken, Schatten
        ]
    ),
    
    # Karte der Sensorstandorte
    html.H2("📍 Sensorstandort", className='text-2xl font-semibold mb-2 mt-4 text-center text-blue-400'),
    dcc.Graph(id="map-graph", style={'height': '400px', 'borderRadius': '10px', 'overflow': 'hidden', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.4)'}), # Abgerundete Ecken, Schatten
    
    # Speicher für Vorhersagedaten und Modellinformationen
    dcc.Store(id='forecast-store'),
    dcc.Store(id='model-store') # Speichert Informationen über das trainierte Modell
])

# --- Callbacks ---

# Callback zum Aktualisieren aller Dashboard-Elemente im Intervall
@app.callback(
    [Output("latest-values-table", "data"),
     Output("all-sensors-graphs-container", "children"),
     Output("map-graph", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard_content(n_intervals):
    global df_all_sensors_data
    logging.info(f"Aktualisiere Dashboard-Inhalte (Intervall: {n_intervals})")
    
    # Lade Daten aus der Datenbank über db_manager
    df_all_sensors_data = db_manager.load_all_sensor_data_from_db(SENSEBOX_ID)

    # Initialisiere leere Daten für den Fall, dass df_all_sensors_data leer ist
    table_data = []
    all_graphs = []
    # Verbesserte Standardkarte
    map_figure = go.Figure(go.Scattermapbox(lat=[SENSEBOX_LAT], lon=[SENSEBOX_LON], mode='markers',
                                           marker=go.scattermapbox.Marker(size=14, color='red'),
                                           text=[SENSEBOX_NAME], hoverinfo='text')).update_layout(
        mapbox_style="carto-darkmatter", # Dunkler Kartenstil
        title_text="Keine Standortdaten verfügbar",
        template='plotly_dark',
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )


    if df_all_sensors_data.empty:
        logging.warning("Keine Daten aus der Datenbank geladen oder Daten sind leer nach der Bereinigung. Es werden keine Graphen angezeigt.")
        empty_figure = px.line(title="Keine Daten verfügbar", template='plotly_dark')
        empty_figure.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        # Leere Daten für Tabelle und Graphen zurückgeben
        for col_name in SENSORS.values():
            all_graphs.append(dcc.Graph(figure=empty_figure, id=f"graph-{col_name}"))
        return table_data, all_graphs, map_figure

    # --- Neueste Werte Tabelle aktualisieren ---
    latest_timestamp_str = df_all_sensors_data.index.max().strftime('%Y-%m-%d %H:%M:%S UTC')
    latest_values = df_all_sensors_data.iloc[-1]
    
    table_data = []
    for display_name, col_name in SENSORS.items():
        if col_name in latest_values:
            value = latest_values[col_name]
            table_data.append({
                "sensor": display_name,
                "value": f"{value:.2f}" if pd.notna(value) else "N/A",
                "timestamp": latest_timestamp_str
            })
        else:
            table_data.append({
                "sensor": display_name,
                "value": "Nicht verfügbar",
                "timestamp": latest_timestamp_str
            })

    # --- Alle Sensorgraphen generieren ---
    # Farbpalette für die Graphen, um visuelle Unterscheidung zu verbessern
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Bold # Mehr Farben hinzufügen
    color_index = 0

    for display_name, col_name in SENSORS.items():
        if col_name in df_all_sensors_data.columns:
            # Verwendung einer Farbe aus der Palette
            line_color = colors[color_index % len(colors)]
            color_index += 1

            fig = px.line(df_all_sensors_data.reset_index(), x="createdAt", y=col_name,
                          title=f"{display_name} Zeitreihe", template='plotly_dark',
                          line_shape='spline', # Glattere Linien
                          color_discrete_sequence=[line_color])
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode="x unified", # Verbesserte Hover-Informationen
                xaxis=dict(gridcolor='#4A5568'), # Dezente Gitterlinien
                yaxis=dict(gridcolor='#4A5568'),
                # Legendenposition für bessere Sichtbarkeit
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.7)', bordercolor='#4A5568', borderwidth=1, font=dict(color='white'))
            )
            all_graphs.append(dcc.Graph(figure=fig, id=f"graph-{col_name}"))
        else:
            logging.warning(f"Spalte '{col_name}' nicht in den geladenen Daten gefunden.")
            empty_figure = px.line(title=f"{display_name} (Daten fehlen)", template='plotly_dark')
            empty_figure.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            all_graphs.append(dcc.Graph(figure=empty_figure, id=f"graph-{col_name}"))


    # --- Karte der Sensorstandorte aktualisieren ---
    map_figure = px.scatter_map(
        pd.DataFrame([{'lat': SENSEBOX_LAT, 'lon': SENSEBOX_LON, 'name': SENSEBOX_NAME}]),
        lat="lat",
        lon="lon",
        zoom=10,
        title="Sensorstandort",
        hover_name="name",
        color_discrete_sequence=['#FF6347'] # Markerfarbe für den Standort
    )
    map_figure.update_layout(
        mapbox_style="carto-darkmatter", # Dunkler Kartenstil
        template='plotly_dark',
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    
    return table_data, all_graphs, map_figure

# Callback zum Trainieren des Modells
@app.callback(
    Output("status-message", "children"),
    Output("model-store", "data"),
    Input("train-btn", "n_clicks"),
    State("sensor-dropdown", "value"),
    prevent_initial_call=True
)
def train_model(n_clicks, sensor_col_name):
    if n_clicks is None:
        return dash.no_update, dash.no_update
    
    if df_all_sensors_data.empty or sensor_col_name not in df_all_sensors_data.columns:
        return "⚠️ Bitte laden Sie zuerst Daten und wählen Sie einen gültigen Sensor aus.", {}

    status_msg = f"Starte Modelltraining für {sensor_col_name}..."
    logging.info(status_msg)

    try:
        df_target_sensor = df_all_sensors_data[[sensor_col_name]].dropna()
        if df_target_sensor.empty or len(df_target_sensor) < 200: # Mindestdaten für robustes Training
            return f"⚠️ Nicht genug Datenpunkte für {sensor_col_name} zum Trainieren. Mindestens 200 werden benötigt.", {}

        # Hier wird load_and_prepare_data_from_df aufgerufen, da die Daten bereits als DataFrame vorliegen
        df_prepared = load_and_prepare_data(df_target_sensor) 
        df_features = prepare_features(df_prepared, sensor_col_name)

        # Kombiniere Features und Original-Ziel für die Trainingsfunktion
        df_for_training = pd.concat([df_features, df_prepared[[sensor_col_name]]], axis=1)

        model, scaler, is_target_detrended = train_and_select_model(df_for_training, sensor_col_name)

        if model and scaler:
            status_msg = f"✅ Modell für {sensor_col_name} erfolgreich trainiert."
            logging.info(status_msg)
            # Speichere Modellinformationen (welcher Sensor, Detrending-Status) in dcc.Store
            model_info = {
                'sensor_col_name': sensor_col_name,
                'is_target_detrended': is_target_detrended,
            }
            return status_msg, model_info
        else:
            status_msg = f"❌ Modelltraining für {sensor_col_name} fehlgeschlagen."
            logging.error(status_msg)
            return status_msg, {}
    except Exception as e:
        status_msg = f"❌ Fehler beim Modelltraining: {str(e)}"
        logging.error(status_msg)
        return status_msg, {}

# Callback zum Erstellen einer Vorhersage und Aktualisieren des Graphen
@app.callback(
    Output({"type": "graph", "index": dash.ALL}, "figure", allow_duplicate=True), # Aktualisiert spezifischen Graphen
    Output("forecast-store", "data"),
    Input("forecast-btn", "n_clicks"),
    State("sensor-dropdown", "value"),
    State("forecast-horizon", "value"),
    State("model-store", "data"),
    State({"type": "graph", "index": dash.ALL}, "figure"), # Holt aktuelle Figuren zur Modifikation
    prevent_initial_call=True
)
def generate_forecast_and_plot(n_clicks, sensor_col_name, forecast_horizon, model_info, current_figures_list):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    if df_all_sensors_data.empty or sensor_col_name not in df_all_sensors_data.columns:
        logging.warning("Keine Daten oder ausgewählte Spalte nicht vorhanden für Vorhersage.")
        raise dash.exceptions.PreventUpdate # Verhindert Update, wenn Daten fehlen
    
    if not model_info or model_info.get('sensor_col_name') != sensor_col_name:
        logging.warning("Modell für diesen Sensor wurde noch nicht trainiert oder ist ungültig.")
        raise dash.exceptions.PreventUpdate # Verhindert Update, wenn Modell für den ausgewählten Sensor nicht trainiert ist

    logging.info(f"Starte Vorhersage für {sensor_col_name} mit Horizont {forecast_horizon}...")

    # Initialisiere ein Dictionary, um Figuren für die Ausgabe zu speichern
    # Dies geht davon aus, dass current_figures_list in der gleichen Reihenfolge wie SENSORS.values() ist
    figures_output = {f"graph-{col_name}": go.Figure(current_figures_list[i])
                      for i, col_name in enumerate(SENSORS.values())}

    try:
        # Lade Modell und Scaler (angenommen, sie wurden von train_and_select_model im MODEL_DIR gespeichert)
        model_path = os.path.join(MODEL_DIR, f'model_{sensor_col_name}.joblib')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{sensor_col_name}.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logging.error("Modell- oder Scaler-Dateien nicht gefunden für die Vorhersage.")
            raise dash.exceptions.PreventUpdate # Stoppt, wenn Modelldateien fehlen

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        is_target_detrended = model_info.get('is_target_detrended', False)

        df_target_sensor = df_all_sensors_data[[sensor_col_name]].dropna()
        if df_target_sensor.empty:
            logging.warning(f"Keine Daten für {sensor_col_name} vorhanden für Vorhersage.")
            raise dash.exceptions.PreventUpdate # Stoppt, wenn Zieldaten leer sind

        forecast_df = forecast_steps(model, scaler, df_target_sensor, sensor_col_name, forecast_horizon, is_target_detrended)

        # Aktualisiere den spezifischen Graphen für den vorhergesagten Sensor
        target_graph_id = f"graph-{sensor_col_name}"
        fig_to_update = figures_output.get(target_graph_id)

        if fig_to_update:
            # Lösche vorhandene Vorhersage-Spuren, falls vorhanden (optional, stellt sicher, dass nur eine Vorhersage angezeigt wird)
            fig_to_update.data = [trace for trace in fig_to_update.data if 'Vorhersage' not in trace.name and 'Vorhersage ±1 STD' not in trace.name]

            # Füge neue Vorhersage-Spur hinzu
            if not forecast_df.empty:
                fig_to_update.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[f'forecast_{sensor_col_name}'], mode='lines', name='Vorhersage', 
                                                   line=dict(color='#00FF7F', dash='dash', width=2))) # Helles Grün, etwas dicker
                
                # Füge Variabilitätsindikator hinzu (einfache Standardabweichung vorerst)
                if len(forecast_df) > 1:
                    forecast_std = forecast_df[f'forecast_{sensor_col_name}'].std()
                    fig_to_update.add_trace(go.Scatter(
                        x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                        y=(forecast_df[f'forecast_{sensor_col_name}'] + forecast_std).tolist() + \
                          (forecast_df[f'forecast_{sensor_col_name}'] - forecast_std).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,255,127,0.15)', # Etwas undurchsichtigeres Füllgrün
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Vorhersage ±1 STD',
                        showlegend=True
                    ))
                fig_to_update.update_layout(showlegend=True) # Stelle sicher, dass die Legende für die Vorhersage sichtbar ist

        forecast_data_for_store = {
            sensor_col_name: forecast_df.reset_index().to_dict(orient='records')
        }
        
        # Bereite die Liste der zurückzugebenden Figuren vor, passend zur Reihenfolge der Ausgaben
        returned_figures_list = [figures_output[f"graph-{col_name}"] for col_name in SENSORS.values()]
        return returned_figures_list, forecast_data_for_store

    except Exception as e:
        logging.error(f"Fehler bei der Vorhersage oder beim Plotten: {str(e)}")
        # Wenn ein Fehler auftritt, setze die spezifische Figur auf eine generische Fehlermeldung zurück
        target_graph_id = f"graph-{sensor_col_name}"
        figures_output[target_graph_id] = px.line(title=f"Fehler bei der Vorhersage für {sensor_col_name}: {str(e)}", template='plotly_dark')
        figures_output[target_graph_id].update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white') # Sicherstellen, dass das Fehlerdiagramm auch dunkel ist
        
        returned_figures_list = [figures_output[f"graph-{col_name}"] for col_name in SENSORS.values()]
        return returned_figures_list, {}

# Callback für Daten-Download
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("forecast-store", "data"),
    State("sensor-dropdown", "value"),
    prevent_initial_call=True
)
def download_data(n_clicks, forecast_data, sensor_col_name):
    if n_clicks is None:
        return None
    
    # Lade die neuesten Daten aus der DB, um sicherzustellen, dass die heruntergeladenen Daten aktuell sind
    df_current_data = db_manager.load_all_sensor_data_from_db(SENSEBOX_ID)
    
    if df_current_data.empty or sensor_col_name not in df_current_data.columns:
        logging.warning("Keine Daten zum Herunterladen verfügbar oder ausgewählter Sensor nicht gefunden.")
        return None

    try:
        df_to_download = df_current_data[[sensor_col_name]].copy()
        df_to_download = df_to_download.reset_index().rename(columns={'createdAt': 'timestamp', sensor_col_name: 'historical_measurement'})
        
        if forecast_data and sensor_col_name in forecast_data:
            forecast_df_raw = pd.DataFrame(forecast_data[sensor_col_name])
            forecast_df = forecast_df_raw.rename(columns={f'forecast_{sensor_col_name}': 'forecast_measurement', 'timestamp': 'timestamp'})
            
            # Führe historische und Vorhersagedaten zusammen
            combined = pd.merge(df_to_download, forecast_df, on='timestamp', how='outer')
            combined = combined.sort_values(by='timestamp')
        else:
            combined = df_to_download
            combined['forecast_measurement'] = np.nan # Füge leere Vorhersagespalte hinzu, wenn keine Vorhersage vorhanden ist

        csv_string = combined.to_csv(index=False)
        logging.info(f"Daten für {sensor_col_name} heruntergeladen.")
        return dict(content=csv_string, filename=f"{sensor_col_name}_data.csv")
    except Exception as e:
        logging.error(f"Fehler beim Herunterladen der Daten für {sensor_col_name}: {str(e)}")
        return dict(content=str(e), filename="error.csv")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

