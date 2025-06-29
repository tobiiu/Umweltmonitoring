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
import joblib # Needed for saving/loading models

# Import functions from forecast_model. Make sure forecast_model.py is in the same directory.
from forecast_model import load_and_prepare_data, remove_outliers, prepare_features, train_and_select_model, forecast_steps

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adjust CSV_FILE path as it's mounted in a 'data' subfolder
CSV_FILE = "data/data.csv"
SENSEBOX_ID = "60d828c48855dd001cf91983" # Replace with your actual SenseBox ID if different
MODEL_DIR = "models" # Directory within the container where models are stored

# --- Helper Functions ---
def get_sensor_location_and_name(sensebox_id):
    """
    Fetches senseBox location and name from the openSenseMap API.
    Uses fallback coordinates and name if API call fails or data is incomplete.
    """
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        metadata = response.json()

        # Handle cases where metadata might be a list or malformed, though typically it's a dict
        if isinstance(metadata, list) and metadata:
            metadata = metadata[0] # Take the first element if it's a list
        elif not isinstance(metadata, dict):
            raise ValueError(f"Unexpected API response format: {type(metadata)}")

        loc_data = metadata.get("loc", {})
        # Ensure loc_data is a dictionary before trying to access 'coordinates'
        if isinstance(loc_data, list) and loc_data:
            loc_data = loc_data[0] # If 'loc' is an unexpected list, take the first item
        elif not isinstance(loc_data, dict):
            loc_data = {} # Fallback to empty dict if 'loc' is not a dict or list

        coordinates = loc_data.get("coordinates", [])

        if len(coordinates) < 2:
            raise ValueError("Coordinates not found or incomplete in API response for 'loc'.")

        lat = coordinates[1]
        lon = coordinates[0]
        name = metadata.get("name", "SenseBox")
        logging.info(f"Sensor location loaded: Lat {lat}, Lon {lon}, Name: {name}")
        return lat, lon, name
    except Exception as e:
        logging.error(f"Error fetching sensor location: {e}. Using default location.")
        return 57.280353, 27.069191, "Default SenseBox" # Fallback coordinates

# Fetch location and name once at startup
SENSEBOX_LAT, SENSEBOX_LON, SENSEBOX_NAME = get_sensor_location_and_name(SENSEBOX_ID)

def load_data_from_csv():
    """
    Loads data from the CSV file, parses timestamps, sets index, and cleans column names.
    """
    try:
        if not os.path.exists(CSV_FILE):
            logging.warning(f"CSV file '{CSV_FILE}' not found. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.read_csv(CSV_FILE, parse_dates=["createdAt"])
        if not df.empty:
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            # Convert 'createdAt' to datetime, ensuring UTC and handling ISO8601 format
            df['createdAt'] = pd.to_datetime(df['createdAt'], utc=True, format='ISO8601')
            df.set_index('createdAt', inplace=True)
            df = df[~df.index.duplicated(keep='first')] # Remove duplicates if any
            df = df.sort_index() # Sort by timestamp

        logging.info(f"Data loaded from CSV. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from CSV: {e}")
        return pd.DataFrame()

# Global DataFrame to hold all loaded sensor data
df_all_sensors_data = load_data_from_csv()

# Define the sensors we expect, mapping display names to exact CSV column names
# These must match the headers in your data.csv exactly (e.g., "Temperature", "PM2.5", "Humidity")
SENSORS = {
    'Temperature': 'Temperature',
    'PM2.5': 'PM2.5',
    'Humidity': 'Humidity',
    'PM10': 'PM10',
    'Pressure': 'Pressure'
}

# --- Dash App Setup ---
# Initialize Dash app with a dark theme from Dash Bootstrap Components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Umweltmonitoring Dashboard"
server = app.server # Expose Flask server for production deployment if needed


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
                className='mb-4 bg-gray-800 text-white border-gray-700 rounded'
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
                className='mb-4 bg-gray-800 text-white border-gray-700 rounded'
            ),
            html.Button("Modell trainieren", id="train-btn", className='bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mb-2 w-full'),
            html.Button("Vorhersage erstellen", id="forecast-btn", className='bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded mb-2 w-full'),
            html.Button("Daten herunterladen", id="download-btn", className='bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded mb-2 w-full'),
            dcc.Download(id="download-data"),
            html.Div(id="status-message", className='mt-4 text-lg text-blue-300') # Statusmeldungen für Training/Vorhersage
        ]),
        # Neueste Werte Tabelle
        html.Div(className='w-full md:w-2/3 px-2 mb-4', children=[
            html.H3("Neueste Sensorwerte", className='text-xl font-semibold mb-2'),
            dash_table.DataTable(
                id='latest-values-table',
                columns=[
                    {"name": "Sensor", "id": "sensor"},
                    {"name": "Wert", "id": "value"},
                    {"name": "Zeitstempel", "id": "timestamp"}
                ],
                style_table={'overflowX': 'auto', 'backgroundColor': '#1F2937'}, # Dark background for table container
                style_cell={
                    'textAlign': 'left',
                    'backgroundColor': '#1F2937', # Dark background for cells
                    'color': 'white', # White text
                    'padding': '10px',
                    'border': '1px solid #374151' # Subtle border
                },
                style_header={
                    'backgroundColor': '#374151', # Darker background for header
                    'fontWeight': 'bold',
                    'color': 'white', # White text
                    'border': '1px solid #374151' # Subtle border
                }
            )
        ])
    ]),
    
    # Zeitreihen-Diagramme für alle Sensoren
    html.H2("Zeitreihen-Diagramme", className='text-2xl font-semibold mb-2 mt-4 text-center'),
    html.Div(
        id='all-sensors-graphs-container', # Container for dynamically generated graphs
        className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4', # Responsive grid
        children=[
            # Placeholder for graphs, they will be updated by callback
            dcc.Graph(id=f"graph-{col_name}", style={'height': '400px'}) for col_name in SENSORS.values()
        ]
    ),
    
    # Karte der Sensorstandorte
    html.H2("📍 Sensorstandort", className='text-2xl font-semibold mb-2 mt-4 text-center'),
    dcc.Graph(id="map-graph", style={'height': '400px'}),
    
    # Speicher für Vorhersagedaten und Modellinformationen
    dcc.Store(id='forecast-store'),
    dcc.Store(id='model-store') # Stores information about the trained model
])

# --- Callbacks ---

# Callback for updating all dashboard elements on interval
@app.callback(
    [Output("latest-values-table", "data"),
     Output("all-sensors-graphs-container", "children"),
     Output("map-graph", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard_content(n_intervals):
    global df_all_sensors_data
    logging.info(f"Aktualisiere Dashboard-Inhalte (Intervall: {n_intervals})")
    
    # Reload data from CSV
    df_all_sensors_data = load_data_from_csv()

    # Initialisiere leere Daten für den Fall, dass df_all_sensors_data leer ist
    table_data = []
    all_graphs = []
    map_figure = go.Figure().update_layout(title="Keine Standortdaten verfügbar", template='plotly_dark')

    if df_all_sensors_data.empty:
        logging.warning("Keine Daten aus CSV geladen oder Daten sind leer nach der Bereinigung. Es werden keine Graphen angezeigt.")
        empty_figure = px.line(title="Keine Daten verfügbar", template='plotly_dark')
        # Return empty data for table and graphs
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
    for display_name, col_name in SENSORS.items():
        if col_name in df_all_sensors_data.columns:
            fig = px.line(df_all_sensors_data.reset_index(), x="createdAt", y=col_name,
                          title=f"{display_name} Zeitreihe", template='plotly_dark')
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
                margin=dict(l=40, r=40, t=40, b=40)
            )
            all_graphs.append(dcc.Graph(figure=fig, id=f"graph-{col_name}"))
        else:
            logging.warning(f"Spalte '{col_name}' nicht in den geladenen Daten gefunden.")
            all_graphs.append(dcc.Graph(figure=px.line(title=f"{display_name} (Daten fehlen)", template='plotly_dark'), id=f"graph-{col_name}"))


    # --- Karte der Sensorstandorte aktualisieren ---
    map_figure = px.scatter_map(
        pd.DataFrame([{'lat': SENSEBOX_LAT, 'lon': SENSEBOX_LON, 'name': SENSEBOX_NAME}]),
        lat="lat",
        lon="lon",
        zoom=10,
        title="Sensorstandort",
        hover_name="name"
    )
    map_figure.update_layout(
        mapbox_style="open-street-map", # You can try 'carto-darkmatter' for a darker map style
        template='plotly_dark',
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    return table_data, all_graphs, map_figure

# Callback for training the model
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
        if df_target_sensor.empty or len(df_target_sensor) < 200: # Minimum data for robust training
            return f"⚠️ Nicht genug Datenpunkte für {sensor_col_name} zum Trainieren. Mindestens 200 werden benötigt.", {}

        df_prepared = load_and_prepare_data_from_df(df_target_sensor)
        df_features = prepare_features(df_prepared, sensor_col_name)

        # Pass the full DataFrame with original target for correct feature preparation inside the model training
        # Concatenate features and original target for the training function
        df_for_training = pd.concat([df_features, df_prepared[[sensor_col_name]]], axis=1)

        model, scaler, is_target_detrended = train_and_select_model(df_for_training, sensor_col_name)

        if model and scaler:
            status_msg = f"✅ Modell für {sensor_col_name} erfolgreich trainiert."
            logging.info(status_msg)
            # Store model info (which sensor, detrending status) in dcc.Store
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

# Callback for creating a forecast and updating the graph
@app.callback(
    Output({"type": "graph", "index": dash.ALL}, "figure", allow_duplicate=True), # Update specific graph
    Output("forecast-store", "data"),
    Input("forecast-btn", "n_clicks"),
    State("sensor-dropdown", "value"),
    State("forecast-horizon", "value"),
    State("model-store", "data"),
    State({"type": "graph", "index": dash.ALL}, "figure"), # Get current figures to modify
    prevent_initial_call=True
)
def generate_forecast_and_plot(n_clicks, sensor_col_name, forecast_horizon, model_info, current_figures_list):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    if df_all_sensors_data.empty or sensor_col_name not in df_all_sensors_data.columns:
        logging.warning("Keine Daten oder ausgewählte Spalte nicht vorhanden für Vorhersage.")
        raise dash.exceptions.PreventUpdate # Prevent update if data is missing
    
    if not model_info or model_info.get('sensor_col_name') != sensor_col_name:
        logging.warning("Modell für diesen Sensor wurde noch nicht trainiert oder ist ungültig.")
        raise dash.exceptions.PreventUpdate # Prevent update if model is not trained for the selected sensor

    logging.info(f"Starte Vorhersage für {sensor_col_name} mit Horizont {forecast_horizon}...")

    # Initialize a dictionary to store figures for the output
    # This assumes `current_figures_list` is in the same order as `SENSORS.values()`
    figures_output = {f"graph-{col_name}": go.Figure(current_figures_list[i])
                      for i, col_name in enumerate(SENSORS.values())}

    try:
        # Load model and scaler (assuming they were saved to MODEL_DIR by train_and_select_model)
        model_path = os.path.join(MODEL_DIR, f'model_{sensor_col_name}.joblib')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{sensor_col_name}.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logging.error("Modell oder Scaler-Dateien nicht gefunden für die Vorhersage.")
            raise dash.exceptions.PreventUpdate # Stop if model files are missing

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        is_target_detrended = model_info.get('is_target_detrended', False)

        df_target_sensor = df_all_sensors_data[[sensor_col_name]].dropna()
        if df_target_sensor.empty:
            logging.warning(f"Keine Daten für {sensor_col_name} vorhanden für Vorhersage.")
            raise dash.exceptions.PreventUpdate # Stop if target data is empty

        forecast_df = forecast_steps(model, scaler, df_target_sensor, sensor_col_name, forecast_horizon, is_target_detrended)

        # Update the specific graph for the forecasted sensor
        target_graph_id = f"graph-{sensor_col_name}"
        fig_to_update = figures_output.get(target_graph_id)

        if fig_to_update:
            # Clear existing forecast traces if any (optional, ensures only one forecast is shown)
            fig_to_update.data = [trace for trace in fig_to_update.data if 'Vorhersage' not in trace.name and 'Vorhersage ±1 STD' not in trace.name]

            # Add new forecast trace
            if not forecast_df.empty:
                fig_to_update.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[f'forecast_{sensor_col_name}'], mode='lines', name='Vorhersage', line=dict(color='green', dash='dash')))

                # Add variability indicator (simple std dev for now)
                if len(forecast_df) > 1:
                    forecast_std = forecast_df[f'forecast_{sensor_col_name}'].std()
                    fig_to_update.add_trace(go.Scatter(
                        x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                        y=(forecast_df[f'forecast_{sensor_col_name}'] + forecast_std).tolist() + \
                          (forecast_df[f'forecast_{sensor_col_name}'] - forecast_std).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,255,0,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Vorhersage ±1 STD',
                        showlegend=True
                    ))
                fig_to_update.update_layout(showlegend=True) # Ensure legend is visible for forecast

        forecast_data_for_store = {
            sensor_col_name: forecast_df.reset_index().to_dict(orient='records')
        }
        
        # Prepare the list of figures to return, matching the order of outputs
        returned_figures_list = [figures_output[f"graph-{col_name}"] for col_name in SENSORS.values()]
        return returned_figures_list, forecast_data_for_store

    except Exception as e:
        logging.error(f"Fehler bei der Vorhersage oder beim Plotten: {str(e)}")
        # If an error occurs, revert the specific figure to a generic error message
        target_graph_id = f"graph-{sensor_col_name}"
        figures_output[target_graph_id] = px.line(title=f"Fehler bei der Vorhersage für {sensor_col_name}: {str(e)}", template='plotly_dark')
        
        returned_figures_list = [figures_output[f"graph-{col_name}"] for col_name in SENSORS.values()]
        return returned_figures_list, {}

# Callback for data download
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
    
    if df_all_sensors_data.empty:
        logging.warning("Keine Daten zum Herunterladen verfügbar.")
        return None

    try:
        df_to_download = df_all_sensors_data[[sensor_col_name]].copy()
        df_to_download = df_to_download.reset_index().rename(columns={'createdAt': 'timestamp', sensor_col_name: 'historical_measurement'})
        
        if forecast_data and sensor_col_name in forecast_data:
            forecast_df_raw = pd.DataFrame(forecast_data[sensor_col_name])
            forecast_df = forecast_df_raw.rename(columns={f'forecast_{sensor_col_name}': 'forecast_measurement', 'timestamp': 'timestamp'})
            
            # Merge historical and forecast data
            combined = pd.merge(df_to_download, forecast_df, on='timestamp', how='outer')
            combined = combined.sort_values(by='timestamp')
        else:
            combined = df_to_download
            combined['forecast_measurement'] = np.nan # Add empty forecast column if no forecast

        csv_string = combined.to_csv(index=False)
        logging.info(f"Daten für {sensor_col_name} heruntergeladen.")
        return dict(content=csv_string, filename=f"{sensor_col_name}_data.csv")
    except Exception as e:
        logging.error(f"Fehler beim Herunterladen der Daten für {sensor_col_name}: {str(e)}")
        return dict(content=str(e), filename="error.csv")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)