import dash
from dash import dcc, html, Input, Output
import pandas as pd
from sqlalchemy import create_engine

import os

db_host = os.getenv("DATABASE_HOST", "localhost")  # <- fallback nur wenn NICHT gesetzt
db_user = os.getenv("DATABASE_USER", "user")
db_password = os.getenv("DATABASE_PASSWORD", "pass")
db_name = os.getenv("DATABASE_NAME", "sensebox")

DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"

app = dash.Dash(__name__)
app.title = "SenseBox Dashboard"

engine = create_engine("postgresql://postgres:postgres@sensebox-db:5433/env_monitoring")

def load_data():
    query = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1000;"
    return pd.read_sql(query, engine)

def load_sensor_types():
    query = "SELECT DISTINCT sensor_type FROM sensor_data;"
    df = pd.read_sql(query, engine)
    return df['sensor_type'].tolist()

app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#39ff14', 'padding': '20px'}, children=[
    html.H1("🌍 SenseBox Dashboard"),
    html.Label("Wähle Sensor:", style={'marginTop': '20px'}),
    dcc.Dropdown(id='sensor-dropdown', options=[], style={'width': '50%'}),
    
    html.Label("Zeitraum:", style={'marginTop': '20px'}),
    dcc.DatePickerRange(id='date-range', start_date_placeholder_text='Start', end_date_placeholder_text='Ende'),

    dcc.Graph(id='sensor-graph', style={'marginTop': '40px', 'backgroundColor': '#111111'})
])

@app.callback(
    Output('sensor-dropdown', 'options'),
    Output('sensor-dropdown', 'value'),
    Input('sensor-graph', 'id')  # Trigger einmal beim Laden
)
def init_dropdown(_):
    sensor_types = load_sensor_types()
    return [{'label': s, 'value': s} for s in sensor_types], sensor_types[0] if sensor_types else None

@app.callback(
    Output('sensor-graph', 'figure'),
    Input('sensor-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
)
def update_graph(sensor_type, start, end):
    if not sensor_type:
        return {}

    query = "SELECT * FROM sensor_data WHERE sensor_type = %s"
    params = [sensor_type]

    if start:
        query += " AND timestamp >= %s"
        params.append(start)
    if end:
        query += " AND timestamp <= %s"
        params.append(end)

    df = pd.read_sql(query, engine, params=params)

    fig = {
        'data': [{
            'x': df['timestamp'],
            'y': df['measurement'],
            'type': 'line',
            'name': sensor_type,
            'line': {'color': '#39ff14'}
        }],
        'layout': {
            'plot_bgcolor': '#111111',
            'paper_bgcolor': '#111111',
            'font': {'color': '#39ff14'},
            'title': f'Messwerte für {sensor_type}'
        }
    }

    return fig

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050)
