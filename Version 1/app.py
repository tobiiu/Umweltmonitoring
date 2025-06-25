import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import sensebox_api  # dein Modul für die Karte

# Daten laden
df = pd.read_csv('sensordaten.csv', parse_dates=['createdAt'])

app = dash.Dash(__name__)

# SenseBox-Daten und Karte
sensebox_id = "67cac102d2a4eb00071d6ac9"
orte = sensebox_api.get_all_locations(sensebox_id)
map_obj = sensebox_api.show_colored_map(orte)
map_html = sensebox_api.get_folium_html(map_obj)

app.layout = html.Div([
    html.H1("SenseBox Dashboard"),

    # Karte als Iframe
    html.Iframe(
        srcDoc=map_html,
        width='100%',
        height='300'
    ),

    # Dropdown zum Sensor auswählen
    dcc.Dropdown(
        id='sensor-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns if col != 'createdAt'],
        value='Luftfeuchte SHT31',
        style={'margin-top': '20px'}
    ),

    # Graph für den Sensor
    dcc.Graph(id='sensor-graph')
])

@app.callback(
    Output('sensor-graph', 'figure'),
    Input('sensor-dropdown', 'value')
)
def update_graph(selected_sensor):
    fig = px.line(df, x='createdAt', y=selected_sensor, title=f'{selected_sensor} über Zeit')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
