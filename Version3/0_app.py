import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from forecast_model import forecast, train_model

# Daten laden
data_file = "data.csv"
location_file = "locations.csv"
df_all = pd.read_csv(data_file)
df_locations = pd.read_csv(location_file)
df_locations['timestamp'] = pd.to_datetime(df_locations['timestamp'])

app = dash.Dash(__name__)
app.title = "Umweltmonitoring Forecast"

app.layout = html.Div([
    html.H1("📈 Umwelt-Daten Dashboard"),
    dcc.Dropdown(
        id="target-dropdown",
        options=[{"label": col, "value": col} for col in ['Temperature', 'PM2.5', 'Humidity', 'PM10', 'Pressure']],
        value="Temperature"
    ),
    dcc.Slider(id="forecast-horizon", min=6, max=60, step=6, value=12,
               marks={i: f"{i} Steps" for i in range(6, 61, 6)}),
    html.Button("📊 Modell trainieren", id="train-btn"),
    html.Button("🔮 Vorhersage erstellen", id="forecast-btn"),
    dcc.Graph(id="time-series-graph"),
    html.H2("📍 Sensorposition"),
    dcc.Graph(id="map-graph"),
    html.Div(id="status-message")
])

@app.callback(
    Output("time-series-graph", "figure"),
    Output("status-message", "children"),
    Input("train-btn", "n_clicks"),
    Input("forecast-btn", "n_clicks"),
    State("target-dropdown", "value"),
    State("forecast-horizon", "value"),
)
def update_graph(train_clicks, forecast_clicks, target, steps):
    ctx = dash.callback_context
    fig = go.Figure()
    message = ""

    try:
        df = pd.read_csv(data_file)
        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)
        df = df.dropna(subset=['createdAt'])  # löscht problematische Zeitstempel
        df = df.sort_values("createdAt")

        fig.add_trace(go.Scatter(x=df['createdAt'], y=df[target], name="Historisch"))

        if ctx.triggered_id == "train-btn":
            train_model(df, target=target, horizon=steps)
            message = f"✅ Modell für {target} trainiert."

        if ctx.triggered_id == "forecast-btn":
            forecast_df = forecast(df, target=target, steps=steps)
            fig.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df[f'predicted_{target}'],
                                     mode='lines+markers', name="Vorhersage"))
            message = f"✅ Vorhersage für {target} generiert."
    except Exception as e:
        message = f"❌ Fehler: {str(e)}"

    fig.update_layout(title=f"Zeitreihe: {target}", xaxis_title="Zeit", yaxis_title=target)
    return fig, message

@app.callback(
    Output("map-graph", "figure"),
    Input("target-dropdown", "value")
)
def update_map(target):
    latest = df_locations.sort_values("timestamp").iloc[-1]
    fig = px.scatter_mapbox(
        df_locations,
        lat="lat",
        lon="lon",
        hover_data={"timestamp": True},
        zoom=8,
        height=400
    )
    fig.update_layout(mapbox_style="open-street-map")
    return fig

if __name__ == '__main__':
    app.run(debug=True)
