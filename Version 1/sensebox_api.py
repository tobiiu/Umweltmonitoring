import requests
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
from matplotlib import colors, cm
import io

def get_all_locations(sensebox_id):
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}/locations"
    params = {
        "from-date": "2015-01-01T00:00:00Z",
        "format": "json"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Fehler beim Abrufen der Daten: {response.status_code}")

    data = response.json()

    locations = []
    for entry in data:
        timestamp = entry.get("timestamp")
        coordinates = entry.get("coordinates")
        if timestamp and coordinates:
            locations.append((timestamp, coordinates))

    return locations

def show_colored_map(locations):
    if not locations:
        print("Keine Koordinaten vorhanden.")
        return None

    locations_sorted = sorted(locations, key=lambda x: x[0])
    timestamps = [datetime.fromisoformat(t[0].replace("Z", "+00:00")) for t in locations_sorted]

    norm = colors.Normalize(vmin=0, vmax=len(timestamps) - 1)
    colormap = cm.get_cmap("plasma")
    colors_hex = [colors.to_hex(colormap(norm(i))) for i in range(len(timestamps))]

    first_coord = locations_sorted[0][1]
    lat = first_coord[1]
    lon = first_coord[0]
    fmap = folium.Map(location=[lat, lon], zoom_start=4)
    marker_cluster = MarkerCluster().add_to(fmap)

    for (timestamp, coord), color in zip(locations_sorted, colors_hex):
        lon = coord[0]
        lat = coord[1]
        alt = coord[2] if len(coord) > 2 else None
        popup_text = f"{timestamp}"
        if alt is not None:
            popup_text += f"<br>Höhe: {alt:.2f} m"

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(marker_cluster)

    return fmap

def get_folium_html(map_obj):
    if map_obj is None:
        return "<p>Keine Karte vorhanden</p>"
    return map_obj.get_root().render()
