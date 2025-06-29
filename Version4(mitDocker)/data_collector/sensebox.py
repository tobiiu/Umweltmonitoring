import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

SENSEBOX_ID = os.getenv("SENSEBOX_ID") # WICHTIG: KEIN Standardwert hier, MUSS über Env Var gesetzt werden.
CSV_FILE_PATH = os.getenv("UPDATER_CSV_PATH", "data/data.csv")

def _get_box_data(sensebox_id):
    """Fetches senseBox metadata."""
    if not sensebox_id: # Füge Prüfung für fehlende SENSEBOX_ID hinzu
        print("SENSEBOX_ID ist nicht gesetzt. Kann keine Box-Metadaten abrufen.")
        return {}
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen der Box-Metadaten: {e}")
        return {}

def _get_sensor_measurements(sensebox_id, sensor, from_date, to_date):
    """Fetches measurements for a single sensor."""
    sensor_id = sensor.get("_id")
    sensor_type = sensor.get("type")
    sensor_unit = sensor.get("unit")
    if not sensor_id:
        return pd.DataFrame(), None

    from_str = from_date.isoformat(timespec='seconds') + 'Z'
    to_str = to_date.isoformat(timespec='seconds') + 'Z'

    api_url = f"https://api.opensensemap.org/boxes/{sensebox_id}/data/{sensor_id}?from-date={from_str}&to-date={to_str}&format=json"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        measurements = response.json()

        if not measurements:
            return pd.DataFrame(), None

        df = pd.DataFrame(measurements)
        if df.empty:
            return pd.DataFrame(), None

        df['createdAt'] = pd.to_datetime(df['createdAt'], utc=True)
        df = df.rename(columns={'createdAt': 'timestamp', 'value': 'measurement'})
        df['sensor_id'] = sensor_id
        df['box_id'] = sensebox_id
        df['unit'] = sensor_unit
        df['sensor_type'] = sensor_type
        # Wähle nur die Spalten aus, die du in der CSV speichern möchtest
        df = df[['timestamp', 'box_id', 'sensor_id', 'measurement', 'unit', 'sensor_type']]

        latest_timestamp_for_sensor = df['timestamp'].max() if not df.empty else None

        return df, latest_timestamp_for_sensor
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Abrufen der Messdaten für Sensor {sensor_id}: {e}")
        return pd.DataFrame(), None

def get_last_timestamp_from_csv(file_path):
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        return None
    try:
        # Lese nur die 'timestamp'-Spalte der letzten Zeile für Effizienz
        df = pd.read_csv(file_path, usecols=['timestamp'], parse_dates=['timestamp'], dayfirst=False)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df['timestamp'].max() # Den neuesten Zeitstempel zurückgeben
    except Exception as e:
        print(f"Fehler beim Lesen des letzten Zeitstempels aus CSV {file_path}: {e}")
    return None

def download_measurements_to_dataframe(sensebox_id, from_date=None):
    if not sensebox_id: # Füge Prüfung für fehlende SENSEBOX_ID hinzu
        print("SENSEBOX_ID ist nicht gesetzt. Überspringe Daten-Download.")
        return pd.DataFrame(), None, {}

    if from_date is None:
        from_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    else:
        from_date = pd.to_datetime(from_date, utc=True).to_pydatetime()

    to_date = datetime.now(timezone.utc)

    box_data = _get_box_data(sensebox_id)
    sensors = box_data.get("sensors", [])

    print(f"📦 Starte parallelen Download von {len(sensors)} Sensoren...")

    sensor_dfs = {}
    sensor_timestamps = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_get_sensor_measurements, sensebox_id, sensor, from_date, to_date): sensor
            for sensor in sensors
        }

        for future in as_completed(futures):
            sensor = futures[future]
            try:
                df, latest_ts = future.result()
                if not df.empty:
                    sensor_dfs[sensor["_id"]] = df
                    sensor_timestamps[sensor["_id"]] = latest_ts
            except Exception as exc:
                print(f'{sensor["_id"]} erzeugte eine Ausnahme: {exc}')

    if not sensor_dfs:
        print("Keine Sensordaten zum Kombinieren gefunden.")
        return pd.DataFrame(), None, {}

    combined_df = pd.concat(sensor_dfs.values()).sort_values(by='timestamp')
    # Entferne Duplikate basierend auf timestamp, box_id und sensor_id, behalte den letzten Eintrag
    combined_df.drop_duplicates(subset=['timestamp', 'box_id', 'sensor_id'], keep='last', inplace=True)

    overall_latest_timestamp = combined_df['timestamp'].max() if not combined_df.empty else None

    return combined_df, overall_latest_timestamp, sensor_timestamps

def main():
    print("Starte Daten-Update-Prozess...")
    try:
        last_timestamp = get_last_timestamp_from_csv(CSV_FILE_PATH)
        if last_timestamp:
            print(f"Letzter gespeicherter Zeitstempel in der CSV: {last_timestamp}")
        else:
            print("Keine Daten in der CSV-Datei gefunden. Starte mit leerer Historie (von 2000-01-01).")
            last_timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)

        combined_df, new_last_timestamp, sensor_timestamps = download_measurements_to_dataframe(
            SENSEBOX_ID,
            from_date=last_timestamp
        )

        if not combined_df.empty:
            print(f"Neue Daten bis: {new_last_timestamp}")

            # Bestimme, ob der Header geschrieben werden soll (nur wenn Datei nicht existiert oder leer ist)
            write_header = not os.path.exists(CSV_FILE_PATH) or os.stat(CSV_FILE_PATH).st_size == 0

            # Hänge Daten an die CSV-Datei an
            combined_df.to_csv(
                CSV_FILE_PATH,
                mode='a', # 'a' für append (anhängen)
                index=False,
                header=write_header
            )
            print(f"✅ Neue Daten an '{CSV_FILE_PATH}' angehängt.")
        else:
            print("Keine neuen Daten zum Anhängen gefunden.")

    except Exception as e:
        print(f"❌ Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    # Die Schleife sorgt dafür, dass der Updater regelmäßig läuft.
    while True:
        main()
        print("Warte x Minuten bis zum nächsten Update...")
        time.sleep(120) # Warte 2 Minuten 