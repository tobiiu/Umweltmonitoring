import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

SENSEBOX_ID = "60d828c48855dd001cf91983"
CSV_FILE = "data.csv"


def download_measurements_to_dataframe(sensebox_id, from_date=None):
    # Zeit korrekt initialisieren (UTC-aware)
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

    # Paralleler Download der Sensordaten
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_get_sensor_measurements, sensebox_id, sensor, from_date, to_date): sensor
            for sensor in sensors
        }

        for future in as_completed(futures):
            sensor = futures[future]
            try:
                sensor_df, timestamps = future.result()
                if not sensor_df.empty:
                    sensor_dfs[sensor["title"]] = sensor_df
                    sensor_timestamps[sensor["title"]] = timestamps
                    print(f"✅ Sensor '{sensor['title']}' geladen ({len(sensor_df)} Einträge)")
                else:
                    print(f"⚠️ Sensor '{sensor['title']}' enthält keine Daten")
            except Exception as e:
                print(f"❌ Fehler bei Sensor '{sensor['title']}': {e}")

    if not sensor_dfs:
        print("❌ Keine Daten geladen.")
        return pd.DataFrame(), None, {}

    # Zusammenführen
    combined_df = pd.concat(sensor_dfs.values(), axis=1, join='outer')
    combined_df = combined_df.sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    last_timestamp = combined_df.index.max().isoformat()

    return combined_df, last_timestamp, sensor_timestamps


def _get_box_data(sensebox_id):
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def _get_sensor_measurements(sensebox_id, sensor, from_date, to_date):
    sensor_id = sensor["_id"]
    sensor_title = sensor["title"]
    step = timedelta(days=30)
    current_start = from_date
    sensor_df = pd.DataFrame()

    while current_start < to_date:
        current_end = min(current_start + step, to_date)
        data = _fetch_measurement_chunk(sensebox_id, sensor_id, current_start, current_end)

        if len(data) == 10000 and step > timedelta(days=1):
            step = timedelta(days=max(step.days // 2, 1))
            continue

        if data:
            batch_df = _parse_measurements_to_df(data, sensor_title)
            sensor_df = pd.concat([sensor_df, batch_df], ignore_index=True)

        current_start = current_end
        step = timedelta(days=30)

    if not sensor_df.empty:
        sensor_df["createdAt"] = pd.to_datetime(sensor_df["createdAt"], utc=True)
        sensor_df = sensor_df.set_index("createdAt")
        sensor_df = sensor_df[~sensor_df.index.duplicated(keep='first')]
        sensor_df = sensor_df.sort_index()

    return sensor_df, sensor_df.index.to_list() if not sensor_df.empty else []


def _fetch_measurement_chunk(sensebox_id, sensor_id, from_dt, to_dt):
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}/data/{sensor_id}"
    params = {
        "format": "json",
        "from-date": from_dt.isoformat().replace("+00:00", "Z"),
        "to-date": to_dt.isoformat().replace("+00:00", "Z")
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ Fehler bei Datenabruf: {from_dt} – {to_dt}")
        return []
    return response.json()


def _parse_measurements_to_df(data, column_name):
    records = []
    for entry in data:
        try:
            records.append({
                "createdAt": entry["createdAt"],
                column_name: float(entry["value"])
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def get_last_timestamp_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file, parse_dates=["createdAt"])
        if not df.empty:
            last_ts = df["createdAt"].max()
            # Sicherheitshalber manuell in datetime umwandeln
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts)

            # Wenn Zeitzone fehlt, UTC setzen
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)

            # 1 Sekunde addieren, um Duplikate zu vermeiden
            return (last_ts + timedelta(seconds=1)).isoformat()
    except FileNotFoundError:
        return None
    return None

def main():
    last_timestamp = get_last_timestamp_from_csv(CSV_FILE)
    if last_timestamp:
        print(f"Letzter gespeicherter Zeitstempel: {last_timestamp}")
    else:
        print("Keine Daten in der CSV-Datei gefunden.")

    combined_df, last_timestamp, sensor_timestamps = download_measurements_to_dataframe(
        SENSEBOX_ID,
        from_date=last_timestamp
    )

    if not combined_df.empty:
        print(f"Neue Daten bis: {last_timestamp}")

        # Index zurücksetzen, damit 'createdAt' als Spalte gespeichert wird
        combined_df = combined_df.reset_index()

        # Datei existiert? Dann Header weglassen
        write_header = not os.path.exists(CSV_FILE)

        combined_df.to_csv(
            CSV_FILE,
            mode='a',
            index=False,
            header=write_header
        )
        print(f"✅ Neue Daten an '{CSV_FILE}' angehängt.")
    else:
        print("Keine neuen Daten zum Anhängen gefunden.")

def run_loop(duration_minutes=5, interval_seconds=10):
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(minutes=duration_minutes)

    while datetime.now(timezone.utc) < end_time:
        print(f"Starte Datencheck um {datetime.now(timezone.utc).isoformat()}")
        main()  # deine bestehende main-Funktion, die Daten lädt und speichert
        print(f"Warte {interval_seconds} Sekunden bis zum nächsten Check...\n")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    run_loop()

