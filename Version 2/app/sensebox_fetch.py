import requests
import json
import pandas
from sqlalchemy import create_engine, text
import os
import time

sensebox_id = "6252afcfd7e732001bb6b9f7"
url = f"https://api.opensensemap.org/boxes/{sensebox_id}?format=json"

def fetch_data():
    result = requests.get(url)
    content = result.json()
    sensor = content["sensors"]
    return pandas.json_normalize(sensor)

def write_sensor_data(engine, box_id, df):
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO sensor_data (
                    timestamp, box_id, sensor_id, measurement,
                    unit, sensor_type, icon
                ) VALUES (
                    :timestamp, :box_id, :sensor_id, :value,
                    :unit, :sensorType, :icon
                )
                ON CONFLICT (timestamp, box_id, sensor_id) DO NOTHING;
            """), {
                'timestamp': row['lastMeasurement.createdAt'],
                'box_id': box_id,
                'sensor_id': row['_id'],
                'value': row['lastMeasurement.value'],
                'unit': row['unit'],
                'sensorType': row['sensorType'],
                'icon': row['icon']
            })

def main():
    time.sleep(10)  # gibt der DB etwas Zeit zum Starten
    df = fetch_data()
    db_url = f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    engine = create_engine(db_url)
    write_sensor_data(engine, sensebox_id, df)
    print("Daten erfolgreich gespeichert.")

if __name__ == "__main__":
    main()
