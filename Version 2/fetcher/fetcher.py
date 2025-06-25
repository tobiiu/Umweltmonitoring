import requests
import json
import pandas
from sqlalchemy import create_engine, text
import time

import os

db_host = os.getenv("DATABASE_HOST", "localhost")  # <- fallback nur wenn NICHT gesetzt
db_user = os.getenv("DATABASE_USER", "user")
db_password = os.getenv("DATABASE_PASSWORD", "pass")
db_name = os.getenv("DATABASE_NAME", "sensebox")

DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"

def fetch_and_store():
    sensebox_id = "6252afcfd7e732001bb6b9f7"
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}?format=json"
    result = requests.get(url)
    content = json.loads(result.content)
    sensor = content["sensors"]
    df = pandas.json_normalize(sensor)

    engine = create_engine("postgresql://postgres:postgres@sensebox-db:5432/env_monitoring")

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
                'box_id': sensebox_id,
                'sensor_id': row['_id'],
                'value': row['lastMeasurement.value'],
                'unit': row['unit'],
                'sensorType': row['sensorType'],
                'icon': row['icon']
            })
    print("✅ Daten gespeichert")

if __name__ == "__main__":
    fetch_and_store()
