import requests
import json
import pandas
from sqlalchemy import create_engine, text

sensebox_id = "6252afcfd7e732001bb6b9f7"
response_format = "json"  # optional

url = f"https://api.opensensemap.org/boxes/{sensebox_id}?format={response_format}"
result = requests.request(method='get', url=url)

content = json.loads(result.content)
sensor = content["sensors"]
df = pandas.json_normalize(sensor)

def write_sensor_data(engine, box_id, df):
    """Write sensor data to database, avoiding duplicates"""
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

engine = create_engine("postgresql://postgres:postgres@localhost:5433/env_monitoring")
write_sensor_data(engine, sensebox_id, df)