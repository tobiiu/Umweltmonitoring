import psycopg2
import pandas as pd
from datetime import datetime, timezone
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables for database connection (from docker-compose)
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "env_monitoring")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_db_connection():
    """Stellt eine Verbindung zur PostgreSQL-Datenbank her und gibt sie zurück."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logging.error(f"Fehler beim Verbinden mit der Datenbank: {e}")
        return None

def get_last_timestamp_from_db(box_id):
    """
    Ruft den neuesten Zeitstempel für eine gegebene box_id aus der Datenbank ab.
    """
    conn = get_db_connection()
    if conn is None:
        return datetime(2000, 1, 1, tzinfo=timezone.utc) # Fallback to a very old date

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT MAX(timestamp) FROM sensor_data WHERE box_id = %s;",
                (box_id,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return result[0].astimezone(timezone.utc)
            return datetime(2000, 1, 1, tzinfo=timezone.utc) # Default if no data for box_id
    except Exception as e:
        logging.error(f"Fehler beim Abrufen des letzten Zeitstempels aus der DB: {e}")
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    finally:
        if conn:
            conn.close()

def insert_measurements_into_db(box_id, measurements_list):
    """
    Fügt eine Liste von Messdaten in die 'sensor_data'-Tabelle ein.
    'measurements_list' sollte eine Liste von Tupeln sein:
    (timestamp, box_id, sensor_id, measurement, unit, sensor_type)
    """
    if not measurements_list:
        return

    insert_query = """
    INSERT INTO sensor_data (timestamp, box_id, sensor_id, measurement, unit, sensor_type)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (timestamp, box_id, sensor_id) DO NOTHING;
    """
    conn = get_db_connection()
    if conn is None:
        logging.error("Keine Datenbankverbindung für den Einfügevorgang.")
        return

    try:
        with conn.cursor() as cursor:
            cursor.executemany(insert_query, measurements_list)
        conn.commit()
        logging.info(f"✅ {len(measurements_list)} neue Datensätze in die Datenbank eingefügt.")
    except Exception as e:
        conn.rollback() # Rollback in case of error
        logging.error(f"❌ Fehler beim Einfügen von Daten in die DB: {e}")
    finally:
        if conn:
            conn.close()

def initial_load_from_csv_to_db(csv_file_path, box_id):
    """
    Lädt historische Daten aus einer CSV-Datei in die Datenbank,
    beginnend nach dem neuesten Zeitstempel in der DB.
    """
    conn = get_db_connection()
    if conn is None:
        logging.error("Keine Datenbankverbindung für den initialen CSV-Ladevorgang.")
        return False

    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file_path):
            logging.info(f"CSV-Datei '{csv_file_path}' nicht gefunden für den initialen Ladevorgang. Überspringe CSV-Import.")
            return False

        # Determine the last timestamp already in the DB for this box_id
        last_db_timestamp = get_last_timestamp_from_db(box_id)
        logging.info(f"Letzter Zeitstempel in der DB für Box {box_id}: {last_db_timestamp}")

        df_csv = pd.read_csv(csv_file_path, parse_dates=["createdAt"])
        if df_csv.empty:
            logging.info(f"CSV-Datei '{csv_file_path}' ist leer. Überspringe CSV-Import.")
            return False

        df_csv['createdAt'] = pd.to_datetime(df_csv['createdAt'], utc=True, format='ISO8601')
        df_csv.set_index('createdAt', inplace=True)
        df_csv = df_csv.loc[~df_csv.index.duplicated(keep='first')] # Remove duplicates
        df_csv = df_csv.sort_index()

        # Filter data from CSV that is newer than the last DB timestamp
        df_new_csv_data = df_csv[df_csv.index > last_db_timestamp].copy()

        if df_new_csv_data.empty:
            logging.info("Keine neuen Daten in der CSV-Datei gefunden, die neuer sind als die DB-Daten. Überspringe CSV-Import.")
            return False

        # Prepare data for insertion (timestamp, box_id, sensor_id, measurement, unit, sensor_type)
        # Note: CSV contains only 'createdAt', 'Temperature', 'PM2.5', 'Humidity', 'PM10', 'Pressure'
        # We need to infer sensor_id, unit, sensor_type. For simplicity, we'll use column name as sensor_type/sensor_id
        # and assign generic unit/type if not available.
        data_to_insert = []
        for index, row in df_new_csv_data.iterrows():
            for col_name in ['Temperature', 'PM2.5', 'Humidity', 'PM10', 'Pressure']:
                if col_name in row and pd.notna(row[col_name]):
                    # Simplified mapping for unit and type based on column name
                    unit = ""
                    sensor_type = col_name # Use column name as sensor type
                    if col_name == 'Temperature': unit = '°C'
                    elif col_name == 'Humidity': unit = '%'
                    elif col_name == 'Pressure': unit = 'hPa'
                    elif col_name == 'PM2.5': unit = 'µg/m³'
                    elif col_name == 'PM10': unit = 'µg/m³'

                    data_to_insert.append((
                        index, # timestamp
                        box_id,
                        col_name, # Use column name as sensor_id
                        float(row[col_name]),
                        unit,
                        sensor_type
                    ))
        
        insert_measurements_into_db(box_id, data_to_insert)
        logging.info(f"✅ {len(data_to_insert)} Datensätze aus '{csv_file_path}' in die Datenbank geladen.")
        return True
    except Exception as e:
        logging.error(f"❌ Fehler beim initialen Laden der CSV in die DB: {e}")
        return False
    finally:
        if conn:
            conn.close()

def load_all_sensor_data_from_db(box_id):
    """
    Lädt alle Sensordaten für eine gegebene Box-ID aus der Datenbank
    und bereitet sie für das Dashboard vor (Zeitstempel als Index, Spalten für Sensortypen).
    """
    conn = get_db_connection()
    if conn is None:
        logging.error("Keine Datenbankverbindung zum Laden aller Sensordaten.")
        return pd.DataFrame()

    try:
        query = """
        SELECT timestamp, sensor_id, measurement, unit, sensor_type
        FROM sensor_data
        WHERE box_id = %s
        ORDER BY timestamp;
        """
        df = pd.read_sql(query, conn, params=(box_id,), parse_dates=['timestamp'])
        
        if df.empty:
            logging.info("Keine Daten aus der Datenbank abgerufen.")
            return pd.DataFrame()

        # Set timestamp as index and ensure UTC
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        df = df.set_index('timestamp')

        # Pivot the table to have sensor_id as columns
        # If sensor_type is more descriptive and unique for plotting, use that.
        # For simplicity, we use sensor_id as column names as they are unique.
        df_pivot = df.pivot(columns='sensor_id', values='measurement')
        
        # Resample to common frequency (e.g., 10min) and interpolate
        # This aligns data points and fills gaps for consistent plotting
        df_resampled = df_pivot.resample('10min').mean().interpolate(method='time')
        
        logging.info(f"Daten aus DB geladen und vorbereitet. Zeilen: {len(df_resampled)}, Spalten: {list(df_resampled.columns)}")
        return df_resampled
    except Exception as e:
        logging.error(f"Fehler beim Laden von Daten aus der Datenbank für das Dashboard: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# Example usage (for testing db_manager.py directly)
if __name__ == "__main__":
    # Test initial CSV load
    SENSEBOX_ID_TEST = "60d828c48855dd001cf91983" # Use your actual SenseBox ID for testing
    CSV_FILE_PATH_TEST = "data/data.csv" # Ensure this path is correct relative to where you run this script

    print(f"\n--- Starte initialen CSV-Ladevorgang für {SENSEBOX_ID_TEST} ---")
    initial_load_from_csv_to_db(CSV_FILE_PATH_TEST, SENSEBOX_ID_TEST)

    print("\n--- Lade alle Sensordaten aus der DB ---")
    df_db = load_all_sensor_data_from_db(SENSEBOX_ID_TEST)
    if not df_db.empty:
        print("Erste 5 Zeilen der geladenen Daten:")
        print(df_db.head())
    else:
        print("Keine Daten aus der DB geladen.")

    # Example of inserting a new measurement (for testing continuous updates)
    # This would typically be done by sensebox.py
    # try:
    #     new_data = [
    #         (datetime.now(timezone.utc), SENSEBOX_ID_TEST, 'Temperature', 22.5, '°C', 'Temperature'),
    #         (datetime.now(timezone.utc), SENSEBOX_ID_TEST, 'Humidity', 60.1, '%', 'Humidity')
    #     ]
    #     insert_measurements_into_db(SENSEBOX_ID_TEST, new_data)
    #     print("\n--- Neue Testdaten eingefügt ---")
    # except Exception as e:
    #     print(f"Fehler beim Einfügen neuer Testdaten: {e}")
