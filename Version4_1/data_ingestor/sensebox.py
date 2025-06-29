import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import logging

# Importiere db_manager aus dem Parent-Verzeichnis oder via PYTHONPATH
# Da db_manager.py nun im 'app'-Verzeichnis liegt und 'sensebox.py' im 'data_ingestor'-Verzeichnis,
# ist ein direkter Import nicht trivial, wenn sie getrennte Container sind.
# Die sauberste Lösung ist, db_manager.py auch in den 'data_ingestor'-Container zu kopieren,
# oder die db_manager-Funktionen direkt hier zu integrieren.
# Für diese Lösung kopieren wir db_manager.py auch in den data_ingestor-Container.
import db_manager # Stelle sicher, dass db_manager.py im PYTHONPATH des Containers ist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SENSEBOX_ID = os.getenv("SENSEBOX_ID", "60d828c48855dd001cf91983")
# Pfad zur initialen CSV-Datei innerhalb des data_ingestor-Containers
# Diese Datei wird von sensebox.py beim Start gelesen und in die DB importiert.
CSV_FILE_PATH_INITIAL = "data.csv" # Relative zum WORKDIR des data_ingestor-Containers

def _get_box_data(sensebox_id):
    """Ruft Metadaten einer SenseBox von der openSenseMap API ab."""
    url = f"https://api.opensensemap.org/boxes/{sensebox_id}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Fehler beim Abrufen der Box-Metadaten für {sensebox_id}: {e}")
        return {}

def _get_sensor_measurements(sensebox_id, sensor_details, from_date, to_date):
    """
    Ruft Messdaten für einen spezifischen Sensor ab und gibt sie als Liste von Tupeln zurück,
    die direkt in die Datenbank eingefügt werden können.
    """
    sensor_id = sensor_details["_id"]
    sensor_title = sensor_details["title"]
    sensor_type = sensor_details.get("sensorType", "unknown")
    sensor_unit = sensor_details.get("unit", "")
    
    step = timedelta(days=30) # Initial step size
    current_start = from_date
    all_measurements_for_db = []

    logging.info(f"Starte Download für Sensor '{sensor_title}' (ID: {sensor_id}) von {from_date} bis {to_date}")

    while current_start < to_date:
        current_end = min(current_start + step, to_date)
        url_format = (
            "https://api.opensensemap.org/boxes/{sensebox_id}/data/{sensor_id}"
            "?from-date={from_date}&to-date={to_date}"
        )
        url = url_format.format(
            sensebox_id=sensebox_id,
            sensor_id=sensor_id,
            from_date=current_start.isoformat(timespec='seconds').replace('+00:00', 'Z'),
            to_date=current_end.isoformat(timespec='seconds').replace('+00:00', 'Z'),
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data:
                for item in data:
                    if "createdAt" in item and "value" in item:
                        try:
                            # Parse timestamp and ensure it's UTC-aware
                            timestamp = pd.to_datetime(item["createdAt"], utc=True).to_pydatetime()
                            measurement_val = float(item["value"]) if item["value"] is not None else None
                            
                            all_measurements_for_db.append(
                                (
                                    timestamp,
                                    sensebox_id,
                                    sensor_id,
                                    measurement_val,
                                    sensor_unit,
                                    sensor_type,
                                )
                            )
                        except (ValueError, TypeError) as parse_error:
                            logging.warning(f"Warnung: Wert '{item['value']}' für Sensor {sensor_title} bei {item['createdAt']} konnte nicht konvertiert werden. Überspringe Punkt. Fehler: {parse_error}")
                # If we received the maximum number of records, it's possible there's more data for this period
                # We need to adjust the step to re-fetch that segment more granularly.
                if len(data) >= 10000 and step.days > 1: # OpenSenseMap API returns max 10000 records
                    step = timedelta(days=max(step.days // 2, 1))
                    logging.info(f"Anzahl maximaler Einträge erreicht für {sensor_title}. Reduziere Schritt auf {step.days} Tage.")
                    continue # Re-run with smaller step
            
            current_start = current_end # Move to the next interval
            time.sleep(0.1) # Be kind to the API

        except requests.exceptions.RequestException as e:
            logging.error(f"Fehler beim Abrufen von Daten für Sensor {sensor_title} von {current_start} bis {current_end}: {e}")
            break # Stop processing this sensor if an error occurs

    logging.info(f"✅ Download für Sensor '{sensor_title}' abgeschlossen. {len(all_measurements_for_db)} Einträge gesammelt.")
    return all_measurements_for_db

def collect_and_store_data(box_id):
    """
    Sammelt die neuesten Sensordaten von der openSenseMap API und speichert sie in der Datenbank.
    """
    logging.info(f"\n--- Starte Datenerfassung für SenseBox ID: {box_id} ---")
    
    # Holen Sie den letzten Zeitstempel aus der Datenbank, um nur neue Daten abzurufen
    last_db_timestamp = db_manager.get_last_timestamp_from_db(box_id)
    logging.info(f"Letzter gespeicherter Zeitstempel in DB: {last_db_timestamp}")

    box_data = _get_box_data(box_id)
    sensors = box_data.get("sensors", [])

    if not sensors:
        logging.warning(f"Keine Sensoren für SenseBox ID: {box_id} gefunden. Überspringe Datenerfassung.")
        return

    logging.info(f"📦 Starte parallelen Download von {len(sensors)} Sensoren ab {last_db_timestamp}...")

    all_data_to_insert = []
    
    with ThreadPoolExecutor(max_workers=5) as executor: # Limit workers to avoid overwhelming API
        futures = {
            executor.submit(_get_sensor_measurements, box_id, sensor, last_db_timestamp, datetime.now(timezone.utc)): sensor
            for sensor in sensors
            if "_id" in sensor # Ensure sensor has an ID
        }

        for future in as_completed(futures):
            sensor_details = futures[future]
            try:
                measurements = future.result()
                if measurements:
                    all_data_to_insert.extend(measurements)
            except Exception as e:
                logging.error(f"Fehler bei Sensor '{sensor_details.get('title', sensor_details.get('_id', 'unknown'))}': {e}")
    
    if all_data_to_insert:
        db_manager.insert_measurements_into_db(box_id, all_data_to_insert)
    else:
        logging.info("Keine neuen Daten von der openSenseMap API zum Einfügen gefunden.")

def run_ingestion_loop(interval_seconds=300): # Standardmäßig alle 5 Minuten
    """
    Führt die Datenerfassung in einer Endlosschleife aus.
    """
    # Führe den initialen CSV-Import einmalig beim Start des Containers aus
    logging.info(f"Starte initialen CSV-Import aus '{CSV_FILE_PATH_INITIAL}' (falls vorhanden)...")
    db_manager.initial_load_from_csv_to_db(CSV_FILE_PATH_INITIAL, SENSEBOX_ID)
    logging.info("Initialer CSV-Import abgeschlossen.")

    while True:
        collect_and_store_data(SENSEBOX_ID)
        logging.info(f"--- Datenerfassung abgeschlossen. Nächster Lauf in {interval_seconds} Sekunden. ---\n")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    run_ingestion_loop()
