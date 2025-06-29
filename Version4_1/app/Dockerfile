# Verwende ein schlankes Python-Image als Basis-Image
FROM python:3.9-slim-buster

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Installiere Systemabhängigkeiten, die für psycopg2 und allgemeine Build-Tools benötigt werden
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Kopiere die requirements-Datei in den Container
COPY requirements.txt .

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere die Anwendungsdateien und das Vorhersagemodul in den Container
COPY app.py .
COPY db_manager.py .      
COPY forecast_model.py .  

# Erstelle ein Verzeichnis für Modelle
RUN mkdir -p models

# Exponiere den Port, auf dem Dash läuft
EXPOSE 8050

# Befehl zum Starten der Dash-Anwendung
CMD ["python", "-u", "app.py"]
