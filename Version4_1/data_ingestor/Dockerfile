# Verwende ein schlankes Python-Image
FROM python:3.9-slim-buster

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Installiere Systemabhängigkeiten, die für psycopg2 benötigt werden
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Kopiere die requirements-Datei
COPY requirements.txt .

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere das Datensammelskript und db_manager (da es importiert wird)
COPY sensebox.py .
COPY db_manager.py . 

# Kopiere die initiale data.csv in den Container
COPY data.csv .

# Befehl zum Starten des Datensammelskripts kontinuierlich
CMD ["python", "-u", "sensebox.py"]
