# app/Dockerfile
# Use a slim Python image for smaller size
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for psycopg2 (even if not currently used, it's good to have)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and forecast model into the container
COPY app.py .
COPY forecast_model.py . # This line is important for forecasting

# Create a directory for models
RUN mkdir -p models # This line is important for saving trained models

# Expose the port that Dash runs on
EXPOSE 8050

# Command to run the Dash application
CMD ["python", "-u", "app.py"]