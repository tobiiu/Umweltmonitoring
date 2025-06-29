# Use a slim Python image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data collection script
COPY sensebox.py .

# Command to run the data collection script continuously
# This assumes your sensebox.py has a function like `run_loop()` that handles continuous data fetching.
# Make sure your sensebox.py's `if __name__ == "__main__":` block calls `run_loop()`.
CMD ["python", "-u", "sensebox.py"]