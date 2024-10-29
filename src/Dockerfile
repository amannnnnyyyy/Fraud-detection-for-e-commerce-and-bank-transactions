# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy and install only dependencies first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Run the application with Gunicorn, limiting workers and threads to reduce memory use
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers=1", "--threads=1", "--timeout=120", "serve_model:app"]
