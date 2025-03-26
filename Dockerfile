# Use a lightweight Python image
FROM python:3.9-slim

# Install FFmpeg and clean up package cache to reduce image size
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv venv && source venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the required port (if your app runs on Flask or FastAPI)
EXPOSE 5000

# Use ENTRYPOINT to activate venv and start the app
ENTRYPOINT ["/bin/sh", "-c", "source venv/bin/activate && python app.py"]  # Change "app.py" if different
