# Use a lightweight Python image
FROM python:3.9-slim

# Install dependencies (FFmpeg) and clean up
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt first (for efficient Docker caching)
COPY requirements.txt .

# Create a virtual environment and install dependencies correctly
RUN python -m venv venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the required port (if using Flask/FastAPI)
EXPOSE 5000

# Ensure the virtual environment is activated when running the app
CMD ["/app/venv/bin/python", "app.py"]  # Change "app.py" if needed
