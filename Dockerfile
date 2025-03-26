# Use an official Python base image
FROM python:3.9-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Create a directory for your app
WORKDIR /app

# Copy your requirements file first (for caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . /app

# Expose the port your Flask app runs on (optional, but good practice)
EXPOSE 10000

# Run the Flask app
CMD ["python", "app.py"]
