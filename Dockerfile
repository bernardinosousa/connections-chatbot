# Use the official latest Python image from Docker Hub
FROM python:latest

# Set work directory
WORKDIR /app

# Optional: Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Default command (can be overridden by docker-compose)
CMD ["python"]
