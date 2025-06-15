# Use the official latest Python image from Docker Hub
FROM python:latest

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run your app
CMD ["python", "index.py"]
