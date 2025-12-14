# Multi-stage Dockerfile for ML model training and serving

# Stage 1: Base image with Python
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Training image
FROM base AS training

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command for training
CMD ["python", "ml/train_genre_model.py"]

# Stage 3: Production/serving image
FROM base AS production

# Copy only necessary files for serving
COPY ml/ ./ml/
COPY preprocessing/ ./preprocessing/
COPY scripts/ ./scripts/

# Expose port for API (if you add a serving API later)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "ml.train_genre_model"]
