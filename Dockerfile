# Multi-stage Dockerfile for ML model training and serving

# Stage 1: Base image with Python
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies (including git and git-lfs for LFS files)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

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
CMD ["python", "src/ml/train_genre_model.py"]

# Stage 3: Production/serving image
FROM base AS production

# Copy application code (including static files, templates, models, and data)
# Note: Models are in src/models/ and will be included when copying src/
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY data/ ./data/
COPY .gitattributes ./.gitattributes

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for FastAPI app
EXPOSE 8000

# Run the FastAPI web application
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
