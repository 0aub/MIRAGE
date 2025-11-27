# Development Dockerfile for MIRAGE Backend
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (required for Python packages and audio processing)
# Note: ffmpeg is large (~350MB with dependencies) but required for YouTube audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port (will be overridden by docker-compose)
EXPOSE ${PORT:-8000}

# Run the application with hot reload enabled
# Using uvicorn with --reload for development
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --reload"]
