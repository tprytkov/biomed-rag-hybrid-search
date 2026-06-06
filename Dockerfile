 
# Use a lightweight official Python runtime base image
FROM python:3.11-slim

# Set system environment processing properties
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/cache/huggingface

# Set the operational directory inside the container
WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration matrices over first to cache layers
COPY requirements.txt .

# Install dependencies directly into system space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model embedding weights during image build phase
# This prevents delays or download timeouts when the container boots up
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy the rest of the application codebase layout into the workspace
COPY . .

# Expose network tracking port 8000
EXPOSE 8000

# Default execution entrypoint command to launch the engine service
CMD ["uvicorn", "src.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
