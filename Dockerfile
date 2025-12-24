# Base Image: Use a slim version for a smaller footprint and reduced attack surface
FROM python:3.9-slim

# Set Environment Variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disc
# PYTHONUNBUFFERED: Ensures logs are streamed immediately to the container logs (Critical for MLOps)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (if any). 
# We clean up the apt cache immediately to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not storing the cache
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .
COPY water_quality.csv .

# Security Best Practice: Create and switch to a non-root user
# Running as root is a security risk in production environments
RUN useradd -m mlops_user
USER mlops_user

# The command that runs when the container starts
CMD ["python", "main.py"]