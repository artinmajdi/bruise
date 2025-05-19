FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and other necessary files
COPY pyproject.toml .
COPY README.md .
COPY MANIFEST.in .

# Install the package and its dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Copy the rest of the application (source code)
COPY src /app/src
COPY scripts /app/scripts

# Create directories for data and output if they don't exist
RUN mkdir -p dataset
RUN mkdir -p logs

# Make sure the directories are writable
RUN chmod -R 777 dataset
RUN chmod -R 777 logs

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the streamlit application (can be overridden)
CMD ["streamlit", "run", "src/claude2/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
