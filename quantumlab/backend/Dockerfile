FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt* ./ 

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    torch \
    schnetpack \
    psutil \
    paramiko \
    cryptography \
    sqlalchemy \
    asyncpg \
    pg8000 \
    websockets \
    python-multipart

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "inference_server.py"]
