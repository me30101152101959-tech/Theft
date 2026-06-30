FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (retry-tolerant, won't fail build on transient network issues)
RUN apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    || echo "WARNING: apt-get install failed (network issue) - continuing build"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload directories
RUN mkdir -p uploads/model uploads/dataset uploads/exports

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
