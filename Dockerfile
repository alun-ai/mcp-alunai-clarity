FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --user --no-warn-script-location -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Default environment variables (can be overridden)
ENV MEMORY_FILE_PATH=/tmp/memory.json

# Create temporary directories
RUN mkdir -p /tmp

# Set permissions
RUN chmod +x setup.sh

ENTRYPOINT ["python", "-m", "memory_mcp"]