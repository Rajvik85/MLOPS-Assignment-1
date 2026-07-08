# Stage 1: Build dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Install compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to user directory to make copying to runner stage cleaner
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Minimal runtime environment
FROM python:3.13-slim AS runner

WORKDIR /app

# Ensure logs are printed immediately to stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy python dependencies from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
# Copy source files
COPY --chown=appuser:appuser src/ /app/src/
# Copy model files (if pre-trained)
COPY --chown=appuser:appuser models/ /app/models/

USER appuser

EXPOSE 8000

# Container healthcheck using built-in urllib
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
