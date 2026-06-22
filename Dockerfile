FROM python:3.11-slim

WORKDIR /app

# System deps for XGBoost (OpenMP) and pgeocode
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pricing_service.py .
COPY models/ ./models/
COPY enrich_zip.py .
# Read at startup to populate /metrics training_benchmarks (the dashboard's
# model-vs-baseline table). Without it, os.path.exists() is false and the
# section renders empty.
COPY eval_results.json .

# Warm pgeocode data cache at build time so first request is fast
RUN python3 -c "import pgeocode; pgeocode.Nominatim('us')" 2>/dev/null || true

EXPOSE 8000

CMD uvicorn pricing_service:app --host 0.0.0.0 --port ${PORT:-8000}
