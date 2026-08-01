# Official Docker Hub Python image (Debian slim).
FROM python:3.12-slim

WORKDIR /app

# Ensure print()/logging appears immediately in `docker logs` (no TTY when detached).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create output directories used by the lesson at runtime.
RUN mkdir -p /app/output /app/.cache

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8766

# -u keeps stdout/stderr line-buffered even if ENV is overridden.
# Serve the live dashboard (re-runs demo; metrics update via /api/metrics).
CMD ["python", "-u", "lesson_code.py", "--serve", "--host", "0.0.0.0", "--port", "8766"]
