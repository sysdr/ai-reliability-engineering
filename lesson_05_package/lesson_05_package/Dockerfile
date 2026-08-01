# Official Docker Hub Python image (Debian slim).
FROM python:3.12-slim

WORKDIR /app

# Create output directories used by the lesson at runtime.
RUN mkdir -p /app/output /app/.cache

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765

# Serve the live dashboard (re-runs demo; metrics update via /api/metrics).
CMD ["python", "lesson_code.py", "--serve", "--host", "0.0.0.0", "--port", "8765"]
