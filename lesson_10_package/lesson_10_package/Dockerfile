FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# Generate dashboard.html, then keep serving it for Docker Desktop.
CMD ["bash", "-c", "python lesson_code.py && python -m http.server 8000 --bind 0.0.0.0"]
