# Basis image Python ringan
FROM python:3.12-slim

WORKDIR /app

COPY prometheus_exporter.py .

# Instalasi dependensi
RUN pip install --no-cache-dir prometheus_client psutil

EXPOSE 8000

CMD ["python", "prometheus_exporter.py"]
