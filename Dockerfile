FROM python:3.10-slim

WORKDIR /app

ENV CHROMA_DIR=/data/chroma
RUN mkdir -p /data/chroma

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8081

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8081} --workers 1 --timeout 120 app:app"]
