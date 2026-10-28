FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Make sure models exist even if the image is built from a clean clone
RUN test -f models/model_meta.json || python -m src.train

EXPOSE 5000
CMD gunicorn app.app:app --workers 2 --bind 0.0.0.0:${PORT} --timeout 120
