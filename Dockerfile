FROM python:3.12-slim

WORKDIR /code

# libgomp1 is required by scikit-learn's OpenMP-backed estimators.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /code

ENV PYTHONPATH=/code \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/var/models

EXPOSE 8000

# Render supplies $PORT; default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
