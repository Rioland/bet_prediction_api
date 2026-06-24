FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /code/pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY . /code
RUN chmod +x /code/scripts/start.sh

ENV PORT=8000
EXPOSE 8000

CMD ["/code/scripts/start.sh"]
