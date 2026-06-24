FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy full source BEFORE pip install so the app package is included in the wheel
COPY . /code

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

RUN chmod +x /code/scripts/start.sh

ENV PYTHONPATH=/code
ENV PORT=8000

EXPOSE 8000

CMD ["/code/scripts/start.sh"]
