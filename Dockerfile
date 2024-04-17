FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

#CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "5", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app"]
# uvicorn --host 0.0.0.0 --port 8000 main:app
# gunicorn --bind 0.0.0.0:8000 --workers 5 --worker-class uvicorn.workers.UvicornWorker main:app