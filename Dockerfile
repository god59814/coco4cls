FROM python:3.11-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/src

COPY requirements.txt pyproject.toml /workspace/
COPY src /workspace/src
COPY scripts /workspace/scripts
COPY app /workspace/app
COPY data /workspace/data
COPY .env.example /workspace/.env.example

RUN pip install --no-cache-dir -r /workspace/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
