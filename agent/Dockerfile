FROM python:3.12-slim

ENV AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY app.py environment.py rollout_driver.py ./
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 agentcore
USER agentcore

EXPOSE 8080

CMD ["python", "-m", "app"]
