FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite+aiosqlite:////data/voice_agent.db \
    CALLS_DIR=/data/calls

WORKDIR /srv
COPY pyproject.toml README.md policy.toml ./
COPY app ./app
COPY static ./static
RUN pip install --no-cache-dir -e . \
    && useradd --create-home agent \
    && mkdir /data && chown agent /data

USER agent
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
