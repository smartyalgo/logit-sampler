# ponytail: single stage — build tools stay in the image; multi-stage it if size matters
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# llama-cpp-python builds llama.cpp from source; libgomp1 is its runtime OpenMP dep
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY tcpip_gen ./tcpip_gen

EXPOSE 8000
ENTRYPOINT [".venv/bin/python", "-m", "tcpip_gen.server"]
