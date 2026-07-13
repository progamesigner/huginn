# syntax=docker/dockerfile:1

ARG BASE=python:3.14.6-slim-bookworm
ARG UV_VERSION=0.11.28

# base
FROM ${BASE} AS base

# uv
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# builder
FROM base AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_LOCKED=1 \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY --from=uv /uv /uvx /usr/local/bin/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# runtime
FROM base AS runtime

ENV HUGINN_SERVER_HOST=0.0.0.0 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

COPY --chown=nobody:nogroup --from=builder /app /app

USER nobody:nogroup

EXPOSE 8000

CMD ["python", "server.py"]
