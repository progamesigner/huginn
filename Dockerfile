# syntax=docker/dockerfile:1

# Images are named in FROM directly so Dependabot can track and bump them.

# base
FROM python:3.15.0-slim-bookworm AS base

# uv
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

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
    uv sync --extra anthropic --extra documents --extra openai --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra anthropic --extra documents --extra openai

# runtime
FROM base AS runtime

ENV HUGINN_SERVER_HOST=0.0.0.0 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

COPY --chown=nobody:nogroup --from=builder /app /app

USER nobody:nogroup

EXPOSE 8000

CMD ["server"]
