# Huginn

[![CI](https://github.com/progamesigner/huginn/actions/workflows/ci.yaml/badge.svg)](https://github.com/progamesigner/huginn/actions/workflows/ci.yaml)

Huginn is a self-hosted agent harness. It runs as a long-lived FastAPI service that owns the agent loop, the tool registry, MCP client connections, and response streaming.

Plain LLM chat is stateless: you send a message, you get one back, and nothing carries over. Huginn turns that into a persistent agent service with state, tools, and memory. Conversations are stored in PostgreSQL, messages come in over multiple channels (Telegram first), and each request is routed to the right agent. New tools and skills plug in without changes to the core.

Configuration and secrets, including agent names and instructions, never live in this repository. That boundary is what keeps the code itself safe to publish.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## Development

```sh
uv sync                 # create .venv and install everything from uv.lock
uv run <cmd>            # run a command inside the project environment
```

Dependencies go through uv. Add them with `uv add <pkg>==<version>`, always pinned to an exact version, never a floating specifier like `^` or `latest`. `pyproject.toml` states what the project wants; `uv.lock` records the exact tree that satisfies it. Both belong in git.

Tests run with `uv run pytest`. Tests in `tests/unit/` carry no marker and must pass without external resources (no network, no database). Tests in `tests/integration/` declare `pytestmark = pytest.mark.integration` at file level and may require real resources (PostgreSQL, Redis, network); deselect them with `-m 'not integration'`. Async tests need no decorator: `asyncio_mode = "auto"` treats every `async def test_*` as an asyncio test.

Types are checked with `uv run ty check`. All rules run at error severity.

CI runs the same checks as a local run. To reproduce the whole pipeline before pushing:

```sh
uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run ty check && uv run pytest -m 'not integration'
```

## Configuration

## Deployment
