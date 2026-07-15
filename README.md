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

The HTTP service is built by the `app.entrypoints.server:create` factory. Start it locally with:

```sh
uv run server
```

`app/entrypoints/server.py` owns the uvicorn runtime layer. It reads three optional environment variables (real process env only, not `.env`), each with the `HUGINN_SERVER_` prefix:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HUGINN_SERVER_HOST` | `127.0.0.1` | Address to bind. |
| `HUGINN_SERVER_PORT` | `8000` | Port to listen on. |
| `HUGINN_SERVER_RELOAD` | `false` | Auto-reload on source changes; local development only. |

The `HUGINN_SERVER_` prefix is deliberate: a bare `HUGINN_PORT` would collide with the `HUGINN_PORT=tcp://<ip>:<port>` value Kubernetes injects for a same-named service, breaking int parsing. These deploy-layer knobs stay separate from `AppSettings` (app behavior), which lives in `app/core/settings.py` and does read `.env`.

`/healthz` answers 200 as long as the process is alive; `/readyz` answers 200 when the service is ready to take traffic:

```sh
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

Runtime objects are wired through a homemade DI container in `lib/container`; `Container` itself only knows bind, resolve, and lifecycle. The application object `App` in `app/core/application.py` extends it with `build(*providers)`, which registers providers in dependency order, and `running_in_mode(*modes)` for mode checks. Each provider subclasses `ProviderContract` with a `__call__(container)` method and an optional `dependencies` tuple of provider classes; providers are callables, so `App.build()` invokes them as `provider(self)`. `create()` in `app/entrypoints/server.py` wires the container explicitly with `App().build(*App.providers, RouterProvider(excludes=...), ProbesEndpointProvider())` and returns the FastAPI app that `RouterProvider` assembles; `app.entrypoints.server:create` is the sole ASGI factory entrypoint. Long-lived resources are bound as shared managers; the lifespan opens the container on startup and closes created instances in reverse order on shutdown. Endpoints receive container objects with `Depends(Provide(SomeType))` from `app.core.dependencies`. Tests swap implementations at the container level: build an `App` from a different provider list, copy the wiring into a fresh container with `container.to(Container)`, or wrap a call in `container.use(SomeType, fake)`.

Logs form a single structlog pipeline: JSON lines on stdout by default, colored console lines when `configure_logging` gets `develop=True`. `configure_logging` in `lib/observability` also bridges stdlib logging into the same processors, so third-party records come out in the same shape and carry the same request context; its `capture` and `silence` parameters take logger names to route through the root pipeline or drop entirely. Application code does not call `structlog.get_logger` itself. `LoggingProvider` owns the policy: it maps the container's `running_in_mode('dev')` to `develop`, captures the uvicorn loggers, silences uvicorn access logs, and binds `Logger` in the container. Endpoints take one with `Depends(Provide(Logger))`, and services receive one at construction time, tagged with `.bind(component=...)`. Code under `lib/` logs like any third-party package, through `logging.getLogger(__name__)`, and the bridge picks those records up. `RequestContextMiddleware` binds a short `request_id` into structlog contextvars for every request except the health probes, and every log line downstream carries it automatically. Never log tokens, Authorization headers, or full message bodies.

Errors follow four rules. Expected failures and absences are return values: `lib/result` provides typed `Result[T, E]` (`Ok` / `Err`) and `Option[T]` (`Some` / `Nothing`), and callers are expected to consume them with `match`. Only unrecoverable failures raise: programming errors, broken invariants, infrastructure that cannot continue, expressed as a subclass of `AppError` in `app/core/exceptions.py`. HTTP boundaries convert uniformly: an endpoint unwraps an `Err` into the matching `AppError`, and a single exception handler, wired through `RouterManager`'s `exception_handlers` constructor argument, is the only place that turns an exception into the `{code, reason, message, details}` envelope, and later the only place that logs or reports it. `Option` is for chained operations or when "not found" and "found an empty value" need to stay distinct; a plain existence check still uses `T | None`, which `ty` narrows natively, and the two coexist without conflict.

Dependencies go through uv. Add them with `uv add <pkg>==<version>`, always pinned to an exact version, never a floating specifier like `^` or `latest`. `pyproject.toml` states what the project wants; `uv.lock` records the exact tree that satisfies it. Both belong in git.

Tests run with `uv run pytest`. Tests in `tests/unit/` carry no marker and must pass without external resources (no network, no database). Tests in `tests/integration/` declare `pytestmark = pytest.mark.integration` at file level and may require real resources (PostgreSQL, Redis, network); deselect them with `-m 'not integration'`. Async tests need no decorator: `asyncio_mode = "auto"` treats every `async def test_*` as an asyncio test.

Types are checked with `uv run ty check`. All rules run at error severity.

CI runs the same checks as a local run. To reproduce the whole pipeline before pushing:

```sh
uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run ty check && uv run pytest -m 'not integration'
```

## Configuration

## Deployment

The service ships as a container image at `ghcr.io/progamesigner/huginn`, built from the multi-stage `Dockerfile` at the repository root. The image is multi-arch (`linux/amd64` and `linux/arm64`), so a mixed cluster pulls the right architecture per node from a single manifest.

CI builds the image on every push and pull request, which surfaces packaging breakage (a missing arm64 wheel, a broken layer) as early as any other check. It only pushes to the registry on `main` and on tags; pull requests build without pushing. Each push is tagged `sha-<short>` for the commit, and `main` additionally moves `latest`.

The runtime stage runs as a non-root user and defaults `HUGINN_SERVER_HOST` to `0.0.0.0` so the process binds inside the container. Override the `HUGINN_SERVER_` knobs (see [Development](#development)) through the environment at deploy time.

Build and run it locally with:

```sh
docker build -t huginn .
docker run --rm -p 8000:8000 huginn
curl http://localhost:8000/healthz
```

Inspect the published multi-arch manifest with:

```sh
docker buildx imagetools inspect ghcr.io/progamesigner/huginn:latest
```
