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

The HTTP service is built by the `bin.server:create` factory. Start it locally with:

```sh
uv run server
```

`bin/server.py` owns the uvicorn runtime layer. It reads three optional environment variables (real process env only, not `.env`), each with the `HUGINN_SERVER_` prefix:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HUGINN_SERVER_HOST` | `127.0.0.1` | Address to bind. |
| `HUGINN_SERVER_PORT` | `8000` | Port to listen on. |
| `HUGINN_SERVER_RELOAD` | `false` | Auto-reload on source changes; local development only. |

The `HUGINN_SERVER_` prefix is deliberate: a bare `HUGINN_PORT` would collide with the `HUGINN_PORT=tcp://<ip>:<port>` value Kubernetes injects for a same-named service, breaking int parsing. These deploy-layer knobs stay separate from `HuginnSettings` (app behavior), which lives in `app/core/settings.py` and does read `.env`.

`/healthz` answers 200 as long as the process is alive; `/readyz` answers 200 when the service is ready to take traffic:

```sh
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

Runtime objects are wired through a homemade DI container in `lib/container`; `Container` itself only knows bind, resolve, and lifecycle. The application object `App` in `app/core/application.py` extends it with `build(*providers)`, which registers providers in dependency order, and `running_in_mode(*modes)` for mode checks. Each provider subclasses `ProviderContract` with a `__call__(container)` method and an optional `dependencies` tuple of provider classes; providers are callables, so `App.build()` invokes them as `provider(self)`. Providers live next to what they wire: core plumbing in `app/core/providers`, each feature slice in its own `providers.py`, split so that a `*Provider` binds services and a sibling `*EndpointProvider` or `*CommandProvider` only mounts routes or commands on top of it. `app/providers.py` is the single composition root above both `app/features` and `app/exts`, holding only aggregate providers with a no-op `__call__` that compose through `dependencies` alone: `HuginnProvider` pulls in every feature's base wiring, `HuginnEndpointProvider` the HTTP surface, `HuginnCommandProvider` the CLI surface. Extensions under `app/exts` stay plain clients with no wiring of their own — the features that use them construct them — so the aggregates reach them transitively. `create()` in `bin/server.py` wires the container explicitly with `App().build(SettingsProvider(), BootstrapProvider(), RouterProvider(excludes=...), HuginnProvider(), HuginnEndpointProvider())` and returns the FastAPI app that `RouterProvider` assembles; `RouterProvider` stays listed there because `build()` would otherwise construct a transitive one without the probe excludes. `bin.server:create` is the sole ASGI factory entrypoint. Long-lived resources are bound as shared managers; the lifespan opens the container on startup and closes created instances in reverse order on shutdown. Endpoints receive container objects with `Depends(Provide(SomeType))` from `app.core.dependencies`. Tests swap implementations at the container level: build an `App` from a different provider list, copy the wiring into a fresh container with `container.to(Container)`, or wrap a call in `container.use(SomeType, fake)`.

Features sit in tiers, declared in `tests/architecture/test_architecture.py` and enforced on every import: adapters that plug into the domain (`channels`, `messages`, `skills`) sit above the domain itself (`agents`), which sits above the machinery it drives (`tools`). Entry gating is its own tier: `auth` turns a credential into the `ApiClient` an adapter above it is handed, and nothing but the composition root wires it. Imports go strictly downward, so the feature graph stays a DAG however many features arrive; same-tier and upward imports fail the build, and a feature absent from the tiers may not cross the feature boundary at all. A lower tier exposes its public face and nothing else — what it declares (`config`, `contracts`, `exceptions`, `types`) and how it is wired (`providers`). Reaching into `services`, `repositories`, `registry`, or any other implementation slice fails the check, and `DECLARED_EXCEPTIONS` is deliberately empty: a feature that needs more than another's public face needs a contract on the other side, not an entry in that table. There are two ways to give it one. When the consumer sits above, the provider declares the surface: `tools/contracts.py` declares `ToolFactory` — bind a configured tool, build tools from an MCP server, check a params/inject overlay — and `ToolBinder` implements it over the registry and the binding machinery, so `agents` orchestrates a turn's tools without ever seeing either. When the consumer sits below, invert instead: the consumer declares an ABC in its own `contracts.py`, the capability implements it and binds it in its own provider, and only the composition root sees both sides — `agents/contracts.py` declares `SkillToolFactory`, `skills` implements it as `SkillToolBuilder` and binds it in `SkillsProvider`, so `agents` never imports `skills`. `app/core` stays out of this: it holds only what the whole application needs — the config mechanism, the container plumbing, and providers for HTTP, Redis, Postgres, the LLM router, the clock, and logging. A capability that only some features use is a feature of its own, owning its config section, its providers, and its contracts, however infrastructural it looks; `app/features/mcp` is exactly that, configured by `mcp_servers` and consumed by `agents` and `skills`.

Extensions under `app/exts` are the outbound direction only, and hold nothing but `client`, `types`, and `exceptions` — no providers, no wiring, no routes. Inbound traffic from a vendor is a feature, not an extension: the payload shape and the outbound client stay in the ext, while the route, the signature check, and the dispatch live in a feature slice that can reach the domain above it. `app/exts/telegram` plus `app/features/channels/telegram` is the working example, and a payment callback belongs in the same shape — one `POST /notify/{vendor}` route in the feature, a registry mapping vendor kind to config model, and a per-vendor slice beneath it.

Logs form a single structlog pipeline: JSON lines on stderr by default (stdout stays reserved for command output), colored console lines when `configure_logging` gets `develop=True`. `configure_logging` in `lib/observability` also bridges stdlib logging into the same processors, so third-party records come out in the same shape and carry the same request context; its `capture` and `silence` parameters take logger names to route through the root pipeline or drop entirely. Application code does not call `structlog.get_logger` itself. `LoggingProvider` owns the policy: it maps the container's `running_in_mode('dev')` to `develop`, captures the uvicorn loggers, silences uvicorn access logs, and binds `Logger` in the container. Endpoints take one with `Depends(Provide(Logger))`, and services receive one at construction time, tagged with `.bind(component=...)`. Code under `lib/` logs like any third-party package, through `logging.getLogger(__name__)`, and the bridge picks those records up. `RequestContextMiddleware` binds a short `request_id` into structlog contextvars for every request except the health probes, and every log line downstream carries it automatically. Never log tokens, Authorization headers, or full message bodies. A webhook lands a receipt before it parses anything, and that receipt carries metadata only — channel, byte count, a truncated sha256 of the payload; the bytes themselves reach the log on exactly one path, when the reader cannot parse them at all and nobody can fix that without seeing them. Vendor readers set `hide_input_in_errors`, so a logged validation failure says what was wrong without quoting what arrived, and a payload gets into a log line only when a caller deliberately puts it there. `develop=True` is a data boundary, not a formatting switch: it drops the root level to DEBUG, where vendor SDKs log their full request bodies — prompts, conversation history and all — so `HUGINN_ENVIRONMENT` must never be `dev` anywhere real traffic lands.

Errors follow four rules. Expected failures and absences are return values: `lib/result` provides typed `Result[T, E]` (`Ok` / `Err`) and `Option[T]` (`Some` / `Nothing`), and callers are expected to consume them with `match`. Only unrecoverable failures raise: programming errors, broken invariants, infrastructure that cannot continue, expressed as a subclass of `AppError` in `app/core/exceptions.py`. HTTP boundaries convert uniformly: an endpoint unwraps an `Err` into the matching `AppError`, and a single exception handler, wired through `RouterManager`'s `exception_handlers` constructor argument, is the only place that turns an exception into the `{code, reason, message, details}` envelope, and later the only place that logs or reports it. `Option` is for chained operations or when "not found" and "found an empty value" need to stay distinct; a plain existence check still uses `T | None`, which `ty` narrows natively, and the two coexist without conflict.

Dependencies go through uv. Add them with `uv add <pkg>==<version>`, always pinned to an exact version, never a floating specifier like `^` or `latest`. `pyproject.toml` states what the project wants; `uv.lock` records the exact tree that satisfies it. Both belong in git.

Tests run with `uv run pytest`. Tests in `tests/unit/` carry no marker and must pass without external resources (no network, no database). Tests in `tests/integration/` declare `pytestmark = pytest.mark.integration` at file level and may require real resources (PostgreSQL, Redis, network); deselect them with `-m 'not integration'`. Async tests need no decorator: `asyncio_mode = "auto"` treats every `async def test_*` as an asyncio test.

The database integration tests apply the Alembic migrations to a scratch database once, then wrap each test in a transaction that is rolled back on teardown, so no test leaves state behind. They resolve their database three ways, in order: an explicit `HUGINN_DATABASE_URL` (this is the CI path); otherwise a throwaway `postgres:16` container started via [testcontainers](https://testcontainers.com/) when Docker is reachable; otherwise they skip. So locally, with Docker available, `uv run pytest -m integration` just works — the container is created and torn down automatically. The devcontainer enables the Docker socket through the `docker-outside-of-docker` feature (rebuild the container once to pick it up).

To run them against your own PostgreSQL instead of a spun container, set the url explicitly:

```sh
HUGINN_DATABASE_URL=postgresql+asyncpg://huginn:huginn@localhost:5432/huginn_test uv run pytest -m integration
```

Migrations are organized under `migrations/<domain>/revisions/`. Each domain maps to one fixed `(database connection, PostgreSQL schema)` pair in `migrations/env.py`. The current lineage lives in `migrations/default/revisions/`; its revisions explicitly target the `public` schema and store their history in `public.migrations`. Apply it against `HUGINN_DATABASE_URL` with:

```sh
uv run alembic upgrade default@head
uv run alembic downgrade default@base
```

A future domain or schema gets its own revision directory, root branch label, and version-table schema. Add that directory to Alembic `version_locations`; do not select the target through runtime `-x database` or `-x schema` arguments.

Types are checked with `uv run ty check`. All rules run at error severity.

CI runs the same checks as a local run. To reproduce the whole pipeline before pushing:

```sh
uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run ty check && uv run pytest -m 'not integration'
```

## Configuration

Agents, MCP server connections, tool wiring, channel routing, LLM providers, and the API keys that may call the HTTP surface are declared in one `huginn.yaml`, validated against the `HuginnConfig` Pydantic schema in `app/core/config/schema.py`. `huginn.example.yaml` at the repository root is a fake-named, fully runnable copy of that schema; it is the only config file this repo ever holds. Real agent names and instructions are treated as secrets — `huginn.yaml` itself is gitignored, and CI validates `huginn.example.yaml` on every push so the schema and the example can never drift apart.

Secrets never sit in the YAML directly. Two indirections keep them out:

- `${ENV_VAR}` string interpolation, for values that vary per environment (a hostname, say).
- `value_from: {env: NAME}` or `value_from: {file: /path}`, which replace the whole node with an environment variable or a file's contents — used for anything that must not be readable in the dumped config, like `mcp_servers.<name>.auth`.

Both are resolved before Pydantic validation runs, so a missing variable or file fails loading with the offending field path (e.g. `mcp_servers.memory.auth`), not a stack trace.

Which LLM answers is a `(provider, model)` pair, resolved the same way everywhere a model can be named — `agents.<name>`, `defaults`, `channels.<name>.router`, `api_keys.<name>.router`. Naming a `model` alone means "whichever provider offers it", the behaviour before providers could be pinned; adding `provider` demands that exact provider and that it list that model, which is what lets one model name reach two routes (native Anthropic and a gateway, say). `defaults.provider` backs `defaults.model` and nothing else: an agent that names its own `model` does not inherit it, because the default provider has no reason to offer a model the default never mentioned. A model that several providers offer cannot be reached by name alone — that is refused at startup with the contending providers listed, rather than resolved to whichever one happened to load last.

`huginn config validate [path]` loads a config file through the same path production uses and prints a summary (agent names, tools, MCP servers, channel bindings — never instructions content), or exits 1 with a field-pathed error. It defaults to `HuginnSettings.config_file` (`HUGINN_CONFIG_FILE`, default `huginn.yaml`) when no path is given. `huginn config schema` prints the `HuginnConfig` JSON Schema for editor completion (e.g. `yaml-language-server`).

The real config is injected at deploy time, never committed: a Windmill secret script renders it, a kustomize `secretGenerator` turns that into a Kubernetes `Secret`, and it lands mounted at `/etc/huginn/config.yaml` inside the running pod. There is no live-reload — a config change means rerunning the Windmill script, `kubectl apply` on the regenerated `Secret`, and a rollout restart to pick it up.

## HTTP chat API

A browser front end talks to Huginn over HTTP, streaming the answer with Server-Sent Events. Three surfaces sit on the same service, the same authentication, and the same turn pipeline; only the wire dialect differs, so an agent gains all three at once.

| Surface | Path | Shape |
| --- | --- | --- |
| Native | `POST /v1/chat` | `event: opened \| routed \| delta \| tool \| unfinished \| done \| error`, each with a JSON payload |
| OpenAI-compatible | `POST /openai/v1/chat/completions` | `data: {chat.completion.chunk}` … `data: [DONE]`, or one `chat.completion` body when `stream` is false |
| Anthropic-compatible | `POST /anthropic/v1/messages` | `message_start` / `content_block_delta` / `message_delta` / `message_stop`, or one `message` body when `stream` is false |

Point an existing SDK at the compatible surfaces by base URL: `https://host/openai/v1` for the OpenAI client, `https://host/anthropic` for the Anthropic one. Both read `model` as an agent name when it matches an agent the caller's key may use, which is how a client whose only knob is a model picker chooses an agent; any other value is left alone, because which LLM answers is the agent's configuration, not the caller's business. Both surfaces ignore the message history a client resends and answer from the transcript this service already keeps — the thread is named by `conversation`, or by `user` when a drop-in client has no field for it, and the native surface's first event reports the id a request was given. A turn nobody named a conversation for gets a fresh one.

Callers authenticate with `Authorization: Bearer <token>` against the `api_keys` config section. A key carries the agents it may reach, optionally a default `agent`, optionally a `router` (an LLM picks among the key's agents, with `fallback` when it cannot), and optionally `tools.deny` to take tools away from what the agent is configured with. Every refusal lands before the stream opens, because an SSE response has spent its status line by the time the first frame is out: an unknown or missing token is `401`, an agent the key may not use is `403`, a key that pins no agent and cannot route is `400`, and a conversation already answering is `409` — one question at a time per conversation, rejected rather than queued, which is what a web client can actually show. A failure after the first frame arrives as a terminal `error` event instead, in each dialect's own idiom.

A turn that answered but stopped short of finishing — it spent its `turn_timeout`, or every iteration `max_iterations` allowed — is not a failure: the answer stands, and the caller is told why it ends where it does. Each surface says so in its own idiom, because only the surface knows how its readers hear it: the native surface emits an `unfinished` event naming the ending before `done`, the compatible surfaces carry it on the stop reason they already have (`finish_reason: length`, `stop_reason: max_tokens`), and Telegram appends a plain sentence inviting the reader to ask for the rest. The note is never folded into the transcript — it is written for the reader, not remembered as the agent's own words.

Streaming details worth knowing: a keepalive comment goes out every 15 seconds so an idle proxy does not close the connection during a long tool call, `Cache-Control: no-cache` and `X-Accel-Buffering: no` keep an ingress from buffering the stream back into one late answer, and a client that hangs up cancels the turn it was reading rather than leaving it to spend tokens on nobody's behalf.

## Deployment

The service ships as a container image at `ghcr.io/progamesigner/huginn`, built from the multi-stage `Dockerfile` at the repository root. The image is multi-arch (`linux/amd64` and `linux/arm64`), so a mixed cluster pulls the right architecture per node from a single manifest.

CI builds the image on every push and pull request, which surfaces packaging breakage (a missing arm64 wheel, a broken layer) as early as any other check. The build waits for `lint` and `test`, so a commit that fails either never reaches the registry, and it only pushes on `main` and on release tags; pull requests build without pushing. Every push is tagged `sha-<short>` for the commit, `main` additionally moves `latest`, and a release tag adds its semantic versions.

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

### Releasing

A release is a signed tag on `main`, and pushing the tag is the whole procedure — there is nothing to click afterwards. `pyproject.toml` holds the version of record, and CI refuses to publish a tag that disagrees with it, so the bump and the tag travel together:

```sh
# set version = "X.Y.Z" in pyproject.toml and commit it, then
git tag -s vX.Y.Z -m 'Release vX.Y.Z'
git push origin vX.Y.Z
```

Tag `vX.Y.Z` publishes the immutable `X.Y.Z`, the floating `X.Y` that follows its patch releases, and the floating `X` that follows its minor releases, alongside the usual `sha-<short>`. The leading `v` is stripped, so `v1.2.3` publishes `1.2.3` and not `v1.2.3` — that part matches how `progamesigner/muninn` names its release images, though the floating tags are huginn's own and muninn publishes neither. `X` is withheld below `1.0.0`, where a floating `0` would claim compatibility across every minor release the project has yet to make.

`vX.Y.Z` is the only accepted spelling, and a tag CI cannot read that way fails rather than publishing. Pre-release tags are deliberately absent, because PEP 440 writes `0.1.0rc1` where semver writes `0.1.0-rc.1` and no single `pyproject.toml` version can match both spellings; `sha-<short>` already names an exact build to try.

`latest` tracks `main`, not the newest release, and a tag does not move it — which is why the workflow sets `flavor: latest=false`, since the default `latest=auto` would move `latest` onto every semver tag. Note that `ghcr.io/progamesigner/muninn:latest` means the newest *release*, so the two images do not read the same way.

Publishing a release does not deploy it. The manifests repository pins an explicit tag through kustomize `newTag`, so a rollout is a separate, deliberate change there — and it should pin an immutable `X.Y.Z` rather than a floating `X.Y` or `X`, so that a pinned deployment cannot drift underneath itself.

## Identity administration

Identity linking requires PostgreSQL and a dedicated HMAC secret of at least 32 bytes:

- `HUGINN_DATABASE_URL` points to the migrated Huginn database.
- `HUGINN_IDENTITY_LINK_SECRET` is generated independently of database and legacy registration credentials. Store it in the deployment secret manager; never log or commit it.

Create canonical users before issuing short-lived, provider- and realm-bound invites:

```sh
huginn identities user create alice "Alice Example"
huginn identities invite alice telegram primary 10m operator --bot-username huginn_bot
huginn identities list alice
huginn identities unlink telegram primary 123456789
```

The invite command prints the plaintext credential once. With `--bot-username` it prints a Telegram deep link; without it, it prints the `link_<token>` payload for the transport layer. It always prints the fallback short code. The database stores only an HMAC digest.

For the legacy registration cutover:

1. Apply all Huginn migrations and create the required canonical users.
2. Do not import Windmill `c_agent_channel.users[]` or any other legacy channel-to-user mapping.
3. Deploy the new identity resolver. Unlinked senders remain isolated anonymous principals and are not denied access merely because they are unlinked.
4. Issue a fresh invite to each user and have them relink through the vendor-verified account.
5. Verify links with `huginn identities list <handle>`, then retire the legacy registration flow and its shared registration secret.
