# Huginn

[![CI](https://github.com/progamesigner/huginn/actions/workflows/ci.yaml/badge.svg)](https://github.com/progamesigner/huginn/actions/workflows/ci.yaml)

Huginn is a self-hosted agent harness. It runs as a long-lived FastAPI service that owns the agent loop, the tool registry, MCP client connections, and response streaming.

Plain LLM chat is stateless: you send a message, you get one back, and nothing carries over. Huginn turns that into a persistent agent service with state, tools, and memory. Conversations are stored in PostgreSQL, messages come in over multiple channels (Telegram first), and each request is routed to the right agent. New tools and skills plug in without changes to the core.

Configuration and secrets, including agent names and instructions, never live in this repository. That boundary keeps the code itself safe to publish.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
  - [Secrets](#secrets)
  - [Choosing the LLM](#choosing-the-llm)
  - [Validating a config file](#validating-a-config-file)
  - [How the config reaches production](#how-the-config-reaches-production)
  - [Scoped agent assets](#scoped-agent-assets)
  - [Personal commands and Telegram menus](#personal-commands-and-telegram-menus)
  - [Constrained subagents](#constrained-subagents)
  - [Collaboration brief](#collaboration-brief)
- [HTTP chat API](#http-chat-api)
- [Deployment](#deployment)
  - [Releasing](#releasing)
- [Identity administration](#identity-administration)
- [Development](#development)
- [Architecture](#architecture)

## Requirements

- Python 3.15
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL, for conversations, file storage and identities. Point `HUGINN_DATABASE_URL` at it and apply the migrations (see [Migrations](#migrations)) before starting the server.
- Redis, for the per-conversation locks of Telegram and the HTTP chat API, and for Telegram update deduplication. Point `HUGINN_REDIS_URL` at it.

Both URLs are optional settings, but `/readyz` checks both, so the service reports ready only when PostgreSQL and Redis are reachable. The test suite does not need either one: it starts throwaway containers when Docker is available (see [Database integration tests](#database-integration-tests)).

## Quick start

Install the dependencies and run commands inside the project environment:

```sh
uv sync                 # create .venv and install everything from uv.lock
uv run <cmd>            # run a command inside the project environment
```

The HTTP service is built by the `bin.server:create` factory. Start it locally with:

```sh
uv run server
```

### Server settings

`bin/server.py` owns the uvicorn runtime layer. It reads three optional environment variables, each with the `HUGINN_SERVER_` prefix. They are read from the real process environment only, not from `.env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HUGINN_SERVER_HOST` | `127.0.0.1` | Address to bind. |
| `HUGINN_SERVER_PORT` | `8000` | Port to listen on. |
| `HUGINN_SERVER_RELOAD` | `false` | Auto-reload on source changes; local development only. |

The `HUGINN_SERVER_` prefix is deliberate. Kubernetes injects `HUGINN_PORT=tcp://<ip>:<port>` for a service with the same name, so a bare `HUGINN_PORT` would collide with that value and break int parsing.

These deploy-layer knobs stay separate from `HuginnSettings`, which holds app behavior. `HuginnSettings` lives in `app/core/settings.py` and does read `.env`.

### Health checks

`/healthz` answers 200 as long as the process is alive. `/readyz` answers 200 when the service is ready to take traffic.

```sh
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

## Configuration

One `huginn.yaml` declares agents, MCP server connections, tool wiring, channel routing, LLM providers, and the API keys that may call the HTTP surface. Huginn validates it against the `HuginnConfig` Pydantic schema in `app/core/config/schema.py`.

`huginn.example.yaml` at the repository root is a fake-named, fully runnable copy of that schema. It is the only config file this repo ever holds. Real agent names and instructions are treated as secrets, so `huginn.yaml` itself is gitignored. CI validates `huginn.example.yaml` on every push so the schema and the example can never drift apart.

### Secrets

Secrets never sit in the YAML directly. Two indirections keep them out:

- `${ENV_VAR}` string interpolation, for values that vary per environment (a hostname, say).
- `value_from: {env: NAME}` or `value_from: {file: /path}`, which replace the whole node with an environment variable or a file's contents. Use this for anything that must not be readable in the dumped config, like `mcp_servers.<name>.auth`.

Huginn resolves both before Pydantic validation runs. A missing variable or file therefore fails loading with the offending field path (e.g. `mcp_servers.memory.auth`) instead of a stack trace.

### Choosing the LLM

Which LLM answers is a `(provider, model)` pair. Huginn resolves it the same way everywhere a model can be named: `agents.<name>`, `defaults`, `channels.<name>.router`, and `api_keys.<name>.router`.

- Naming a `model` alone means "whichever provider offers it". This was the behaviour before providers could be pinned.
- Adding `provider` demands that exact provider, and that provider must list that model. This lets one model name reach two routes (native Anthropic and a gateway, say).
- `defaults.provider` backs `defaults.model` and nothing else. An agent that names its own `model` does not inherit `defaults.provider`, because the default provider has no reason to offer a model the default never mentioned.
- A model that several providers offer cannot be reached by name alone. Huginn refuses that at startup and lists the contending providers. It does not resolve the name to whichever provider happened to load last.

### Validating a config file

The `huginn` command is the project's CLI. Run it as `uv run huginn ...` in a checkout; inside the container image it is on `PATH`, so `huginn ...` works directly. The `huginn config` and `huginn identities` commands below use it.

`huginn config validate [path]` loads a config file through the same path production uses.

- On success it prints a summary: agent names, tools, MCP servers, and channel routing. It never prints instructions content.
- On failure it exits 1 with a field-pathed error.
- When no path is given, it defaults to `HuginnSettings.config_file` (`HUGINN_CONFIG_FILE`, default `huginn.yaml`).

`huginn config schema` prints the `HuginnConfig` JSON Schema for editor completion (e.g. `yaml-language-server`).

### How the config reaches production

The real config lives in the separate deployment repository, never in this one:

1. A kustomize `configMapGenerator` turns the config file into a Kubernetes `ConfigMap`, which is mounted at `/etc/huginn/config.yaml` inside the pod.
2. The same `ConfigMap` sets `HUGINN_CONFIG_FILE=/etc/huginn/config.yaml`, which is how the server finds that file instead of the default `huginn.yaml`.
3. Secrets stay out of the config file. A kustomize `secretGenerator` builds a Kubernetes `Secret` from an env file that is not committed, plus the database and Redis URLs, and the pod receives them as environment variables. The config refers to them through `${VAR}` interpolation or `value_from`.

There is no live-reload. A config change is applied by redeploying the manifests. The generated `ConfigMap` gets a new name when its content changes, so applying it rolls the pod and the new process reads the new file.

### Scoped agent assets

Assets are the skills, commands, subagents, and collaboration brief that an agent loads from a named library. Use this section when you want an agent to pick up those files from a memory service (over MCP) or from a directory on disk.

Declare named libraries in `assets.sources`. Each agent selects its own source for `skills`, `commands`, `subagents`, and `brief`.

```yaml
assets:
  max_entries: 256
  sources:
    personal:
      kind: mcp
      server: memory
      inject:
        agent: default
      bind:
        user: memory_scope
    local:
      kind: filesystem
      root: /opt/huginn/assets
      globs:
        skill: skills/**/SKILL.md
        command: commands/*.md
        subagent: agents/*/PROMPT.md
agents:
  advisor:
    description: Advice and planning
    instructions: Help the user plan their work.
    skills: {enabled: true, source: personal}
    commands: {enabled: true, source: personal}
    subagents: {enabled: true, source: personal, max_concurrency: 1}
    brief: {enabled: false, writable: false, source: personal}
```

#### Selecting a source

- An omitted source means `default`, never the first available library.
- The first three selectors (`skills`, `commands`, `subagents`) have `enabled: false` and `allow: []` defaults.
- Once a selector is enabled, an empty allowlist offers all valid assets in that selected source and scope. A nonempty list narrows it.

#### Source kinds

| Kind | Fields | Notes |
| --- | --- | --- |
| `mcp` | `server`, `list_tool` (default `list_memory_notes`), `read_tool` (default `read_memory_note`), `globs`, `inject`, and `bind` | Default globs are `Agents/skills/**/SKILL.md`, `Agents/commands/*.md`, and `Agents/agents/*/PROMPT.md`. |
| `filesystem` | `root` and `globs` | Neither glob traversal nor symlink escapes are allowed. |

Rules for MCP sources:

- Binding resolves trusted turn identity, including the account's approved `memory_scope`. It does not assume that the display handle names an existing memory vault.
- Missing binding values withhold the library.
- Shared scopes require explicit injection.

Config interpolation and `value_from` work as elsewhere in the configuration.

#### Caching and refresh

- Discovery caches are keyed by source, kind, and the complete resolved binding. They have a 15-minute TTL and a bounded entry count.
- Concurrent refreshes share a read. Cancelling the last reader reclaims that work.
- Bodies and the fixed `Agents/AGENTS.md` path are read each time they are used.
- The public asset service's `invalidate(source)` permits controlled refresh. Inventory changes appear on the next interaction after refresh or expiry.
- Permission checks are repeated independently of inventory and menu caches.
- Config changes still require a restart.

#### Migrating from `skills.sources`

Existing `skills.sources` and legacy `skills.source` remain supported. Migrate one source at a time, preserving each name, role selector, injection, binding, and allowlist:

| Existing configuration | Asset configuration |
| --- | --- |
| `skills.source` | `assets.sources.default` |
| `skills.sources.<name>` | `assets.sources.<name>` |
| source `glob` | source `globs.skill` (copy the exact value) |
| source `kind`, `server`, `root`, `list_tool`, `read_tool`, `inject`, `bind` | same fields on the named asset source |
| `agents.<role>.skills.{source,enabled,allow}` | unchanged |

Migration rules:

- Distinct names may coexist during migration. Duplicate names across the old and new sections are rejected.
- A legacy source is never implicitly used for commands, subagents, or the brief.
- Real vault content and credentials belong outside this repository.
- Deployment config and manifests need the same migration before enabling the new capabilities.

### Personal commands and Telegram menus

A personal command is a prompt template that a user stores in their own asset source and runs as a slash command. Enable `commands` on an agent when you want users to call their own templates from Telegram or over HTTP.

#### Command files

A command is `Agents/commands/<name>.md`. Its frontmatter has a required `description` and an optional `name` and `agent`. The prompt is the body.

- Only `$ARGUMENTS` is replaced, verbatim. Without it, arguments are appended after the body.
- Built-in names are reserved.
- A template's agent must be authorized and use the same resolved source and scope as that template.
- Ambiguous names across sources are rejected.

#### Dispatch

- Telegram dispatch, personal `/help`, and menu discovery use one scoped resolver.
- Template commands enter the normal turn pipeline. They are also available to HTTP callers with a trusted asserted identity.
- Leading-slash text that names no command available to the sender remains ordinary input. Telegram still answers an unknown command itself.
- Mixed identity batches cannot borrow another sender's personal command.
- The original command remains attached to the expanded prompt for provenance. Template text cannot authorize a brief update.

#### Telegram menus

- Private menus use chat scope. Group menus use chat-member scope.
- Menus are recomputed on interaction, including after in-chat linking, and after inventory refresh.
- An external unlink or permission change is enforced at the next interaction. Menus never grant access.
- Telegram menu syntax and capacity can omit items. Those items remain available through `/help` and runtime authorization.
- Disabling commands withdraws a menu this process published at the next interaction. Menus published before a restart stay until commands are enabled again.

#### Menu synchronization details

- Menus are synchronized in the background only when an authorized agent enables commands, so a slow identity, inventory or Telegram call never delays the conversation.
- Publishing is serialized per bot, chat and member. While one run waits its turn, further requests for that menu are absorbed by it.
- Publishing deduplicates identical content and removes stale entries after a failed replacement.
- Publishing retries at most 3 times per interaction with backoff, and stops 30 seconds after the run starts.
- A failed sync remains eligible for retry.
- Shutdown waits briefly for running synchronizations, then cancels them.
- Restarting rebuilds these caches.

### Constrained subagents

A subagent is a child agent that the parent agent calls as a tool to handle one task. Enable `subagents` when an agent should hand work to a narrower prompt while staying inside the parent's limits.

#### Defining a subagent

`agent(name, task)` loads `Agents/agents/<name>/PROMPT.md`. That file has a required `description`, optional `model` and `tools`, and the instructions in the body.

- Additional descriptive metadata is retained by the asset layer.
- Execution limits, provider and MCP settings cannot come from vault frontmatter.
- `subagents.definitions.<name>` may instead define `description`, `instructions`, optional `model`, `tools`, `max_iterations`, `max_completion_tokens`, and `turn_timeout`.
- Config wins a name collision. The collision is logged as a warning when that scope's subagents are listed; the request itself does not fail.

#### Delegation rules

- Delegation requires `subagents.enabled: true`.
- The default concurrency is 1. `max_concurrency` accepts positive integers.
- Every call shares the parent's semaphore, cancellation, and monotonic deadline, and reserves at least one parent iteration to collect the answer. Queueing and prompt reads spend that budget.
- Delegation uses its reserved deadline instead of the ordinary `tool_timeout`. Other tools keep their configured timeout.
- An external MCP tool is treated as delegation only when listed in that reference's `delegate`, which must be a subset of its `allow`. Such tools are never inherited by children.

#### What a child can and cannot do

- Children cannot delegate, update the brief, exceed parent iterations or output limits, or select another provider.
- Requested child tools must be a subset of those available to the parent.
- Children receive their task and instructions. They do not receive parent history, attachments, conversation document grants, or a separate stored conversation.
- Only completed final text returns to the parent. Intermediate drafts do not stream to the reader.
- Files a child produces are delivered with the parent's answer.
- Ordinary inherited tools can still have side effects.

#### Limits and logging

- Logs identify the parent turn and child execution without logging task or instruction text.
- There is no aggregate token or spending cap across children.
- The current streaming contract does not expose model token-usage counters.

### Collaboration brief

The collaboration brief is a short shared note, `Agents/AGENTS.md`, that the agent reads at the start of every turn. Enable it when the agent should carry recent agreements with the user from one conversation to the next.

The brief contains recent agreements, summaries and links. Current task progress belongs in Muninn's per-scope `HEARTBEAT.md`, which only its `update_task_heartbeat` tool writes. Detailed decisions and durable knowledge remain in their canonical notes.

#### Reading the brief

- `brief.enabled` reads the selected scope's `Agents/AGENTS.md` afresh each turn.
- `brief.enabled` and `brief.writable` default to `false`, `brief.source` defaults to `default`, and `brief.writers` defaults to an empty list. `brief.writable` requires `brief.enabled`; a config that sets only `writable` is rejected.
- The brief follows the static system prompt's cache breakpoint. It does not modify that static prefix.
- A missing brief is acceptable.

#### Letting the agent update the brief

`brief.writable` additionally permits `update_brief`, and only against an original, explicit user request to change the brief. The request must meet all of these conditions:

- It carries a matching message identifier.
- It contains an exact quote, within one sentence, that names `AGENTS.md`, the collaboration brief or 協作簡報 together with a change verb.

Huginn reads the sentence whole:

- A negation, condition, report or question before the brief is named denies the request. So does a later clause that retracts the change.
- Text after a colon or inside quotation marks is content to record.
- Polite forms such as 「可以幫我更新 AGENTS.md 嗎？」 still count.
- The check errs towards denying, so a request mixing content and instruction may need restating.
- Discussion, template text, tool output and child tasks do not authorize writes.

This language check is a conservative policy aid, not an infallible natural-language authorization mechanism. The server's scope, writable setting, managed tool boundary and version check are separate enforced restrictions.

Huginn blocks other ways of changing the file:

- Raw MCP brief wrappers are blocked.
- Generic memory mutations or property edits targeting the root `Agents/AGENTS.md` (or moving `Agents`) are blocked. Huginn assumes Muninn's default `Agents` directory.
- Other memory writes remain available.

#### Who may ask for a change

Who may ask for a change depends on the source binding.

| Source binding | Brief ownership | Whose request counts |
| --- | --- | --- |
| Binds a personal value (`account`, `actor`, `handle` or `memory_scope`) | The brief belongs to the sender. | Only the sender's own request. |
| Any other binding | The brief is shared. | The brief stays read-only unless `brief.writers` lists the handles whose requests may change it. |

#### Write requirements

- Writes require Muninn's versioned `update_agents_file` wrapper. It accepts `content`, `expected_version` and `expected_absent` alongside fixed `agent` and `user` bindings, and returns the new `version`.
- Existing reads must return a `version` when requested with `include_version: true`. Huginn negotiates this field through the read tool schema and preserves structured MCP metadata and error codes.
- Updates replace the entire file. Conflicts preserve it.
- The client rejects more than 200 lines or 65,536 UTF-8 bytes without truncation.
- The client audits success or failure against the originating message.
- The client disables writes when the service lacks this capability.
- The matching Muninn implementation is on its `feat/m7-briefs` branch and has not shipped yet. Roll it out before enabling writes.

#### Measurement

- The brief's byte/character count is logged without its content.
- Prefix stability is tested with fake models.
- Actual cached-token hits and incremental token cost require a later measurement against the selected provider. No paid model calls are part of ordinary tests.

#### Integration test

The integration test uses a locally built Muninn binary, independent loopback servers sharing a disposable vault, and a scripted model (no provider calls):

```bash
MUNINN_TEST_BINARY=/absolute/path/to/muninn/target/debug/muninn \
  uv run pytest tests/integration/app/features/briefs/test_muninn_brief.py -q
```

- Without that variable the test skips. It never falls back to a remote server.
- The test clears the server environment.
- It checks creation, next-turn reads, approved memory bindings, read-only enforcement, managed-path protection, and conflicting updates/initial creations across independent servers.
- Build the binary from Muninn's matching `feat/m7-briefs` branch before running it.

## HTTP chat API

A browser front end talks to Huginn over HTTP and receives the answer as a stream of Server-Sent Events. Three surfaces sit on the same service, the same authentication, and the same turn pipeline. Only the wire dialect differs, so an agent gains all three at once.

| Surface | Path | Shape |
| --- | --- | --- |
| Native | `POST /v1/chat` | `event: opened \| routed \| delta \| tool \| attachment \| unfinished \| done \| error`, each with a JSON payload |
| OpenAI-compatible | `POST /openai/v1/chat/completions` | `data: {chat.completion.chunk}` … `data: [DONE]`, or one `chat.completion` body when `stream` is false |
| Anthropic-compatible | `POST /anthropic/v1/messages` | `message_start` / `content_block_delta` / `message_delta` / `message_stop`, or one `message` body when `stream` is false |

### Using the compatible surfaces

Point an existing SDK at the compatible surfaces by base URL: `https://host/openai/v1` for the OpenAI client, `https://host/anthropic` for the Anthropic one.

- Both surfaces read `model` as an agent name when it matches an agent the caller's key may use. A client whose only knob is a model picker chooses an agent this way.
- Any other `model` value is left alone, because the agent's configuration decides which LLM answers and the caller does not.
- Both surfaces ignore the message history a client resends and answer from the transcript this service already keeps.
- The thread is named by `conversation`, or by `user` when a drop-in client has no field for it.
- The native surface's first event reports the id a request was given.
- A turn nobody named a conversation for gets a fresh one.

### Authentication and refusals

Callers authenticate with `Authorization: Bearer <token>` against the `api_keys` config section. A key carries:

- the agents it may reach;
- optionally a default `agent`;
- optionally a `router`, where an LLM picks among the key's agents, with `fallback` when it cannot;
- optionally `tools.deny`, to take tools away from what the agent is configured with.

Every refusal lands before the stream opens, because an SSE response has spent its status line by the time the first frame is out.

| Status | Cause |
| --- | --- |
| `401` | The token is unknown or missing. |
| `403` | The key may not use the agent. |
| `400` | The key pins no agent and cannot route. |
| `409` | The conversation is already answering. |

Huginn allows one question at a time per conversation. A second question is rejected and not queued, because a rejection is what a web client can actually show. A failure after the first frame arrives as a terminal `error` event instead, in each dialect's own idiom.

### Unfinished turns

A turn can answer and still stop short of finishing. This happens when it spends its `turn_timeout`, or every iteration that `max_iterations` allowed. Each limit is set per agent, falls back to `defaults`, and then to 300 seconds and 10 iterations.

Such a turn is not a failure. The answer stands, and the caller is told why it ends where it does. Each surface says so in its own idiom, because only the surface knows how its readers hear it:

| Surface | How it reports an unfinished turn |
| --- | --- |
| Native | Emits an `unfinished` event naming the ending before `done`. |
| Compatible | Carry it on the stop reason they already have (`finish_reason: length`, `stop_reason: max_tokens`). |
| Telegram | Appends a plain sentence inviting the reader to ask for the rest. |

The note is never folded into the transcript. It is written for the reader and is not remembered as the agent's own words.

### Sending files

The native request takes `attachments`. Each attachment is one of:

- `{"blob": "<id>"}`, naming a file uploaded with `POST /v1/blobs`. The upload uses the multipart field `file` and is answered `201` with its `id`, `kind`, `mime` and `size`.
- `{"data": "<base64>", "name": "..."}`, inline, for anything up to 1 MB.

The compatible surfaces read the parts their SDKs already send:

- OpenAI: `image_url` as a `data:` URI, `input_audio` and `file`.
- Anthropic: `image` and `document` blocks with a `base64` source.

Huginn hands these parts to the same pipeline Telegram uses, so a voice note is transcribed and a document read before the turn starts. What a file is comes from its bytes, never from the label on the request.

These refusals also land before the stream opens:

| Status | Cause |
| --- | --- |
| `400` | A link to a remote file. Huginn refuses it and does not fetch it. |
| `415` | An unreadable type. |
| `413` | A file over its kind's limit. |
| `404` | A blob uploaded with another key. |

An upload no turn ever uses lapses with the `blobs.retention.generated` period.

### Receiving files

A file a tool produced (an image, a voice note, a document) never travels inside the stream.

- The native surface announces it with an `attachment` event carrying `blob`, `kind`, `mime`, `size`, `name` and a relative `url`. The bytes come from `GET /v1/blobs/{id}` with the same bearer token.
- A key can fetch only what was produced in its own turns or uploaded with it. Anything else answers `404`, never `403`, so a guessed id learns nothing.
- The compatible surfaces have no wire shape for a file. There the file arrives as a Markdown link appended to the answer text, because their SDKs would not understand an event.

### Streaming details

- A keepalive comment goes out every 15 seconds so an idle proxy does not close the connection during a long tool call.
- `Cache-Control: no-cache` and `X-Accel-Buffering: no` keep an ingress from buffering the stream back into one late answer.
- A client that hangs up cancels the turn it was reading, so the turn does not keep spending tokens on nobody's behalf.

## Deployment

The service ships as a container image at `ghcr.io/progamesigner/huginn`, built from the multi-stage `Dockerfile` at the repository root. The image is multi-arch (`linux/amd64` and `linux/arm64`), so a mixed cluster pulls the right architecture per node from a single manifest.

The runtime stage runs as a non-root user and defaults `HUGINN_SERVER_HOST` to `0.0.0.0` so the process binds inside the container. Override the `HUGINN_SERVER_` knobs (see [Server settings](#server-settings)) through the environment at deploy time.

### Image builds in CI

- CI builds the image on every push and pull request. This surfaces packaging breakage (a missing arm64 wheel, a broken layer) as early as any other check.
- The build waits for `lint` and `test`, so a commit that fails either never reaches the registry.
- The build only pushes on `main` and on release tags. Pull requests build without pushing.
- Every image pushed to the registry (from `main` or a release tag) is tagged `sha-<short>` for its commit. A push to `main` additionally moves `latest`, and a release tag adds its semantic versions.

### Building and inspecting the image

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

A release is a signed tag on `main`. Pushing the tag is the whole procedure, and there is nothing to click afterwards. `pyproject.toml` holds the version of record, and CI refuses to publish a tag that disagrees with it, so the bump and the tag travel together:

```sh
# set version = "X.Y.Z" in pyproject.toml and commit it, then
git tag -s vX.Y.Z -m 'Release vX.Y.Z'
git push origin vX.Y.Z
```

Tag `vX.Y.Z` publishes these image tags, alongside the usual `sha-<short>`:

| Image tag | Kind | Behavior |
| --- | --- | --- |
| `X.Y.Z` | Immutable | Names this release. |
| `X.Y` | Floating | Follows its patch releases. |
| `X` | Floating | Follows its minor releases. Withheld below `1.0.0`, where a floating `0` would claim compatibility across every minor release the project has yet to make. |

Tag naming rules:

- The leading `v` is stripped, so `v1.2.3` publishes `1.2.3` and not `v1.2.3`. That part matches how `progamesigner/muninn` names its release images. The floating tags are huginn's own, and muninn publishes neither.
- `vX.Y.Z` is the only accepted spelling. A tag CI cannot read that way fails and does not publish.
- Pre-release tags are deliberately absent. PEP 440 writes `0.1.0rc1` where semver writes `0.1.0-rc.1`, and no single `pyproject.toml` version can match both spellings. `sha-<short>` already names an exact build to try.

`latest` tracks `main`, not the newest release, and a tag does not move it. The workflow sets `flavor: latest=false` for this reason, since the default `latest=auto` would move `latest` onto every semver tag. Note that `ghcr.io/progamesigner/muninn:latest` means the newest *release*, so the two images do not read the same way.

Publishing a release does not deploy it. The manifests repository pins an explicit tag through kustomize `newTag`, so a rollout is a separate, deliberate change there. That change should pin an immutable `X.Y.Z` and not a floating `X.Y` or `X`, so that a pinned deployment cannot drift underneath itself.

## Identity administration

Huginn links external identities (a Telegram user, or a user ID asserted by an HTTP application) to a canonical account with its own memory scope. Registration is invitation-only: an operator chooses the target and approves the memory ownership before a recipient claims an account.

### Prerequisites

Identity linking requires PostgreSQL and a dedicated HMAC secret of at least 32 bytes:

- `HUGINN_DATABASE_URL` points to the migrated Huginn database.
- `HUGINN_IDENTITY_LINK_SECRET` is generated independently of database and legacy registration credentials. Store it in the deployment secret manager; never log or commit it.

### Issuing registration invites

```sh
# Verify that this user scope is unused in Muninn before authorizing fresh memory.
huginn identities registration-invite primary operator --new-user example-user --confirm-new-scope
# Continue an unowned, existing memory scope; the recipient chooses their handle.
huginn identities registration-invite http:web operator --existing-scope prior-memory
# Revoke an unused registration or link invite by the printed UUID.
huginn identities revoke <invite-id>
```

Memory options:

- `--new-user` locks both the approved handle and its new memory user scope. The operator must check Muninn before passing `--confirm-new-scope`: absence from Huginn's database does not prove that a vault is unused.
- `--existing-scope` locks the existing scope instead.
- These options are mutually exclusive.
- Neither option may name memory already owned by another Huginn account. That account must issue a link invite.
- Reserved names such as `default`, `shared`, `admin`, `system`, `root`, and `anonymous` cannot be new scopes or handles. An operator may explicitly authorize an unowned existing scope with such a name.
- The reserved names apply to account handles and to the Muninn user scope they create. They do not restrict the Muninn `agent` key or scopes that the config injects, such as a shared library with `inject: {agent: default, user: default}`.
- Live invitations reserve their scope until expiry or revocation.
- Registration expires in 10 minutes by default, with a maximum lifetime of 1 day.

Targets and credentials:

- Targets are configured channel names or API key names. Use `channel:<name>` or `http:<key-name>` when a name is ambiguous.
- An invite cannot be redeemed through a different provider or realm.
- Only the operator CLI issues registration invites. Self-service `/invite` always links the sender's existing account.
- The credentials are printed once, and only their HMAC digest is stored.

### Registering from Telegram

In a Telegram private chat, send `/register <code>` or open `https://t.me/<bot>?start=register_<token>`, then send `/iam <handle>` to finish registration.

| Command | Effect |
| --- | --- |
| `/register <code>` | Starts registration with an invite code. |
| `/iam <handle>` | Finishes registration. On a linked identity, `/iam` reports its existing handle. |
| `/identity` | Reports the link. |
| `/invite <target>` | Issues a 10-minute link code. |
| `/displayname <name>` | Updates the display name independently. |

Identity commands refuse group chats and answer "Manage your identity in a private chat." A group member registers or links in a private chat with the bot. The identity belongs to the Telegram user rather than the chat, so the same person is recognized in groups afterwards. Legacy `link_<token>` deep links remain supported.

### Registering over HTTP

HTTP clients use the same bearer token as chat and send `X-Huginn-User` with the application's stable user ID (1 to 256 printable characters). Identity resolves as `(http, API key name, asserted ID)`. The API key holder is trusted to assert this ID; the header is not an individual login credential.

- `POST /v1/identity/claim`: `{"code":"...","handle":"example-user","display_name":"Example Person"}`.
- `POST /v1/identity/link`: `{"code":"..."}` for an existing account's link invite.
- `GET /v1/identity`: show the current link.
- `POST /v1/identity/invite`: `{"target":"primary"}` to link your own account.
- `PATCH /v1/identity`: `{"display_name":"New Display Name"}`.

### Redemption rules

- Claim and link requests reject extra fields, so recipients cannot change the invite's target or memory mode.
- Account creation, external identity, memory binding, and invite consumption commit together.
- Repeated successful redemption by the same external identity is safe. Another identity cannot reuse the code.
- Expected registration failures leave an audit record without gaining memory access.
- Handles use `[a-z0-9._-]{1,64}`, excluding `.` and `..`. Migration refuses incompatible existing handles before altering the schema.

### Asserted users on the chat surfaces

- All three chat surfaces honor `X-Huginn-User`.
- Asserted users get separate conversation namespaces within the application. Compatible `user` still names the conversation when `conversation` is absent.
- Omitting the header retains the existing machine actor and conversation namespace.
- File grants remain shared within an API key, including uploads and generated files. This feature does not provide individual file isolation.
- The application must enforce its own user authentication and file access policy.

### Memory scope binding

- Accounts retain their approved memory user scope.
- Binding `user: memory_scope` selects it explicitly. Deployed `user: handle` bindings also use that approved scope.
- Other `handle` bindings retain the actual handle.
- The agent axis remains controlled by source configuration.
- Unlinked senders have no personal memory binding.
- Handle renaming and memory migration are not supported.

### Administering accounts directly

Operators can still administer canonical accounts and existing links directly:

```sh
huginn identities user create alice "Alice Example"
huginn identities invite alice telegram primary 10m operator --bot-username huginn_bot
huginn identities list alice
huginn identities unlink telegram primary 123456789
```

### Legacy registration cutover

1. Apply all Huginn migrations and create the required canonical users.
2. Do not import Windmill `c_agent_channel.users[]` or any other legacy channel-to-user mapping.
3. Deploy the new identity resolver. Unlinked senders remain isolated anonymous principals and are not denied access merely because they are unlinked.
4. Issue a fresh invite to each user and have them relink through the vendor-verified account.
5. Verify links with `huginn identities list <handle>`, then retire the legacy registration flow and its shared registration secret.

## Development

### Dependencies

Dependencies go through uv. Add them with `uv add <pkg>==<version>`, always pinned to an exact version, never a floating specifier like `^` or `latest`. `pyproject.toml` states what the project wants, and `uv.lock` records the exact tree that satisfies it. Both belong in git.

### Tests

Tests run with `uv run pytest`.

| Directory | Marker | External resources |
| --- | --- | --- |
| `tests/unit/` | No marker. | None. These tests must pass without network or database. |
| `tests/integration/` | `pytestmark = pytest.mark.integration` at file level. | May require real resources (PostgreSQL, Redis, network). Deselect them with `-m 'not integration'`. |

Async tests need no decorator: `asyncio_mode = "auto"` treats every `async def test_*` as an asyncio test.

### Database integration tests

The database integration tests apply the Alembic migrations to a scratch database once. They then wrap each test in a transaction that is rolled back on teardown, so no test leaves state behind.

The tests resolve their database three ways, in order:

1. An explicit `HUGINN_DATABASE_URL`. This is the CI path.
2. Otherwise, a throwaway `postgres:16` container started via [testcontainers](https://testcontainers.com/) when Docker is reachable.
3. Otherwise, they skip.

So locally, with Docker available, `uv run pytest -m integration` just works: the container is created and torn down automatically. The devcontainer enables the Docker socket through the `docker-outside-of-docker` feature (rebuild the container once to pick it up).

To run them against your own PostgreSQL instead of a spun container, set the url explicitly:

```sh
HUGINN_DATABASE_URL=postgresql+asyncpg://huginn:huginn@localhost:5432/huginn_test uv run pytest -m integration
```

### Migrations

Migrations are organized under `migrations/<domain>/revisions/`. Each domain maps to one fixed `(database connection, PostgreSQL schema)` pair in `migrations/env.py`.

The current lineage lives in `migrations/default/revisions/`. Its revisions explicitly target the `public` schema and store their history in `public.migrations`. Apply it against `HUGINN_DATABASE_URL` with:

```sh
uv run alembic upgrade default@head
uv run alembic downgrade default@base
```

A future domain or schema gets its own revision directory, root branch label, and version-table schema. Add that directory to Alembic `version_locations`; do not select the target through runtime `-x database` or `-x schema` arguments.

### Type checks

Types are checked with `uv run ty check`. All rules run at error severity.

### Reproducing CI locally

CI runs the same checks as a local run. To reproduce the whole pipeline before pushing:

```sh
uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run ty check && uv run pytest -m 'not integration'
```

## Architecture

This section describes how the code is organized and which rules the architecture tests enforce. Read it before you add a provider, a feature, or an extension.

### Dependency injection container

Runtime objects are wired through a homemade DI container in `lib/container`.

- `Container` itself only knows bind, resolve, contribute and collect, and lifecycle. It has no request scopes.
- The application object `App` in `app/core/application.py` extends it with `build(*providers)`, which registers providers in dependency order, and `running_in_mode(*modes)` for mode checks.
- Long-lived resources are bound as shared managers. The lifespan opens the container on startup and closes created instances in reverse order on shutdown.
- Endpoints receive container objects with `Depends(Provide(SomeType))` from `app.core.dependencies`.

`create()` in `bin/server.py` builds `App().build(SettingsProvider(), BootstrapProvider(), HuginnProvider(), HuginnEndpointProvider())` and returns the FastAPI app that `RouterProvider` assembles. `bin.server:create` is the sole ASGI factory entrypoint.

### Providers

- Each provider subclasses `ProviderContract` with a `__call__(container)` method and an optional `dependencies` tuple of provider classes.
- A provider takes no constructor arguments. Options come from `HuginnSettings`, so a provider wires the same whether it is listed or pulled in transitively.
- `SettingsProvider` binds settings only when none are bound yet. A test uses this to pin its own settings before calling `build()`.
- Providers live next to what they wire: core plumbing in `app/core/providers`, each feature slice in its own `providers.py`.
- Within a feature the providers are split. A `*Provider` binds services, and a sibling `*EndpointProvider` or `*CommandProvider` only mounts routes or commands on top of it.

`app/providers/` is the single composition root above both `app/features` and `app/exts`:

| File | Role |
| --- | --- |
| `features.py`, `endpoints.py` and `commands.py` | Hold the aggregates `HuginnProvider`, `HuginnEndpointProvider` and `HuginnCommandProvider`, which compose through `dependencies` alone. |
| `contributions.py` | The only place that fills a registry. |
| `telegram.py` | Hands the Telegram transport (its gateway factory and webhook route) to the feature that uses it. |

### Registries

Registries are built once and never change.

- A provider declares an item with `container.contribute(Item, factory)`.
- The registry's owner binds the registry from `container.collect(Item)`.
- The factories run at first resolution, so `SectionRegistry`, `ToolRegistry`, `ToolSourceRegistry`, `ChannelRegistry`, `CommandRegistry` and `BlobReferenceRegistry` see every contribution whatever order the providers ran in.
- An architecture test keeps `contribute` out of everything but `app/providers/`.

### Database access

Database writes take their transaction explicitly. `DatabaseContract` offers `session()` for reads, `transaction()` for writes (commit on a clean exit, roll back on an error) and `ping()`. A service opens one and hands it to the repositories it builds.

### Swapping implementations in tests

Tests swap implementations at the container level in one of three ways:

- Build an `App` from a different provider list.
- Copy the wiring into a fresh container with `container.to(Container)`.
- Wrap a call in `container.use(SomeType, fake)`.

### Feature boundaries

Which feature may import which is read off the wiring instead of being declared beside it.

- A feature may name another only when one of its providers depends, directly or transitively, on one of the other's providers.
- The feature graph this yields must stay acyclic.
- Both rules run in `tests/architecture/test_architecture.py`. Declaring the dependency in `providers.py` is what licenses the import, so the two cannot drift.

A feature exposes its public face and nothing else:

- The public face is `config` (the sections other sections validate their references into), `contracts`, `entities`, `exceptions`, and `providers`.
- Reaching into `services`, `repositories`, `sources`, `schemas`, or any other slice fails the check, and there is no exception list.
- A feature that needs more than another's public face needs a contract on the other side.

Naming and placement inside a feature:

- A feature's contracts front its services and are named `*ServiceContract` (`TurnServiceContract`, `TranscriptServiceContract`, `BlobServiceContract`), never a repository.
- Extension points a feature invites others to implement (`Tool`, `ToolSource`, `BlobReferenceSource`) carry no `Contract` suffix.
- `agents/contracts.py` declares `Tool` and `ToolSource`. `tools`, `mcp` and `skills` implement them and contribute their sources at the root, so `agents` never imports any of them.
- A feature's `entities.py` holds domain values built on `AppEntity` (anything with an identity) or `AppValue` from `app/core/schema.py`, strict and frozen like the `lib/llm` values.
- HTTP shapes live in the feature's `schemas.py` on `AppRequest` / `AppResponse`, and the ORM records sit behind `sources/`.

### What belongs in `app/core` and `lib/`

- `app/core` stays out of the feature rules. It holds only what the whole application needs: the config mechanism, the container plumbing, and providers for HTTP, Redis, Postgres, the LLM router, the clock, and logging.
- A capability that only some features use is a feature of its own, however infrastructural it looks. It owns its config section, its providers, and its contracts. `app/features/mcp` is exactly that: it is configured by `mcp_servers` and consumed by `agents` and `skills`.
- Packages under `lib/` never import one another, which an import-linter `independence` contract enforces.
- Where one `lib/` package needs another's capability, it declares the narrow shape it uses. `lib/llm`, `lib/decision` and `lib/mcp` each take an `HTTPConnector` that `HTTPClientManager` satisfies structurally.

### Extensions

- Extensions under `app/exts` hold vendor vocabulary and transport only: clients, payload types, exceptions, and for a transport its inbound endpoints and its gateways. They never hold providers or wiring. Which file names an ext uses is left to review.
- A feature reaches an ext only from its `repositories` or `sources`, and never names a vendor payload type.
- A transport is the other direction. It implements an `app/contracts` contract, and `app/providers/` wires it to the feature, so the feature is handed `ChannelGatewayFactoryContract` and never imports the ext.
- Inbound traffic from a vendor is a feature concern. The payload shape stays in the ext, while the signature check and the dispatch live in a feature slice that can reach the domain above it.

`app/exts/telegram` plus `app/features/channels/telegram` is the working example. A payment callback belongs in the same shape: one `POST /notify/{vendor}` route, a registry mapping vendor kind to config model, and a per-vendor slice beneath it.

### Logging

Logs form a single structlog pipeline. By default it writes JSON lines on stderr, and stdout stays reserved for command output. It writes colored console lines when `configure_logging` gets `develop=True`.

How the pipeline is set up:

- `configure_logging` in `lib/observability` also bridges stdlib logging into the same processors, so third-party records come out in the same shape and carry the same request context.
- Its `capture` and `silence` parameters take logger names to route through the root pipeline or drop entirely.
- Nothing outside `lib/observability` looks a logger up, by `logging.getLogger` or `structlog.get_logger`. An architecture test holds that line.
- `LoggingProvider` owns the policy. It maps the container's `running_in_mode('dev')` to `develop`, captures the uvicorn loggers, silences uvicorn access logs, and binds `Logger` in the container.

How code gets a logger:

- Endpoints take one with `Depends(Provide(Logger))`.
- Services receive one at construction time and tag it with `.bind(component=...)`.
- Code under `lib/` is handed a logger the same way, typed as structlog's `BoundLogger` so no lib package depends on another. `configure_logging` formats the printf-style arguments those calls pass.
- `RequestContextMiddleware` binds a short `request_id` into structlog contextvars and onto the request for every request except the paths in `HUGINN_UNTRACED_PATHS` (the health probes by default). Every log line downstream carries it automatically.

What must stay out of the logs:

- Never log tokens, Authorization headers, or full message bodies.
- A webhook lands a receipt before it parses anything. That receipt carries metadata only: channel, byte count, and a truncated sha256 of the payload.
- The payload bytes reach the log on exactly one path: when the reader cannot parse them at all and nobody can fix that without seeing them.
- Vendor readers set `hide_input_in_errors`, so a logged validation failure says what was wrong without quoting what arrived. A payload gets into a log line only when a caller deliberately puts it there.
- `develop=True` is a data boundary, not a formatting switch. It drops the root level to DEBUG, where vendor SDKs log their full request bodies, including prompts and conversation history. `HUGINN_ENVIRONMENT` must therefore never be `dev` anywhere real traffic lands.

### Error handling

Errors follow four rules.

1. Expected failures and absences are return values.
   - A repository answers `T | None`.
   - A service answers `Result[T, E]` (`Ok` / `Err` from `lib/result`) whose `Err` holds an `*Error` instance.
   - `Option[T]` (`Some` / `Nothing`) is for chained operations, or for when "not found" and "found an empty value" need to stay distinct.
2. Only unrecoverable failures raise: programming errors, broken invariants, and infrastructure that cannot continue.
3. HTTP boundaries convert uniformly.
   - An endpoint calls `.unwrap()` on a `Result`, which raises an `Err`'s error.
   - Every error the API answers with subclasses `AppError` in `app/core/exceptions.py`, with a `status` and a `code` from the single `Code` table in `app/core/codespaces.py`. In that table the number is the code and the name is the error, so no two errors share one.
4. A single set of exception handlers is the only place that turns an exception into the `{code, error, message, request_id}` envelope.
   - The handlers are wired through `RouterManager`'s `exception_handlers` constructor argument.
   - Request validation failures add `details` naming where and what was wrong, never the input that arrived. Nothing else carries details.

Until `ty` narrows a generic class inside `match`, test `isinstance(result, Err)` or `isinstance(result, Ok)` first and read the other case on the fall-through, where the type arguments survive.
