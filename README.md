# OpenSteward MCP Server

OpenSteward is a read-only [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
server for open-source maintainers. It helps answer:

> What deserves maintainer attention, why, which expertise is required, and
> what should happen next?

It combines pull-request readiness, repository policy, decision memory,
related historical work, review-cost intelligence, maintainer attention
routing, and deterministic recommended actions.

OpenSteward supports maintainer decisions. It does **not** decide whether a
pull request should merge, and it does **not** evaluate a contributor's skill
or trustworthiness. Core evidence extraction, scoring, and routing are
deterministic. The default server does not require an LLM.

## Table of contents

- [Project overview](#project-overview)
- [Why OpenSteward](#why-opensteward)
- [Main capabilities](#main-capabilities)
- [Available MCP tools](#available-mcp-tools)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Trust and security model](#trust-and-security-model)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [GitHub App setup](#github-app-setup)
- [Environment configuration](#environment-configuration)
- [Starting the server](#starting-the-server)
- [Health and readiness](#health-and-readiness)
- [Testing with MCP Inspector](#testing-with-mcp-inspector)
- [Connecting an OpenAI Agents SDK agent](#connecting-an-openai-agents-sdk-agent)
- [Calling get_maintainer_brief](#calling-get_maintainer_brief)
- [Repository policy](#repository-policy)
- [Development](#development)
- [Testing and linting](#testing-and-linting)
- [Current limitations](#current-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Project overview

OpenSteward exposes structured maintainer intelligence over Streamable HTTP.
Its live GitHub tools authenticate as a GitHub App installation, collect
read-only evidence, and return validated results that an MCP client or agent
can explain.

The recommended complete workflow is `get_maintainer_brief`. It combines the
same underlying evidence used by the lower-level assessment, related-work, and
review-cost tools into one credential-redacted result.

## Why OpenSteward

Maintainers often need to reconstruct context before they can review a pull
request:

- Is the contribution ready for detailed review?
- Which repository rules apply?
- Did similar work succeed, get rejected, or remain unresolved?
- Which parts of the change make review expensive?
- Does the change need security, database, deployment, architecture, or
  general expertise?
- Is author action or maintainer attention the appropriate next step?

OpenSteward makes that reasoning explicit and machine-readable. It reports
evidence, weighted contributions, provenance, and completeness warnings
instead of returning an unexplained verdict.

## Main capabilities

### Contribution readiness

`assess_pull_request` retrieves live pull-request metadata, changed files,
review history, effective reviews, approvals, check runs, and the repository's
policy. The policy is loaded from the pull request's base commit, not its
contributor-controlled head commit.

The result includes:

- pull-request state, draft and mergeability evidence;
- additions, deletions, and file-collection completeness;
- review and approval counts;
- check-run summaries;
- policy provenance and whether defaults were used;
- structured policy findings and suggested next actions.

### Decision memory

`find_related_work` builds a bounded snapshot of:

- recently updated closed GitHub issues;
- recently updated closed GitHub pull requests;
- changed paths for a bounded subset of historical pull requests;
- Markdown architecture decision records (ADRs) in supported repository
  directories.

Search is lexical and deterministic by default. Optional local embeddings can
add semantic relevance, and an optional Groq reranker can rerank a bounded
top-k subset. Semantic features are disabled by default and are not required
for the server.

Every result reports source-history completeness, ranking coverage, and
result truncation. OpenSteward does not claim to retrieve complete repository
history.

### Review-cost intelligence

`assess_review_cost` derives expected maintainer effort from live evidence
using five deterministic signals:

- change size;
- change dispersion;
- risk-sensitive paths;
- validation gaps;
- historical complexity.

The score is accompanied by exact weights, weighted contributions,
explanations, evidence, reducers, and coverage warnings. Review cost means
expected maintainer effort—not code quality, contributor ability, or trust.

### Maintainer brief

`get_maintainer_brief` is the flagship tool. It combines:

- contribution readiness;
- repository-policy evidence;
- related historical work;
- review-cost intelligence;
- maintainer priority;
- specialist review routes;
- deterministic recommended actions;
- completeness warnings.

Its top-level output includes the recommendation, review routes, review-cost
score and level, number of related-work matches returned, and an overall
`complete` flag.

## Available MCP tools

All current tools are read-only. “Live GitHub” means the tool creates a
repository-scoped installation token and calls the GitHub REST API.

| Tool | Purpose | Live GitHub | Recommended use |
| --- | --- | --- | --- |
| `system_status` | Returns server name, version, environment, stage, and read-only mode. | No | First connectivity smoke test. |
| `estimate_review_cost` | Older estimator that scores seven normalized factors supplied by the caller. | No | Use when factors already exist outside OpenSteward. |
| `evaluate_repository_policy` | Evaluates caller-supplied contribution facts against supplied YAML, the server's local `.opensteward.yml`, or defaults. | No | Policy testing and local policy inspection. |
| `assess_pull_request` | Collects live PR evidence and evaluates trusted base-commit policy. | Yes | Readiness and policy analysis without historical ranking. |
| `find_related_work` | Searches bounded closed issues, closed PRs, changed paths, and ADRs. | Yes | Focused historical-context lookup. |
| `assess_review_cost` | Derives review effort from live PR, policy, checks, paths, reviews, and history. | Yes | Explainable effort analysis. |
| `get_maintainer_brief` | Combines readiness, policy, history, review cost, attention routing, and next actions. | Yes | Recommended complete PR analysis. |

The server also exposes the read-only resource
`steward://repository/policy`. It reads the active `.opensteward.yml` from the
server process's working directory and includes source metadata. It does not
fetch a remote repository policy.

The minimal live-tool inputs are:

| Tool | Required inputs |
| --- | --- |
| `assess_pull_request` | `installation_id`, `repository`, `pull_number` |
| `find_related_work` | `installation_id`, `repository`, `git_ref`, `query` |
| `assess_review_cost` | `installation_id`, `repository`, `pull_number` |
| `get_maintainer_brief` | `installation_id`, `repository`, `pull_number` |

`repository` has exactly two required fields: `owner` and `name`.
GitHub-backed tools also accept bounded optional settings visible in their MCP
schemas.

## How it works

1. An MCP caller authenticates to OpenSteward with a configured bearer token.
2. OpenSteward verifies that the caller may use the requested GitHub
   installation ID.
3. OpenSteward signs a short-lived GitHub App JWT and creates a
   repository-scoped installation token with only the read permissions needed
   by that tool.
4. OpenSteward collects bounded, read-only GitHub evidence.
5. Deterministic services construct validated structured results.
6. The MCP result is returned to the client or agent.
7. An agent may explain the structured evidence, but it must not invent
   missing evidence or turn the result into a merge decision.

Optional semantic scoring affects related-work ranking only. With semantic
scoring disabled—the default—ranking is lexical and deterministic. With it
enabled, local embeddings and optional Groq scores can affect relevance
ordering; maintainer routing and recommended-action construction remain
deterministic.

## Architecture

```mermaid
flowchart TD
    A[Agent or MCP client] -->|Streamable HTTP + bearer token| B[OpenSteward MCP server]
    B --> C[Pull-request assessment]
    B --> D[Repository policy]
    B --> E[Decision memory and related work]
    B --> F[Review-cost intelligence]
    B --> G[Maintainer attention routing]
    C --> H[GitHub App installation authentication]
    D --> H
    E --> H
    F --> H
    G --> H
    H -->|Repository-scoped read token| I[GitHub REST API]
```

The FastAPI application provides process endpoints and mounts a stateless
FastMCP Streamable HTTP application at `/mcp`. Live runners share an HTTP
client and an in-memory installation-token cache until application shutdown.
There is no database or background index.

## Trust and security model

### Read-only GitHub access

OpenSteward requests only these repository permissions:

| GitHub App repository permission | Level | Used for |
| --- | --- | --- |
| Contents | Read-only | Trusted policy files, Git trees, and ADR content |
| Pull requests | Read-only | PR metadata, files, and reviews |
| Issues | Read-only | Historical issues and PR labels used by related work |
| Checks | Read-only | Check runs for PR head commits |

Individual installation tokens are further narrowed to the requested
repository and the permissions needed by the selected tool.

OpenSteward performs no repository write operations. It does not create
comments, labels, approvals, requested changes, check runs, merges, closes, or
edits—even if policy automation fields contain other values.

### Separate trust boundaries

- The GitHub App private key stays on the OpenSteward server.
- Short-lived GitHub installation tokens stay inside the server and are
  cached only in memory.
- MCP clients receive neither the private key nor an installation token.
- MCP bearer tokens authenticate clients to OpenSteward; they are separate
  from GitHub credentials.
- Each configured MCP caller has an allowlist of GitHub installation IDs.
  Authorization is checked before a live runner executes.
- Private keys and bearer tokens use secret-backed settings and are not
  serialized into tool results.
- High-level composed outputs redact invocation and internal authentication
  data. The legacy `assess_pull_request` contract still includes the
  non-secret installation ID, but never credentials.
- Incomplete history, incomplete ranking coverage, and result truncation are
  explicit output fields and warnings.
- Repository text is treated as untrusted data by the optional Groq reranker.

The current MCP authentication mechanism is a static map of opaque bearer
tokens configured by the operator. This is appropriate for local and private
deployments. A public multi-user service would normally add standardized
OAuth/OIDC authorization, token lifecycle management, and an external secret
store. OpenSteward does not currently implement that layer.

> [!WARNING]
> Anyone with an MCP caller token can invoke tools for every installation ID
> assigned to that caller. Generate strong unique tokens, use HTTPS outside
> localhost, keep allowlists narrow, and rotate exposed credentials.

## Requirements

- Python **3.12 or newer**
- Git
- A GitHub account
- A GitHub App installed on the repositories to inspect
- Node.js and `npx` only when using MCP Inspector
- An OpenAI API key only when running the separate OpenAI Agents SDK example

OpenSteward itself does not need an OpenAI API key. Groq credentials are also
unnecessary unless optional Groq reranking is explicitly enabled.

## Quick start

### Windows PowerShell

```powershell
git clone https://github.com/NaimurRahmannn/OpenSteward-MCP-Server
cd OpenSteward-MCP-Server

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .

Copy-Item .env.example .env
```

### Unix-like shells

```bash
git clone https://github.com/NaimurRahmannn/OpenSteward-MCP-Server
cd OpenSteward-MCP-Server

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

cp .env.example .env
```

### Optional installation with uv

The repository includes a `uv.lock`, but `uv` is not required.

Windows:

```powershell
uv venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev]"
```

Unix:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
```

Next, [create the GitHub App](#github-app-setup), edit `.env`, and
[start the server](#starting-the-server).

## GitHub App setup

OpenSteward is request-driven and does not consume webhooks. No webhook secret,
callback URL, or user OAuth authorization is required.

1. In GitHub, open **Settings → Developer settings → GitHub Apps → New GitHub
   App**. For an organization-owned app, use the organization's developer
   settings.
2. Choose a unique name and an appropriate homepage URL. Disable **Active**
   under Webhook because OpenSteward does not receive webhook events.
3. Under **Repository permissions**, configure:

   | Permission | Access |
   | --- | --- |
   | Contents | Read-only |
   | Pull requests | Read-only |
   | Issues | Read-only |
   | Checks | Read-only |

   No account or organization permissions are required by the current
   runtime.
4. Choose whether only the owning account or any account may install the App.
   A private App is usually simplest for a private deployment.
5. Create the App and record its numeric **App ID**. The App ID is not the
   client ID.
6. On the App settings page, generate a private key and download the `.pem`
   file. Store it outside the repository with access limited to the server
   operator.
7. Install the App on the user or organization that owns the target
   repositories. Prefer **Only select repositories** when broad access is not
   needed.
8. Record the numeric installation ID. It appears in the installation URL
   after installation; operators can also retrieve it through GitHub's
   authenticated App installation endpoints.
9. Put the App ID, private-key path, and installation ID allowlist into
   `.env` as shown below.

GitHub references:

- [Registering a GitHub App](https://docs.github.com/en/apps/creating-github-apps/registering-a-github-app/registering-a-github-app)
- [Generating a private key](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/managing-private-keys-for-github-apps)
- [Installing your own GitHub App](https://docs.github.com/en/apps/using-github-apps/installing-your-own-github-app)
- [Finding an installation ID](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-an-installation-access-token-for-a-github-app)

## Environment configuration

Settings are read from process environment variables and a `.env` file in the
current working directory. Names are case-insensitive, but the uppercase names
below are recommended.

### Safe minimal example

```dotenv
OPENSTEWARD_APP_NAME=OpenSteward
OPENSTEWARD_ENVIRONMENT=development
OPENSTEWARD_HOST=127.0.0.1
OPENSTEWARD_PORT=8000
OPENSTEWARD_LOG_LEVEL=INFO

OPENSTEWARD_MCP_AUTHORIZED_CALLERS={"local-agent":{"token":"REPLACE_WITH_A_RANDOM_TOKEN_AT_LEAST_32_CHARACTERS","installation_ids":[12345678]}}

OPENSTEWARD_GITHUB_APP_ID=123456
OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH=C:/keys/opensteward.pem

OPENSTEWARD_GITHUB_API_URL=https://api.github.com
OPENSTEWARD_GITHUB_API_VERSION=2026-03-10
OPENSTEWARD_GITHUB_USER_AGENT=OpenSteward/0.1.0
OPENSTEWARD_GITHUB_REQUEST_TIMEOUT_SECONDS=15
OPENSTEWARD_GITHUB_RETRY_TIME_BUDGET_SECONDS=120
```

All IDs and credentials above are placeholders.

### Application and caller settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENSTEWARD_APP_NAME` | `OpenSteward` | Name returned by health and status responses. |
| `OPENSTEWARD_ENVIRONMENT` | `development` | One of `development`, `test`, or `production`. |
| `OPENSTEWARD_HOST` | `127.0.0.1` | Uvicorn bind host used by the `opensteward` command. |
| `OPENSTEWARD_PORT` | `8000` | Uvicorn bind port. |
| `OPENSTEWARD_LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `OPENSTEWARD_MCP_AUTHORIZED_CALLERS` | `{}` | JSON map of caller IDs to bearer tokens and installation-ID allowlists. |

Production mode refuses to start without at least one configured MCP caller.
Caller IDs must be non-empty and trimmed. Tokens must be unique, at least 32
characters, ASCII, trimmed, and contain no whitespace. Every caller needs at
least one positive GitHub installation ID.

Generate a caller token with Python on Windows or Unix:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### GitHub settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENSTEWARD_GITHUB_APP_ID` | unset | Positive numeric GitHub App ID. Required with either private-key setting. |
| `OPENSTEWARD_GITHUB_PRIVATE_KEY` | unset | Inline RSA or PKCS#8 PEM. Literal `\n` sequences are converted to newlines. |
| `OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH` | unset | Path to an RSA or PKCS#8 PEM file. Recommended for local deployments. |
| `OPENSTEWARD_GITHUB_API_URL` | `https://api.github.com` | HTTPS GitHub REST API base URL. |
| `OPENSTEWARD_GITHUB_API_VERSION` | `2026-03-10` | API version in `YYYY-MM-DD` form. |
| `OPENSTEWARD_GITHUB_USER_AGENT` | `OpenSteward/0.1.0` | Non-empty GitHub `User-Agent`. |
| `OPENSTEWARD_GITHUB_REQUEST_TIMEOUT_SECONDS` | `15` | Per-request timeout; greater than 0 and at most 120 seconds. |
| `OPENSTEWARD_GITHUB_RETRY_TIME_BUDGET_SECONDS` | `120` | Total transient-retry wait budget; 0 to 3,600 seconds. |

Configure exactly one of `OPENSTEWARD_GITHUB_PRIVATE_KEY` and
`OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH`. The App ID and private key must either
both be configured or both be absent.

The three identities have different roles:

- **MCP caller token:** authenticates an MCP client or agent to OpenSteward.
- **GitHub App ID:** identifies the GitHub App whose private key signs App
  JWTs.
- **GitHub installation ID:** identifies one installation of that App. It is
  passed to live tools and must appear in the caller's configured allowlist.

### Optional semantic-ranking settings

These settings affect related-work ranking in `find_related_work`,
`assess_review_cost`, and `get_maintainer_brief`. They are optional and
disabled by default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENSTEWARD_SEMANTIC_ENABLED` | `false` | Enable bounded local embedding scoring. |
| `OPENSTEWARD_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model name. |
| `OPENSTEWARD_EMBEDDING_MAX_TOKENS` | `512` | Per-query/document truncation bound. |
| `OPENSTEWARD_EMBEDDING_BATCH_SIZE` | `4` | Local embedding batch size. |
| `OPENSTEWARD_EMBEDDING_THREADS` | `2` | CPU embedding threads. |
| `OPENSTEWARD_EMBEDDING_CACHE_DIR` | unset | Optional model cache directory. |
| `OPENSTEWARD_SEMANTIC_MAX_DOCUMENTS` | `100` | Maximum lexical candidates sent to semantic scoring. |
| `OPENSTEWARD_GROQ_ENABLED` | `false` | Enable optional bounded Groq reranking. Requires semantic scoring. |
| `OPENSTEWARD_GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model identifier. |
| `OPENSTEWARD_GROQ_API_KEY` | unset | Required only when Groq reranking is enabled. |
| `OPENSTEWARD_GROQ_API_URL` | `https://api.groq.com/openai/v1` | HTTPS Groq-compatible API base URL. |
| `OPENSTEWARD_GROQ_MAX_CANDIDATES` | `10` | Maximum local candidates sent to Groq. |
| `OPENSTEWARD_GROQ_REQUEST_TIMEOUT_SECONDS` | `20` | Groq request timeout. |
| `OPENSTEWARD_GROQ_MAX_INPUT_CHARACTERS` | `24000` | Bounded reranking input size. |

The local embedding model may be downloaded on first use. If semantic scoring
is unavailable, related-work search reports fallback/coverage information
instead of silently claiming complete hybrid ranking. Groq failures fall back
to local embedding scores.

## Starting the server

The installed console command has no command-line flags. It reads host, port,
and log level from environment settings, then starts Uvicorn. Do not use
`opensteward --help`; this CLI does not implement a help screen.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
opensteward
```

Unix:

```bash
source .venv/bin/activate
opensteward
```

The defaults are `127.0.0.1:8000`. Stop the server with `Ctrl+C`.

As a troubleshooting alternative, invoke Uvicorn directly.

Windows:

```powershell
.\.venv\Scripts\python.exe -m uvicorn opensteward.app:app --host 127.0.0.1 --port 8000
```

Unix:

```bash
.venv/bin/python -m uvicorn opensteward.app:app --host 127.0.0.1 --port 8000
```

The direct Uvicorn command uses the explicit flags shown; the installed
`opensteward` command uses the `OPENSTEWARD_HOST`,
`OPENSTEWARD_PORT`, and `OPENSTEWARD_LOG_LEVEL` settings.

## Health and readiness

| Endpoint | Authentication | Purpose |
| --- | --- | --- |
| `GET /health` | None | Process-level liveness and package identity. |
| `GET /ready` | None | Sanitized configuration and external-dependency readiness. |
| `/mcp` | Bearer token | Stateless Streamable HTTP MCP endpoint. |

### Health

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Unix:

```bash
curl http://127.0.0.1:8000/health
```

Response shape:

```json
{
  "status": "ok",
  "name": "OpenSteward",
  "version": "0.1.0"
}
```

`/health` remains available even when GitHub or authentication dependencies
are not ready.

### Readiness

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Unix:

```bash
curl http://127.0.0.1:8000/ready
```

Ready response shape:

```json
{
  "status": "ready",
  "environment": "development",
  "checks": {
    "mcp": "ready",
    "mcp_authentication": "ready",
    "github_credentials": "ready",
    "github_api": "ready",
    "github_installations": "ready"
  },
  "issues": []
}
```

Readiness validates:

- MCP caller authentication is configured;
- GitHub App configuration and private-key signing work;
- GitHub accepts the App identity;
- each installation assigned to a caller exists;
- each installation grants Contents, Pull requests, Issues, and Checks access.

A non-ready result uses HTTP `503`, sets individual checks to `not_ready` or
`not_checked`, and includes sanitized messages in `issues`. GitHub probes are
cached for 30 seconds.

## Testing with MCP Inspector

Install Node.js, start OpenSteward, then launch the official Inspector:

```bash
npx @modelcontextprotocol/inspector
```

In the Inspector UI, configure:

- **Transport:** Streamable HTTP
- **URL:** `http://127.0.0.1:8000/mcp`
- **Authorization:** `Bearer YOUR_MCP_CALLER_TOKEN`

Use the configured OpenSteward caller token—not the Inspector proxy session
token and not a GitHub token. If the UI presents custom headers instead of a
bearer-token field, add:

```text
Authorization: Bearer YOUR_MCP_CALLER_TOKEN
```

Suggested smoke-test order:

1. `system_status`
2. `assess_pull_request`
3. `find_related_work`
4. `assess_review_cost`
5. `get_maintainer_brief`

For `find_related_work`, provide a Git ref and a query, for example:

```json
{
  "installation_id": 12345678,
  "repository": {
    "owner": "NaimurRahmannn",
    "name": "OpenSteward-MCP-Server"
  },
  "git_ref": "main",
  "query": {
    "text": "authentication retry behavior"
  }
}
```

For the flagship tool:

```json
{
  "installation_id": 12345678,
  "repository": {
    "owner": "NaimurRahmannn",
    "name": "OpenSteward-MCP-Server"
  },
  "pull_number": 42
}
```

These are placeholders. The GitHub App must be installed on the repository,
the caller token must be authorized for that installation ID, and the pull
request must exist.

## Connecting an OpenAI Agents SDK agent

The runnable example is
[`examples/openai_agents_sdk.py`](examples/openai_agents_sdk.py). It is a
separate client environment; `openai-agents` is deliberately not an
OpenSteward server dependency.

Install the external SDK:

```bash
pip install openai-agents
```

Configure the agent process:

```dotenv
OPENAI_API_KEY=replace-me
OPENSTEWARD_MCP_URL=http://127.0.0.1:8000/mcp
OPENSTEWARD_MCP_TOKEN=replace-with-the-configured-caller-token
OPENSTEWARD_INSTALLATION_ID=12345678
OPENSTEWARD_REPOSITORY_OWNER=NaimurRahmannn
OPENSTEWARD_REPOSITORY_NAME=OpenSteward-MCP-Server
OPENSTEWARD_PULL_NUMBER=42
```

`OPENSTEWARD_REPOSITORY_OWNER`, `OPENSTEWARD_REPOSITORY_NAME`, and
`OPENSTEWARD_PULL_NUMBER` are optional; when omitted, the example prompts for
them. The URL, caller token, and installation ID are required. The OpenAI
Agents SDK reads `OPENAI_API_KEY` through its normal environment behavior.

Run:

```bash
python examples/openai_agents_sdk.py
```

The example:

- connects with `MCPServerStreamableHttp`;
- sends the configured bearer token in the `Authorization` header;
- prefers structured MCP content;
- registers OpenSteward on an `Agent`;
- requires the agent to use `get_maintainer_brief`;
- gives historical collection a longer timeout;
- directs the agent to explain warnings and incomplete coverage;
- prohibits invented evidence and merge decisions;
- prints the final agent response.

See the
[OpenAI Agents SDK MCP guide](https://openai.github.io/openai-agents-python/mcp/)
for the current transport API.

## Calling get_maintainer_brief

The minimal input is:

```json
{
  "installation_id": 12345678,
  "repository": {
    "owner": "OWNER",
    "name": "REPOSITORY"
  },
  "pull_number": 42
}
```

Optional inputs include:

- `policy_path` (default: `.opensteward.yml`);
- `explicit_categories`;
- `conversion_options`;
- `snapshot_options`;
- `related_work_options`;
- `review_cost_options`.

Use the MCP-advertised schema as the authoritative definition for advanced
options. The result includes:

- repository and PR identity;
- a credential-redacted pull-request assessment;
- related-work matches and coverage metadata;
- five weighted review-cost contributions;
- readiness and path-risk summaries;
- attention recommendation and specialist routes;
- deterministic recommended actions;
- warnings and the overall `complete` flag.

Treat `complete: false` and non-empty warnings as part of the result, not as
incidental logging. The result supports a maintainer's judgment; it is not a
merge authorization.

## Repository policy

The policy file is `.opensteward.yml` at the repository root by default. It is
optional:

- Live PR tools read it from the pull request's trusted base commit.
- `evaluate_repository_policy` and the policy resource read the local file
  from the server's current working directory when no YAML is supplied.
- A missing or empty file activates built-in defaults.
- Results identify the policy source and expose `used_defaults`.

The version-1 schema supports:

- contribution categories that require linked issues;
- contribution categories that require changed tests;
- a preferred maximum diff size;
- exact required GitHub check-run names;
- protected repository-relative path patterns and risk levels;
- default, public-API, and security approval counts;
- a maximum pending-review setting;
- automation-policy metadata.

A small valid policy:

```yaml
version: 1

pull_requests:
  linked_issue_required_for:
    - public_api
    - architecture
  tests_required_for:
    - bug_fix
    - observable_behavior
    - public_api
  preferred_maximum_diff_lines: 500
  required_checks:
    - tests

protected_paths:
  - pattern: "src/security/**"
    risk: critical
    human_review_required: true

review:
  maximum_pending_reviews_per_reviewer: 8
  required_approvals:
    default: 1
    public_api: 2
    security: 2

automation:
  publish_check_runs: false
  publish_comments: false
  apply_labels: false
  require_human_approval: true
```

`preferred_maximum_diff_lines` produces explainable size findings.
Protected-path matches influence policy findings, review-cost path risk, and
specialist routing. Required checks are exact check-run names configured in
the policy; OpenSteward does not automatically import branch-protection rules.

Automation fields do not enable writes in the current server. The runtime
remains read-only.

## Development

Install the project with development dependencies.

Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Unix:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The package uses a `src/` layout, Hatchling for builds, pytest with strict
asyncio mode, and Ruff with a Python 3.12 target.

## Testing and linting

Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m ruff check src tests examples
```

Unix:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check src tests examples
```

To validate only the external example:

```bash
python -m compileall examples
python -m ruff check examples
```

Tests use mocks for GitHub runtime behavior unless explicitly configured for a
live integration path. Never put production keys or tokens into fixtures.

## Current limitations

- Historical collection is bounded. Defaults collect at most 100 closed
  issues, 100 closed pull requests, paths for 50 historical pull requests, and
  50 ADR files; scan-page, per-file, and total-byte limits also apply.
- Historical GitHub Discussions, issue/PR comments, review comments, and event
  timelines are not collected.
- ADR discovery is limited to Markdown files in configured architecture
  directories at one exact Git ref. Per-file commit history is not collected.
- Semantic ranking is disabled by default. When enabled, it operates only on a
  bounded lexical candidate set and may require a first-use model download.
- Required-check intelligence uses exact check-run names from
  `.opensteward.yml`; it does not discover branch-protection requirements.
- The maintainer-brief evaluation harness is a deterministic regression
  benchmark, not a real-world accuracy study with maintainers.
- MCP caller authentication uses static configured bearer tokens. Public
  multi-user OAuth/OIDC is not implemented.
- There is no persistence, background indexing, queue, or cross-process token
  cache. Evidence is collected on demand.
- There are no GitHub write operations by design.
- Docker and hosted deployment instructions are not currently provided.
- CI and release workflows are not currently present.

## Roadmap

Potential production-hardening work:

- container packaging and deployment guidance;
- CI, release automation, and signed artifacts;
- observability and operational dashboards;
- external secret-store integration and security review;
- concurrency and load testing;
- evaluation with real maintainers and a public demonstration repository;
- OAuth/OIDC for public multi-user deployments.

No timelines are implied.

## License

OpenSteward is available under the [MIT License](LICENSE).
