# Container deployment

## Scope

Phase 7.1 provides a production-oriented OpenSteward image, a reproducible
local build, and a Docker Compose workflow. It is intended for private,
operator-controlled deployments.

This phase does not provide Kubernetes manifests, high availability,
autoscaling, managed hosting, production certification, automatic TLS, or
public-client authorization.

## Prerequisites

You need:

- Docker Engine or Docker Desktop with Linux containers;
- Docker Compose v2 (`docker compose`);
- a configured GitHub App with the read-only permissions documented in the
  [main README](../README.md#github-app-setup);
- the GitHub App ID, installation ID, and generated private-key file;
- an MCP caller bearer token of at least 32 non-whitespace ASCII characters;
- a repository-root `.env` file; and
- Node.js only when using MCP Inspector.

The private key should be stored outside the repository. The installation ID
belongs in the MCP caller's installation allowlist and is also supplied in
live tool calls.

## Security model

OpenSteward runs as UID and GID `10001`, requests only read permissions from
GitHub, and starts through its installed `opensteward` entry point. The Compose
service drops Linux capabilities, prevents privilege escalation, uses a
read-only root filesystem, and provides only an ephemeral `/tmp`.

MCP caller authentication and GitHub authentication are separate:

- a caller sends its configured MCP bearer token to `/mcp`;
- OpenSteward verifies that caller's allowed GitHub installation IDs;
- the server reads the mounted GitHub App private key and creates short-lived
  installation tokens; and
- neither the private key nor an installation token is returned to the MCP
  client.

OpenSteward does not comment, label, approve, request changes, merge, close, or
edit GitHub resources.

Static bearer tokens are appropriate for controlled private deployments. The
server does not implement OAuth or OIDC for arbitrary public clients. Add a
standardized authorization layer before exposing it as a multi-user public
service.

## Required configuration

Copy the example and generate a caller token:

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Configure at least these values in `.env`:

```dotenv
OPENSTEWARD_ENVIRONMENT=production
OPENSTEWARD_LOG_LEVEL=INFO
OPENSTEWARD_MCP_AUTHORIZED_CALLERS={"local-client":{"token":"REPLACE_WITH_A_RANDOM_TOKEN_AT_LEAST_32_CHARACTERS","installation_ids":[12345678]}}

OPENSTEWARD_GITHUB_APP_ID=123456
OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH=C:/absolute/path/outside-the-repository/opensteward-github-app.pem
```

`OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH` is a Compose interpolation variable,
not the private key. Compose uses it as the host side of a bind mount and
overrides the application's `OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH` with
`/run/secrets/opensteward-github-app.pem`.

Do not set `OPENSTEWARD_GITHUB_PRIVATE_KEY` for the container workflow. It
would put the key in the container environment and conflict with the mounted
key-path setting. Host and port are also overridden by Compose to
`0.0.0.0:8000`.

See [Environment configuration](../README.md#environment-configuration) for
all other settings. If local semantic scoring is enabled, Compose directs its
model cache to ephemeral `/tmp`; the model is downloaded again after the
container is recreated.

## GitHub App private-key handling

Never copy the GitHub App private key into the image, and never commit the key
or `.env`. Keep it outside the repository when practical and mount it
read-only at runtime. The host path and the fixed container path are different.
The MCP client never receives this key.

PowerShell, for the current terminal:

```powershell
$env:OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH = "C:/secure/opensteward-github-app.pem"
```

Unix-like shell:

```bash
export OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH=/secure/opensteward-github-app.pem
```

Compose reads interpolation values from the shell first and then the
repository-root `.env`. You may therefore put the non-secret absolute host
path in `.env` instead of exporting it.

On native Linux, the bind-mounted file must be readable by container UID
`10001`. Prefer a narrow filesystem ACL, for example:

```bash
sudo setfacl -m u:10001:r /secure/opensteward-github-app.pem
```

Do not make the key broadly writable. On Docker Desktop, ensure the key's
parent location is shared with Docker and that Docker Desktop can read it.

## Build the image

Build directly:

```bash
docker build -t opensteward:local .
```

Or use Compose:

```bash
docker compose build
```

The build uses the checked-in `uv.lock` with frozen, production-only
installation. It does not install the development dependency group.

## Start with Docker Compose

Start in the foreground:

```bash
docker compose up --build
```

Or start in detached mode:

```bash
docker compose up --build -d
```

The endpoints are then available at:

- health: `http://127.0.0.1:8000/health`
- readiness: `http://127.0.0.1:8000/ready`
- MCP: `http://127.0.0.1:8000/mcp`

The service uses `restart: unless-stopped`. Remove or change that policy for a
one-off development container if desired.

## Start with `docker run`

Build the image first, set
`OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH` in the current shell, and keep the
application configuration in `.env`.

PowerShell:

```powershell
docker run --rm `
  --name opensteward `
  --env-file .env `
  -e OPENSTEWARD_HOST=0.0.0.0 `
  -e OPENSTEWARD_PORT=8000 `
  -e OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH=/run/secrets/opensteward-github-app.pem `
  -e OPENSTEWARD_EMBEDDING_CACHE_DIR=/tmp/opensteward-fastembed `
  --mount "type=bind,source=$env:OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH,target=/run/secrets/opensteward-github-app.pem,readonly" `
  --read-only `
  --tmpfs "/tmp:rw,noexec,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=1777" `
  --security-opt no-new-privileges:true `
  --cap-drop ALL `
  -p 8000:8000 `
  opensteward:local
```

Unix-like shell:

```bash
docker run --rm \
  --name opensteward \
  --env-file .env \
  -e OPENSTEWARD_HOST=0.0.0.0 \
  -e OPENSTEWARD_PORT=8000 \
  -e OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH=/run/secrets/opensteward-github-app.pem \
  -e OPENSTEWARD_EMBEDDING_CACHE_DIR=/tmp/opensteward-fastembed \
  --mount "type=bind,source=$OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH,target=/run/secrets/opensteward-github-app.pem,readonly" \
  --read-only \
  --tmpfs "/tmp:rw,noexec,nosuid,nodev,size=256m,uid=10001,gid=10001,mode=1777" \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  -p 8000:8000 \
  opensteward:local
```

An unset or incorrect host path makes the bind mount fail instead of falling
back to an image-bundled credential.

## Verify health

Docker's image and Compose health checks call `/health`, which confirms only
that the process is alive.

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
docker compose ps
```

Unix:

```bash
curl http://127.0.0.1:8000/health
docker compose ps
```

## Verify readiness

`/ready` checks configured MCP callers, GitHub credentials, GitHub API
connectivity, and the authorized installations. It may return HTTP `503` while
`/health` succeeds and Docker reports the container as healthy. That
distinction prevents a temporary dependency failure from being treated as a
dead application process.

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Unix:

```bash
curl http://127.0.0.1:8000/ready
```

Readiness responses are sanitized and do not expose credentials. See
[Health and readiness](../README.md#health-and-readiness) for the response
fields.

## Connect MCP Inspector

Start Inspector:

```bash
npx @modelcontextprotocol/inspector
```

Configure:

- transport: **Streamable HTTP**
- URL: `http://127.0.0.1:8000/mcp`
- header: `Authorization: Bearer <your-configured-MCP-caller-token>`

First call `system_status`. Then call `get_maintainer_brief` with an
installation ID allowed for that caller, a repository on which the GitHub App
is installed, and an existing pull-request number.

## Connect an agent

Agents use the same Streamable HTTP URL and bearer header. The repository's
[OpenAI Agents SDK example](../examples/openai_agents_sdk.py) reads:

- `OPENSTEWARD_MCP_URL`
- `OPENSTEWARD_MCP_TOKEN`
- `OPENSTEWARD_INSTALLATION_ID`
- optional repository owner, repository name, and pull-request number

Run it from a separate environment that has `openai-agents` installed:

```bash
pip install openai-agents
python examples/openai_agents_sdk.py
```

The agent dependency and `OPENAI_API_KEY` are not required by the OpenSteward
container itself.

## View logs

```bash
docker compose logs -f opensteward
```

Logs are written to standard output and standard error. Do not add commands
that print `.env`, private keys, caller tokens, or installation tokens.

## Stop and restart

Stop and remove the service:

```bash
docker compose down
```

Start it again without rebuilding:

```bash
docker compose up -d
```

Restart the existing service:

```bash
docker compose restart opensteward
```

Compose allows 30 seconds for graceful SIGTERM shutdown. Configuration and the
private key remain outside the image.

## Upgrade the container

For a local source-based upgrade:

```bash
git pull
docker compose build --pull
docker compose up -d
```

Review upstream configuration changes before replacing a running deployment.
The `.env` file and mounted private key are not baked into the rebuilt image.

## Troubleshooting

### Private-key file not found

Set `OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH` to an absolute host path. Run
`docker compose config` to inspect the resolved mount path without printing
credential contents. A missing interpolation value fails with an explicit
Compose error.

### Private-key permission denied

Ensure Docker can read the host file and that the mounted file is readable by
UID `10001`. On Linux, use a narrow ACL as described above. On Docker Desktop,
check file-sharing access. Keep the mount read-only.

### Missing GitHub App ID

Set `OPENSTEWARD_GITHUB_APP_ID` to the numeric App ID in `.env`. Do not use the
client ID or installation ID in its place.

### Missing MCP authorized callers

Set `OPENSTEWARD_MCP_AUTHORIZED_CALLERS` to the documented JSON mapping. In
`production` mode, startup rejects an empty mapping.

### Unauthorized installation ID

Add the installed GitHub App installation ID to the correct caller's
`installation_ids` allowlist, then recreate the container. Do not grant a
caller installations it should not access.

### `/health` succeeds but `/ready` fails

The process is live, but a configuration or external dependency check failed.
Read the sanitized readiness check names and inspect container logs. Common
causes are invalid GitHub credentials, an unavailable API, a missing App
installation, or insufficient read permissions.

### Port 8000 is already in use

Stop the conflicting service, or change only the host side of the port
mapping, for example `8080:8000`. The container must continue listening on
port `8000`.

### Container repeatedly restarts

Inspect `docker compose logs opensteward`. A production configuration with no
authorized caller fails during application settings initialization. Other
configuration and GitHub failures are normally visible through `/ready`.

### MCP client receives HTTP 401

Send `Authorization: Bearer <token>` and ensure the token exactly matches one
configured caller. Tokens must be at least 32 non-whitespace ASCII characters.

### MCP client receives HTTP 403

The caller is authenticated but is not authorized for the requested
installation ID. Check the caller-specific installation allowlist.

### GitHub rate limits or connectivity failure

Readiness or live tools report the sanitized failure. OpenSteward uses bounded
transient retries, but it does not bypass GitHub rate limits. Wait for the
reported reset window or restore network access.

### Compose interpolation problems on Windows

Use an absolute path with forward slashes in `.env`, for example
`C:/secure/opensteward-github-app.pem`. Avoid surrounding the `.env` value
with extra shell quoting. `docker compose config` shows whether interpolation
succeeded.

If Compose reports an unexpected character or invalid variable name, inspect
the reported `.env` line. Every non-blank line must be either a `#` comment or
a `NAME=value` assignment; wrapped explanatory text must retain `#` on every
line.

### Stale image after source changes

Force a rebuild and recreate the service:

```bash
docker compose build --no-cache
docker compose up -d --force-recreate
```

## Production deployment considerations

Before an internet-facing deployment, add:

- HTTPS termination through a reverse proxy or managed ingress;
- centralized secret management instead of local bind-mounted files;
- centralized logs, metrics, tracing, and alerting;
- edge rate limiting and request-size controls;
- OAuth/OIDC when accepting arbitrary public clients;
- image vulnerability scanning and a patching process;
- pinned release tags and image digests with controlled promotion; and
- load and security testing.

Backups become relevant only if future versions add persistence. This phase
does not provide automatic TLS or any of the controls above.

## Current deployment limitations

- The Compose workflow runs one local service; it is not a high-availability
  deployment.
- The server has no persistence, background index, or durable embedding cache.
- GitHub history collection remains bounded and request-driven.
- Static bearer tokens assume a private trust boundary; public OAuth/OIDC is
  not implemented.
- TLS termination, managed secrets, observability, image publishing,
  Kubernetes manifests, and cloud-specific deployment are outside this phase.
- The server remains deliberately read-only and provides no GitHub write
  operations.
