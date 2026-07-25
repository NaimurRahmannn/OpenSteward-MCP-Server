FROM python:3.12-slim AS builder

ARG UV_VERSION=0.11.31

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/opensteward/.venv

WORKDIR /build

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENSTEWARD_HOST=0.0.0.0 \
    OPENSTEWARD_PORT=8000 \
    PATH="/opt/opensteward/.venv/bin:${PATH}"

LABEL org.opencontainers.image.title="OpenSteward MCP Server" \
    org.opencontainers.image.description="Read-only MCP server for open-source maintainers" \
    org.opencontainers.image.source="https://github.com/NaimurRahmannn/OpenSteward-MCP-Server"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 opensteward \
    && useradd --uid 10001 --gid 10001 --create-home \
        --home-dir /home/opensteward --shell /usr/sbin/nologin opensteward

WORKDIR /app

COPY --from=builder --chown=10001:10001 \
    /opt/opensteward/.venv /opt/opensteward/.venv

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()"]

STOPSIGNAL SIGTERM

CMD ["opensteward"]
