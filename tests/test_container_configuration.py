"""Static tests for the container and deployment contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "compose.yaml"
DEPLOYMENT_GUIDE = ROOT / "docs" / "deployment.md"
README = ROOT / "README.md"

PRIVATE_KEY_PATH = "/run/secrets/opensteward-github-app.pem"
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)


def _read(path: Path) -> str:
    assert path.is_file(), f"Required repository file is missing: {path.name}"
    return path.read_text(encoding="utf-8")


def _assert_no_credentials(text: str, *, source: str) -> None:
    for pattern in SECRET_PATTERNS:
        assert pattern.search(text) is None, (
            f"{source} contains credential-looking material matching "
            f"{pattern.pattern!r}"
        )


def _normalized_ignore_rules() -> set[str]:
    rules = set()
    for line in _read(DOCKERIGNORE).splitlines():
        rule = line.strip()
        if rule and not rule.startswith("#"):
            rules.add(rule.rstrip("/"))
    return rules


def test_dockerfile_uses_reproducible_multistage_python_build() -> None:
    text = _read(DOCKERFILE)
    lower = text.lower()
    from_lines = re.findall(r"(?im)^from\s+([^\s]+)(?:\s+as\s+\w+)?\s*$", text)

    assert len(from_lines) >= 2, "Dockerfile must use a multi-stage build"
    assert all(image == "python:3.12-slim" for image in from_lines)
    assert "latest" not in lower
    assert "uv sync" in text
    assert "--frozen" in text
    assert "--no-dev" in text
    assert "--no-editable" in text
    assert "UV_PROJECT_ENVIRONMENT=/opt/opensteward/.venv" in text
    assert (
        "/opt/opensteward/.venv /opt/opensteward/.venv"
        in text.replace("\\\n", "")
    )
    assert "pytest" not in lower
    assert "ruff" not in lower


def test_dockerfile_copies_only_installation_inputs() -> None:
    text = _read(DOCKERFILE)
    copy_lines = re.findall(r"(?im)^\s*copy\s+.*$", text)
    joined = "\n".join(copy_lines).lower()

    assert not re.search(r"(?im)^\s*copy\s+\.\s+\.", text)
    assert ".env" not in joined
    assert ".pem" not in joined
    assert ".key" not in joined
    assert "copy src ./src" in joined
    _assert_no_credentials(text, source="Dockerfile")


def test_dockerfile_runs_non_root_through_installed_entry_point() -> None:
    text = _read(DOCKERFILE)
    final_stage = re.split(r"(?im)^from\s+", text)[-1]

    assert "groupadd --gid 10001 opensteward" in text
    assert "useradd --uid 10001 --gid 10001" in text
    assert re.search(r"(?im)^user\s+10001:10001\s*$", final_stage)
    assert "EXPOSE 8000" in final_stage
    assert "OPENSTEWARD_HOST=0.0.0.0" in final_stage
    assert "OPENSTEWARD_PORT=8000" in final_stage
    assert 'CMD ["opensteward"]' in final_stage
    assert "STOPSIGNAL SIGTERM" in final_stage
    assert "--reload" not in final_stage
    assert not re.search(r"(?im)^cmd\s+(?!\[)", final_stage)


def test_dockerfile_healthcheck_is_liveness_only() -> None:
    text = _read(DOCKERFILE)
    healthcheck = text[text.index("HEALTHCHECK") :]
    healthcheck = healthcheck[: healthcheck.index("STOPSIGNAL")]

    assert "/health" in healthcheck
    assert "/ready" not in healthcheck
    assert "urllib.request" in healthcheck
    assert "curl" not in text.lower()


def test_dockerignore_protects_local_and_secret_files() -> None:
    rules = _normalized_ignore_rules()

    assert ".git" in rules
    assert {".venv", "venv"} <= rules
    assert "**/__pycache__" in rules
    assert "**/*.pyc" in rules
    assert {".pytest_cache", ".ruff_cache", ".mypy_cache"} <= rules
    assert {".env", ".env.*", "!.env.example"} <= rules
    assert {"*.pem", "*.key", "secrets", "**/secrets"} <= rules


def test_dockerignore_keeps_package_installation_inputs() -> None:
    rules = _normalized_ignore_rules()
    required_inputs = {"pyproject.toml", "uv.lock", "README.md", "LICENSE", "src"}

    assert required_inputs.isdisjoint(rules)


def test_compose_defines_local_service_and_container_environment() -> None:
    text = _read(COMPOSE)

    assert re.search(r"(?m)^version\s*:", text) is None
    assert re.search(r"(?m)^  opensteward:\s*$", text)
    assert "image: opensteward:local" in text
    assert "context: ." in text
    assert "dockerfile: Dockerfile" in text
    assert re.search(r"(?m)^\s+- \.env\s*$", text)
    assert 'OPENSTEWARD_HOST: 0.0.0.0' in text
    assert 'OPENSTEWARD_PORT: "8000"' in text
    assert '"8000:8000"' in text


def test_compose_mounts_private_key_read_only_at_fixed_path() -> None:
    text = _read(COMPOSE)

    assert "OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH" in text
    assert ":?Set OPENSTEWARD_GITHUB_PRIVATE_KEY_HOST_PATH" in text
    assert f"target: {PRIVATE_KEY_PATH}" in text
    assert f"OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH: {PRIVATE_KEY_PATH}" in text
    assert "read_only: true" in text


def test_compose_applies_runtime_security_controls() -> None:
    text = _read(COMPOSE)
    lower = text.lower()

    assert re.search(r"(?m)^\s+read_only:\s+true\s*$", text)
    assert "no-new-privileges:true" in text
    assert re.search(r"(?ms)cap_drop:\s*\n\s+- ALL", text)
    assert "tmpfs:" in text
    assert "restart: unless-stopped" in text
    assert "stop_grace_period: 30s" in text
    assert "privileged:" not in lower
    assert "network_mode:" not in lower
    assert "cap_add:" not in lower
    assert "/var/run/docker.sock" not in lower
    assert not re.search(r"(?m)^\s*-\s+\.:/", text)
    _assert_no_credentials(text, source="compose.yaml")


def test_compose_healthcheck_is_liveness_only() -> None:
    text = _read(COMPOSE)
    healthcheck = text[text.index("healthcheck:") :]

    assert "/health" in healthcheck
    assert "/ready" not in healthcheck
    assert "urllib.request" in healthcheck


def test_deployment_documentation_covers_container_workflow() -> None:
    guide = _read(DEPLOYMENT_GUIDE)
    readme = _read(README)

    required_guide_text = (
        "docker build -t opensteward:local .",
        "docker compose build",
        "docker compose up --build",
        "docker run --rm",
        "/health",
        "/ready",
        "/mcp",
        "Streamable HTTP",
        "Bearer",
        "read-only",
        "never commit",
    )
    for expected in required_guide_text:
        assert expected in guide, f"Deployment guide is missing {expected!r}"

    assert "docs/deployment.md" in readme
    assert "does not implement OAuth or OIDC" in guide
    assert "does not provide automatic TLS" in guide
    _assert_no_credentials(guide, source="deployment guide")
    _assert_no_credentials(readme, source="README")
