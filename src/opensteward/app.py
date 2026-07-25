"""FastAPI application for OpenSteward."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from opensteward import __version__
from opensteward.mcp.server import mcp
from opensteward.readiness import (
    ReadinessChecks,
    ReadinessService,
    ReadinessStatus,
)
from opensteward.settings import get_settings


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: Literal["ok"]
    name: str
    version: str


class ReadinessResponse(BaseModel):
    """Response returned by the readiness endpoint."""

    status: ReadinessStatus
    environment: str
    checks: ReadinessChecks
    issues: list[str]


# Calling streamable_http_app creates the MCP HTTP application
# and initializes the session-manager object.
mcp_http_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Start and stop application-level services."""

    async with mcp.session_manager.run():
        yield


def create_app() -> FastAPI:
    """Create and configure the OpenSteward FastAPI application."""

    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        description=(
            "MCP-based maintainer intelligence for open-source projects."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.state.readiness_service = ReadinessService(
        settings=settings
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Operations"],
    )
    async def health() -> HealthResponse:
        """Report whether the application process is running."""

        return HealthResponse(
            status="ok",
            name=settings.app_name,
            version=__version__,
        )

    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ReadinessResponse,
            },
        },
        tags=["Operations"],
    )
    async def ready(
        response: Response,
    ) -> ReadinessResponse:
        """Report whether the application is ready to receive requests."""

        assessment = await (
            application.state.readiness_service.assess()
        )

        if assessment.status != ReadinessStatus.READY:
            response.status_code = (
                status.HTTP_503_SERVICE_UNAVAILABLE
            )

        return ReadinessResponse(
            status=assessment.status,
            environment=settings.environment,
            checks=assessment.checks,
            issues=assessment.issues,
        )

    # The MCP sub-application already defines its own /mcp route,
    # so it is mounted at the root of the parent application.
    application.mount("/", mcp_http_app)

    return application


app = create_app()
