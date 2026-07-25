"""Command-line entry point for OpenSteward."""

import uvicorn

from opensteward.settings import get_settings


def main() -> None:
    """Run the OpenSteward ASGI application."""

    settings = get_settings()
    uvicorn.run(
        "opensteward.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
