"""Tests for the installed OpenSteward command."""

from importlib.metadata import EntryPoint
from types import SimpleNamespace

import opensteward.main as main_module


def test_console_entry_point_resolves() -> None:
    entry_point = EntryPoint(
        name="opensteward",
        value="opensteward.main:main",
        group="console_scripts",
    )

    assert entry_point.load() is main_module.main


def test_main_runs_configured_application(
    monkeypatch,
) -> None:
    settings = SimpleNamespace(
        host="0.0.0.0",
        port=9000,
        log_level="WARNING",
    )
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        main_module.uvicorn,
        "run",
        lambda application, **kwargs: calls.append((application, kwargs)),
    )

    main_module.main()

    assert calls == [
        (
            "opensteward.app:app",
            {
                "host": "0.0.0.0",
                "port": 9000,
                "log_level": "warning",
            },
        )
    ]
