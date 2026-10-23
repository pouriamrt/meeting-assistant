import pytest

from assistant import answer, uistate


@pytest.fixture(autouse=True)
def isolate_ui_state(tmp_path, monkeypatch):
    """Keep every test off the real %LOCALAPPDATA%\\meeting-assistant\\ui.json.

    Autouse and unconditional. The overlay saves on quit and on a debounce
    timer, so any test that builds one can reach the file the developer is
    actually using: it would rewrite their window position from a test, and
    leak state between tests in whatever order they happen to run.
    """
    monkeypatch.setattr(uistate, "_state_path", lambda: tmp_path / "ui.json")


@pytest.fixture(autouse=True)
def reset_fast_mode_downgrade(monkeypatch):
    """A fast-mode 4xx sticks for the process; never let one leak between tests."""
    monkeypatch.setattr(answer, "_fast_refused", False)
