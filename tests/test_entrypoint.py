"""The frozen entry point. A windowed exe has no console, so every failure
here is silent unless this module makes it otherwise."""

import sys

import pytest

import run_assistant


@pytest.fixture(autouse=True)
def capture_dialogs(monkeypatch):
    shown = []
    monkeypatch.setattr(run_assistant, "_show_fatal", shown.append)
    return shown


def _fake_main(monkeypatch, fn):
    """Stand in for assistant.main.main, which opens audio devices."""
    import assistant.main

    monkeypatch.setattr(assistant.main, "main", fn)


def test_a_clean_run_returns_zero(monkeypatch, tmp_path, capture_dialogs):
    monkeypatch.setattr(run_assistant, "_setup_logging", lambda: tmp_path / "a.log")
    _fake_main(monkeypatch, lambda: None)
    assert run_assistant.main() == 0
    assert capture_dialogs == []


def test_a_crash_reports_and_returns_one(monkeypatch, tmp_path, capture_dialogs):
    monkeypatch.setattr(run_assistant, "_setup_logging", lambda: tmp_path / "a.log")

    def boom():
        raise RuntimeError("capture exclusion failed")

    _fake_main(monkeypatch, boom)
    assert run_assistant.main() == 1
    assert capture_dialogs, "a crash produced no dialog, so the exe just vanishes"
    assert "capture exclusion failed" in capture_dialogs[0]
    assert "RuntimeError" in capture_dialogs[0], "the message alone can be empty"


def test_logging_that_cannot_start_still_reports(monkeypatch, capture_dialogs):
    # The failure this module exists to prevent. With _setup_logging outside
    # the guard, an unwritable %LOCALAPPDATA% gives no log AND no dialog.
    def boom():
        raise OSError("read-only volume")

    monkeypatch.setattr(run_assistant, "_setup_logging", boom)
    _fake_main(monkeypatch, lambda: None)
    assert run_assistant.main() == 1
    assert capture_dialogs, "logging failed and the app vanished without a word"
    assert "read-only volume" in capture_dialogs[0]


def test_a_deliberate_exit_code_is_preserved(monkeypatch, tmp_path, capture_dialogs):
    # sys.exit(1) means 1. Reporting 0 would tell a caller the run succeeded.
    monkeypatch.setattr(run_assistant, "_setup_logging", lambda: tmp_path / "a.log")

    def quit_hard():
        sys.exit(3)

    _fake_main(monkeypatch, quit_hard)
    assert run_assistant.main() == 3
    assert capture_dialogs == [], "a deliberate exit is not worth a dialog"


def test_ctrl_c_is_not_reported_as_a_crash(monkeypatch, tmp_path, capture_dialogs):
    monkeypatch.setattr(run_assistant, "_setup_logging", lambda: tmp_path / "a.log")

    def interrupt():
        raise KeyboardInterrupt

    _fake_main(monkeypatch, interrupt)
    assert run_assistant.main() == 0
    assert capture_dialogs == []
