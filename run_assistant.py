"""Frozen-app entry point. Only used by the PyInstaller build.

`uv run meeting-assistant` still goes straight to assistant.main:main; this
exists because a windowed exe has no console, which breaks two things that are
free when a terminal is attached.
"""

import logging
import multiprocessing
import sys
import traceback
from pathlib import Path


def _log_dir() -> Path:
    from assistant import uistate

    return uistate._state_path().parent


def _setup_logging() -> Path:
    """Log to a file, because a windowed exe has no stderr to log to.

    sys.stderr is None under `console=False`, and the logging module writing to
    it raises. Configured before assistant.main runs, since basicConfig is a
    no-op once handlers exist, so this wins over the one in main().
    """
    path = _log_dir() / "assistant.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8")],
    )
    return path


def _show_fatal(message: str) -> None:
    """A windowed app that raises just vanishes. Say something first."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Meeting assistant could not start", message)
        root.destroy()
    except Exception:
        pass          # no display, nothing left to try


def main() -> int:
    # ctranslate2 and onnxruntime can spawn workers. Without this a frozen
    # child re-runs the whole program instead of the worker entry point, which
    # on Windows means the app launching itself repeatedly.
    multiprocessing.freeze_support()

    log_path = _setup_logging()
    try:
        from assistant.main import main as run

        run()
        return 0
    except Exception as exc:
        logging.exception("fatal")
        _show_fatal(
            f"{exc}\n\nFull details: {log_path}"
        )
        return 1
    except BaseException:
        # KeyboardInterrupt and SystemExit are not failures worth a dialog,
        # but the traceback still belongs in the log.
        logging.info("stopped: %s", traceback.format_exc(limit=0).strip())
        return 0


if __name__ == "__main__":
    sys.exit(main())
