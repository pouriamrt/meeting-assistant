"""Overlay size, position, font and opacity, remembered between runs.

The bounds below are the single source of truth for what the overlay may be
set to. `overlay.py` imports them for its live input clamps and `clamped()`
applies the same numbers to whatever comes off disk, so a file cannot restore
a value the UI has no way to change back.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

log = logging.getLogger(__name__)

MIN_W, MIN_H = 240, 120     # below this the text is unreadable and the grip
                            # gets hard to hit, leaving no way to grow it back
FONT_MIN, FONT_MAX = 8, 32
ALPHA_MIN, ALPHA_MAX = 0.25, 1.0

# How much of the window has to stay on the desktop. The overlay is
# borderless, so the only way to move it is the drag bar along its top edge -
# leave that off-screen and the window cannot be recovered with a mouse.
_MIN_VISIBLE = 90


def _state_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".config"
    return root / "meeting-assistant" / "ui.json"


@dataclass(frozen=True)
class UiState:
    w: int
    h: int
    x: int
    y: int
    font_size: int
    alpha: float

    def clamped(self, desktop: tuple[int, int, int, int]) -> "UiState":
        """Pull every field back into range for the desktop it will open on.

        `desktop` is the union of all monitors as (left, top, right, bottom).
        Left and top can be negative: Windows gives screens placed left of or
        above the primary a negative origin, so this cannot clamp at zero.
        """
        left, top, right, bottom = desktop
        # A junk rect must not produce a zero-sized window, so the minimum
        # wins over the desktop width rather than the other way round.
        w = max(MIN_W, min(self.w, max(MIN_W, right - left)))
        h = max(MIN_H, min(self.h, max(MIN_H, bottom - top)))
        return replace(
            self,
            w=w,
            h=h,
            # Horizontally the window may hang off either edge, as long as a
            # grabbable strip stays on screen.
            x=max(left - w + _MIN_VISIBLE, min(self.x, right - _MIN_VISIBLE)),
            # Vertically it may not go above the top edge at all: the drag bar
            # is the top strip, and above `top` there is nothing to grab.
            y=max(top, min(self.y, bottom - _MIN_VISIBLE)),
            font_size=max(FONT_MIN, min(FONT_MAX, self.font_size)),
            alpha=max(ALPHA_MIN, min(ALPHA_MAX, self.alpha)),
        )


DEFAULT = UiState(w=460, h=300, x=40, y=40, font_size=12, alpha=ALPHA_MAX)


def load() -> UiState:
    """Whatever is on disk, or the defaults. Never raises.

    A missing, unreadable, malformed or wrong-shaped file all mean the same
    thing to the caller: start from the defaults. Refusing to launch because
    a remembered window size could not be parsed would be a far worse failure
    than forgetting it.
    """
    try:
        raw = json.loads(_state_path().read_text(encoding="utf-8"))
        state = UiState(
            w=int(raw["w"]),
            h=int(raw["h"]),
            x=int(raw["x"]),
            y=int(raw["y"]),
            font_size=int(raw["font_size"]),
            alpha=float(raw["alpha"]),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return DEFAULT
    return state


def save(state: UiState) -> None:
    """Write the state, or log and carry on. Never raises.

    Written to a temporary file and moved into place, because os.replace is
    atomic: a crash or a power cut mid-write leaves either the old file or the
    new one, never half of either. Callers include the quit path, where an
    exception would be raised while the app is already tearing down.
    """
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        log.warning("could not save ui state to %s", path, exc_info=True)
