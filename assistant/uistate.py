"""Overlay size, position, font and opacity, remembered between runs.

The bounds below are the single source of truth for what the overlay may be
set to. `overlay.py` imports them for its live input clamps and `clamped()`
applies the same numbers to whatever comes off disk, so a file cannot restore
a value the UI has no way to change back.
"""

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# (left, top, right, bottom) in the units Tk positions windows with.
Rect = tuple[int, int, int, int]

log = logging.getLogger(__name__)

MIN_W, MIN_H = 240, 120     # below this the text is unreadable and the grip
                            # gets hard to hit, leaving no way to grow it back
FONT_MIN, FONT_MAX = 8, 32
ALPHA_MIN, ALPHA_MAX = 0.25, 1.0

# How much of the window has to stay on the desktop. The overlay is
# borderless, so the only way to move it is the drag bar along its top edge -
# leave that off-screen and the window cannot be recovered with a mouse.
_MIN_VISIBLE = 90


def _clamp(low: float, high: float, value: float) -> float:
    return max(low, min(high, value))


def _reachable_on(x: int, y: int, w: int, h: int, monitor: "Rect") -> bool:
    """Can the window be grabbed if it opens here, on this one monitor?

    Recoverable means the drag bar is clickable, and the drag bar is the
    window's top strip, so the top edge has to be on the screen rather than
    merely some part of the window.
    """
    left, top, right, bottom = monitor
    if not top <= y <= bottom - _MIN_VISIBLE:
        return False
    return min(x + w, right) - max(x, left) >= _MIN_VISIBLE


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

    def clamped(self, monitors: "Sequence[Rect]") -> "UiState":
        """Pull every field back into range for the screens it will open on.

        Takes the monitors individually, primary first, each as
        (left, top, right, bottom). Left and top can be negative: Windows gives
        screens placed left of or above the primary a negative origin, so this
        cannot clamp at zero.

        Deliberately NOT their bounding box. Two monitors in an L or a diagonal
        leave dead space inside that box which belongs to no screen, and a
        saved position landing there passes a union check while sitting on
        nothing. Checking each monitor is the only version that holds for a
        layout that is not one rectangle.
        """
        usable = [m for m in monitors if m[2] > m[0] and m[3] > m[1]]
        if not usable:
            # A junk or empty probe still has to yield a usable window.
            usable = [(0, 0, MIN_W, MIN_H)]

        w = int(_clamp(MIN_W, max(MIN_W, max(m[2] - m[0] for m in usable)), self.w))
        h = int(_clamp(MIN_H, max(MIN_H, max(m[3] - m[1] for m in usable)), self.h))

        x, y = self.x, self.y
        if not any(_reachable_on(x, y, w, h, m) for m in usable):
            # Nowhere reachable, so put it back on the primary.
            left, top, right, bottom = usable[0]
            x = int(_clamp(left - w + _MIN_VISIBLE, right - _MIN_VISIBLE, x))
            y = int(_clamp(top, max(top, bottom - _MIN_VISIBLE), y))

        return replace(
            self,
            w=w,
            h=h,
            x=x,
            y=y,
            font_size=int(_clamp(FONT_MIN, FONT_MAX, self.font_size)),
            alpha=_clamp(ALPHA_MIN, ALPHA_MAX, self.alpha),
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
    except Exception:
        # Deliberately broad. Narrow tuples kept missing cases: json.loads
        # accepts Infinity and NaN, so int() raises OverflowError, which is
        # neither a parse error nor a type error. The contract this function
        # owes its caller is that the app starts, so anything at all that makes
        # the file unusable has to end up here. Logged rather than swallowed.
        log.warning("unusable ui state file, using defaults", exc_info=True)
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
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(state), indent=2))
            fh.flush()
            os.fsync(fh.fileno())   # replace is atomic, not durable
        tmp.replace(path)
    except OSError:
        log.warning("could not save ui state to %s", path, exc_info=True)
