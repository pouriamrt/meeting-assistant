"""Screen capture for the vision hotkey."""

import mss
import mss.tools
import numpy as np

from . import config


def grab() -> list[bytes]:
    """One PNG per physical monitor, at native resolution.

    Not stitched. Two 1920-wide monitors side by side are 3840 px across, and
    the API downscales anything past 1568 on the long edge - that lands the
    pair at 441 px tall, where source code stops being legible. Separate
    blocks each get their own downscale budget instead, so a second monitor
    costs tokens rather than readability.

    No resize here either: at or below 1568 nothing is scaled, and above it
    the API does the same scaling server-side. Doing it locally would buy a
    smaller upload at the cost of a Pillow dependency.
    """
    shots: list[bytes] = []
    with mss.MSS() as sct:   # mss.mss() is deprecated as of mss 10
        # monitors[0] is mss's union of every screen - it *is* the stitched
        # image this function exists to avoid, and every real monitor follows
        # it, so including it would duplicate all of them.
        for monitor in sct.monitors[1:]:
            image = sct.grab(monitor)
            data, size = image.rgb, tuple(image.size)
            if config.SCREEN_HALF_RES:
                data, size = _halve(data, image.size.width, image.size.height)
            # to_png returns None only when handed an `output` path to write
            # to. Without one it always returns the bytes.
            png = mss.tools.to_png(data, size)
            assert png is not None
            shots.append(png)
    return shots


def _halve(rgb: bytes, width: int, height: int) -> tuple[bytes, tuple[int, int]]:
    """Drop every other row and column.

    Nearest-neighbour on purpose: the alternative is Pillow for one resize,
    and a proper filter would blur the thin strokes that make small code
    legible rather than preserve them. numpy is already a dependency.

    `tobytes()` is what makes the strided view contiguous again - mss.tools
    reads a flat buffer and a view would hand it the wrong bytes.
    """
    arr = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
    small = arr[::2, ::2]
    return small.tobytes(), (small.shape[1], small.shape[0])
