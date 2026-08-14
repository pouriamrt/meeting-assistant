"""Screen capture for the vision hotkey."""

import mss
import mss.tools


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
            # to_png returns None only when handed an `output` path to write
            # to. Without one it always returns the bytes.
            png = mss.tools.to_png(image.rgb, image.size)
            assert png is not None
            shots.append(png)
    return shots
