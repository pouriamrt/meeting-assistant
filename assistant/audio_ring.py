"""Rolling window of mono float32 samples, safe across threads."""

import threading
from collections import deque

import numpy as np


class AudioRing:
    """Keeps the most recent `max_samples`, dropping the oldest on overflow.

    Writers are WASAPI callbacks, so `write` never blocks on a slow reader —
    stalling that callback causes audible dropouts in the live call. Readers
    track position with an absolute sample index and are told where the data
    they actually received starts, so a reader that fell behind can detect the
    gap instead of splicing two non-adjacent spans together.
    """

    def __init__(self, max_samples: int) -> None:
        self._max = max_samples
        self._chunks: deque[np.ndarray] = deque()
        self._held = 0        # samples currently retained
        self._dropped = 0     # absolute index of the oldest retained sample
        self._lock = threading.Lock()

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._dropped + self._held

    def write(self, samples: np.ndarray) -> None:
        if len(samples) == 0:
            return
        with self._lock:
            self._chunks.append(samples)
            self._held += len(samples)

            # Drop whole chunks while the one after them still covers the cap.
            while self._chunks and self._held - len(self._chunks[0]) >= self._max:
                gone = len(self._chunks.popleft())
                self._held -= gone
                self._dropped += gone

            # Trim the new leftmost chunk to land exactly on the cap.
            overflow = self._held - self._max
            if overflow > 0:
                self._chunks[0] = self._chunks[0][overflow:]
                self._held -= overflow
                self._dropped += overflow

    def read_from(self, abs_index: int) -> tuple[np.ndarray, int]:
        """Return samples from `abs_index` on, plus where they actually start.

        A returned start greater than `abs_index` means audio was dropped
        before the reader got to it.
        """
        with self._lock:
            start = max(abs_index, self._dropped)
            end = self._dropped + self._held
            if start >= end:
                return np.zeros(0, dtype=np.float32), start
            joined = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
            return joined[start - self._dropped:].copy(), start
