# Local Meeting Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows assistant that captures both sides of a live call, transcribes locally, and paints cue cards into a screen-capture-invisible overlay on a hotkey.

**Architecture:** Two WASAPI capture threads write mono 16 kHz float32 into per-source ring buffers. Two VAD gate threads watch those rings and hand complete utterances to a single lock-guarded whisper model, which appends tagged turns to an in-memory transcript. A hotkey spawns a transient answer thread that streams Claude tokens onto a queue drained by the Tkinter main thread.

**Tech Stack:** Python 3.13 (fall back to 3.12 if `pyaudiowpatch` has no wheel), `uv`, `pyaudiowpatch`, `faster-whisper`, `soxr`, `numpy`, `anthropic`, `pynput`, `tkinter` (stdlib), `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-13-meeting-assistant-design.md`

## Deviations from the spec

Two dependency findings during planning. Both were verified against library
documentation, not assumed.

1. **`silero-vad` is not needed and is not installed.** `faster-whisper` already
   ships a Silero VAD running on `onnxruntime`, exposed as
   `faster_whisper.vad.get_speech_timestamps(audio, VadOptions)`. The standalone
   `silero-vad` package depends on `torch` and `torchaudio` — roughly 200 MB of
   wheels and about two seconds of import time — to reach the same ONNX model.
   The spec's module table lists `silero-vad`; the plan uses `faster_whisper.vad`.

2. **Resampling was a gap in the spec.** WASAPI shared mode hands you the
   endpoint's mix format, typically 48 kHz stereo. Whisper and Silero both
   require exactly 16 kHz mono. Nothing in the spec's dependency list converts
   between them. `soxr` is added for this — one call per audio chunk, and the
   only new dependency in the plan.

## Global Constraints

- **Platform:** Windows 10 build 19041 or later. `SetWindowDisplayAffinity` with
  `WDA_EXCLUDEFROMCAPTURE` does not exist below that build.
- **Sample rate:** exactly `16000` Hz, mono, float32 everywhere past `capture.py`.
  Both whisper and Silero require it.
- **Whisper model:** `base.en`, `device="cpu"`, `compute_type="int8"`. There is no
  CUDA on this machine. `small.en` and above are outside the latency budget.
- **Model:** `claude-opus-5`, `speed="fast"`, `betas=["fast-mode-2026-02-01"]`,
  `output_config={"effort": "low"}`, `max_tokens=2000`.
- **Never send** `temperature`, `top_p`, or `top_k` — each is a 400 on Opus 5.
- **Never disable thinking.** It leaks `<thinking>` tags into visible output.
- **Cache prefix is byte-frozen.** No `datetime`, no meeting name, no per-request
  f-string anywhere in `SYSTEM` or in a transcript turn already appended.
- **API key** is read from `ANTHROPIC_API_KEY` at call time. Never hardcoded,
  never logged, never written into the transcript.
- **Audio devices fail loudly.** A missing mic or loopback endpoint raises. There
  is no fallback to a different device.
- **Capture exclusion fails loudly.** If the Win32 call fails, the overlay refuses
  to show rather than appearing unprotected.

---

## File Structure

| File | Responsibility |
|---|---|
| `assistant/config.py` | All tunable constants; lazy API key read |
| `assistant/audio_ring.py` | Thread-safe drop-oldest sample ring with absolute indexing |
| `assistant/capture.py` | WASAPI device discovery, two capture streams, mono/16k conversion |
| `assistant/transcribe.py` | `Turn`, `Transcript`, VAD gate loop, shared whisper model |
| `assistant/prompts.py` | `SYSTEM` constant and the two mode tails |
| `assistant/answer.py` | Request assembly with cache breakpoints, streaming, usage logging |
| `assistant/overlay.py` | Always-on-top window, capture exclusion, queue drain |
| `assistant/main.py` | Wiring, hotkeys, lifecycle |
| `tests/test_assistant.py` | All automated tests |

`audio_ring.py` is split out from `capture.py` because it is the one piece of
capture that is testable without audio hardware, and it owns the trickiest
boundary condition in the project.

---

### Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml` (via `uv init`)
- Create: `assistant/__init__.py`
- Create: `assistant/config.py`
- Create: `tests/test_assistant.py`

**Interfaces:**
- Consumes: nothing
- Produces: `config.SAMPLE_RATE: int`, `config.RING_SAMPLES: int`,
  `config.WHISPER_MODEL: str`, `config.WHISPER_THREADS: int`,
  `config.SILENCE_END_MS: int`, `config.MAX_UTTERANCE_S: int`,
  `config.MAX_TURNS: int`, `config.NAME: str`,
  `config.EXCLUDE_FROM_CAPTURE: bool`, `config.api_key() -> str`

- [ ] **Step 1: Initialize the project**

```bash
cd /c/Users/pouri/Python/AI/Test/AI-interview-helper
uv init --python 3.13 --name assistant --no-readme
```

- [ ] **Step 2: Add dependencies and confirm `pyaudiowpatch` has a wheel**

`pyaudiowpatch` is a C extension and is the one dependency likely to lack a
Python 3.13 wheel. Run this first, alone, so a failure is unambiguous:

```bash
uv add pyaudiowpatch
```

Expected: resolves and installs.

If it fails with a build error mentioning `Microsoft Visual C++` or
`error: subprocess-exited-with-error`, there is no 3.13 wheel. Drop to 3.12 —
this is a supported fallback, not a workaround:

```bash
uv python pin 3.12
rm -rf .venv && uv sync
uv add pyaudiowpatch
```

- [ ] **Step 3: Add the remaining dependencies**

```bash
uv add faster-whisper soxr numpy anthropic pynput
uv add --dev pytest
```

- [ ] **Step 4: Verify every import resolves**

`tkinter` is stdlib but is a separate installer component on some Python builds,
so it gets checked here rather than discovered in Task 7.

```bash
uv run python -c "import pyaudiowpatch, faster_whisper, soxr, numpy, anthropic, pynput, tkinter; from faster_whisper.vad import get_speech_timestamps, VadOptions; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Write the failing test**

Create `tests/test_assistant.py`:

```python
import pytest

from assistant import config


def test_api_key_missing_raises_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        config.api_key()


def test_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    assert config.api_key() == "sk-ant-not-a-real-key"


def test_api_key_is_not_module_state(monkeypatch):
    # Read lazily so the value never sits in module globals, where a repr of
    # the module or a traceback frame could surface it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    config.api_key()
    assert not any(
        isinstance(v, str) and "sk-ant" in v for v in vars(config).values()
    )
```

- [ ] **Step 6: Run the test to verify it fails**

```bash
uv run pytest tests/test_assistant.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'assistant.config'`

- [ ] **Step 7: Write the implementation**

Create `assistant/__init__.py` as an empty file, then `assistant/config.py`:

```python
"""Tunable constants. Anything that varies between runs lives here."""

import os

# Your first name, interpolated into the system prompt once at import.
NAME = os.environ.get("ASSISTANT_NAME", "the user")

# Whisper and Silero both require exactly this rate. Not a preference.
SAMPLE_RATE = 16_000

RING_SECONDS = 30
RING_SAMPLES = SAMPLE_RATE * RING_SECONDS

WHISPER_MODEL = "base.en"   # distil-small.en if accuracy hurts more than latency
WHISPER_THREADS = 4         # 15W part sharing the CPU with a live video call

SILENCE_END_MS = 700        # trailing silence that closes an utterance
MAX_UTTERANCE_S = 30        # force a transcribe on an unbroken monologue

MAX_TURNS = 400

# False to debug the overlay — with exclusion on you cannot screenshot it.
EXCLUDE_FROM_CAPTURE = True


def api_key() -> str:
    """Read the key at call time so it never lives in module state."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in the environment; "
            "never hardcode it."
        )
    return key
```

- [ ] **Step 8: Run the tests to verify they pass**

```bash
uv run pytest tests/test_assistant.py -v
```

Expected: 3 passed

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock assistant/ tests/
git commit -m "feat: project scaffold and configuration"
```

---

### Task 2: Audio ring buffer

The one part of capture testable without audio hardware, and the one with a real
boundary condition. It drops the oldest samples rather than blocking, because the
writer is a WASAPI callback — blocking it causes audible dropouts in the live
call, which is a far worse failure than losing stale audio.

**Files:**
- Create: `assistant/audio_ring.py`
- Test: `tests/test_assistant.py`

**Interfaces:**
- Consumes: nothing
- Produces: `AudioRing(max_samples: int)` with
  `write(samples: np.ndarray) -> None`,
  `read_from(abs_index: int) -> tuple[np.ndarray, int]` returning
  `(samples, actual_start_abs_index)`,
  and property `total_written -> int`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_assistant.py`:

```python
import numpy as np

from assistant.audio_ring import AudioRing


def test_ring_returns_everything_below_cap():
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(4, dtype=np.float32))
    samples, start = ring.read_from(0)
    assert np.array_equal(samples, np.arange(4, dtype=np.float32))
    assert start == 0
    assert ring.total_written == 4


def test_ring_keeps_exact_tail_across_the_wrap():
    # Writes straddle the cap: 6 then 8 samples into a 10-sample ring.
    # The retained window must be the true last 10 samples, with no sample
    # dropped from the middle and none duplicated at the seam.
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(0, 6, dtype=np.float32))
    ring.write(np.arange(6, 14, dtype=np.float32))

    samples, start = ring.read_from(0)
    assert np.array_equal(samples, np.arange(4, 14, dtype=np.float32))
    assert start == 4
    assert ring.total_written == 14
    assert len(samples) == 10


def test_ring_read_from_cursor_returns_only_new_samples():
    ring = AudioRing(max_samples=100)
    ring.write(np.arange(0, 5, dtype=np.float32))
    ring.write(np.arange(5, 9, dtype=np.float32))
    samples, start = ring.read_from(5)
    assert np.array_equal(samples, np.arange(5, 9, dtype=np.float32))
    assert start == 5


def test_ring_reports_the_gap_when_the_reader_fell_behind():
    # A stalled consumer must be able to detect that audio was dropped,
    # rather than silently splicing two non-adjacent spans together.
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(0, 20, dtype=np.float32))
    samples, start = ring.read_from(0)
    assert start == 10, "reader asked for sample 0, oldest surviving is 10"
    assert np.array_equal(samples, np.arange(10, 20, dtype=np.float32))


def test_ring_read_past_the_end_is_empty_not_an_error():
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(3, dtype=np.float32))
    samples, start = ring.read_from(3)
    assert len(samples) == 0
    assert start == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_assistant.py -k ring -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'assistant.audio_ring'`

- [ ] **Step 3: Write the implementation**

Create `assistant/audio_ring.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_assistant.py -k ring -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add assistant/audio_ring.py tests/test_assistant.py
git commit -m "feat: add drop-oldest audio ring buffer"
```

---

### Task 3: WASAPI capture

**Files:**
- Create: `assistant/capture.py`
- Test: `tests/test_assistant.py`

**Interfaces:**
- Consumes: `config.SAMPLE_RATE`, `config.RING_SAMPLES`, `AudioRing`
- Produces: `to_mono_16k(raw: bytes, channels: int, src_rate: int) -> np.ndarray`,
  `Capture()` with `.mic: AudioRing`, `.sys: AudioRing`, `.start() -> None`,
  `.stop() -> None`

- [ ] **Step 1: Write the failing tests**

`to_mono_16k` is pure, so it carries the tests. Device discovery is verified by
hand in Step 6 — mocking `PyAudio` would only prove the mock works.

Append to `tests/test_assistant.py`:

```python
from assistant.capture import to_mono_16k


def test_to_mono_16k_downmixes_stereo():
    # Two interleaved channels: left all 1.0, right all -1.0 averages to 0.0.
    stereo = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    out = to_mono_16k(stereo.tobytes(), channels=2, src_rate=16_000)
    assert len(out) == 2
    assert np.allclose(out, 0.0)


def test_to_mono_16k_passes_through_at_native_rate():
    mono = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = to_mono_16k(mono.tobytes(), channels=1, src_rate=16_000)
    assert np.allclose(out, mono)


def test_to_mono_16k_resamples_48k_to_16k():
    # One second of 48 kHz must come back as one second of 16 kHz.
    mono = np.zeros(48_000, dtype=np.float32)
    out = to_mono_16k(mono.tobytes(), channels=1, src_rate=48_000)
    assert abs(len(out) - 16_000) <= 2, f"expected ~16000 samples, got {len(out)}"


def test_to_mono_16k_always_returns_float32():
    stereo = np.array([0.5, 0.5], dtype=np.float32)
    out = to_mono_16k(stereo.tobytes(), channels=2, src_rate=44_100)
    assert out.dtype == np.float32


def test_to_mono_16k_handles_an_empty_packet():
    # An idle output endpoint delivers zero-length packets rather than erroring.
    out = to_mono_16k(b"", channels=2, src_rate=48_000)
    assert len(out) == 0
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_assistant.py -k mono -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'assistant.capture'`

- [ ] **Step 3: Write the implementation**

Create `assistant/capture.py`:

```python
"""Dual WASAPI capture: microphone and system output, both to 16 kHz mono."""

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from . import config
from .audio_ring import AudioRing

_FORMAT = pyaudio.paFloat32
_FRAMES_PER_BUFFER = 1024


def to_mono_16k(raw: bytes, channels: int, src_rate: int) -> np.ndarray:
    """Interleaved float32 bytes at the device rate to mono float32 at 16 kHz."""
    samples = np.frombuffer(raw, dtype=np.float32)
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if src_rate != config.SAMPLE_RATE:
        samples = soxr.resample(samples, src_rate, config.SAMPLE_RATE)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _loopback_device(p: pyaudio.PyAudio) -> dict:
    try:
        return p.get_default_wasapi_loopback()
    except OSError as exc:
        raise RuntimeError(
            "WASAPI is unavailable. This runs on Windows only."
        ) from exc
    except LookupError as exc:
        raise RuntimeError(
            "No WASAPI loopback device found. Run `python -m pyaudiowpatch` "
            "to list devices."
        ) from exc


def _mic_device(p: pyaudio.PyAudio) -> dict:
    try:
        return p.get_default_input_device_info()
    except OSError as exc:
        raise RuntimeError(
            "No default input device. Plug in a microphone and set it as the "
            "Windows default."
        ) from exc


class Capture:
    """Two capture streams feeding two rings. No fallback on a missing device."""

    def __init__(self) -> None:
        self.mic = AudioRing(config.RING_SAMPLES)
        self.sys = AudioRing(config.RING_SAMPLES)
        self._pa: pyaudio.PyAudio | None = None
        self._streams: list = []

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        # Resolve both devices before opening either, so a missing one fails
        # before we have a half-started capture to unwind.
        mic = _mic_device(self._pa)
        spk = _loopback_device(self._pa)
        self._streams.append(self._open(mic, self.mic))
        self._streams.append(self._open(spk, self.sys))

    def _open(self, device: dict, ring: AudioRing):
        rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])

        def callback(in_data, frame_count, time_info, status):
            # Must return fast: this is the WASAPI thread, and stalling it
            # causes audible dropouts in the call itself.
            ring.write(to_mono_16k(in_data, channels, rate))
            return (None, pyaudio.paContinue)

        return self._pa.open(
            format=_FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=_FRAMES_PER_BUFFER,
            stream_callback=callback,
        )

    def stop(self) -> None:
        for stream in self._streams:
            stream.stop_stream()
            stream.close()
        self._streams.clear()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_assistant.py -k mono -v
```

Expected: 5 passed

- [ ] **Step 5: Verify capture against real hardware**

Automated tests cannot reach the audio stack. Run this by hand:

```bash
uv run python -c "
import time, numpy as np
from assistant.capture import Capture
c = Capture(); c.start()
print('Play audio and talk for 5 seconds...')
time.sleep(5)
c.stop()
m, _ = c.mic.read_from(0); s, _ = c.sys.read_from(0)
print(f'mic  {len(m):>7} samples  peak {np.abs(m).max() if len(m) else 0:.3f}')
print(f'sys  {len(s):>7} samples  peak {np.abs(s).max() if len(s) else 0:.3f}')
"
```

Expected: both counts near 80000 (5s at 16 kHz), both peaks above 0.01.
A peak of 0.000 on `sys` means nothing was playing — play music and rerun.

- [ ] **Step 6: Verify the loud-failure path**

Confirm that a missing device raises rather than silently picking another one.
In Windows Sound settings, disable the default input device, then:

```bash
uv run python -c "from assistant.capture import Capture; Capture().start()"
```

Expected: `RuntimeError: No default input device.` Re-enable the device after.

- [ ] **Step 7: Commit**

```bash
git add assistant/capture.py tests/test_assistant.py
git commit -m "feat: add dual WASAPI capture with 16k mono conversion"
```

---

### Task 4: VAD gate and transcription

**Files:**
- Create: `assistant/transcribe.py`
- Test: `tests/test_assistant.py`

**Interfaces:**
- Consumes: `config.*`, `AudioRing`
- Produces: `Turn(src: str, text: str, ts: float)` frozen dataclass;
  `Transcript(max_turns: int)` with `append(turn) -> None`, `render() -> str`,
  `__len__`; `speech_spans(audio: np.ndarray) -> list[dict]`;
  `transcribe(audio: np.ndarray) -> str`;
  `gate_loop(ring, src, transcript, stop_event) -> None`

- [ ] **Step 1: Write the failing tests**

The silent-start test lives here. An idle output endpoint sends no packets, so
the gate sees an empty or all-zero buffer — that must read as silence, not as an
error and not as a timeout.

Append to `tests/test_assistant.py`:

```python
import time

from assistant.transcribe import Transcript, Turn, speech_spans


def test_silent_start_empty_buffer_is_silence():
    # The WASAPI trap: an idle endpoint delivers nothing at all. The gate must
    # treat that as ordinary silence — no exception, no special case.
    assert speech_spans(np.zeros(0, dtype=np.float32)) == []


def test_silent_start_digital_silence_is_silence():
    assert speech_spans(np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32)) == []


def test_sub_window_buffer_is_silence_not_an_error():
    # Silero needs 512 samples. Anything shorter must not reach it.
    assert speech_spans(np.zeros(100, dtype=np.float32)) == []


def test_transcript_renders_with_source_tags():
    t = Transcript(max_turns=10)
    t.append(Turn(src="them", text="what's your postgres experience", ts=1.0))
    t.append(Turn(src="me", text="about four years", ts=2.0))
    assert t.render() == (
        "[them] what's your postgres experience\n[me] about four years"
    )


def test_transcript_render_carries_no_timestamps():
    # Rendering a timestamp would rewrite frozen bytes on every press and
    # silently invalidate the whole cached prefix.
    now = time.time()
    t = Transcript(max_turns=10)
    t.append(Turn(src="them", text="hello", ts=now))
    assert t.render() == "[them] hello"
    assert str(int(now)) not in t.render()


def test_transcript_drops_oldest_past_the_cap():
    t = Transcript(max_turns=2)
    for i in range(4):
        t.append(Turn(src="me", text=f"turn{i}", ts=float(i)))
    assert t.render() == "[me] turn2\n[me] turn3"
    assert len(t) == 2


def test_turn_is_frozen():
    turn = Turn(src="me", text="hello", ts=1.0)
    with pytest.raises(Exception):
        turn.text = "changed"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_assistant.py -k "silent or transcript or turn or sub_window" -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'assistant.transcribe'`

- [ ] **Step 3: Write the implementation**

Create `assistant/transcribe.py`:

```python
"""VAD-gated local transcription into a tagged in-memory transcript."""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from . import config

# Silero processes 512-sample windows; anything shorter cannot be evaluated.
_MIN_VAD_SAMPLES = 512

# Our own trailing-silence timer closes utterances, so the VAD's internal
# silence threshold is set short and left to just find span boundaries.
_GATE_OPTS = VadOptions(
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=300,
    speech_pad_ms=200,
)

_SILENCE_SAMPLES = config.SAMPLE_RATE * config.SILENCE_END_MS // 1000
_MAX_UTTERANCE_SAMPLES = config.SAMPLE_RATE * config.MAX_UTTERANCE_S

_model: WhisperModel | None = None
_model_lock = threading.Lock()


@dataclass(frozen=True)
class Turn:
    """One utterance. Frozen: a turn's bytes must never change after append."""

    src: str    # "them" for system audio, "me" for the microphone
    text: str
    ts: float


class Transcript:
    def __init__(self, max_turns: int) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._lock = threading.Lock()

    def append(self, turn: Turn) -> None:
        with self._lock:
            self._turns.append(turn)

    def render(self) -> str:
        """Append-only text. No timestamps — they would break the cache prefix."""
        with self._lock:
            return "\n".join(f"[{t.src}] {t.text}" for t in self._turns)

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)


def speech_spans(audio: np.ndarray) -> list[dict]:
    """Speech regions in `audio`, or [] for silence.

    An idle WASAPI output endpoint sends no packets, so this is called with an
    empty array as a matter of course. That is silence, not an error.
    """
    if len(audio) < _MIN_VAD_SAMPLES:
        return []
    return get_speech_timestamps(audio, _GATE_OPTS)


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=config.WHISPER_THREADS,
        )
    return _model


def transcribe(audio: np.ndarray) -> str:
    """Decode one utterance. Serialized: one model shared by both gates."""
    with _model_lock:
        segments, _ = _get_model().transcribe(
            audio,
            language="en",
            beam_size=1,                      # greedy; beams cost latency we lack
            vad_filter=False,                 # already gated upstream
            condition_on_previous_text=False,  # avoids hallucination loops
        )
        return " ".join(s.text.strip() for s in segments).strip()


def gate_loop(
    ring,
    src: str,
    transcript: Transcript,
    stop_event: threading.Event,
) -> None:
    """Watch one ring, transcribe each utterance as its trailing silence lands."""
    pending = np.zeros(0, dtype=np.float32)
    cursor = 0

    while not stop_event.is_set():
        new, start = ring.read_from(cursor)
        if start > cursor:
            # We fell behind and audio was dropped. The partial utterance in
            # hand is no longer contiguous with what arrived, so discard it
            # rather than splice a gap into the middle of a sentence.
            pending = np.zeros(0, dtype=np.float32)
        cursor = start + len(new)
        if len(new):
            pending = np.concatenate((pending, new))

        spans = speech_spans(pending)
        if not spans:
            # Keep a short tail so a word starting right now is not clipped.
            # `cursor` tracks the ring, not `pending`, so trimming here is safe.
            if len(pending) > _SILENCE_SAMPLES:
                pending = pending[-_SILENCE_SAMPLES:]
            stop_event.wait(0.2)
            continue

        last_end = spans[-1]["end"]
        trailing_silence = len(pending) - last_end
        if trailing_silence >= _SILENCE_SAMPLES or len(pending) >= _MAX_UTTERANCE_SAMPLES:
            speech = pending[spans[0]["start"]:last_end]
            pending = pending[last_end:]
            text = transcribe(speech)
            if text:
                transcript.append(Turn(src=src, text=text, ts=time.time()))

        stop_event.wait(0.2)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_assistant.py -k "silent or transcript or turn or sub_window" -v
```

Expected: 7 passed

The first run downloads the Silero ONNX model — expect a pause of a few seconds.

- [ ] **Step 5: Verify transcription end to end**

The model download for `base.en` is about 150 MB on first run.

```bash
uv run python -c "
import threading, time
from assistant.capture import Capture
from assistant.transcribe import Transcript, gate_loop
from assistant import config

cap = Capture(); cap.start()
tr = Transcript(config.MAX_TURNS)
stop = threading.Event()
for ring, src in ((cap.mic, 'me'), (cap.sys, 'them')):
    threading.Thread(target=gate_loop, args=(ring, src, tr, stop), daemon=True).start()
print('Talk into the mic and play some speech for 20 seconds...')
time.sleep(20)
stop.set(); cap.stop()
print(tr.render() or '(nothing transcribed)')
"
```

Expected: tagged lines with recognizable text. If `[me]` is empty but `[them]`
works, the mic is not the Windows default input.

- [ ] **Step 6: Commit**

```bash
git add assistant/transcribe.py tests/test_assistant.py
git commit -m "feat: add VAD gate and local whisper transcription"
```

---

### Task 5: Prompts and request assembly

The cache prefix invariant is what this task exists to pin. A cache miss is
silent — it costs ten times as much and looks identical — so the invariant gets a
test rather than a comment.

**Files:**
- Create: `assistant/prompts.py`
- Create: `assistant/answer.py`
- Test: `tests/test_assistant.py`

**Interfaces:**
- Consumes: `config.NAME`, `config.api_key()`
- Produces: `prompts.SYSTEM: str`, `prompts.last_question_tail() -> str`,
  `prompts.query_tail(query: str) -> str`;
  `answer.build_request(transcript_text: str, tail: str) -> dict`,
  `answer.stream_answer(client, transcript_text, tail, ui_queue) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_assistant.py`:

```python
from datetime import datetime

from assistant import answer, prompts


def test_cache_prefix_is_identical_across_both_modes():
    # Both hotkeys must share one cached prefix. If this drifts, every press
    # pays full price and nothing in the output looks wrong.
    text = "[them] what's your postgres experience"
    a = answer.build_request(text, prompts.last_question_tail())
    b = answer.build_request(text, prompts.query_tail("migrations?"))

    assert a["system"] == b["system"]
    assert a["messages"][0]["content"][0] == b["messages"][0]["content"][0]
    assert a["messages"][0]["content"][1] != b["messages"][0]["content"][1]


def test_system_prompt_carries_nothing_volatile():
    assert str(datetime.now().year) not in prompts.SYSTEM
    assert "{" not in prompts.SYSTEM and "}" not in prompts.SYSTEM


def test_system_prompt_is_stable_across_calls():
    first = answer.build_request("x", "tail one")["system"]
    second = answer.build_request("y", "tail two")["system"]
    assert first == second


def test_exactly_two_cache_breakpoints():
    req = answer.build_request("transcript", "tail")
    marked = [req["system"][0]]
    marked += [b for b in req["messages"][0]["content"] if "cache_control" in b]
    assert len(marked) == 2, "one on system, one on the transcript, none on the tail"
    assert all(b["cache_control"]["ttl"] == "1h" for b in marked)


def test_transcript_is_a_single_block():
    # One block per turn would exhaust the 20-block cache lookback in about ten
    # exchanges and stop hitting, silently.
    req = answer.build_request("[them] a\n[me] b\n[them] c", "tail")
    assert len(req["messages"][0]["content"]) == 2


def test_request_omits_the_parameters_that_400_on_opus_5():
    req = answer.build_request("t", "tail")
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in req


def test_request_pins_fast_mode_and_effort():
    req = answer.build_request("t", "tail")
    assert req["model"] == "claude-opus-5"
    assert req["speed"] == "fast"
    assert req["betas"] == ["fast-mode-2026-02-01"]
    assert req["output_config"] == {"effort": "low"}
    assert req["max_tokens"] == 2000


def test_query_tail_embeds_the_typed_question():
    assert "how do you handle migrations" in prompts.query_tail(
        "how do you handle migrations"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_assistant.py -k "cache or system_prompt or request or query_tail or transcript_is" -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'assistant.prompts'`

- [ ] **Step 3: Write the prompts module**

Create `assistant/prompts.py`. `NAME` is substituted with `.replace` rather than
an f-string so the prompt body can contain braces without escaping, and so there
is no format call anywhere near the cached prefix:

```python
"""Prompt text. SYSTEM is built once at import and never rebuilt."""

from . import config

_ROLE = """You are a live assistant for {NAME} during a technical conversation.
Someone else is talking; you produce notes {NAME} glances at and speaks from.

OUTPUT SHAPE
- 3-5 bullets, roughly 12 words max each.
- Fragments, not sentences. Keywords over prose.
- Concrete: name the tool, the number, the tradeoff. Never a generality.
- Strongest point first. They may only get through bullet one.
- Any caveat goes last, four words.

These are cue cards, not a script. Anything that reads like written prose
defeats the purpose: it will get read verbatim and sound read.

IF YOU DON'T KNOW
Say what you'd do to find out. "Haven't used X, closest is Y" is a real
answer. Never invent a specific."""

_HUMANIZER = """VOCABULARY. These mark an answer as machine-written. Don't use:
leverage, delve, robust, seamless, comprehensive, utilize, streamline,
holistic, cutting-edge, unlock, supercharge, empower.
Use the plain verb. "Use", not "leverage". "Cuts export from four clicks
to one", not "streamlines the workflow".

NEVER OPEN WITH: "Great question", "Absolutely", "Certainly",
"I'd be happy to". Start on the answer.

NEVER CLOSE WITH: "In conclusion", "In summary", or a bullet that
restates the other bullets.

BANNED CADENCE: "It's not just X, it's Y" and "not only X but also Y".
State the real claim and drop the negation - it's there to inflate a
thin point.

RULE OF THREE: don't force a third parallel item when you have two.
Two is fine. Four is fine.

VARY BULLET LENGTH. Five bullets of identical length and shape is the
strongest tell in a list. Put a four-word bullet next to a twelve-word one.

HAVE A POSITION. "I'd start with X, though Y if writes dominate" beats
listing options neutrally. They're asking to hear judgment, not a menu.

UNCERTAINTY IS HUMAN. "Not sure at that scale" is a real bullet.
Confident filler is the failure mode; admitting a gap isn't."""

# Built once at import. Never rebuild this per request — one stray byte and the
# cache misses silently at ten times the price.
SYSTEM = _ROLE.replace("{NAME}", config.NAME) + "\n\n" + _HUMANIZER

_PREAMBLE = (
    "Transcript above is live. [them] is the other person, [me] is you.\n"
)


def last_question_tail() -> str:
    return _PREAMBLE + (
        "Find the last question [them] asked that's aimed at [me] and still "
        "unanswered, and answer it. Several asked? Take the last. Nothing "
        "asked? Give the single most useful thing to say next."
    )


def query_tail(query: str) -> str:
    return _PREAMBLE + (
        "Answer this using the transcript as context: " + query
    )
```

- [ ] **Step 4: Write the answer module**

Create `assistant/answer.py`:

```python
"""Request assembly and token streaming."""

import logging
import queue

from . import prompts

log = logging.getLogger(__name__)

_CACHE = {"type": "ephemeral", "ttl": "1h"}


def build_request(transcript_text: str, tail: str) -> dict:
    """Assemble the request. Two cache breakpoints, volatile tail last.

    No temperature, top_p, top_k, or thinking key: the first three are a 400 on
    Opus 5, and disabling thinking leaks <thinking> tags into visible output.
    max_tokens covers thinking plus text together, so it is sized well above
    the four bullets the prompt asks for.
    """
    return {
        "model": "claude-opus-5",
        "max_tokens": 2000,
        "speed": "fast",
        "betas": ["fast-mode-2026-02-01"],
        "output_config": {"effort": "low"},
        "system": [
            {"type": "text", "text": prompts.SYSTEM, "cache_control": _CACHE}
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": transcript_text, "cache_control": _CACHE},
                    {"type": "text", "text": tail},
                ],
            }
        ],
    }


def stream_answer(
    client,
    transcript_text: str,
    tail: str,
    ui_queue: "queue.Queue",
) -> None:
    """Stream one answer onto the UI queue. Runs on its own thread per press."""
    ui_queue.put(("clear", None))
    try:
        with client.beta.messages.stream(
            **build_request(transcript_text, tail)
        ) as stream:
            for text in stream.text_stream:
                ui_queue.put(("token", text))
            final = stream.get_final_message()
    except Exception as exc:                     # surface it, never swallow it
        log.exception("answer failed")
        ui_queue.put(("token", f"\n[error: {type(exc).__name__}]"))
        ui_queue.put(("done", None))
        return

    ui_queue.put(("done", None))

    # A cache miss is invisible in the output and costs 10x. Zero cache reads
    # across consecutive presses means an invalidator got into the prefix.
    usage = final.usage
    log.info(
        "cache_read=%s cache_write=%s in=%s out=%s stop=%s",
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        usage.input_tokens,
        usage.output_tokens,
        final.stop_reason,
    )
    if final.stop_reason == "max_tokens":
        log.warning("answer truncated at max_tokens - raise it in answer.py")
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_assistant.py -v
```

Expected: all tests pass

- [ ] **Step 6: Verify a real answer streams and the cache engages**

Two presses against the same transcript. The second must report a non-zero
`cache_read`:

```bash
uv run python -c "
import logging, queue, anthropic
from assistant import answer, prompts, config
logging.basicConfig(level=logging.INFO)
client = anthropic.Anthropic(api_key=config.api_key())
t = '[them] so how would you scale the write path on this?'
for i in (1, 2):
    q = queue.Queue()
    answer.stream_answer(client, t, prompts.last_question_tail(), q)
    out = ''.join(v for k, v in iter(q.get, ('done', None)) if k == 'token')
    print(f'--- press {i} ---'); print(out)
"
```

Expected: bullets both times, and `cache_read=` above 0 on the second press.
A second `cache_read=0` means the prefix is not stable — check for anything
volatile in `SYSTEM`.

- [ ] **Step 7: Commit**

```bash
git add assistant/prompts.py assistant/answer.py tests/test_assistant.py
git commit -m "feat: add prompts and cached request assembly"
```

---

### Task 6: Overlay with capture exclusion

**Files:**
- Create: `assistant/overlay.py`
- Test: manual (Step 5) — this is a Win32 call against a live window, and mocking
  it would only prove the mock works

**Interfaces:**
- Consumes: `config.EXCLUDE_FROM_CAPTURE`
- Produces: `Overlay(ui_queue: queue.Queue)` with `.root: tk.Tk`,
  `.run() -> None`, `.toggle_visible() -> None`

- [ ] **Step 1: Write the implementation**

Create `assistant/overlay.py`. The `use_last_error=True` on the library load is
load-bearing: plain `ctypes.windll` does not capture the Win32 error, so
`ctypes.get_last_error()` returns 0 and the raised error reads as success —
exactly the silent failure the loud-failure rule exists to prevent.

```python
"""Always-on-top cue card overlay, excluded from screen capture."""

import ctypes
import queue
import tkinter as tk

from . import config

# use_last_error is required. Loading through plain ctypes.windll leaves
# ctypes.get_last_error() at 0 even when the call failed, so the raised error
# would say "success" while the overlay sits visible in the screen share.
_user32 = ctypes.WinDLL("user32", use_last_error=True)

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

_BG = "#101014"
_FG = "#e8e8ea"


def set_capture_excluded(root: tk.Tk, excluded: bool) -> None:
    """Raises on failure. The caller must not show the window if this raises."""
    # winfo_id() is usually the CHILD hwnd and the toplevel is its parent, but
    # an overrideredirect window may have no wrapper — GetParent returns 0 and
    # winfo_id() is already the toplevel. Fall back rather than pass 0.
    child = root.winfo_id()
    hwnd = _user32.GetParent(child) or child
    flag = WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
    if not _user32.SetWindowDisplayAffinity(hwnd, flag):
        raise ctypes.WinError(ctypes.get_last_error())


class Overlay:
    """Owns the main thread. Nothing else touches a widget."""

    def __init__(self, ui_queue: "queue.Queue") -> None:
        self.ui_queue = ui_queue
        self.root = tk.Tk()
        self.root.title("assistant")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=_BG)
        self.root.geometry("460x300+40+40")

        self.text = tk.Text(
            self.root,
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 12),
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=14,
            pady=12,
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        self._visible = True
        # Fully transparent until exclusion is confirmed. The window has to be
        # mapped before it has an HWND, so it cannot simply stay withdrawn —
        # alpha 0 gets it mapped without ever painting a visible frame.
        self.root.attributes("-alpha", 0.0)
        self.root.after(0, self._apply_exclusion_then_show)
        self.root.after(50, self._drain)

    def _apply_exclusion_then_show(self) -> None:
        self.root.update_idletasks()   # force the HWND into existence
        if config.EXCLUDE_FROM_CAPTURE:
            try:
                set_capture_excluded(self.root, True)
            except OSError as exc:
                self.root.destroy()
                raise RuntimeError(
                    "Screen-capture exclusion failed; refusing to show the "
                    "overlay unprotected. Needs Windows 10 build 19041+."
                ) from exc
        self.root.attributes("-alpha", 1.0)

    def _write(self, chunk: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", chunk)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "clear":
                    self._clear()
                elif kind == "token":
                    self._write(payload)
                elif kind == "status":
                    self._clear()
                    self._write(payload)
                elif kind == "toggle":
                    self.toggle_visible()
                elif kind == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(50, self._drain)

    def toggle_visible(self) -> None:
        # Alpha, not withdraw/deiconify: unmapping can hand back a different
        # HWND on remap, and display affinity lives on the HWND. Hiding must
        # not be able to silently drop capture exclusion.
        self._visible = not self._visible
        self.root.attributes("-alpha", 1.0 if self._visible else 0.0)

    def run(self) -> None:
        self.root.mainloop()
```

- [ ] **Step 2: Verify the overlay renders**

```bash
uv run python -c "
import queue, threading, time
from assistant.overlay import Overlay
q = queue.Queue()
o = Overlay(q)
def feed():
    time.sleep(1)
    for chunk in ['- Postgres, four years\n', '- Partitioned the writes table\n', '- Not sure past 10k TPS\n']:
        q.put(('token', chunk)); time.sleep(0.4)
threading.Thread(target=feed, daemon=True).start()
o.run()
"
```

Expected: a dark always-on-top box, top left, filling in line by line.
Close it with Alt+F4 or Ctrl+C in the terminal.

- [ ] **Step 3: Verify capture exclusion — the security check**

With the overlay from Step 2 still running, press `Win+Shift+S` and drag a
selection over the overlay region.

Expected: the captured image shows the desktop behind the overlay. The overlay
must be **absent**.

If it appears in the capture, stop and diagnose before continuing. Two known
causes: passing `winfo_id()` without `GetParent`, and calling before the window
is mapped.

- [ ] **Step 4: Verify the loud-failure path**

Confirm a failed exclusion refuses to show rather than coming up unprotected:

```bash
uv run python -c "
import queue, assistant.overlay as ov
ov.WDA_EXCLUDEFROMCAPTURE = 0x00000099   # invalid flag, forces the call to fail
try:
    ov.Overlay(queue.Queue()).run()
    print('FAIL: overlay showed anyway')
except RuntimeError as e:
    print('PASS:', e)
"
```

Expected: `PASS: Screen-capture exclusion failed; ...` and no window appears.

- [ ] **Step 5: Commit**

```bash
git add assistant/overlay.py
git commit -m "feat: add capture-excluded always-on-top overlay"
```

---

### Task 7: Wiring and hotkeys

**Files:**
- Create: `assistant/main.py`
- Modify: `pyproject.toml` (add the script entry point)

**Interfaces:**
- Consumes: everything above
- Produces: `main() -> None`, and a `meeting-assistant` console script

- [ ] **Step 1: Write the implementation**

Create `assistant/main.py`. Hotkey callbacks run on pynput's own thread, so they
only ever put onto a queue or spawn a thread — never touch a widget:

```python
"""Wiring: capture, gates, hotkeys, overlay."""

import logging
import queue
import threading
import tkinter as tk
import tkinter.simpledialog

import anthropic

from . import answer, config, prompts
from .capture import Capture
from .overlay import Overlay
from .transcribe import Transcript, gate_loop

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    client = anthropic.Anthropic(api_key=config.api_key())
    transcript = Transcript(config.MAX_TURNS)
    ui_queue: queue.Queue = queue.Queue()
    stop = threading.Event()

    capture = Capture()
    capture.start()
    for ring, src in ((capture.mic, "me"), (capture.sys, "them")):
        threading.Thread(
            target=gate_loop,
            args=(ring, src, transcript, stop),
            daemon=True,
            name=f"gate-{src}",
        ).start()

    overlay = Overlay(ui_queue)

    def answer_thread(tail: str) -> None:
        answer.stream_answer(client, transcript.render(), tail, ui_queue)

    def on_last_question() -> None:
        if len(transcript) == 0:
            ui_queue.put(("status", "nothing transcribed yet"))
            return
        threading.Thread(
            target=answer_thread, args=(prompts.last_question_tail(),), daemon=True
        ).start()

    def on_query() -> None:
        # Prompt on the main thread; the hotkey fires on pynput's thread.
        def ask() -> None:
            q = tk.simpledialog.askstring("query", "ask:", parent=overlay.root)
            if q:
                threading.Thread(
                    target=answer_thread, args=(prompts.query_tail(q),), daemon=True
                ).start()

        overlay.root.after(0, ask)

    def on_toggle() -> None:
        ui_queue.put(("toggle", None))

    def on_quit() -> None:
        ui_queue.put(("quit", None))

    from pynput import keyboard

    hotkeys = keyboard.GlobalHotKeys(
        {
            "<ctrl>+<space>": on_last_question,
            "<ctrl>+<shift>+<space>": on_query,
            "<ctrl>+<alt>+h": on_toggle,
            "<ctrl>+<alt>+q": on_quit,
        }
    )
    hotkeys.start()

    log.info(
        "ready - ctrl+space answer, ctrl+shift+space query, "
        "ctrl+alt+h hide, ctrl+alt+q quit"
    )
    try:
        overlay.run()          # blocks: Tkinter owns the main thread
    finally:
        stop.set()
        hotkeys.stop()
        capture.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the entry point**

Append to `pyproject.toml`:

```toml
[project.scripts]
meeting-assistant = "assistant.main:main"
```

- [ ] **Step 3: Run the full test suite**

```bash
uv run pytest tests/test_assistant.py -v
```

Expected: all tests pass

- [ ] **Step 4: Verify end to end**

Set the key, then run against a real conversation — a YouTube interview through
the speakers works for a solo test:

```bash
export ASSISTANT_NAME="Pouria"
uv run meeting-assistant
```

Check each of these:
1. Overlay appears, top left, always on top.
2. Speech through the speakers turns into `[them]` turns (watch the log).
3. Speech into the mic turns into `[me]` turns.
4. `Ctrl+Space` streams bullets within a few seconds.
5. `Ctrl+Shift+Space` opens a box, and a typed question gets answered.
6. A second press logs `cache_read=` above 0.
7. `Win+Shift+S` over the overlay captures the desktop behind it.
8. `Ctrl+Alt+Q` exits cleanly with no stuck audio device.

- [ ] **Step 5: Commit**

```bash
git add assistant/main.py pyproject.toml
git commit -m "feat: wire capture, transcription, hotkeys, and overlay"
```

---

### Task 8: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write it**

```markdown
# Local meeting assistant

Captures both sides of a call, transcribes locally with whisper, and puts cue
cards in an overlay that screen capture cannot see.

Design: `docs/superpowers/specs/2026-08-13-meeting-assistant-design.md`

## Requirements

Windows 10 build 19041 or later. CPU only — no GPU needed.

## Setup

    uv sync
    export ANTHROPIC_API_KEY=sk-ant-...
    export ASSISTANT_NAME="Your Name"
    uv run meeting-assistant

## Hotkeys

| Key | Does |
|---|---|
| `Ctrl+Space` | Answer the last question they asked |
| `Ctrl+Shift+Space` | Answer a typed question, using the transcript |
| `Ctrl+Alt+H` | Hide or show the overlay |
| `Ctrl+Alt+Q` | Quit |

## Tuning

Everything adjustable is in `assistant/config.py`.

- Transcription too inaccurate: `WHISPER_MODEL = "distil-small.en"`
- Machine bogged down: lower `WHISPER_THREADS`
- Cutting you off mid-sentence: raise `SILENCE_END_MS`
- Debugging the overlay: `EXCLUDE_FROM_CAPTURE = False`, because you cannot
  screenshot the overlay while exclusion is on

## What it does not hide

Exclusion covers screen capture and screen share. It does not cover an HDMI
capture card, a camera pointed at the screen, or your eyes tracking text.

## Scope

Personal use. Pointing this at work calls reopens the consent and
recording-policy question, which is a separate decision.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Notes for the executor

**Watch the log on the first real conversation.** Three numbers say whether the
design is working: `cache_read` above zero from the second press on, the gap
between someone finishing a sentence and their turn appearing (should be under
four seconds), and `stop_reason` never reading `max_tokens`.

**Expect to tune the vocabulary banlist.** The spec flags it: prohibition lists
can anchor toward what they ban. If the bullets come back stilted after about
twenty real answers, `_HUMANIZER` in `prompts.py` is the first suspect, not the
last.

**`WHISPER_THREADS = 4` is a guess.** It is a 15 W part with two performance
cores, sharing the CPU with a live video call. If the call itself starts
stuttering, drop it to 2 before touching anything else.
