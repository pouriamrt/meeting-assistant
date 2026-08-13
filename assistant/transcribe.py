"""VAD-gated local transcription into a tagged in-memory transcript."""

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from . import config
from .audio_ring import AudioRing

log = logging.getLogger(__name__)

# Defensive, not load-bearing: the library already returns [] on a short
# buffer. Silero's window is 512 samples and a shorter buffer can't contain
# speech worth evaluating, so the call is skipped rather than relied on to
# no-op.
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

# Ticks of a completely unchanged ring before the stall warning fires.
# Derived from the poll interval rather than hardcoded, so changing the poll
# rate cannot silently retune this to 15s or 60s.
_STALL_TICKS = int(30 / config.GATE_POLL_S)

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
    return get_speech_timestamps(audio, _GATE_OPTS, sampling_rate=config.SAMPLE_RATE)


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


def _is_degenerate(text: str) -> bool:
    """True for whisper's loop-on-near-silence output.

    Observed live: a 0.8s low-energy segment decoded as "There it goes."
    repeated twenty-nine times. `condition_on_previous_text=False` reduces
    this but does not eliminate it. Left in, it pollutes the transcript, the
    prompt built from it, and the cached prefix - so it is dropped at the
    source rather than filtered downstream.
    """
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    return len(parts) >= 4 and len(set(parts)) == 1


def warm_model() -> None:
    """Load whisper now rather than on the first utterance.

    Construction costs about a second on this machine, and lazily it lands on
    the first question of the meeting - the one moment latency is least
    affordable.
    """
    _get_model()


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
        text = " ".join(s.text.strip() for s in segments).strip()
    if _is_degenerate(text):
        log.info("dropped a degenerate decode: %.60s...", text)
        return ""
    return text


def gate_loop(
    ring: AudioRing,
    src: str,
    transcript: Transcript,
    stop_event: threading.Event,
) -> None:
    """Watch one ring, transcribe each utterance as its trailing silence lands."""
    pending = np.zeros(0, dtype=np.float32)
    cursor = 0
    errors = 0
    # Detects a stream that has stopped delivering entirely - unplug a
    # headset or switch the default device mid-meeting and WASAPI simply
    # stops invoking the callback, no exception, nothing for `errors` to
    # count. total_written only advances on real writes, so a run of ticks
    # with no change means nothing arrived at all.
    last_total, stalled_ticks = ring.total_written, 0
    last_arrival = time.monotonic()

    while not stop_event.is_set():
        try:
            new, start = ring.read_from(cursor)
            if start > cursor:
                # We fell behind and audio was dropped. The partial utterance
                # in hand is no longer contiguous with what arrived, so
                # discard it rather than splice a gap into a sentence.
                pending = np.zeros(0, dtype=np.float32)
            cursor = start + len(new)
            if len(new):
                pending = np.concatenate((pending, new))
                last_arrival = time.monotonic()

            spans = speech_spans(pending)
            if not spans:
                # Keep a short tail so a word starting right now is not
                # clipped. `cursor` tracks the ring, not `pending`, so
                # trimming here is safe.
                if len(pending) > _SILENCE_SAMPLES:
                    pending = pending[-_SILENCE_SAMPLES:]
            else:
                last_end = spans[-1]["end"]
                trailing = len(pending) - last_end
                # Silence measured two ways, because on the loopback channel
                # only one of them ever happens. Trailing silence inside the
                # buffer needs packets to keep arriving to accumulate - but an
                # idle WASAPI output endpoint sends nothing at all, so when the
                # other person stops talking their last utterance stops growing
                # and would sit here undelivered until they spoke again. That is
                # exactly the moment you press the hotkey to answer them, so the
                # one turn you need would be the one missing. Wall-clock silence
                # since the last packet closes it instead.
                idle = (time.monotonic() - last_arrival) * 1000
                if (
                    trailing >= _SILENCE_SAMPLES
                    or idle >= config.SILENCE_END_MS
                    or len(pending) >= _MAX_UTTERANCE_SAMPLES
                ):
                    speech = pending[spans[0]["start"]:last_end]
                    pending = pending[last_end:]
                    text = transcribe(speech)
                    if text:
                        transcript.append(Turn(src=src, text=text, ts=time.time()))
        except Exception:
            # This runs on a daemon thread, and both calls above reach native
            # libraries (ONNX for the VAD, CTranslate2 for the decode) that
            # can throw under CPU pressure or on a malformed frame. An
            # uncaught exception kills the thread, and that source's
            # transcription is then silently dead for the rest of the
            # session — indistinguishable from nobody talking. Same reasoning
            # as the capture callback: stay alive, drop the buffer that
            # failed so a poison frame cannot spin the loop, and log the
            # first failure plus every hundredth.
            errors += 1
            if errors == 1 or errors % 100 == 0:
                log.exception("gate loop failed on %s (%d errors)", src, errors)
            pending = np.zeros(0, dtype=np.float32)

        total = ring.total_written
        stalled_ticks = 0 if total != last_total else stalled_ticks + 1
        last_total = total
        if stalled_ticks == _STALL_TICKS:
            # An idle output endpoint legitimately delivers nothing during
            # quiet stretches, so `them` will trip this - warning, not
            # error, and 30s is long enough that ordinary silence won't.
            log.warning(
                "%s capture has delivered nothing for 30s - device gone?", src
            )

        stop_event.wait(config.GATE_POLL_S)
