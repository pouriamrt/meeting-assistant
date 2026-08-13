"""VAD-gated local transcription into a tagged in-memory transcript."""

import logging
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
    ring: AudioRing,
    src: str,
    transcript: Transcript,
    stop_event: threading.Event,
) -> None:
    """Watch one ring, transcribe each utterance as its trailing silence lands."""
    pending = np.zeros(0, dtype=np.float32)
    cursor = 0
    errors = 0

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
                if trailing >= _SILENCE_SAMPLES or len(pending) >= _MAX_UTTERANCE_SAMPLES:
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

        stop_event.wait(0.2)
