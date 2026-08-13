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
