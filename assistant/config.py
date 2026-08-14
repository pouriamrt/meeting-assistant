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
GATE_POLL_S = 0.1           # how often the gate checks for new audio
MAX_UTTERANCE_S = 30        # force a transcribe on an unbroken monologue

MAX_TURNS = 400

# False to debug the overlay — with exclusion on you cannot screenshot it.
EXCLUDE_FROM_CAPTURE = True

# Send screenshots at half resolution. Measured on this machine, one 1536x864
# monitor: TTFT 2.37s -> 2.03s, total 9.24s -> 8.09s, input 2859 -> 868 tokens
# (about $0.010 a press). Legibility was tested rather than assumed — at half
# resolution the model still transcribed a 14pt IDE-font stub correctly, one
# hyphen aside. Set False if a real interview pad ever reads worse.
#
# Do not expect much more than this: output dominates the wall clock. Screen
# answers run ~400 tokens at roughly 50 tok/s, so most of those 8 seconds is
# generation, not the image.
SCREEN_HALF_RES = True

# Fast mode is a research preview and needs the org to be allowlisted for it.
# Without that entitlement EVERY request 429s with "rate limit of 0 fast mode
# input tokens per minute" - the app is unusable rather than merely slower, so
# this defaults off. Turn it on if you get access: it buys up to 2.5x output
# tokens/sec, which is the difference between reading a cue card mid-sentence
# and after it.
FAST_MODE = False


def api_key() -> str:
    """Read the key at call time so it never lives in module state."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in the environment; "
            "never hardcode it."
        )
    return key
