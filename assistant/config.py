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

# Send screenshots at half resolution.
#
# This is a COST saving, not a speed fix. Controlled A/B on one 1536x864
# monitor, same question, four runs each alternating: input 2847 -> 856 tokens
# (about $0.010 a press, deterministic), TTFT median 2.05s -> 1.80s, total
# median 3.43s -> 3.34s. The totals overlap run to run — one full-res run came
# back faster than both half-res runs — so treat the wall-clock difference as
# noise and the token difference as real.
#
# An earlier "9.2s -> 4.3s" figure was wrong: it compared two different
# questions with different answer lengths, not the same one at two
# resolutions.
#
# Legibility was tested rather than assumed — at half resolution the model
# still transcribed a 14pt IDE-font stub correctly, one hyphen aside. Set
# False if a real interview pad ever reads worse.
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
