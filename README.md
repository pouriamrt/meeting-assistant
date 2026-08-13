# Local meeting assistant

Captures both sides of a call, transcribes locally with whisper, and puts cue
cards in an overlay that screen capture cannot see.

Design: `docs/superpowers/specs/2026-08-13-meeting-assistant-design.md`

## Requirements

Windows 10 build 19041 or later. CPU only — no GPU needed.

**Use headphones.** On speakers the microphone hears the other person's voice
coming out of them, so every question they ask is transcribed twice — once as
`[them]` and again as `[me]`. That breaks the core feature: `Ctrl+Space` looks
for the last question `[them]` asked that `[me]` has not answered, and a
duplicate under your own tag makes it look answered. Measured, not theoretical —
a sentence played through the speakers came back under both tags.

## Setup

    uv sync
    export ANTHROPIC_API_KEY=sk-ant-...
    export ASSISTANT_NAME="Your Name"
    uv run meeting-assistant

Missing `ANTHROPIC_API_KEY` will fail immediately on startup with a clear error.

## Hotkeys

| Key | Does |
|---|---|
| `Ctrl+Space` | Answer the last question they asked |
| `Ctrl+Shift+Space` | Answer a typed question, using the transcript |
| `Ctrl+Alt+H` | Hide or show the overlay |
| drag the box | Move it — there is no titlebar to grab |
| `Ctrl+Alt+Q` | Quit |

## Tuning

Everything adjustable is in `assistant/config.py`.

- Transcription too inaccurate: `WHISPER_MODEL = "distil-small.en"`
- Machine bogged down: lower `WHISPER_THREADS`
- Cutting you off mid-sentence: raise `SILENCE_END_MS` (700ms today; a
  natural pause mid-question otherwise splits it into two turns)
- Want lower latency at some CPU cost: lower `GATE_POLL_S`
- Debugging the overlay: `EXCLUDE_FROM_CAPTURE = False`, because you cannot
  screenshot the overlay while exclusion is on
- Got fast-mode access: `FAST_MODE = True`. It is off by default because
  without the entitlement every request fails with `rate limit of 0 fast mode
  input tokens per minute` — unusable, not just slower

## What it does not hide

Exclusion covers screen capture and screen share. It does not cover an HDMI
capture card, a camera pointed at the screen, or your eyes tracking text.

## Scope

Personal use. Pointing this at work calls reopens the consent and
recording-policy question, which is a separate decision.
