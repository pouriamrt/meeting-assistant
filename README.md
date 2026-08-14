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
| `Ctrl+Alt+Space` | Read the screen and answer what's on it, in code |
| `Ctrl+Alt+H` | Hide or show the overlay |
| drag the box | Move it — there is no titlebar to grab |
| drag the corner grip | Resize it — the small square, bottom right |
| `Ctrl` + scroll wheel | Bigger or smaller text, 8pt to 32pt |
| `Ctrl+Alt+Q` | Quit |

## Screen mode

`Ctrl+Alt+Space` sends a screenshot of every monitor **and** the transcript.
The screen carries the signature, the starter code, the error. The transcript
carries the constraints that are only ever spoken — "no built-ins", "optimize
for memory". Neither alone is enough, which is why both go.

It answers under a second system prompt that writes real code instead of the
spoken cue cards `Ctrl+Space` produces. On a diagram or a dashboard it falls
back to bullets on its own.

Measured, one monitor at 1536x864: 140 ms to grab, 407 KB of PNG, about 7
seconds to a finished answer, roughly $0.025 a press. Output tokens are most
of that — code is much longer than four bullets.

Monitors are sent as one image each, never stitched into one wide picture.
Two 1920-wide screens stitched are 3840 across, and the API scales anything
past 1568 on the long edge, which lands the pair at 441 px tall — too short to
read code off. Separate images each get their own scaling budget.

The overlay never appears in the screenshot: `WDA_EXCLUDEFROMCAPTURE` hides it
from this capture path too, so the model cannot read its own last answer back.
Verified by sampling the pixel under the overlay with exclusion on and off.

## Tuning

Everything adjustable is in `assistant/config.py`.

- Transcription too inaccurate: `WHISPER_MODEL = "distil-small.en"`
- Machine bogged down: lower `WHISPER_THREADS`
- Cutting you off mid-sentence: raise `SILENCE_END_MS` (700ms today; a
  natural pause mid-question otherwise splits it into two turns)
- Want lower latency at some CPU cost: lower `GATE_POLL_S`
- Debugging the overlay: `EXCLUDE_FROM_CAPTURE = False`, because you cannot
  screenshot the overlay while exclusion is on
- Overlay too small or text too fine: drag the corner grip, or `Ctrl`+scroll.
  Neither persists across restarts — change `geometry(...)` and `_FONT` in
  `assistant/overlay.py` to move the defaults
- Got fast-mode access: `FAST_MODE = True`. It is off by default because
  without the entitlement every request fails with `rate limit of 0 fast mode
  input tokens per minute` — unusable, not just slower

## What it does not hide

Exclusion covers screen capture and screen share. It does not cover an HDMI
capture card, a camera pointed at the screen, or your eyes tracking text.

## Scope

Personal use. Pointing this at work calls reopens the consent and
recording-policy question, which is a separate decision.
