# Local meeting assistant — design

Date: 2026-08-13
Status: implemented on `feat/meeting-assistant`; amended where review proved
the design wrong (see Caching, Testing, and the note in Prompts)

## What this is

A local assistant for live technical conversations on Windows. It listens to both
sides of a call, transcribes locally, and on a hotkey puts cue cards in an
always-on-top overlay that is invisible to screen capture.

It replaces SMOG. SMOG's free plan streams raw audio to a realtime model, which
costs roughly 800-1100 tokens per answer against a hard 8000/day cap, and bills
per second of audio even while nobody is talking. Transcribing locally makes the
audio free and bills only the text answer, so the daily ceiling disappears.

## Hardware constraint

This shapes more of the design than anything else, so it goes first.

```
GPU  Intel Iris Xe (integrated) — no CUDA
CPU  i7-1255U — 2 P-core + 8 E-core, 15W ultrabook part
RAM  15.7 GB
```

There is no NVIDIA GPU. Whisper runs CPU-only int8 through CTranslate2. Measured
ceiling for the ~30s segments that matter for live use:

| model | decode | verdict |
|---|---|---|
| `tiny.en` | ~1-2s | usable, sloppy |
| `base.en` | ~2-4s | usable — the default |
| `distil-small.en` | ~3-5s | usable, better accuracy |
| `small.en` | ~6-10s | borderline |
| `medium`, `large-v3` | 30s+ | unusable |

Second constraint: this is a 15W part that throttles, and transcription competes
with the video call already running. Budget for contention, not the benchmark.

Start on `base.en`. Move to `distil-small.en` if accuracy hurts more than latency.

## Architecture

```
MAIN THREAD                     Tkinter mainloop (must own main thread)
  └── root.after(50ms) ──────── drains ui_queue, paints overlay

THREAD mic ────► ring_mic ──┐
THREAD loopback ► ring_sys ─┤
                            ▼
THREAD stt        silero VAD gate ─► speech? ─► faster-whisper int8
                            │                        │
                       silence: ~0% CPU              ▼
                                            transcript.append(
                                              Turn(src="them"|"me", text, ts)
                                            )

HOTKEY LISTENER (pynput, own thread)
  Ctrl+Space ───────► answer thread, mode="last_question"
  Ctrl+Shift+Space ─► inline entry ─► answer thread, mode="query"
  Ctrl+Alt+H ───────► hide / show
  Ctrl+Alt+Q ───────► quit

THREAD answer (transient, one per press)
  build prompt ─► client.beta.messages.stream(...) ─► ui_queue.put(token)
```

Only the main thread touches widgets. Everything else hands work across a
`queue.Queue`. That single rule is the whole thread-safety story.

### Why threads

Tkinter must own the main thread on Windows. `faster-whisper` runs CTranslate2,
which is C++ and releases the GIL, so a worker thread gets real parallelism.

asyncio was rejected: tk's mainloop and the event loop both want the thread, and
integrating them means driving `root.update()` by hand for no gain.
Multiprocessing was rejected: it duplicates the model load and buys nothing a
GIL-releasing thread doesn't already provide.

### Why VAD-gated continuous transcription

The alternative was an on-demand ring buffer — hold 60s of raw audio, transcribe
only on a press. That keeps CPU at zero but adds 3-5s of STT to a press that
already waits 3-8s on the model.

Gating a continuous loop behind Silero VAD gets the fast press-to-answer while
silence still costs nothing. It also disarms the WASAPI trap for free: an idle
output endpoint sends no packets, which the VAD gate reads as silence like any
other. Starting into a dead endpoint is the normal path, not an error path.

### Rejected: pipecat

Pipecat covers neither WASAPI loopback nor dual tagged streams, which are the two
hardest parts here, and its push-driven pipeline fights a hotkey-triggered
design. Using it would mean writing a custom transport anyway and inheriting a
frame framework to save about 55 lines of wrapper. Revisit if the assistant ever
needs to speak back.

### Rejected: Rust

Whisper and VAD are already C++ in either language, and CTranslate2 releases the
GIL, so the usual no-GIL argument doesn't apply. There is no official Anthropic
SDK for Rust, so streaming, cache breakpoints, and the fast-mode beta flag would
all be hand-rolled against raw HTTP. Revisit if a single distributable `.exe`
becomes a goal.

## Modules

| File | Owns | Depends on |
|---|---|---|
| `capture.py` | dual WASAPI capture, two ring buffers | `pyaudiowpatch`, `soxr` |
| `transcribe.py` | VAD gate, whisper worker, tagged transcript | `faster-whisper` |
| `prompts.py` | system prompt, humanizer block, mode tails | — |
| `answer.py` | prompt assembly, cache breakpoints, streaming | `anthropic`, `prompts` |
| `overlay.py` | always-on-top window, token painting, capture exclusion | `tkinter` |
| `main.py` | wiring, hotkeys, queues | all |
| `audio_ring.py` | drop-oldest sample ring, absolute indexing | — |
| `test_assistant.py` | 49 tests; see Testing | `pytest` |

The transcript is one in-memory list of `Turn(src, text, ts)`, capped. `src` is
`"them"` for loopback and `"me"` for mic. That tag is what makes "answer the last
question they asked" possible at all. Nothing is persisted.

Two dependency corrections found while planning, both verified against library
documentation. There is no standalone `silero-vad` package: `faster-whisper`
already ships the same Silero model on `onnxruntime` as
`faster_whisper.vad.get_speech_timestamps`, and the standalone package would drag
in `torch` and `torchaudio` to reach it. And `soxr` was missing entirely —
WASAPI hands over the endpoint mix format, usually 48 kHz stereo, while whisper
and Silero both require exactly 16 kHz mono. Nothing else converts between them.

## Model and request shape

Claude Opus 5 with fast mode. Cost lands near $0.50 for a 45-minute conversation
against SMOG's quota wall, so the model was chosen on latency and answer quality
rather than price.

```python
with client.beta.messages.stream(
    model="claude-opus-5",
    max_tokens=2000,
    speed="fast",
    betas=["fast-mode-2026-02-01"],
    output_config={"effort": "low"},
    system=[{"type": "text", "text": SYSTEM,
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    messages=[{"role": "user", "content": [
        {"type": "text", "text": transcript_text},
        {"type": "text", "text": tail},
    ]}],
) as stream:
    for text in stream.text_stream:
        ui_queue.put(text)
```

`stream()` is a context manager. Assigning it without `with` gets an unstarted
stream object and no tokens.

Leave thinking on. Disabling it on Opus 5 leaks `<thinking>` tags into visible
output, which would be garbage on screen; `effort: "low"` is the cheaper and
faster lever. No `temperature`, `top_p`, or `top_k` — all three are a 400 on
Opus 5.

Thinking is on by default and `display` defaults to omitted on Opus 5, so
thinking text never reaches the client and `text_stream` yields answer tokens
only. The overlay needs no filtering.

`max_tokens` caps thinking plus response text together. A 600-token cap sized
around a four-bullet answer can be consumed entirely by thinking on a hard
question, and the overlay shows nothing after a five-second wait. 2000 leaves
room; the bullet-count instruction, not the cap, is what keeps answers short.
Check `stop_reason == "max_tokens"` on every response and log it — truncation
is otherwise indistinguishable from a short answer.

`effort: "low"` is for latency and cost, not brevity. Opus 5 writes longer by
default and effort does not reliably shorten visible output. The bullet-count
instruction in the system prompt is what keeps answers short. If answers come
back long, tighten that line and leave effort alone.

## Caching

```
┌─ system ────────────────────────────────────────┐
│  role + answer format                           │  frozen forever
│  distilled humanizer block                      │
│  worked example                                 │
└─────────────────────────────────── [BREAKPOINT 1]
┌─ user, block 1 ─────────────────────────────────┐
│  transcript, single joined text block           │  append-only, uncached
└─────────────────────────────────────────────────┘
┌─ user, block 2 ─────────────────────────────────┐
│  mode instruction (+ typed query)               │  volatile, uncached
└─────────────────────────────────────────────────┘
```

Both hotkeys send the full transcript and vary only the tail, so they share one
cached prefix. A question referencing something from twenty minutes earlier still
resolves. Focus is the instruction's job, not the context window's.

The transcript is one joined text block rather than one block per turn. Per-turn
blocks would blow through the 20-block cache lookback window in about ten
exchanges and silently stop hitting.

1-hour TTL on the one breakpoint that remains.

**An earlier version of this design put a second breakpoint on the transcript.
That was wrong, and expensively so.** Caching is a prefix match evaluated at
block boundaries. The transcript grows between every press, so that block never
matches a prior prefix: it wrote a fresh entry every press at the 1-hour TTL's
2× write premium, and read one back essentially never. The largest and
fastest-growing block in the prompt was billed at 2× where no breakpoint at all
would have billed it at 1×.

The diagnostic could not see it either. Logging `cache_read_input_tokens` was
supposed to catch exactly this, but the system breakpoint genuinely does hit, so
the number reads non-zero while the transcript misses every time. With one
breakpoint left, "zero cache reads means an invalidator got in" is a true signal
again.

Second constraint, found the same way: Opus 5 will not cache a prefix below 512
tokens, silently. `SYSTEM` measured around 516 — four tokens of margin — and
`config.NAME` is substituted into it, so a short `ASSISTANT_NAME` could drop it
under the floor and kill caching entirely with no error. The worked example in
part 3 of the prompt exists partly to buy that margin.

### Cache killers

| Don't | Why |
|---|---|
| `datetime.now()` or meeting name in `SYSTEM` | prefix differs every press, nothing caches |
| Relative timestamps on old turns | re-renders frozen bytes, invalidates the rest |
| Toggling `speed="fast"` mid-session | switching speed invalidates the cache |
| Rebuilding `SYSTEM` with an f-string | one stray byte and you pay full price silently |

Turns are frozen at append. `SYSTEM` is a module constant built once at import,
not a function.

Log `usage.cache_read_input_tokens` on every press. Zero across consecutive
presses means an invalidator got in. This is the only signal — a cache miss is
silent and just costs 10x quietly.

## Capture exclusion

```python
import ctypes

# use_last_error is required: plain ctypes.windll does not capture the
# Win32 error, and ctypes.get_last_error() then always returns 0
_user32 = ctypes.WinDLL("user32", use_last_error=True)

WDA_NONE               = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

def set_capture_excluded(root, excluded: bool) -> None:
    # winfo_id() is the CHILD hwnd; the toplevel is its parent
    hwnd = _user32.GetParent(root.winfo_id())
    flag = WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
    if not _user32.SetWindowDisplayAffinity(hwnd, flag):
        raise ctypes.WinError(ctypes.get_last_error())
```

Needs Windows 10 build 19041 or later. Three ways to break it silently:

- Passing `winfo_id()` directly. Succeeds and does nothing, because it is the
  child handle.
- Calling before the window is mapped. Fails because the HWND doesn't exist yet.
  Call from `root.after(0, ...)`.
- Loading `user32` through `ctypes.windll` and reading `ctypes.get_last_error()`.
  Verified: the call fails with 1400 `ERROR_INVALID_WINDOW_HANDLE`, and
  `get_last_error()` still reports 0. The raised error then says "success",
  which is precisely the failure mode the loud-failure rule exists to prevent.

It must fail loudly. Believing you are hidden while visible is worse than knowing
you are visible, so a failed call means the overlay refuses to show rather than
coming up unprotected.

| | |
|---|---|
| Windows Graphics Capture (Teams, Zoom, Meet, Discord, OBS) | hidden |
| Desktop Duplication, BitBlt | hidden |
| PrintScreen, Snipping Tool, Win+Shift+S | hidden |
| HDMI capture card on monitor output | visible — reads the physical framebuffer |
| Camera pointed at the screen | visible |
| Kernel-mode or hooked capture | visible |

On a video call this hides the overlay from the screen share, not from the
webcam. Eyes tracking text read clearly on camera. The feature solves the capture
problem and not the tell problem.

Config flag, default on. With exclusion active you cannot screenshot your own
overlay, so UI debugging needs it off.

## Prompts

Output is cue cards, not an answer. A polished paragraph gets read aloud and
sounds exactly like someone reading aloud. Fragments force a glance and then
speech in your own words, which is the only version that survives a live
conversation.

### System, part 1 — role and format

```
You are a live assistant for {NAME} during a technical conversation.
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
answer. Never invent a specific.
```

`{NAME}` is interpolated once at import from config, so every request stays
byte-identical and the cache holds.

### System, part 2 — humanizer

Distilled from the `humanizer` and `structural-humanizer` skills. The full skills
run about 10K tokens and would cost roughly $0.37 per conversation cached, so
this was a steering decision rather than a cost one: most of
`structural-humanizer` targets narrative arcs, stated lessons, and embodied
emotion, none of which exist in a four-bullet cue card. Only the rules with a
real target survived.

```
VOCABULARY. These mark an answer as machine-written. Don't use:
leverage, delve, robust, seamless, comprehensive, utilize, streamline,
holistic, cutting-edge, unlock, supercharge, empower.
Use the plain verb. "Use", not "leverage". "Cuts export from four clicks
to one", not "streamlines the workflow".

NEVER OPEN WITH: "Great question", "Absolutely", "Certainly",
"I'd be happy to". Start on the answer.

NEVER CLOSE WITH: "In conclusion", "In summary", or a bullet that
restates the other bullets.

BANNED CADENCE: "It's not just X, it's Y" and "not only X but also Y".
State the real claim and drop the negation — it's there to inflate a
thin point.

RULE OF THREE: don't force a third parallel item when you have two.
Two is fine. Four is fine.

VARY BULLET LENGTH. Five bullets of identical length and shape is the
strongest tell in a list. Put a four-word bullet next to a twelve-word one.

HAVE A POSITION. "I'd start with X, though Y if writes dominate" beats
listing options neutrally. They're asking to hear judgment, not a menu.

UNCERTAINTY IS HUMAN. "Not sure at that scale" is a real bullet.
Confident filler is the failure mode; admitting a gap isn't.
```

The vocabulary ban is a prohibition list, and prohibition lists can anchor toward
what they ban. It is kept to twelve words and paired with the positive form for
that reason. Expect to tune it after twenty real answers — if bullets start
reading stilted, the banlist is the first suspect.

### System, part 3 — worked example

A short worked cue-card answer is appended to `SYSTEM`. It demonstrates the
fragments-not-prose shape, varied bullet length, and a stated position — the
things the rules above describe but do not show.

It earns its place twice. One example of the target shape steers a model harder
than a paragraph of instructions about that shape, and it also carries `SYSTEM`
clear of the 512-token caching floor described above. See `assistant/prompts.py`
for the current text.

### Mode tails (uncached)

```
Ctrl+Space
  Transcript above is live. [them] is the other person, [me] is you.
  Find the last question [them] asked that's aimed at [me] and still
  unanswered, and answer it. Several asked? Take the last. Nothing
  asked? Give the single most useful thing to say next.

Ctrl+Shift+Space
  Transcript above is live. [them] is the other person, [me] is you.
  Answer this using the transcript as context: {query}
```

## Security

API key from the environment. Never hardcoded, never logged, never written to the
transcript. A missing audio device fails loudly rather than falling back to the
wrong one.

## Testing

Two checks were specified up front, both pinning things that fail silently in
production:

1. **Silent start.** Capture begins against an idle output endpoint, receives no
   packets, and the VAD gate treats it as silence. No timeout, no exception, and
   speech that arrives later is still transcribed.
2. **Buffer trim boundary.** The ring buffer trims at its cap without dropping or
   duplicating a sample across the wrap.

The suite grew to 49 during implementation, and the growth is the interesting
part: nearly all of it came from review finding that a guard, an invariant, or a
failure path had no test — the transcription loop's whole state machine, the
cache-prefix invariant, the reentrancy guard, and the "API key never reaches the
log" requirement all arrived that way.

Three of those tests were originally written so that they passed against
deliberately broken code. Each was caught by mutation — break the thing the test
claims to pin, and check the test actually dies. That is now the standard for any
test guarding a property rather than a return value, and it is worth more here
than coverage percentage.

Anything needing real audio hardware, a real screen, or a paid API call cannot be
gated automatically. Those live in `docs/verification-checklist.md` as nine
ordered manual checks. Capture exclusion is among them: it is a Win32 call
against a live window, and mocking it would only prove the mock works.

## Scope

In: dual tagged capture, VAD-gated local STT, four hotkeys, streaming answers,
always-on-top overlay, capture exclusion, prompt caching.

Two of those hotkeys are utility rather than feature. An `overrideredirect`
always-on-top window has no titlebar and no close button, so hide (Ctrl+Alt+H)
and quit (Ctrl+Alt+Q) have no other route. They were not in the first draft of
this scope line and are recorded here so the count stops looking like drift.

Out: persistent transcript storage, meeting summaries, speaker diarization,
packaging or installer, settings GUI, cloud sync.

Personal use. Not for Bell Canada calls — pointing this at work calls reopens the
consent and recording-policy question, which is a separate task and must be
settled before, not after.
