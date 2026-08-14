# Verification checklist

Every gate during the build was a unit test, a code read, or a falsification harness —
all of which a machine can do, and none of which prove the thing works. That is still
why this file exists.

This file used to open by claiming these were "the checks a machine could not perform."
That was wrong, and believing it costs real time. On 2026-08-14 all of them were
attempted headless and most simply ran: §§2, 4, 6, 7, 8 fully, and 11 of §9's 13 items.
What actually resists automation is narrower than it looks — **real acoustic input, a
foreign capture API, and a genuine OS device state.** Three things, not nine.

Each section below is marked **Scripted** or **Needs a human**. Run every Scripted one
first and in order; a failure there stops you before a manual step wastes your time.
Techniques that unlock the non-obvious ones are in *Driving it headless* at the bottom.

**Wear headphones for anything involving both channels.** On speakers the mic
picks up the system audio and the same sentence is transcribed under both tags.

Set up first:

    uv sync
    export ANTHROPIC_API_KEY=sk-ant-...
    export ASSISTANT_NAME="Your Name"

---

## 1. Capture reaches both rings

**Scripted.** Playing `sys_audio.npy` out of the default output device covers the
speaker side; the mic side only needs a nonzero peak, which room noise supplies.

Play music through the speakers and talk into the mic while this runs.

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

**Expect:** both counts near 80000 (5 s at 16 kHz), both peaks above 0.01.
A `sys` peak of 0.000 means nothing was playing. A `mic` peak of 0.000 means your
microphone is not the Windows default input.

## 2. A missing device fails loudly

**Scripted, with one gap left for a human.** Monkeypatching
`get_default_input_device_info` / `get_default_wasapi_loopback` to raise proves the
guard converts the error and leaves nothing running — including the case where the
*second* stream fails with the first already live, which is the leak the except block
in `Capture.start` exists to prevent. All of that passes.

What it cannot prove is that a really-disabled device raises `OSError` rather than some
other driver error, which is the whole point of the manual step. Do it once, by hand,
below.

Disable the default input device in Windows Sound settings, then:

    uv run python -c "from assistant.capture import Capture; Capture().start()"

**Expect:** `RuntimeError: No default input device. Plug in a microphone and set it
as the Windows default.` Re-enable the device afterwards.

If you see a driver error instead of that message, the guard added for it is not
working — that was a real defect found in review and the message is the evidence it
was fixed.

## 3. Transcription produces tagged turns

**Scripted for `[them]`, needs a human for `[me]`.** Playing `sys_audio.npy` out of the
speakers drives the whole loopback path — real device, real VAD gate, real whisper —
and produces a tagged `[them]` turn. It also exercises the wall-clock silence close,
since an idle output endpoint sends no packets at all. `[me]` needs someone to talk:
the mic path is provably live (rms ~0.007 on room noise) but VAD correctly refuses
anything that is not speech.

First run downloads `base.en`, about 150 MB.

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

**Expect:** tagged lines with recognisable text. Watch the latency between someone
finishing a sentence and their turn appearing — it should be under about four
seconds. If `[them]` works but `[me]` is empty, the mic is not the Windows default.

## 4. The overlay renders

**Scripted.** Position, topmost, overrideredirect and the streamed fill are all
readable with `winfo_*` after pumping `root.update()` in a loop instead of calling
`run()`, which would block on the mainloop.

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

**Expect:** a dark always-on-top box, top left, filling in line by line. Close with
Alt+F4 or Ctrl+C in the terminal.

## 5. The overlay is invisible to capture — the security check

**Needs a human. Do not skip it because a script said PASS.** Exclusion is provable
against the GDI/BitBlt path (what `mss` uses) by sampling the pixel under the overlay
with exclusion on and off — the off run is mandatory, because "the pixel is not the
overlay colour" is equally true when the coordinates are simply wrong. That path
passes, with the entry open too.

`Win+Shift+S` and modern share tools use Windows Graphics Capture, a **different API**.
Nothing scripted here touches it. This is the security check; it stays manual.

With the overlay from step 4 still running, press `Win+Shift+S` and drag a selection
over the overlay region.

**Expect:** the captured image shows the desktop *behind* the overlay. The overlay
must be **absent**.

If it appears, stop and diagnose before going further. Two known causes: passing
`winfo_id()` without `GetParent`, and calling before the window is mapped.

## 6. A failed exclusion refuses to show

**Scripted.** Confirmed on Windows 11 build 26200: raises and shows no window.

    uv run python -c "
    import queue, assistant.overlay as ov
    ov.WDA_EXCLUDEFROMCAPTURE = 0x00000002   # rejected flag, forces the call to fail
    try:
        ov.Overlay(queue.Queue()).run()
        print('FAIL: overlay showed anyway')
    except RuntimeError as e:
        print('PASS:', e)
    "

**Expect:** `PASS: Screen-capture exclusion failed; ...` and no window appears.

The injected flag is build-dependent. `0x2` is rejected on Windows 11 build 26200;
`0x99` is *accepted* there, which would make this report a failure of its own premise
rather than of the guard. If this ever reports `FAIL`, probe the flag directly before
concluding the overlay is broken.

## 7. The answer streams, and the cache engages

**Scripted.** Measured: `cache_read=1317` on consecutive presses, and the entry
survives across separate processes.

Costs money — two real API calls.

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

**Expect:** cue-card bullets both times, and `cache_read=` above 0 on the second
press. A second `cache_read=0` means the system prefix is not stable — check for
anything volatile in `SYSTEM`.

## 8. The system prompt clears the caching floor

**Scripted.** Measured: `SYSTEM` 866 tokens, `CODE_SYSTEM` 1322, and both unchanged
with `ASSISTANT_NAME="Al"` — the short-name risk noted below is not real at this size.
Run it for both prompts; screen mode has its own prefix and its own floor to clear.

Also costs money, though `count_tokens` is cheap.

    uv run python -c "
    import anthropic
    from assistant import prompts, config
    c = anthropic.Anthropic(api_key=config.api_key())
    r = c.messages.count_tokens(
        model='claude-opus-5',
        system=[{'type': 'text', 'text': prompts.SYSTEM}],
        messages=[{'role': 'user', 'content': 'x'}],
    )
    print(r)
    "

**Expect:** comfortably above 512. Opus 5 will not cache a prefix below that, and it
fails silently — no error, just no caching.

`ASSISTANT_NAME` is substituted into `SYSTEM` at two sites, so re-run this with the
shortest name you would realistically use. If it lands under about 600, add more
content to `SYSTEM` rather than trimming it.

## 9. End to end, against a real conversation

**11 of these 13 are scripted.** Only 3 and 6 need a person; both are marked below.

A YouTube interview through the speakers works for a solo test.

    uv run meeting-assistant

Check each:

1. Overlay appears, top left, always on top.
2. Speech through the speakers becomes `[them]` turns (watch the log).
3. Speech into the mic becomes `[me]` turns. **Needs a human** — say one sentence.
   Nothing else here requires your voice.
4. `Ctrl+Space` streams bullets within a few seconds.
5. `Ctrl+Shift+Space` opens an inline entry **inside the overlay** — not a separate
   window. Type a question, press Return, get an answer. Escape cancels.
6. **`Win+Shift+S` over the overlay while the query entry is open. Needs a human** —
   a scripted GDI probe passes with the entry open, but that is not this API. The entry must be
   absent from the capture too. This was a real Critical: the original implementation
   used a modal dialog, which is a separate window and was fully visible to screen
   share while displaying the question being typed.
7. Press `Ctrl+Space` twice a second apart. The second answer must replace the first
   cleanly, not interleave with it.
8. A second press logs `cache_read=` above 0.
9. **`Ctrl+Alt+Space` with code visible on screen.** The answer must be real code,
   not bullets — that is the whole reason screen mode has its own system prompt.
   A second press logs `cache_read=` above 0 for that prompt too; both prefixes
   are cached separately.
10. **Both inputs actually reach the model.** Say a constraint out loud that
    appears nowhere on screen — "don't use any imports" — then press
    `Ctrl+Alt+Space` on a coding prompt. The answer must honour it. If it does
    not, the transcript is not riding along and the screenshot is being sent
    alone.
11. **Press `Ctrl+Alt+Space` while an earlier answer is still on the overlay.**
    The new answer must not react to that text. Capture exclusion is what keeps
    the overlay out of the screenshot; if it ever stops covering this path the
    model starts reading its own last answer back as input, and the only sign is
    an answer that quietly drifts off-topic.
12. `Ctrl+Alt+H` hides and shows.
13. `Ctrl+Alt+Q` exits cleanly, with no audio device left open.

---

## Driving it headless

Three techniques unlock most of the list. None needs a new dependency.

**Hotkeys.** Do not synthesise a keypress: pynput's listener ignores keystrokes
injected by pynput's own `Controller`, so a synthetic press proves nothing either way.
Capture the mapping instead and call the bound callable — which also proves the binding
exists.

    class FakeHotkeys:
        captured = {}
        def __init__(self, mapping): FakeHotkeys.captured = mapping
        def start(self): pass
        def stop(self): pass

    from pynput import keyboard as kb
    real, kb.GlobalHotKeys = kb.GlobalHotKeys, FakeHotkeys
    try:
        main._install_hotkeys(client, transcript, ui_queue, main._AnswerGate())
    finally:
        kb.GlobalHotKeys = real
    FakeHotkeys.captured["<ctrl>+<alt>+<space>"]()      # the real callback

**Real audio without a person.** `sys_audio.npy` in the repo root is 6.2 s of real
captured speech. Play it out of the default output device with pyaudio while `Capture`
is running and the loopback stream picks it up through the genuine WASAPI path. It
transcribes as "how would you scale the *right* path" — whisper mishearing "write",
a homophone, not a defect.

    out = pa.open(format=pyaudio.paFloat32, channels=1, rate=16_000, output=True)
    out.write(np.load("sys_audio.npy").astype(np.float32).tobytes())

**Anything visual needs a control run.** Measure the failing configuration too, or a
broken probe reads as a passing check. Three probes written for this list were wrong
rather than the code: `"import " not in answer` fired on the prose "not an import"; a
blanket brace ban fired on `seen = {}` inside a code example; and `winfo_viewable()`
was simply the wrong question for `toggle_visible`, which hides with **alpha on
purpose** — unmapping can hand back a different HWND, and display affinity lives on the
HWND, so withdraw/deiconify could silently drop capture exclusion. Each looked like a
defect until the code was read.

## What this does not cover

Exclusion covers screen capture and screen share. It does not cover an HDMI capture
card on the monitor output, a camera pointed at the screen, or your eyes visibly
tracking text. It solves the capture problem, not the tell problem.
