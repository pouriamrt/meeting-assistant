"""Wiring: capture, gates, hotkeys, overlay."""

import logging
import queue
import threading
from collections.abc import Callable
from typing import cast

import anthropic

from . import answer, config, prompts
from .capture import Capture
from .overlay import Overlay
from .transcribe import Transcript, gate_loop

log = logging.getLogger(__name__)


class _AnswerGate:
    """Latest-press-wins reentrancy guard for `answer.stream_answer`.

    Every hotkey press spawns its own answer thread, all writing to one
    ui_queue; two overlapping presses would otherwise interleave tokens from
    two different answers. A lock that drops the second press would ignore
    it - pressing again means "give me a new answer", not "ignore me" - so
    this hands out an increasing generation number instead, and each press
    gets a queue view that silently drops once a later press starts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0

    def start(self, ui_queue: "queue.Queue") -> "_GenerationQueue":
        """Claim the next generation and hand back a queue scoped to it."""
        with self._lock:
            self._generation += 1
            generation = self._generation
        return _GenerationQueue(ui_queue, generation, self)

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation


class _GenerationQueue:
    """`ui_queue.put`, but dropped once a newer press supersedes this one.

    No discrimination by message kind: once superseded, a stray `clear` or
    `done` from the old answer is dropped exactly like its tokens, so it
    cannot disrupt the live one.
    """

    def __init__(
        self, ui_queue: "queue.Queue", generation: int, gate: _AnswerGate
    ) -> None:
        self._ui_queue = ui_queue
        self._generation = generation
        self._gate = gate

    def put(self, item: tuple) -> None:
        if self._gate.is_current(self._generation):
            self._ui_queue.put(item)


def _guarded(label: str, fn: Callable[[], None]) -> Callable[[], None]:
    """Wrap a hotkey callback so one failure logs instead of killing
    pynput's listener thread - the same event-loop-dies-silently shape
    already guarded in the capture callback and the transcription gate
    loop, now closed for the third caller on this codebase's list."""

    def wrapped() -> None:
        try:
            fn()
        except Exception:
            log.exception("hotkey callback %r failed", label)

    return wrapped


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    client = anthropic.Anthropic(api_key=config.api_key())
    transcript = Transcript(config.MAX_TURNS)
    ui_queue: queue.Queue = queue.Queue()
    stop = threading.Event()
    answer_gate = _AnswerGate()

    capture = Capture()
    capture.start()

    hotkeys = None
    try:
        for ring, src in ((capture.mic, "me"), (capture.sys, "them")):
            threading.Thread(
                target=gate_loop,
                args=(ring, src, transcript, stop),
                daemon=True,
                name=f"gate-{src}",
            ).start()

        # Overlay() raises RuntimeError if capture exclusion fails, and must
        # be allowed to reach the user rather than show an overlay visible to
        # a screen share. capture is already running by this point, so
        # construction has to stay inside this try - otherwise a raise here
        # would skip the finally below and leak a running capture with an
        # open audio device.
        overlay = Overlay(ui_queue)

        def start_answer(tail: str) -> None:
            gen_queue = answer_gate.start(ui_queue)

            def run() -> None:
                answer.stream_answer(
                    client, transcript.render(), tail, cast(queue.Queue, gen_queue)
                )

            threading.Thread(target=run, daemon=True).start()

        def on_last_question() -> None:
            if len(transcript) == 0:
                # Routed through the gate too: this also counts as "a press",
                # and must supersede whatever a previous press is still
                # streaming rather than write over it directly.
                answer_gate.start(ui_queue).put(("status", "nothing transcribed yet"))
                return
            start_answer(prompts.last_question_tail())

        def on_query() -> None:
            # This fires on pynput's thread, which must never touch a widget.
            # Hand the dialog to the main thread over the queue, like
            # everything else, and get the typed string back through this
            # callback.
            def on_typed(typed: str) -> None:
                start_answer(prompts.query_tail(typed))

            ui_queue.put(("prompt", on_typed))

        def on_toggle() -> None:
            ui_queue.put(("toggle", None))

        def on_quit() -> None:
            ui_queue.put(("quit", None))

        from pynput import keyboard

        hotkeys = keyboard.GlobalHotKeys(
            {
                "<ctrl>+<space>": _guarded("last_question", on_last_question),
                "<ctrl>+<shift>+<space>": _guarded("query", on_query),
                "<ctrl>+<alt>+h": _guarded("toggle", on_toggle),
                "<ctrl>+<alt>+q": _guarded("quit", on_quit),
            }
        )
        hotkeys.start()

        log.info(
            "ready - ctrl+space answer, ctrl+shift+space query, "
            "ctrl+alt+h hide, ctrl+alt+q quit"
        )
        overlay.run()  # blocks: Tkinter owns the main thread
    finally:
        stop.set()
        if hotkeys is not None:
            try:
                hotkeys.stop()
            except Exception:
                log.exception("hotkey listener failed to stop cleanly")
        capture.stop()


if __name__ == "__main__":
    main()
