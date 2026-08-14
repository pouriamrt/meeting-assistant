"""Wiring: capture, gates, hotkeys, overlay."""

import hashlib
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import anthropic

from . import answer, apikey, config, prompts, screen
from .capture import Capture
from .overlay import Overlay
from .transcribe import Transcript, gate_loop, warm_model

if TYPE_CHECKING:  # pynput touches the platform input stack on import
    from pynput import keyboard

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

    def put_if_current(
        self, generation: int, ui_queue: "queue.Queue", item: tuple
    ) -> None:
        """Check the generation and put in one atomic step.

        Checking and then putting are two operations, and a second press
        landing between them lets a superseded token through - the exact
        interleaving this guard exists to prevent. A lock around the counter
        alone does not make check-then-act atomic. `queue.Queue.put` on an
        unbounded queue never blocks, so holding the lock across it is cheap
        and cannot deadlock.
        """
        with self._lock:
            if generation == self._generation:
                ui_queue.put(item)


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
        self._gate.put_if_current(self._generation, self._ui_queue, item)


def _guarded(label: str, fn: Callable[[], None]) -> Callable[[], None]:
    """Wrap a hotkey callback so one failure logs instead of killing
    pynput's listener thread - the same event-loop-dies-silently shape
    already guarded in the capture callback, the transcription gate loop,
    and the overlay drain."""

    def wrapped() -> None:
        try:
            fn()
        except Exception:
            log.exception("hotkey callback %r failed", label)

    return wrapped


def _install_hotkeys(
    client: "anthropic.Anthropic",
    transcript: Transcript,
    ui_queue: "queue.Queue",
    answer_gate: _AnswerGate,
) -> "keyboard.GlobalHotKeys":
    """Build the five hotkey callbacks and start the listener.

    Split out of main() for length. The callbacks all close over the same four
    objects, which is why those arrive as parameters rather than through
    module state.
    """

    def start_answer(
        tail: str,
        images: Sequence[bytes] = (),
        system: str | None = None,
    ) -> None:
        gen_queue = answer_gate.start(ui_queue)

        def run() -> None:
            # stream_answer guards its own streaming call, but not the
            # ui_queue.put("clear") or usage-logging around it - same
            # dies-silently shape as _guarded, so it gets the same wrap.
            try:
                answer.stream_answer(
                    client, transcript.render(), tail, gen_queue, images, system
                )
            except Exception:
                log.exception("answer thread failed")

        threading.Thread(target=run, daemon=True).start()

    def status(message: str) -> None:
        """Show one line instead of an answer.

        Routed through the gate rather than straight onto the queue: this
        counts as "a press" too, and must supersede whatever a previous press
        is still streaming rather than be written over by it.
        """
        answer_gate.start(ui_queue).put(("status", message))

    def on_last_question() -> None:
        if len(transcript) == 0:
            status("nothing transcribed yet")
            return
        start_answer(prompts.last_question_tail())

    def on_query() -> None:
        # This fires on pynput's thread, which must never touch a widget. Hand
        # the query entry to the main thread over the queue, like everything
        # else, and get the typed string back through this callback.
        def on_typed(typed: str) -> None:
            start_answer(prompts.query_tail(typed))

        ui_queue.put(("prompt", on_typed))

    def on_screen() -> None:
        # Grabbed here, on pynput's thread, rather than inside the answer
        # thread: the screenshot must be of what was on screen when the key
        # went down. Measured at 80-140ms for one 1536x864 monitor, nearly all
        # of it PNG encoding - invisible unless a second hotkey is pressed
        # inside that window.
        try:
            images = screen.grab()
        except Exception:
            log.exception("screen grab failed")
            status("screen grab failed")
            return
        if not images:
            status("no monitor to capture")
            return
        # A digest per press, so "it answered the previous question" is
        # answerable from the log instead of by guesswork: two presses with
        # the same digest captured identical pixels, and the screen never
        # changed under the grab. Different digests move the search to the
        # request. ~1ms on 400KB.
        digest = hashlib.sha256(b"".join(images)).hexdigest()[:12]
        log.info(
            "screen: %d monitor(s), %d KB, sha=%s",
            len(images), sum(len(i) for i in images) // 1024, digest,
        )
        start_answer(prompts.screen_tail(), images, prompts.CODE_SYSTEM)

    def on_toggle() -> None:
        ui_queue.put(("toggle", None))

    def on_quit() -> None:
        ui_queue.put(("quit", None))

    # Imported here rather than at module scope: pynput touches the platform
    # input stack on import, and nothing else in this package needs it.
    from pynput import keyboard as kb

    hotkeys = kb.GlobalHotKeys(
        {
            "<ctrl>+<space>": _guarded("last_question", on_last_question),
            "<ctrl>+<shift>+<space>": _guarded("query", on_query),
            "<ctrl>+<alt>+<space>": _guarded("screen", on_screen),
            "<ctrl>+<alt>+h": _guarded("toggle", on_toggle),
            "<ctrl>+<alt>+q": _guarded("quit", on_quit),
        }
    )
    hotkeys.start()
    return hotkeys


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Environment, then the stored key, then a setup window. Before
    # anything else starts, so a missing key costs nothing but a prompt.
    client = anthropic.Anthropic(api_key=apikey.resolve())
    transcript = Transcript(config.MAX_TURNS)
    ui_queue: queue.Queue = queue.Queue()
    stop = threading.Event()
    answer_gate = _AnswerGate()

    # Off the critical path: the first decode would otherwise pay ~1s of
    # model construction on top of its own latency.
    threading.Thread(
        target=warm_model,
        args=(lambda text: ui_queue.put(("status", text)),),
        daemon=True,
        name="warm",
    ).start()

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

        hotkeys = _install_hotkeys(client, transcript, ui_queue, answer_gate)

        log.info(
            "ready - ctrl+space answer, ctrl+shift+space query, "
            "ctrl+alt+space screen, ctrl+alt+h hide, ctrl+alt+q quit"
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
