"""Overlay tests: the capture-excluded window and every gesture on it.

Split out of test_assistant.py, which had grown past the 800-line ceiling in
rules/common/coding-style.md. These are the tests that build a real Tk window,
so they are also the slow ones and the only ones that need a display.
"""

import logging
import queue
import tkinter as tk

import pytest

import assistant.overlay as overlay_mod
from assistant import uistate


class _FakeEntry:
    """Stand-in for tk.Entry: records what the widget code does."""

    def __init__(self, master, **kwargs):
        self.master = master
        self.kwargs = kwargs
        self.packed = False
        self.pack_kwargs = {}
        self.focused = False
        self.destroyed = False
        self.bindings = {}
        self.text = ""

    def pack(self, **kwargs):
        self.packed = True
        self.pack_kwargs = kwargs

    def focus_set(self):
        self.focused = True

    def bind(self, sequence, func):
        self.bindings[sequence] = func

    def get(self):
        return self.text

    def destroy(self):
        self.destroyed = True


class _FakeUiQueue:
    def __init__(self, items):
        self._items = list(items)

    def get_nowait(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


def _bare_overlay():
    """Overlay.__new__ skips __init__ - no Tk window, no capture-exclusion
    syscall - so _drain's message handling can be unit tested in isolation."""
    ov = overlay_mod.Overlay.__new__(overlay_mod.Overlay)
    ov.root = object()  # never touched except as the fake Entry's master
    ov.text = object()  # the entry packs relative to it; identity is all we need
    ov._entry = None
    ov._alive = False  # skip _drain's after() reschedule, no real Tk loop
    # __new__ skips __init__, so anything __init__ establishes has to be set
    # by hand. _open_query_entry reads this to size a lazily-created entry.
    ov._font_size = uistate.DEFAULT.font_size
    return ov


def _font_size_of(widget) -> int:
    """Tk's cget("font") returns a font SPEC STRING ("{Segoe UI} 12"), not the
    tuple it was configured with, so indexing it walks characters."""
    import tkinter.font as tkfont

    return tkfont.Font(font=widget.cget("font")).cget("size")


def test_prompt_message_opens_one_inline_entry_in_the_root(monkeypatch):
    monkeypatch.setattr(overlay_mod.tk, "Entry", _FakeEntry)
    ov = _bare_overlay()
    ov.ui_queue = _FakeUiQueue([("prompt", lambda typed: None)])

    ov._drain()

    assert isinstance(ov._entry, _FakeEntry)
    # `before` is what stops the entry being allocated zero space behind a
    # widget packed with expand=True/fill=both. Without it pack() still
    # succeeds and the entry silently never appears.
    assert ov._entry.pack_kwargs.get("before") is ov.text
    assert ov._entry.master is ov.root
    assert ov._entry.master is ov.root, "entry must live in the excluded root, not a new window"
    assert ov._entry.packed
    assert ov._entry.focused
    assert ov._entry.kwargs["bg"] == overlay_mod._BG
    assert ov._entry.kwargs["fg"] == overlay_mod._FG
    assert ov._entry.kwargs["borderwidth"] == 0


def test_prompt_message_reuses_an_already_open_entry(monkeypatch):
    monkeypatch.setattr(overlay_mod.tk, "Entry", _FakeEntry)
    ov = _bare_overlay()
    ov.ui_queue = _FakeUiQueue([("prompt", lambda typed: None)])
    ov._drain()
    first = ov._entry
    first.focused = False  # simulate focus having moved elsewhere

    ov.ui_queue = _FakeUiQueue([("prompt", lambda typed: None)])
    ov._drain()

    assert ov._entry is first, "a second press must not stack a second entry"
    assert first.focused


def test_entry_return_destroys_before_calling_back_with_the_typed_text(monkeypatch):
    monkeypatch.setattr(overlay_mod.tk, "Entry", _FakeEntry)
    ov = _bare_overlay()
    received = []
    ov.ui_queue = _FakeUiQueue([("prompt", received.append)])
    ov._drain()
    entry = ov._entry
    entry.text = "how do you handle migrations"

    entry.bindings["<Return>"]()

    assert entry.destroyed, "must be destroyed before the callback runs"
    assert ov._entry is None
    assert received == ["how do you handle migrations"]


def test_entry_return_with_empty_text_does_not_call_back(monkeypatch):
    monkeypatch.setattr(overlay_mod.tk, "Entry", _FakeEntry)
    ov = _bare_overlay()
    called = []
    ov.ui_queue = _FakeUiQueue([("prompt", lambda typed: called.append(typed))])
    ov._drain()
    entry = ov._entry
    entry.text = ""

    entry.bindings["<Return>"]()

    assert entry.destroyed
    assert ov._entry is None
    assert called == []


def test_entry_escape_destroys_without_calling_back(monkeypatch):
    monkeypatch.setattr(overlay_mod.tk, "Entry", _FakeEntry)
    ov = _bare_overlay()
    called = []
    ov.ui_queue = _FakeUiQueue([("prompt", lambda typed: called.append(typed))])
    ov._drain()
    entry = ov._entry
    entry.text = "should not be sent"

    entry.bindings["<Escape>"]()

    assert entry.destroyed
    assert ov._entry is None
    assert called == []


def test_entry_return_with_a_raising_callback_logs_and_does_not_propagate(
    monkeypatch, caplog
):
    # submit is a Tk event binding, not code inside _drain's try - an
    # uncaught exception here would otherwise reach Tk's
    # report_callback_exception (stderr only) instead of this module's log.
    monkeypatch.setattr(overlay_mod.tk, "Entry", _FakeEntry)
    ov = _bare_overlay()

    def boom(typed):
        raise RuntimeError("callback exploded")

    ov.ui_queue = _FakeUiQueue([("prompt", boom)])
    ov._drain()
    entry = ov._entry
    entry.text = "trigger the boom"

    with caplog.at_level(logging.ERROR):
        entry.bindings["<Return>"]()  # must not raise

    assert entry.destroyed
    assert ov._entry is None
    assert caplog.records, "the failure was swallowed without a log"


def test_query_entry_is_actually_displayed_inside_the_root_window(monkeypatch):
    # Regression. The entry was packed after a widget with expand=True and
    # fill="both", which claims the entire cavity - pack() succeeded, the
    # entry stayed unmapped at 1x1, and Ctrl+Shift+Space silently did
    # nothing. Nothing in the suite checked that a widget was displayed, only
    # that it existed, so this shipped through every review.
    import queue as _queue

    from assistant import config as _config

    monkeypatch.setattr(_config, "EXCLUDE_FROM_CAPTURE", False)
    from assistant.overlay import Overlay

    ov = _tk_root_or_skip(lambda: Overlay(_queue.Queue()))

    try:
        ov.root.attributes("-alpha", 0.0)
        ov._open_query_entry(lambda _text: None)
        ov.root.update_idletasks()

        entry = ov._entry
        assert entry is not None
        assert entry.winfo_ismapped(), "entry was packed but never displayed"
        assert entry.winfo_width() > 100, f"entry collapsed to {entry.winfo_width()}px"
        assert entry.winfo_height() > 5, f"entry collapsed to {entry.winfo_height()}px"
        # The whole point of an inline entry: a Toplevel would be a second
        # HWND, and SetWindowDisplayAffinity is per-HWND and not inherited.
        assert entry.winfo_toplevel() is ov.root
    finally:
        ov.root.destroy()


def _live_overlay(monkeypatch):
    """A real Overlay with capture exclusion off, or skip if there is no display."""
    import queue as _queue

    from assistant import config as _config

    monkeypatch.setattr(_config, "EXCLUDE_FROM_CAPTURE", False)
    from assistant.overlay import Overlay

    ov = _tk_root_or_skip(lambda: Overlay(_queue.Queue()))
    ov.root.attributes("-alpha", 0.0)
    return ov


def _tk_root_or_skip(build):
    """Build a Tk-owning object, retrying once, or skip if Tk is really absent.

    Creating and tearing down this many Tk roots in one process intermittently
    fails to resolve the Tcl library - measured at roughly half of all suite
    runs, on a different test each time, and under at least four different
    messages ("Can't find a usable init.tcl" with an EMPTY directory list,
    "Can't find a usable tk.tcl", 'invalid command name "tcl_findLibrary"',
    and a missing auto.tcl). Matching on the wording missed half of them, so
    any TclError is retried.

    The blanket skip this replaces turned a transient failure into a silently
    missing test, which is the one thing a green suite must never mean. A
    second attempt has always succeeded; failing twice is a real broken
    environment and still skips.
    """
    import tkinter as _tk

    try:
        return build()
    except _tk.TclError:
        # A failure partway through construction leaves a live root behind:
        # tk.Tk() succeeded, something after it did not. Leaking one into the
        # retry is how a flake becomes a cascade, since the whole problem here
        # is too many roots in one process.
        stray = getattr(_tk, "_default_root", None)
        if stray is not None:
            try:
                stray.destroy()
            except _tk.TclError:
                pass
        try:
            return build()
        except _tk.TclError as again:
            pytest.skip(f"tk unavailable twice: {again}")


def _spy_on_raising(ov):
    """Record lift() and -topmost writes on a real Tk window."""
    calls = []
    real_lift, real_attr = ov.root.lift, ov.root.attributes

    def lift(*a):
        calls.append("lift")
        return real_lift(*a)

    def attributes(*a):
        if len(a) >= 2 and a[0] == "-topmost":
            calls.append(("topmost", a[1]))
        return real_attr(*a)

    ov.root.lift = lift
    ov.root.attributes = attributes
    return calls


def test_showing_the_overlay_raises_it_not_just_its_alpha(monkeypatch):
    # Alpha cannot change z-order, which is exactly why Ctrl+Alt+H could not
    # bring the overlay back once another app buried it. Nothing can detect
    # the burial either - Tk reports -topmost 1 and viewable True throughout -
    # so raising has to be unconditional on the user's action.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        ov.toggle_visible()          # hide
        ov.root.update()
        calls = _spy_on_raising(ov)
        ov.toggle_visible()          # show again
        ov.root.update()

        assert "lift" in calls, "showing the overlay never lifted it"
        # off-then-on is what re-applies topmost on Windows; setting it to
        # True when it is already True is a no-op.
        assert ("topmost", False) in calls and ("topmost", True) in calls, (
            f"topmost was not re-asserted: {calls}"
        )
        assert calls.index(("topmost", False)) < calls.index(("topmost", True))
    finally:
        ov.root.destroy()


def test_a_new_answer_raises_the_overlay(monkeypatch):
    # The other half of the recovery: if it is buried, pressing any answer
    # hotkey should surface it along with the answer.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        calls = _spy_on_raising(ov)
        ov.ui_queue.put(("clear", None))
        ov._drain()
        ov.root.update()
        assert "lift" in calls, "a new answer did not raise the overlay"
    finally:
        ov.root.destroy()


def test_raising_does_not_change_the_hwnd_or_the_alpha(monkeypatch):
    # Display affinity lives on the HWND, and alpha is the hide state. A
    # recovery that disturbed either would trade one bug for a worse one.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        hwnd = ov.root.winfo_id()
        alpha = float(ov.root.attributes("-alpha"))
        ov._raise()
        ov.root.update()
        assert ov.root.winfo_id() == hwnd
        assert float(ov.root.attributes("-alpha")) == alpha
    finally:
        ov.root.destroy()


def test_grip_resizes_without_moving_the_window(monkeypatch):
    # overrideredirect(True) removes the frame, so there is no resize border
    # and no window manager to do the work - hand-rolled like the drag.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("460x300+100+100")
        ov.root.update()
        hwnd_before = ov.root.winfo_id()

        ov._grip.event_generate("<Button-1>", x=5, y=5, rootx=560, rooty=400)
        ov.root.update()
        ov._grip.event_generate("<B1-Motion>", rootx=660, rooty=480)
        ov.root.update()

        assert (ov.root.winfo_width(), ov.root.winfo_height()) == (560, 380), (
            f"got {ov.root.winfo_width()}x{ov.root.winfo_height()}, want 560x380"
        )
        # Growing must not drag the window across the screen.
        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (100, 100)
        assert ov.root.winfo_id() == hwnd_before, "HWND changed - exclusion would drop"
    finally:
        ov.root.destroy()


def test_grip_drag_does_not_also_move_the_window(monkeypatch):
    # Tk runs bindtags widget -> class -> TOPLEVEL -> all, and <Button-1> is
    # already bound on self.root for dragging. Without "break" on the grip
    # handlers the window resizes AND slides at the same time. Measured: the
    # handlers fired ['grip', 'root'] without it.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("460x300+100+100")
        ov.root.update()
        ov._grip.event_generate("<Button-1>", x=5, y=5, rootx=560, rooty=400)
        ov.root.update()
        assert ov._drag_from is None, "the grip click reached the drag handler"
    finally:
        ov.root.destroy()


def test_grip_will_not_shrink_the_window_into_uselessness(monkeypatch):
    # Past the floor the text is unreadable and the grip itself gets hard to
    # hit, which would leave no way to grow it back.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("460x300+100+100")
        ov.root.update()
        ov._grip.event_generate("<Button-1>", x=5, y=5, rootx=560, rooty=400)
        ov.root.update()
        ov._grip.event_generate("<B1-Motion>", rootx=100, rooty=100)
        ov.root.update()

        assert ov.root.winfo_width() >= overlay_mod._MIN_W
        assert ov.root.winfo_height() >= overlay_mod._MIN_H
    finally:
        ov.root.destroy()


def test_the_drag_bar_moves_the_window(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("460x300+100+100")
        ov.root.update()
        hwnd = ov.root.winfo_id()

        ov._drag_bar.event_generate("<Button-1>", rootx=150, rooty=110)
        ov.root.update()
        ov._drag_bar.event_generate("<B1-Motion>", rootx=210, rooty=150)
        ov.root.update()

        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (160, 140)
        assert (ov.root.winfo_width(), ov.root.winfo_height()) == (460, 300)
        assert ov.root.winfo_id() == hwnd
    finally:
        ov.root.destroy()


def test_the_drag_bar_is_visible_along_the_top(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("460x300+100+100")
        ov.root.update()
        bar = ov._drag_bar
        assert bar.winfo_ismapped()
        assert bar.winfo_rooty() == ov.root.winfo_y(), "bar is not at the top"
        assert bar.winfo_width() > 400, "bar does not span the width"
        # An empty Frame collapses to 1px unless propagation is off, which
        # would leave a drag surface too thin to hit.
        assert bar.winfo_height() >= 10, f"bar is {bar.winfo_height()}px tall"
    finally:
        ov.root.destroy()


def test_dragging_the_body_selects_text_instead_of_moving_the_window(monkeypatch):
    # The point of the drag bar. <Button-1> used to be bound on self.text AND
    # on self.root, and bindtags run the toplevel binding for events on any
    # child - so both had to go, or a drag across the text would still move
    # the window instead of selecting.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("460x300+100+100")
        ov._write("- Redis, 5min TTL\n- Invalidate on write\n")
        ov.root.update()

        ov.text.event_generate("<Button-1>", x=20, y=10, rootx=150, rooty=150)
        ov.root.update()
        ov.text.event_generate("<B1-Motion>", x=200, y=10, rootx=310, rooty=190)
        ov.root.update()

        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (100, 100), (
            "dragging across the text moved the window"
        )
        assert ov._drag_from is None, "the text still starts a window drag"
    finally:
        ov.root.destroy()


def test_ctrl_c_copies_the_selection(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov._write("- Redis, 5min TTL\n")
        ov.root.update()
        ov.root.clipboard_clear()
        ov.root.clipboard_append("STALE")
        ov.text.tag_add("sel", "1.0", "1.7")
        ov.root.update()

        # Both delivery paths, because which one fires depends on whether a
        # child holds Tk focus - and focus does not reliably land in a process
        # that has created and destroyed many Tk roots, which is exactly what
        # this suite does.
        for target in (ov.text, ov.root):
            ov.root.clipboard_clear()
            ov.root.clipboard_append("STALE")
            # focus_force, not focus_set: a key event generated for an
            # unfocused widget is not delivered to it, and plain focus_set
            # does not take in a process that has churned through many Tk
            # roots. This still drives the real binding.
            target.focus_force()
            ov.root.update()
            target.event_generate("<Control-c>")
            ov.root.update()
            assert ov.root.clipboard_get() == "- Redis", (
                f"Ctrl+C on {target} left the clipboard at "
                f"{ov.root.clipboard_get()!r}"
            )
    finally:
        ov.root.destroy()


def test_ctrl_c_with_nothing_selected_does_not_raise(monkeypatch):
    # A disabled Text raises TclError for sel.first when there is no
    # selection. Unhandled, that reaches Tk's report_callback_exception and
    # only ever shows up on stderr.
    ov = _live_overlay(monkeypatch)
    try:
        ov._write("- Redis\n")
        ov.root.update()
        ov.root.clipboard_clear()
        ov.root.clipboard_append("UNTOUCHED")

        # Asserting "the clipboard did not change" alone has no teeth: it is
        # equally true when the binding raises, because Tk routes exceptions
        # from a callback to report_callback_exception and prints to stderr
        # rather than propagating. Watch that handler instead.
        blew_up = []
        ov.root.report_callback_exception = lambda *a: blew_up.append(a)

        ov.text.focus_force()
        ov.root.update()
        ov.text.event_generate("<Control-c>")
        ov.root.update()

        assert not blew_up, f"the binding raised with no selection: {blew_up}"
        assert ov.root.clipboard_get() == "UNTOUCHED"
    finally:
        ov.root.destroy()


def test_ctrl_wheel_scales_the_font_both_ways(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        start = ov._font_size

        ov.text.event_generate("<Control-MouseWheel>", delta=120)
        ov.root.update()
        bigger = ov._font_size
        # Checked HERE, while enlarged, not after scrolling back. Asserting
        # only the restored state passes on code that never reconfigures the
        # widget at all - it simply never left `start`. Caught by mutation.
        assert bigger > start, "scroll up did not enlarge the font"
        assert _font_size_of(ov.text) == bigger, (
            "the counter moved but the widget kept its old font"
        )

        ov.text.event_generate("<Control-MouseWheel>", delta=-120)
        ov.root.update()
        assert ov._font_size == start, "scroll down did not restore it"
        assert _font_size_of(ov.text) == start
    finally:
        ov.root.destroy()


def test_font_size_is_clamped_at_both_ends(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        for _ in range(80):
            ov.text.event_generate("<Control-MouseWheel>", delta=120)
        ov.root.update()
        assert ov._font_size == overlay_mod._FONT_MAX

        for _ in range(160):
            ov.text.event_generate("<Control-MouseWheel>", delta=-120)
        ov.root.update()
        assert ov._font_size == overlay_mod._FONT_MIN
    finally:
        ov.root.destroy()


def test_a_query_entry_opened_after_scaling_uses_the_current_size(monkeypatch):
    # The entry is created lazily and read `font=_FONT`, the module constant.
    # Left alone it would open at 12pt beside 20pt body text.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        for _ in range(5):
            ov.text.event_generate("<Control-MouseWheel>", delta=120)
        ov.root.update()

        ov._open_query_entry(lambda _t: None)
        ov.root.update()
        assert ov._entry is not None
        assert _font_size_of(ov._entry) == ov._font_size
        assert _font_size_of(ov._entry) != uistate.DEFAULT.font_size, (
            "entry fell back to the module constant"
        )
    finally:
        ov.root.destroy()


def test_an_already_open_query_entry_scales_too(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        ov._open_query_entry(lambda _t: None)
        ov.root.update()

        ov.text.event_generate("<Control-MouseWheel>", delta=120)
        ov.root.update()
        assert ov._entry is not None
        assert _font_size_of(ov._entry) == ov._font_size
        assert _font_size_of(ov.text) == ov._font_size
    finally:
        ov.root.destroy()


@pytest.mark.parametrize(
    "sequence, delta",
    [("<Control-MouseWheel>", 120), ("<Shift-MouseWheel>", -120)],
)


def test_wheel_gestures_keep_the_hwnd(monkeypatch, sequence, delta):
    # Display affinity lives on the HWND, so any gesture that rebuilt the
    # window - the obvious wrong way to implement hiding, and a plausible
    # wrong way to implement fading - would silently drop capture exclusion.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        before = ov.root.winfo_id()
        for _ in range(6):
            ov.text.event_generate(sequence, delta=delta)
        ov.root.update()
        assert ov.root.winfo_id() == before, "HWND changed - exclusion would drop"
    finally:
        ov.root.destroy()


def test_shift_wheel_changes_the_opacity_both_ways(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        start = ov._alpha

        ov.text.event_generate("<Shift-MouseWheel>", delta=-120)
        ov.root.update()
        # Asserted while faded, not after scrolling back: checking only the
        # restored value passes on code that never moved in the first place.
        assert ov._alpha < start, "scroll down did not fade the overlay"
        # One notch is one step, not two. The handler is bound on the Text AND
        # on the toplevel, and Tk runs widget -> class -> toplevel bindtags, so
        # only the "break" keeps a single notch from being counted twice.
        assert ov._alpha == pytest.approx(start - overlay_mod._ALPHA_STEP), (
            "one wheel notch did not move exactly one step"
        )
        assert float(ov.root.attributes("-alpha")) == pytest.approx(ov._alpha), (
            "the counter moved but the window kept its old alpha"
        )

        ov.text.event_generate("<Shift-MouseWheel>", delta=120)
        ov.root.update()
        assert ov._alpha == pytest.approx(start), "scroll up did not restore it"
        assert float(ov.root.attributes("-alpha")) == pytest.approx(start)
    finally:
        ov.root.destroy()


def test_opacity_is_clamped_at_both_ends(monkeypatch):
    # The floor is the load-bearing half. An overrideredirect window at alpha 0
    # has no titlebar to find and no frame to click, so an unclamped fade is a
    # one-way trip out of the UI.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        for _ in range(80):
            ov.text.event_generate("<Shift-MouseWheel>", delta=-120)
        ov.root.update()
        assert ov._alpha == pytest.approx(overlay_mod._ALPHA_MIN)
        assert overlay_mod._ALPHA_MIN > 0.0, "a zero floor is an invisible window"

        for _ in range(160):
            ov.text.event_generate("<Shift-MouseWheel>", delta=120)
        ov.root.update()
        assert ov._alpha == pytest.approx(overlay_mod._ALPHA_MAX)
    finally:
        ov.root.destroy()


def test_hiding_and_showing_preserves_the_chosen_opacity(monkeypatch):
    # toggle_visible drives the same -alpha the user is now setting. Restoring
    # a hard-coded 1.0 on show would silently throw the setting away.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        for _ in range(4):
            ov.text.event_generate("<Shift-MouseWheel>", delta=-120)
        ov.root.update()
        chosen = ov._alpha
        assert chosen < 1.0, "the fade never took effect, so the test proves nothing"

        ov.toggle_visible()          # hide
        ov.root.update()
        assert float(ov.root.attributes("-alpha")) == 0.0, "hiding must be fully clear"

        ov.toggle_visible()          # show
        ov.root.update()
        assert ov._alpha == pytest.approx(chosen)
        assert float(ov.root.attributes("-alpha")) == pytest.approx(chosen), (
            "showing reset the opacity instead of restoring it"
        )
    finally:
        ov.root.destroy()


def test_fading_while_hidden_does_not_reveal_the_overlay(monkeypatch):
    # Ctrl+Alt+H hides it; a stray Shift+wheel must not paint it back onto a
    # screen share. The setting is stored, the window stays clear.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        ov.toggle_visible()          # hide
        ov.root.update()

        ov.text.event_generate("<Shift-MouseWheel>", delta=-120)
        ov.root.update()
        assert float(ov.root.attributes("-alpha")) == 0.0, (
            "adjusting opacity un-hid the overlay"
        )
        assert ov._alpha < 1.0, "the adjustment was dropped instead of stored"
    finally:
        ov.root.destroy()


def test_shift_wheel_does_not_also_scroll_the_text(monkeypatch):
    # Shift+wheel is horizontal scroll on a Text by default. Without "break"
    # the view slides sideways every time the opacity is nudged.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        ov.text.configure(wrap="none")
        ov._write("x" * 4000 + "\n")
        ov.root.update()
        before = ov.text.xview()

        for _ in range(5):
            ov.text.event_generate("<Shift-MouseWheel>", delta=-120)
        ov.root.update()
        assert ov.text.xview() == before, f"the text scrolled sideways: {before}"
    finally:
        ov.root.destroy()


def test_monitor_rects_are_converted_out_of_physical_pixels():
    # mss reports physical pixels, Tk here is DPI-unaware and positions windows
    # in scaled ones. Measured on this machine: a 1920-wide screen at 125% is
    # 1536 to Tk. Clamping in the wrong units puts the window a quarter of a
    # screen off the right edge, which is the failure the clamp exists to stop.
    assert overlay_mod._scale_rect((0, 0, 1920, 1080), 1536 / 1920) == (0, 0, 1536, 864)
    # A screen left of the primary keeps its negative origin through the scale.
    assert overlay_mod._scale_rect((-1920, 0, 1920, 1080), 0.8) == (-1536, 0, 1536, 864)
    # No scaling is a no-op, not a rounding drift.
    assert overlay_mod._scale_rect((0, 0, 2560, 1440), 1.0) == (0, 0, 2560, 1440)


def test_scale_comes_from_the_primary_not_from_monitors_index_one(monkeypatch):
    # monitors[1] is just first in EnumDisplayMonitors order, which is not
    # always the primary. Taking the scale from the wrong monitor yields 1.0 on
    # a scaled display, the rects stay in physical pixels, and the clamp then
    # happily approves a position past the right-hand edge of Tk's space. That
    # is the guard rail failing in the direction it exists to prevent.
    import mss

    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        tk_width = ov.root.winfo_screenwidth()

        secondary = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        primary = {"left": 1920, "top": 0, "width": tk_width * 2, "height": 2160,
                   "is_primary": True}

        class FakeSct:
            monitors = [{"left": 0, "top": 0, "width": 1, "height": 1},
                        secondary, primary]
            primary_monitor = primary

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(mss, "MSS", FakeSct)
        rects = ov._monitors()

        # scale = tk_width / (tk_width * 2) = 0.5, applied to every rect.
        assert rects[0] == (960, 0, 960 + tk_width, 1080), (
            f"primary rect {rects[0]} was not halved, so the scale came from "
            "the wrong monitor"
        )
        assert rects[0][2] - rects[0][0] == tk_width
    finally:
        ov.root.destroy()


def test_monitors_fall_back_to_the_primary_screen_when_mss_fails(monkeypatch):
    # mss.exception.ScreenShotError subclasses Exception, not OSError, so a
    # narrow except tuple leaves this fallback unreachable for the one library
    # it guards and the failure reaches Overlay.__init__ instead.
    import mss
    import mss.exception

    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()

        def boom(*a, **k):
            raise mss.exception.ScreenShotError("no monitors")

        monkeypatch.setattr(mss, "MSS", boom)
        assert ov._monitors() == [
            (0, 0, ov.root.winfo_screenwidth(), ov.root.winfo_screenheight())
        ]
    finally:
        ov.root.destroy()


def test_the_clamp_uses_the_same_units_tk_positions_windows_in(monkeypatch):
    # The end-to-end version of the above: whatever _desktop reports, a hostile
    # x must land inside the area Tk itself calls the screen.
    uistate.save(uistate.UiState(w=460, h=300, x=99999, y=40, font_size=12, alpha=1.0))
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        assert ov.root.winfo_x() < ov.root.winfo_screenwidth(), (
            f"x={ov.root.winfo_x()} is off a {ov.root.winfo_screenwidth()}px screen"
        )
    finally:
        ov.root.destroy()


def test_a_saved_state_is_applied_on_startup(monkeypatch):
    # The whole point of the feature: what you set last time is what you get.
    uistate.save(uistate.UiState(w=612, h=388, x=140, y=96, font_size=19, alpha=0.55))

    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        assert ov._font_size == 19, "font size was not restored"
        assert ov._alpha == pytest.approx(0.55), "opacity was not restored"
        assert _font_size_of(ov.text) == 19, "the widget kept the default font"
        assert (ov.root.winfo_width(), ov.root.winfo_height()) == (612, 388)
        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (140, 96)
    finally:
        ov.root.destroy()


def test_a_hostile_saved_state_cannot_produce_an_unusable_window(monkeypatch):
    # The file is the back door around every live clamp. Zero alpha and an
    # off-screen origin are each enough to make the overlay unrecoverable.
    uistate.save(
        uistate.UiState(w=5, h=5, x=99999, y=-9999, font_size=400, alpha=0.0)
    )

    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        assert ov._alpha >= uistate.ALPHA_MIN, "restored an invisible overlay"
        assert ov._font_size <= uistate.FONT_MAX
        assert ov.root.winfo_width() >= uistate.MIN_W
        assert ov.root.winfo_height() >= uistate.MIN_H
        assert ov.root.winfo_y() >= 0, "drag bar restored above the screen top"
        assert ov.root.winfo_x() < ov.root.winfo_screenwidth(), "restored off-screen"
    finally:
        ov.root.destroy()


def test_quitting_writes_the_current_geometry(monkeypatch):
    # The debounce timer needs a running mainloop, so quit has to flush. It
    # also has to do so BEFORE destroy(), which is what makes this worth a
    # test: _save_now reads the geometry off the live window.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("533x341+77+88")
        ov.text.event_generate("<Control-MouseWheel>", delta=120)
        ov.root.update()
        expected_font = ov._font_size

        ov.ui_queue.put(("quit", None))
        ov._drain()
    finally:
        try:
            ov.root.destroy()
        except Exception:
            pass          # the quit path already destroyed it

    saved = uistate.load()
    assert (saved.w, saved.h) == (533, 341), f"geometry not saved: {saved}"
    assert (saved.x, saved.y) == (77, 88)
    assert saved.font_size == expected_font


def test_a_wheel_change_schedules_exactly_one_save(monkeypatch):
    # Debounced, not per event. A spun wheel is a burst; writing the file on
    # every notch would be hundreds of writes to change one number.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        scheduled = []
        for _ in range(12):
            ov.text.event_generate("<Control-MouseWheel>", delta=120)
            ov.root.update()
            scheduled.append(ov._save_job)

        assert ov._save_job is not None, "no save was scheduled"
        # Each notch cancels the one before, so of the twelve ids handed out
        # exactly one is still queued: the newest. Counting every pending timer
        # would be a weaker claim, since the drain loop has one of its own.
        queued = set(ov.root.tk.call("after", "info"))
        still = [i for i in scheduled if i in queued]
        assert still == [ov._save_job], (
            f"{len(still)} of 12 save timers survived, expected 1"
        )
    finally:
        ov.root.destroy()


def test_a_failing_save_still_lets_the_app_quit(monkeypatch):
    # Asserting "nothing blew up" proved nothing: _drain catches everything, so
    # that held even on code that never quits. What matters is that the window
    # is gone, because _alive is already False by then - a raise that skipped
    # destroy() would leave the overlay up and deaf to every hotkey.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        monkeypatch.setattr(
            uistate, "save", lambda _s: (_ for _ in ()).throw(OSError("nope"))
        )
        with pytest.raises(OSError):
            uistate.save(uistate.DEFAULT)   # the patch really is in force

        ov.ui_queue.put(("quit", None))
        ov._drain()

        try:
            alive = bool(ov.root.winfo_exists())
        except tk.TclError:
            # "application has been destroyed": the interpreter went with the
            # window, which is a stronger result than winfo_exists saying 0.
            alive = False
        assert not alive, "a failed save left the window up"
    finally:
        try:
            ov.root.destroy()
        except tk.TclError:
            pass      # already gone via the quit path, which is the point


def test_overlay_can_be_dragged_and_keeps_its_hwnd(monkeypatch):
    # overrideredirect(True) means no titlebar and no window manager to move
    # the window, so dragging is hand-rolled. It must move by the drag delta -
    # not jump so the grab point lands under the cursor - and it must do so
    # without recreating the toplevel: display affinity lives on the HWND, so
    # a new HWND would silently drop capture exclusion mid-drag.
    import queue as _queue

    from assistant import config as _config

    monkeypatch.setattr(_config, "EXCLUDE_FROM_CAPTURE", False)
    from assistant.overlay import Overlay

    ov = _tk_root_or_skip(lambda: Overlay(_queue.Queue()))

    try:
        ov.root.attributes("-alpha", 0.0)
        ov.root.geometry("+100+100")
        ov.root.update()
        hwnd_before = ov.root.winfo_id()

        # Drive the real handlers rather than setting geometry ourselves -
        # otherwise this passes with no drag code at all. Driven through the
        # drag bar since the text body became selectable.
        ov._drag_bar.event_generate("<Button-1>", rootx=150, rooty=150)
        ov.root.update()
        assert ov._drag_from == (50, 50), "grab did not record the offset"

        ov._drag_bar.event_generate("<B1-Motion>", rootx=210, rooty=190)
        ov.root.update()

        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (160, 140), (
            f"moved to {ov.root.winfo_x()},{ov.root.winfo_y()} - expected 160,140"
        )
        assert ov.root.winfo_id() == hwnd_before, "HWND changed - exclusion would drop"

        ov._drag_bar.event_generate("<ButtonRelease-1>", rootx=210, rooty=190)
        ov.root.update()
        assert ov._drag_from is None, "release did not clear the drag state"

        # A motion event with no grab must not move anything.
        ov._drag_bar.event_generate("<B1-Motion>", rootx=400, rooty=400)
        ov.root.update()
        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (160, 140)
    finally:
        ov.root.destroy()


def test_quit_cancels_a_pending_save_timer(monkeypatch):
    # A debounce timer still queued when the window is destroyed fires into a
    # dead interpreter, and Tk prints `invalid command name` to stderr. Seen
    # for real while exercising the launch/quit cycle by hand.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        ov.text.event_generate("<Control-MouseWheel>", delta=120)
        ov.root.update()
        pending = ov._save_job
        assert pending is not None, "nothing was scheduled, so nothing is proven"

        ov.ui_queue.put(("quit", None))
        ov._drain()

        assert ov._save_job is None, "the save timer outlived the window"
        assert pending not in set(ov.root.tk.call("after", "info"))
    finally:
        try:
            ov.root.destroy()
        except tk.TclError:
            pass
