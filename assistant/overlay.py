"""Always-on-top cue card overlay, excluded from screen capture."""

import ctypes
import logging
import queue
import tkinter as tk
from collections.abc import Callable
from ctypes import wintypes

from . import config

log = logging.getLogger(__name__)

# use_last_error is required. Loading through plain ctypes.windll leaves
# ctypes.get_last_error() at 0 even when the call failed, so the raised error
# would say "success" while the overlay sits visible in the screen share.
_user32 = ctypes.WinDLL("user32", use_last_error=True)

# Declare the signatures. Without these ctypes marshals HWNDs as C int, which
# is 32 bits on 64-bit Windows — a truncation hazard on the one call chain
# whose correctness is a security property.
_user32.GetParent.argtypes = [wintypes.HWND]
_user32.GetParent.restype = wintypes.HWND
_user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
_user32.SetWindowDisplayAffinity.restype = wintypes.BOOL

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

_BG = "#101014"
_FG = "#e8e8ea"
_FONT_FAMILY = "Segoe UI"
_FONT_SIZE = 12             # starting size only; Ctrl+wheel moves it at runtime

_GRIP = "#2a2a32"           # just visible against _BG, not attention-grabbing
_GRIP_PX = 16
_DRAG_BAR_PX = 18           # the only surface that moves the window, so the
                            # text body underneath stays selectable
_MIN_W, _MIN_H = 240, 120   # below this the text is unreadable and the grip
                            # gets hard to hit, leaving no way to grow it back
_FONT_MIN, _FONT_MAX = 8, 32
# The floor is not cosmetic: overrideredirect(True) leaves no titlebar and no
# frame, so an overlay faded to 0 is gone with nothing left to grab it by.
_ALPHA_MIN, _ALPHA_MAX = 0.25, 1.0
_ALPHA_STEP = 0.05


def set_capture_excluded(root: tk.Tk, excluded: bool) -> None:
    """Raises on failure. The caller must not show the window if this raises."""
    # winfo_id() is usually the CHILD hwnd and the toplevel is its parent, but
    # an overrideredirect window may have no wrapper — GetParent returns 0 and
    # winfo_id() is already the toplevel. Fall back rather than pass 0.
    child = root.winfo_id()
    hwnd = _user32.GetParent(child) or child
    flag = WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
    if not _user32.SetWindowDisplayAffinity(hwnd, flag):
        raise ctypes.WinError(ctypes.get_last_error())


class Overlay:
    """Owns the main thread. Nothing else touches a widget."""

    def __init__(self, ui_queue: "queue.Queue") -> None:
        self.ui_queue = ui_queue
        self.root = tk.Tk()
        self.root.title("assistant")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=_BG)
        self.root.geometry("460x300+40+40")

        # Packed before self.text so it claims the top strip; self.text then
        # takes the rest with expand/fill.
        self._drag_bar = tk.Frame(
            self.root, bg=_GRIP, height=_DRAG_BAR_PX, cursor="fleur"
        )
        # Keeps its configured height because it has no children to shrink to
        # - pack_propagate(False) was tried here and measured to change
        # nothing, so it is not present.
        self._drag_bar.pack(side="top", fill="x")

        self.text = tk.Text(
            self.root,
            bg=_BG,
            fg=_FG,
            font=(_FONT_FAMILY, _FONT_SIZE),
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=14,
            pady=12,
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        self._drag_from: tuple[int, int] | None = None
        # Only the bar. Not self.text, and not self.root either: bindtags run
        # the toplevel binding for events on any child, so a <Button-1> bound
        # on root would fire for clicks in the text body and start a window
        # drag instead of a selection.
        self._make_draggable(self._drag_bar)
        self._make_selectable()

        self._visible = True
        self._alive = True  # cleared on quit, so the drain stops rescheduling
        self._entry: tk.Entry | None = None  # inline query input, see _open_query_entry

        # After _entry: _apply_font reads it, and binding the handler before
        # the attribute exists leaves a window - narrow, but the kind that
        # only shows up as an AttributeError on a keystroke.
        self._font_size = _FONT_SIZE
        self._alpha = _ALPHA_MAX
        self._resize_from: tuple[int, int, int, int] | None = None
        self._grip = self._make_resizable()
        self._make_font_scalable()
        self._make_opacity_scalable()
        # Fully transparent until exclusion is confirmed. The window has to be
        # mapped before it has an HWND, so it cannot simply stay withdrawn —
        # alpha 0 gets it mapped without ever painting a visible frame.
        self.root.attributes("-alpha", 0.0)
        # Called directly here, not scheduled via root.after(0, ...): a raise
        # inside an `after` callback is caught by Tk's own callback wrapper
        # and only handed to report_callback_exception (default: printed to
        # stderr), never re-raised to whoever called run(). That would
        # silently swallow the one exception this code exists to surface, so
        # the overlay would end up destroyed-but-not-raised instead of
        # visibly refusing to start. Run before mainloop even starts, on the
        # plain Python call stack, so a raise here propagates normally out of
        # __init__ to the caller of Overlay(...). Verified empirically:
        # update_idletasks() maps the window and produces a usable HWND
        # without the mainloop running.
        self._apply_exclusion_then_show()
        self.root.after(50, self._drain)

    def _apply_exclusion_then_show(self) -> None:
        self.root.update_idletasks()   # force the HWND into existence
        if config.EXCLUDE_FROM_CAPTURE:
            try:
                set_capture_excluded(self.root, True)
            except OSError as exc:
                self.root.destroy()
                raise RuntimeError(
                    "Screen-capture exclusion failed; refusing to show the "
                    "overlay unprotected. Needs Windows 10 build 19041+."
                ) from exc
        self._apply_alpha()

    def _make_draggable(self, widget: tk.Misc) -> None:
        """Drag the window by its body.

        overrideredirect(True) removes the titlebar, so there is nothing to
        grab and no window manager to do the moving - both have to be
        hand-rolled. Moving does not touch the HWND, so capture exclusion
        survives being dragged.
        """

        def grab(event: "tk.Event") -> None:
            self._drag_from = (
                event.x_root - self.root.winfo_x(),
                event.y_root - self.root.winfo_y(),
            )

        def drag(event: "tk.Event") -> None:
            if self._drag_from is None:
                return
            dx, dy = self._drag_from
            self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

        def drop(_event: "tk.Event") -> None:
            self._drag_from = None

        widget.bind("<Button-1>", grab)
        widget.bind("<B1-Motion>", drag)
        widget.bind("<ButtonRelease-1>", drop)

    def _make_selectable(self) -> None:
        """Select with the mouse, copy with Ctrl+C.

        Selection already worked on a state="disabled" Text - copying does not
        modify it - so nothing here fights the read-only display. What blocked
        it was the drag binding claiming <Button-1> first.

        Focus is taken on release because an overrideredirect window has no
        titlebar to click, and a keyboard binding never fires for a widget
        that was never focused.
        """
        self.text.bind("<ButtonRelease-1>", lambda _e: self.text.focus_set())
        for sequence in ("<Control-c>", "<Control-C>", "<<Copy>>"):
            # Bound on the toplevel as well as the Text. A keyboard event goes
            # to the focused widget, and if the window has OS focus while no
            # child holds Tk focus it arrives at the toplevel instead - where
            # a Text-only binding would never see it. _copy_selection returns
            # "break", so the two bindings cannot both run for one keypress.
            self.text.bind(sequence, self._copy_selection)
            self.root.bind(sequence, self._copy_selection)

    def _copy_selection(self, _event: "tk.Event | None" = None) -> str:
        """Copy the selection, or do nothing if there is none.

        A Text with no selection raises TclError for sel.first rather than
        returning empty. Unhandled inside a Tk binding that reaches
        report_callback_exception and lands on stderr, which nobody is
        watching mid-interview.
        """
        try:
            selected = self.text.get("sel.first", "sel.last")
        except tk.TclError:
            return "break"
        if selected:
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
        return "break"

    def _make_resizable(self) -> tk.Frame:
        """Bottom-right grip. Same reason as the drag: overrideredirect(True)
        removes the frame, so there is no resize border and no window manager.

        Every handler returns "break". Tk runs bindtags widget -> class ->
        TOPLEVEL -> all, and <Button-1> is already bound on self.root to start
        a drag, so without it the window resizes and slides at the same time.
        Measured: the handlers fire ['grip', 'root'] without the break.
        """
        grip = tk.Frame(
            self.root, bg=_GRIP, width=_GRIP_PX, height=_GRIP_PX,
            cursor="size_nw_se",
        )
        # place(), not pack(): self.text is packed expand/fill and owns the
        # whole cavity, so a packed grip is allocated nothing and stays
        # unmapped at 1x1 - the exact trap that hid the query entry.
        grip.place(relx=1.0, rely=1.0, anchor="se")

        def grab(event: "tk.Event") -> str:
            self._resize_from = (
                event.x_root,
                event.y_root,
                self.root.winfo_width(),
                self.root.winfo_height(),
            )
            return "break"

        def drag(event: "tk.Event") -> str:
            if self._resize_from is None:
                return "break"
            px, py, w0, h0 = self._resize_from
            width = max(_MIN_W, w0 + event.x_root - px)
            height = max(_MIN_H, h0 + event.y_root - py)
            # x and y are pinned: the window grows from its top-left corner
            # rather than wandering across the screen while being resized.
            self.root.geometry(
                f"{width}x{height}+{self.root.winfo_x()}+{self.root.winfo_y()}"
            )
            return "break"

        def drop(_event: "tk.Event") -> str:
            self._resize_from = None
            return "break"

        grip.bind("<Button-1>", grab)
        grip.bind("<B1-Motion>", drag)
        grip.bind("<ButtonRelease-1>", drop)
        return grip

    def _make_font_scalable(self) -> None:
        """Ctrl+wheel resizes the text. Bigger window shows more lines; this
        is for when 12pt is simply hard to read at a glance."""

        def scale(event: "tk.Event") -> str:
            step = 1 if event.delta > 0 else -1
            self._font_size = max(
                _FONT_MIN, min(_FONT_MAX, self._font_size + step)
            )
            self._apply_font()
            return "break"   # otherwise the Text scrolls as well

        for widget in (self.root, self.text):
            widget.bind("<Control-MouseWheel>", scale)

    def _make_opacity_scalable(self) -> None:
        """Shift+wheel fades the window, so what is behind it stays readable."""

        def scale(event: "tk.Event") -> str:
            step = _ALPHA_STEP if event.delta > 0 else -_ALPHA_STEP
            self._alpha = max(
                _ALPHA_MIN, min(_ALPHA_MAX, self._alpha + step)
            )
            self._apply_alpha()
            return "break"   # Shift+wheel scrolls a Text sideways otherwise

        for widget in (self.root, self.text):
            widget.bind("<Shift-MouseWheel>", scale)

    def _apply_alpha(self) -> None:
        """The only place that writes -alpha once the overlay is running.

        Opacity and hiding share one channel, so one function owns it and
        hidden always wins: a stray Shift+wheel while the overlay is hidden
        must not paint it back onto the monitor. Not a capture concern - with
        EXCLUDE_FROM_CAPTURE on, the window is already absent from any share -
        but Ctrl+Alt+H is also what you press when someone walks up behind you,
        and that only works if nothing else can undo it. The chosen value is
        still stored, and shows up again on the way back.
        """
        self.root.attributes("-alpha", self._alpha if self._visible else 0.0)

    def _apply_font(self) -> None:
        """Push the current size onto every widget that shows text.

        The size lives on the instance rather than in a module constant
        because the query entry is built lazily - reading the constant would
        open it at 12pt beside 20pt body text. _open_query_entry calls this
        instead of composing the font itself, so a new entry cannot drift.
        """
        font = (_FONT_FAMILY, self._font_size)
        self.text.configure(font=font)
        if self._entry is not None:
            self._entry.configure(font=font)

    def _write(self, chunk: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", chunk)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def _open_query_entry(self, payload: Callable[[str], None]) -> None:
        """Inline text entry packed into `self.root` instead of a modal
        dialog. A dialog is a Toplevel with its own HWND, and
        SetWindowDisplayAffinity is per-HWND and not inherited - a second
        window would be visible to the screen share. One window, one HWND,
        already excluded.
        """
        if self._entry is not None:
            self._entry.focus_set()
            return

        entry = tk.Entry(
            self.root,
            bg=_BG,
            fg=_FG,
            insertbackground=_FG,
            borderwidth=0,
            highlightthickness=0,
        )
        # `before=self.text` is load-bearing, not cosmetic. self.text is
        # packed with expand=True/fill=both, so it claims the whole cavity;
        # anything packed after it is allocated nothing and silently stays
        # unmapped at 1x1 - pack() succeeds, the widget never appears, and
        # the typed-query hotkey looks like it does nothing at all.
        entry.pack(
            side="bottom", fill="x", padx=14, pady=(0, 12), before=self.text
        )
        entry.focus_set()

        def submit(_event: "tk.Event | None" = None) -> None:
            typed = entry.get()
            entry.destroy()
            self._entry = None
            # Destroyed before invoking payload, so a raising callback
            # cannot leave a stray widget behind. Guarded like every other
            # caller-supplied callback in this file: submit is a Tk event
            # binding, not code running inside _drain's try, so an
            # uncaught exception here would reach Tk's
            # report_callback_exception (stderr only) instead of this
            # module's log.
            if typed:
                try:
                    payload(typed)
                except Exception:
                    log.exception("query callback failed")

        def cancel(_event: "tk.Event | None" = None) -> None:
            entry.destroy()
            self._entry = None

        entry.bind("<Return>", submit)
        entry.bind("<Escape>", cancel)
        self._entry = entry
        # Sized here rather than in the constructor above, so there is exactly
        # one place that decides what font a widget gets. Composing it twice
        # is how an entry opened after Ctrl+wheel ends up at the old size.
        self._apply_font()

    def _drain(self) -> None:
        try:
            while True:
                item = self.ui_queue.get_nowait()
                try:
                    # Unpack inside the guard too: a malformed item must not
                    # raise past both excepts.
                    kind, payload = item
                    if kind == "clear":
                        # Every answer starts with a clear, so this is where
                        # a buried overlay comes back: press any hotkey and
                        # it surfaces along with the answer.
                        self._clear()
                        self._raise()
                    elif kind == "token":
                        self._write(payload)
                    elif kind == "status":
                        self._clear()
                        self._write(payload)
                        self._raise()
                    elif kind == "prompt":
                        # Entry must be built on the thread that owns the
                        # mainloop; payload is a callback taking the typed
                        # string.
                        self._open_query_entry(payload)
                    elif kind == "toggle":
                        self.toggle_visible()
                    elif kind == "quit":
                        self._alive = False
                        self.root.destroy()
                        return
                    elif kind != "done":
                        # "done" is deliberately unhandled here. Anything
                        # else is protocol drift and worth a log line
                        # instead of silently doing nothing.
                        log.warning("unknown ui message %r", kind)
                except Exception:
                    # One bad message must not kill the drain. This callback
                    # is the overlay's only channel — including its own quit
                    # path — and Tk swallows exceptions raised inside an
                    # `after` callback to stderr, so the death would be both
                    # silent and permanent. `prompt` payloads are
                    # caller-supplied code and the likeliest trigger.
                    log.exception("overlay message %r failed", item)
        except queue.Empty:
            pass
        finally:
            # In `finally`, not after the try: an escaping exception would
            # otherwise skip the reschedule and stop the loop forever.
            if self._alive:
                self.root.after(50, self._drain)

    def _raise(self) -> None:
        """Put the window back on top of whatever buried it.

        Always a recovery the user triggers, never something the app notices:
        Tk reports -topmost 1 and winfo_viewable True even while another
        window covers this one, so there is nothing to detect. And alpha
        cannot change z-order, which is why Ctrl+Alt+H alone could not bring
        it back.

        Toggling -topmost off and on is what re-applies it; setting it True
        when Tk already believes it is True does nothing, and lift() alone
        does not outrank a window that is genuinely topmost. Neither touches
        the HWND, so capture exclusion survives.

        Deliberately not on a timer. A window that reclaims the front every
        second fights every other topmost window on the desktop, which is a
        worse neighbour than one you raise on purpose.
        """
        self.root.lift()
        self.root.attributes("-topmost", False)
        self.root.attributes("-topmost", True)

    def toggle_visible(self) -> None:
        # Alpha, not withdraw/deiconify: unmapping can hand back a different
        # HWND on remap, and display affinity lives on the HWND. Hiding must
        # not be able to silently drop capture exclusion.
        self._visible = not self._visible
        # _apply_alpha, not a literal: showing must replay whatever Shift+wheel
        # was set to, or hiding quietly discards the user's choice.
        self._apply_alpha()
        if self._visible:
            self._raise()

    def run(self) -> None:
        self.root.mainloop()
