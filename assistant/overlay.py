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
_FONT = ("Segoe UI", 12)


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

        self.text = tk.Text(
            self.root,
            bg=_BG,
            fg=_FG,
            font=_FONT,
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            padx=14,
            pady=12,
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        self._drag_from: tuple[int, int] | None = None
        self._make_draggable(self.text)
        self._make_draggable(self.root)

        self._visible = True
        self._alive = True  # cleared on quit, so the drain stops rescheduling
        self._entry: tk.Entry | None = None  # inline query input, see _open_query_entry
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
        self.root.attributes("-alpha", 1.0)

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
            font=_FONT,
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

    def _drain(self) -> None:
        try:
            while True:
                item = self.ui_queue.get_nowait()
                try:
                    # Unpack inside the guard too: a malformed item must not
                    # raise past both excepts.
                    kind, payload = item
                    if kind == "clear":
                        self._clear()
                    elif kind == "token":
                        self._write(payload)
                    elif kind == "status":
                        self._clear()
                        self._write(payload)
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

    def toggle_visible(self) -> None:
        # Alpha, not withdraw/deiconify: unmapping can hand back a different
        # HWND on remap, and display affinity lives on the HWND. Hiding must
        # not be able to silently drop capture exclusion.
        self._visible = not self._visible
        self.root.attributes("-alpha", 1.0 if self._visible else 0.0)

    def run(self) -> None:
        self.root.mainloop()
