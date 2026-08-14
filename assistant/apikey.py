"""Find the Anthropic key, or ask for it once and remember it.

Order is environment, then the stored blob, then a prompt. Environment first so
a developer export always wins over whatever a previous run saved, which is
what makes the key swappable without hunting for a file.

Stored through Windows DPAPI rather than as plain text. The blob is tied to the
Windows account, so another user on the same machine cannot read it, and it
costs no dependency: crypt32 is already there. Not encryption the user manages
- anything they would have to type on every launch defeats the point of
remembering it at all.
"""

import ctypes
import logging
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

from . import config, uistate

log = logging.getLogger(__name__)

# Secondary key. It ships inside the exe, so it is not a secret and does not
# defend against someone reading the binary. What it does buy: another
# program running as this user cannot decrypt the blob by calling
# CryptUnprotectData with no entropy, which is the realistic case.
_ENTROPY = b"meeting-assistant"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


_crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_crypt32.CryptProtectData.argtypes = [
    ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob),
]
_crypt32.CryptProtectData.restype = wintypes.BOOL
_crypt32.CryptUnprotectData.argtypes = [
    ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob),
]
_crypt32.CryptUnprotectData.restype = wintypes.BOOL
# Declared for the same reason the overlay declares its user32 calls: without
# argtypes ctypes marshals a pointer as C int, which is 32 bits on 64-bit
# Windows.
_kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
_kernel32.LocalFree.restype = wintypes.HLOCAL


def _key_path() -> Path:
    # Beside ui.json but not inside it: that file is rewritten on every resize,
    # and a credential should not ride along with window geometry.
    return uistate._state_path().with_name("key.bin")


def _as_blob(data: bytes) -> "tuple[_Blob, ctypes.Array]":
    # The buffer is returned as well as the blob. It owns the memory the blob
    # points at, and letting it fall out of scope frees it under the API.
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf


def _dpapi(fn: "Callable[..., int]", data: bytes) -> bytes:
    """Protect or unprotect, with _ENTROPY as the secondary key both ways.

    Entropy is the third parameter, NOT the second. The second is szDataDescr,
    a description string that is pure metadata and changes nothing about the
    encryption - passing the salt there would look right, encrypt fine, and
    protect nothing. It must match on both calls or unprotect fails.
    """
    blob, _owns = _as_blob(data)
    entropy, _owns_entropy = _as_blob(_ENTROPY)
    out = _Blob()
    if not fn(ctypes.byref(blob), None, ctypes.byref(entropy),
              None, None, 0, ctypes.byref(out)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        _kernel32.LocalFree(out.pbData)


def stored() -> "str | None":
    """The saved key, or None. Never raises.

    A blob written by a different Windows account, or a truncated file, both
    mean the same thing to the caller: there is no usable key, ask for one.
    Refusing to start would leave no way to correct it from inside the app.
    """
    path = _key_path()
    if not path.exists():
        return None
    try:
        return _dpapi(_crypt32.CryptUnprotectData, path.read_bytes()).decode()
    except Exception:
        log.warning("stored key is unreadable, asking again", exc_info=True)
        return None


def save(key: str) -> None:
    """Encrypt and write the key. Never raises, and never logs the key."""
    path = _key_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_dpapi(_crypt32.CryptProtectData, key.encode()))
    except Exception:
        # Deliberately no key material in the message. Losing the convenience
        # of a remembered key is survivable; leaking it into a log is not.
        log.warning("could not save the key to %s", path, exc_info=True)


def prompt() -> "str | None":
    """Ask for a key in a small window. None if the user closes it.

    Its own Tk root, not the overlay: this runs before the overlay exists, and
    the overlay refuses to start at all until capture exclusion is confirmed.
    """
    import tkinter as tk

    root = tk.Tk()
    root.title("Meeting assistant setup")
    root.configure(bg="#101014")
    root.geometry("520x190")
    root.resizable(False, False)

    typed: list[str] = []
    common = {"bg": "#101014", "fg": "#e8e8ea"}
    tk.Label(root, text="Anthropic API key", font=("Segoe UI", 12, "bold"),
             **common).pack(pady=(22, 4))
    tk.Label(root, text="Starts with sk-ant-. Stored for this Windows account\n"
                        "only, encrypted with DPAPI.",
             font=("Segoe UI", 9), justify="center",
             bg="#101014", fg="#9aa0aa").pack()

    entry = tk.Entry(root, show="*", width=46, font=("Consolas", 10),
                     insertbackground="#e8e8ea", **common)
    entry.pack(pady=14)
    entry.focus_set()

    def submit(_event: "object | None" = None) -> None:
        value = entry.get().strip()
        if value:
            typed.append(value)
            root.destroy()

    tk.Button(root, text="Save and start", command=submit).pack()
    entry.bind("<Return>", submit)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()
    return typed[0] if typed else None


def resolve() -> str:
    """Environment, then the stored key, then ask. Raises if all three fail."""
    try:
        return config.api_key()          # env wins, and stays out of module state
    except RuntimeError:
        pass

    key = stored()
    if key:
        return key

    key = prompt()
    if not key:
        raise RuntimeError(
            "No Anthropic API key. Set ANTHROPIC_API_KEY, or enter one when "
            "the setup window asks."
        )
    save(key)
    return key
