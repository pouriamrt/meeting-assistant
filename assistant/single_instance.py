"""One copy at a time.

Two instances both capture the same audio, both run whisper on it, and both
register the same hotkeys. CPU roughly doubles and every decode queues behind
another one, so transcription falls behind speech and answers take longer -
without anything looking broken. Easy to do by accident once this is an exe
someone double-clicks, where there is no terminal to show it already running.
"""

import ctypes
import logging
from ctypes import wintypes

log = logging.getLogger(__name__)

_ERROR_ALREADY_EXISTS = 183

# No "Global\" prefix on purpose: the mutex is per logon session, so two
# different users on one machine may each run their own copy.
_NAME = "meeting-assistant-single-instance"

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.CreateMutexW.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL

_handle: "int | None" = None


def acquire() -> bool:
    """True if this process is the only copy running.

    The handle is stored rather than returned because it has to outlive this
    call: closing it frees the name and lets a second copy start.
    """
    global _handle
    handle = _kernel32.CreateMutexW(None, False, _NAME)
    if not handle:
        # Cannot tell either way, so allow the launch. Refusing to start over a
        # failed advisory check would be worse than the duplicate it prevents.
        # No exception is in flight here, so exc_info would print
        # "NoneType: None". use_last_error is set precisely so the reason is
        # readable; read it.
        log.warning(
            "could not create the instance mutex: %s",
            ctypes.WinError(ctypes.get_last_error()),
        )
        return True
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        _kernel32.CloseHandle(handle)
        return False
    _handle = handle
    return True


def release() -> None:
    """Drop the claim. The OS does this on exit; tests need it sooner."""
    global _handle
    if _handle is not None:
        _kernel32.CloseHandle(_handle)
        _handle = None
