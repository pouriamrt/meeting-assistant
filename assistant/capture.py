"""Dual WASAPI capture: microphone and system output, both to 16 kHz mono."""

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from . import config
from .audio_ring import AudioRing

_FORMAT = pyaudio.paFloat32
_FRAMES_PER_BUFFER = 1024


def to_mono_16k(raw: bytes, channels: int, src_rate: int) -> np.ndarray:
    """Interleaved float32 bytes at the device rate to mono float32 at 16 kHz."""
    samples = np.frombuffer(raw, dtype=np.float32)
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if src_rate != config.SAMPLE_RATE:
        samples = soxr.resample(samples, src_rate, config.SAMPLE_RATE)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _loopback_device(p: pyaudio.PyAudio) -> dict:
    try:
        return p.get_default_wasapi_loopback()
    except OSError as exc:
        raise RuntimeError(
            "WASAPI is unavailable. This runs on Windows only."
        ) from exc
    except LookupError as exc:
        raise RuntimeError(
            "No WASAPI loopback device found. Run `python -m pyaudiowpatch` "
            "to list devices."
        ) from exc


def _mic_device(p: pyaudio.PyAudio) -> dict:
    try:
        return p.get_default_input_device_info()
    except OSError as exc:
        raise RuntimeError(
            "No default input device. Plug in a microphone and set it as the "
            "Windows default."
        ) from exc


class Capture:
    """Two capture streams feeding two rings. No fallback on a missing device."""

    def __init__(self) -> None:
        self.mic = AudioRing(config.RING_SAMPLES)
        self.sys = AudioRing(config.RING_SAMPLES)
        self._pa: pyaudio.PyAudio | None = None
        self._streams: list = []

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        # Resolve both devices before opening either, so a missing one fails
        # before we have a half-started capture to unwind.
        mic = _mic_device(self._pa)
        spk = _loopback_device(self._pa)
        self._streams.append(self._open(mic, self.mic))
        self._streams.append(self._open(spk, self.sys))

    def _open(self, device: dict, ring: AudioRing):
        rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])

        def callback(in_data, frame_count, time_info, status):
            # Must return fast: this is the WASAPI thread, and stalling it
            # causes audible dropouts in the call itself.
            ring.write(to_mono_16k(in_data, channels, rate))
            return (None, pyaudio.paContinue)

        return self._pa.open(
            format=_FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=_FRAMES_PER_BUFFER,
            stream_callback=callback,
        )

    def stop(self) -> None:
        for stream in self._streams:
            stream.stop_stream()
            stream.close()
        self._streams.clear()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
