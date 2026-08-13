import logging
import threading
import time
from datetime import datetime

import pytest
import numpy as np

from assistant import answer, config, prompts
from assistant.audio_ring import AudioRing
from assistant.capture import to_mono_16k
from assistant.transcribe import Transcript, Turn, speech_spans


def test_api_key_missing_raises_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        config.api_key()


def test_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    assert config.api_key() == "sk-ant-not-a-real-key"


def test_api_key_is_not_module_state(monkeypatch):
    # Read lazily so the value never sits in module globals, where a repr of
    # the module or a traceback frame could surface it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    config.api_key()
    assert not any(
        isinstance(v, str) and "sk-ant" in v for v in vars(config).values()
    )


def test_ring_returns_everything_below_cap():
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(4, dtype=np.float32))
    samples, start = ring.read_from(0)
    assert np.array_equal(samples, np.arange(4, dtype=np.float32))
    assert start == 0
    assert ring.total_written == 4


def test_ring_keeps_exact_tail_across_the_wrap():
    # Writes straddle the cap: 6 then 8 samples into a 10-sample ring.
    # The retained window must be the true last 10 samples, with no sample
    # dropped from the middle and none duplicated at the seam.
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(0, 6, dtype=np.float32))
    ring.write(np.arange(6, 14, dtype=np.float32))

    samples, start = ring.read_from(0)
    assert np.array_equal(samples, np.arange(4, 14, dtype=np.float32))
    assert start == 4
    assert ring.total_written == 14
    assert len(samples) == 10


def test_ring_read_from_cursor_returns_only_new_samples():
    ring = AudioRing(max_samples=100)
    ring.write(np.arange(0, 5, dtype=np.float32))
    ring.write(np.arange(5, 9, dtype=np.float32))
    samples, start = ring.read_from(5)
    assert np.array_equal(samples, np.arange(5, 9, dtype=np.float32))
    assert start == 5


def test_ring_reports_the_gap_when_the_reader_fell_behind():
    # A stalled consumer must be able to detect that audio was dropped,
    # rather than silently splicing two non-adjacent spans together.
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(0, 20, dtype=np.float32))
    samples, start = ring.read_from(0)
    assert start == 10, "reader asked for sample 0, oldest surviving is 10"
    assert np.array_equal(samples, np.arange(10, 20, dtype=np.float32))


def test_ring_read_past_the_end_is_empty_not_an_error():
    ring = AudioRing(max_samples=10)
    ring.write(np.arange(3, dtype=np.float32))
    samples, start = ring.read_from(3)
    assert len(samples) == 0
    assert start == 3


def test_ring_evicts_whole_chunks_under_many_small_writes():
    # The realistic capture pattern: many small same-size writes. This drives
    # the whole-chunk eviction path, which the straddle test never reaches —
    # there the overflow lands inside the final chunk and only trims it.
    ring = AudioRing(max_samples=5)
    for i in range(0, 20, 2):
        ring.write(np.arange(i, i + 2, dtype=np.float32))
    samples, start = ring.read_from(0)
    assert np.array_equal(samples, np.arange(15, 20, dtype=np.float32))
    assert start == 15
    assert ring.total_written == 20


def test_ring_does_not_retain_the_callers_buffer():
    # A capture backend may reuse one buffer for every frame. What the ring
    # hands back must not change when the caller overwrites its own array.
    ring = AudioRing(max_samples=10)
    buf = np.arange(4, dtype=np.float32)
    ring.write(buf)
    buf[:] = 99.0
    samples, _ = ring.read_from(0)
    assert np.array_equal(samples, np.arange(4, dtype=np.float32))


def test_to_mono_16k_downmixes_stereo():
    # Two interleaved channels: left all 1.0, right all -1.0 averages to 0.0.
    stereo = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    out = to_mono_16k(stereo.tobytes(), channels=2, src_rate=16_000)
    assert len(out) == 2
    assert np.allclose(out, 0.0)


def test_to_mono_16k_passes_through_at_native_rate():
    mono = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = to_mono_16k(mono.tobytes(), channels=1, src_rate=16_000)
    assert np.allclose(out, mono)


def test_to_mono_16k_resamples_48k_to_16k():
    # One second of 48 kHz must come back as one second of 16 kHz.
    mono = np.zeros(48_000, dtype=np.float32)
    out = to_mono_16k(mono.tobytes(), channels=1, src_rate=48_000)
    assert abs(len(out) - 16_000) <= 2, f"expected ~16000 samples, got {len(out)}"


def test_to_mono_16k_always_returns_float32():
    stereo = np.array([0.5, 0.5], dtype=np.float32)
    out = to_mono_16k(stereo.tobytes(), channels=2, src_rate=44_100)
    assert out.dtype == np.float32


def test_to_mono_16k_handles_an_empty_packet():
    # An idle output endpoint delivers zero-length packets rather than erroring.
    out = to_mono_16k(b"", channels=2, src_rate=48_000)
    assert len(out) == 0
    assert out.dtype == np.float32


def test_to_mono_16k_raises_on_a_truncated_packet():
    # Documents why the stream callback guards. A partial frame is possible,
    # and an unguarded raise on the WASAPI thread tears the stream down —
    # capture would die mid-meeting with an empty transcript as the only sign.
    with pytest.raises(ValueError):
        to_mono_16k(b"\x00\x00\x00", channels=1, src_rate=16_000)


def test_silent_start_empty_buffer_is_silence():
    # The WASAPI trap: an idle endpoint delivers nothing at all. The gate must
    # treat that as ordinary silence — no exception, no special case.
    assert speech_spans(np.zeros(0, dtype=np.float32)) == []


def test_silent_start_digital_silence_is_silence():
    assert speech_spans(np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32)) == []


def test_sub_window_buffer_is_silence_not_an_error():
    # Silero needs 512 samples. Anything shorter must not reach it.
    assert speech_spans(np.zeros(100, dtype=np.float32)) == []


def test_transcript_renders_with_source_tags():
    t = Transcript(max_turns=10)
    t.append(Turn(src="them", text="what's your postgres experience", ts=1.0))
    t.append(Turn(src="me", text="about four years", ts=2.0))
    assert t.render() == (
        "[them] what's your postgres experience\n[me] about four years"
    )


def test_transcript_render_carries_no_timestamps():
    # Rendering a timestamp would rewrite frozen bytes on every press and
    # silently invalidate the whole cached prefix.
    now = time.time()
    t = Transcript(max_turns=10)
    t.append(Turn(src="them", text="hello", ts=now))
    assert t.render() == "[them] hello"
    assert str(int(now)) not in t.render()


def test_transcript_drops_oldest_past_the_cap():
    t = Transcript(max_turns=2)
    for i in range(4):
        t.append(Turn(src="me", text=f"turn{i}", ts=float(i)))
    assert t.render() == "[me] turn2\n[me] turn3"
    assert len(t) == 2


def test_turn_is_frozen():
    turn = Turn(src="me", text="hello", ts=1.0)
    with pytest.raises(Exception):
        turn.text = "changed"


# --- gate_loop -------------------------------------------------------------
#
# The loop carries two positions at once: `cursor` into the ring and `pending`
# as a local accumulator. That interaction is the hardest logic in the project
# and needs no hardware to test — a scripted ring and a stubbed VAD reach every
# branch.

import assistant.transcribe as tm  # noqa: E402


class FakeRing:
    """Hands out scripted (samples, start) pairs in place of a real ring."""

    def __init__(self, reads):
        self._reads = list(reads)

    def read_from(self, abs_index):
        if self._reads:
            return self._reads.pop(0)
        return np.zeros(0, dtype=np.float32), abs_index


class StopAfter(threading.Event):
    """Ends the loop after N waits, so it runs a bounded number of
    iterations instantly instead of on a 200 ms timer."""

    def __init__(self, n):
        super().__init__()
        self._left = n

    def wait(self, timeout=None):
        self._left -= 1
        if self._left <= 0:
            self.set()
        return True


def _spans_over_nonzero(buf):
    """Stand-in for Silero: treats any nonzero sample as speech."""
    nz = np.nonzero(buf)[0]
    if len(nz) == 0:
        return []
    return [{"start": int(nz[0]), "end": int(nz[-1]) + 1}]


def _speech(n):
    return np.full(n, 0.5, dtype=np.float32)


def _silence(n):
    return np.zeros(n, dtype=np.float32)


def test_gate_loop_emits_a_turn_when_trailing_silence_closes_an_utterance(monkeypatch):
    monkeypatch.setattr(tm, "speech_spans", _spans_over_nonzero)
    monkeypatch.setattr(tm, "transcribe", lambda a: "hello there")
    reads = [(_speech(16_000), 0), (_silence(16_000), 16_000)]
    tr = Transcript(max_turns=10)
    tm.gate_loop(FakeRing(reads), "them", tr, StopAfter(4))
    assert tr.render() == "[them] hello there"


def test_gate_loop_discards_pending_when_the_ring_reports_a_gap(monkeypatch):
    # The second read starts far past the cursor: audio was dropped. The
    # partial utterance must be thrown away, not stitched across the gap.
    lengths = []
    monkeypatch.setattr(tm, "speech_spans", _spans_over_nonzero)
    monkeypatch.setattr(tm, "transcribe", lambda a: lengths.append(len(a)) or "x")
    reads = [
        (_speech(16_000), 0),
        (_speech(16_000), 16_000 * 5),   # gap: start >> cursor
        (_silence(16_000), 16_000 * 6),
    ]
    tm.gate_loop(FakeRing(reads), "me", Transcript(max_turns=10), StopAfter(5))
    assert lengths == [16_000], "transcribed audio spanned the dropped region"


def test_gate_loop_trims_pending_during_silence_and_emits_nothing(monkeypatch, caplog):
    monkeypatch.setattr(tm, "speech_spans", _spans_over_nonzero)
    monkeypatch.setattr(tm, "transcribe", lambda a: "should not be called")
    tr = Transcript(max_turns=10)
    with caplog.at_level(logging.ERROR):
        tm.gate_loop(FakeRing([(_silence(16_000 * 2), 0)]), "them", tr, StopAfter(4))
    assert len(tr) == 0
    # The log assertion is what gives this test teeth. "No turn emitted" is
    # equally true if the silence branch threw and the loop's except swallowed
    # it, so without this the test passes on a broken trim.
    assert not caplog.records, f"gate loop logged an error: {caplog.text}"


def test_gate_loop_forces_a_close_at_the_max_utterance_length(monkeypatch):
    # An unbroken monologue never produces trailing silence; the length cap
    # is what closes it.
    monkeypatch.setattr(tm, "speech_spans", _spans_over_nonzero)
    monkeypatch.setattr(tm, "transcribe", lambda a: "long answer")
    tr = Transcript(max_turns=10)
    reads = [(_speech(config.SAMPLE_RATE * config.MAX_UTTERANCE_S), 0)]
    tm.gate_loop(FakeRing(reads), "them", tr, StopAfter(3))
    assert tr.render() == "[them] long answer"


def test_gate_loop_survives_a_decode_failure(monkeypatch):
    # gate_loop runs on a daemon thread. An uncaught exception would kill it,
    # leaving that source's transcription silently dead for the rest of the
    # session — indistinguishable from nobody talking.
    calls = []

    def boom(_audio):
        calls.append(1)
        raise RuntimeError("ctranslate2 exploded")

    monkeypatch.setattr(tm, "speech_spans", _spans_over_nonzero)
    monkeypatch.setattr(tm, "transcribe", boom)
    reads = [
        (_speech(16_000), 0),
        (_silence(16_000), 16_000),
        (_speech(16_000), 16_000 * 2),
        (_silence(16_000), 16_000 * 3),
    ]
    tr = Transcript(max_turns=10)
    tm.gate_loop(FakeRing(reads), "them", tr, StopAfter(9))
    assert len(calls) >= 2, "the loop died after the first failure"
    assert len(tr) == 0


# --- prompts and request assembly ------------------------------------------


def test_cache_prefix_is_identical_across_both_modes():
    # Both hotkeys must share one cached prefix. If this drifts, every press
    # pays full price and nothing in the output looks wrong.
    text = "[them] what's your postgres experience"
    a = answer.build_request(text, prompts.last_question_tail())
    b = answer.build_request(text, prompts.query_tail("migrations?"))

    assert a["system"] == b["system"]
    assert a["messages"][0]["content"][0] == b["messages"][0]["content"][0]
    assert a["messages"][0]["content"][1] != b["messages"][0]["content"][1]

    # Identity, not just equality. Equality alone passes even if SYSTEM were
    # rebuilt per call with a clock value in it, because two calls issued
    # microseconds apart land in the same wall-clock second. Identity proves
    # the request reuses the frozen module constant.
    assert a["system"][0]["text"] is prompts.SYSTEM
    assert b["system"][0]["text"] is prompts.SYSTEM


def test_system_prompt_carries_nothing_volatile():
    assert str(datetime.now().year) not in prompts.SYSTEM
    assert "{" not in prompts.SYSTEM and "}" not in prompts.SYSTEM


def test_system_prompt_is_stable_across_calls():
    first = answer.build_request("x", "tail one")["system"]
    second = answer.build_request("y", "tail two")["system"]
    assert first == second
    assert first[0]["text"] is second[0]["text"] is prompts.SYSTEM


def test_exactly_two_cache_breakpoints():
    req = answer.build_request("transcript", "tail")
    system_block = req["system"][0]
    assert "cache_control" in system_block, "system block lost its breakpoint"
    marked = [system_block]
    marked += [b for b in req["messages"][0]["content"] if "cache_control" in b]
    assert len(marked) == 2, "one on system, one on the transcript, none on the tail"
    assert all(b["cache_control"]["ttl"] == "1h" for b in marked)


def test_transcript_is_a_single_block():
    # One block per turn would exhaust the 20-block cache lookback in about ten
    # exchanges and stop hitting, silently.
    req = answer.build_request("[them] a\n[me] b\n[them] c", "tail")
    assert len(req["messages"][0]["content"]) == 2


def test_request_omits_the_parameters_that_400_on_opus_5():
    req = answer.build_request("t", "tail")
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in req


def test_request_pins_fast_mode_and_effort():
    req = answer.build_request("t", "tail")
    assert req["model"] == "claude-opus-5"
    assert req["speed"] == "fast"
    assert req["betas"] == ["fast-mode-2026-02-01"]
    assert req["output_config"] == {"effort": "low"}
    assert req["max_tokens"] == 2000


def test_query_tail_embeds_the_typed_question():
    assert "how do you handle migrations" in prompts.query_tail(
        "how do you handle migrations"
    )
