import logging
import queue
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


def test_ring_read_from_a_ring_that_has_never_been_written():
    # gate_loop's first tick, before any audio arrives. Without the empty-deque
    # guard np.concatenate([]) raises and both gate threads log on every tick.
    samples, start = AudioRing(max_samples=10).read_from(0)
    assert len(samples) == 0
    assert start == 0


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


#
# The loop carries two positions at once: `cursor` into the ring and `pending`
# as a local accumulator. That interaction is the hardest logic in the project
# and needs no hardware to test — a scripted ring and a stubbed VAD reach every
# branch.

import assistant.transcribe as tm  # noqa: E402


class FakeRing:
    """Hands out scripted (samples, start) pairs in place of a real ring.

    total_written mirrors the real AudioRing's: it advances with each
    scripted read that delivers samples, and holds once the script runs dry
    - gate_loop's stall detector reads it exactly like the real ring.
    """

    def __init__(self, reads):
        self._reads = list(reads)
        self.total_written = 0

    def read_from(self, abs_index):
        if self._reads:
            samples, start = self._reads.pop(0)
            self.total_written = start + len(samples)
            return samples, start
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


def test_gate_loop_logs_every_turn_it_appends(monkeypatch, caplog):
    # The only observable sign that capture is alive. The overlay shows
    # answers and never the transcript, so with nothing logged here a working
    # mic and a dead one produce identical output - which is exactly what
    # happened: the log showed whisper running and gave no way to tell whether
    # a turn had landed. The checklist's own "watch the log" was impossible.
    monkeypatch.setattr(tm, "speech_spans", _spans_over_nonzero)
    monkeypatch.setattr(tm, "transcribe", lambda a: "four years of postgres")
    reads = [(_speech(16_000), 0), (_silence(16_000), 16_000)]
    tr = Transcript(max_turns=10)
    with caplog.at_level(logging.INFO):
        tm.gate_loop(FakeRing(reads), "me", tr, StopAfter(4))

    assert len(tr) == 1, "precondition: a turn must have been appended"
    logged = [r.getMessage() for r in caplog.records]
    assert any("[me]" in m and "four years of postgres" in m for m in logged), (
        f"the appended turn was not logged: {logged}"
    )


def test_gate_loop_logs_nothing_when_no_turn_is_appended(monkeypatch, caplog):
    # The obvious way to break the test above is to log unconditionally, which
    # would report turns that were never appended - worse than silence,
    # because it looks like proof the mic works.
    monkeypatch.setattr(tm, "speech_spans", _spans_over_nonzero)
    monkeypatch.setattr(tm, "transcribe", lambda a: "")
    tr = Transcript(max_turns=10)
    with caplog.at_level(logging.INFO):
        tm.gate_loop(FakeRing([(_speech(16_000), 0), (_silence(16_000), 16_000)]),
                     "me", tr, StopAfter(4))
    assert len(tr) == 0
    assert not any("[me]" in r.getMessage() for r in caplog.records)


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


def test_gate_loop_warns_once_when_the_ring_stalls(monkeypatch, caplog):
    # Unplugging a headset mid-meeting produces no exception - WASAPI just
    # stops invoking the callback, so total_written never moves. Lower the
    # threshold via the module constant so this doesn't need a real 30s run.
    monkeypatch.setattr(tm, "_STALL_TICKS", 3)
    monkeypatch.setattr(tm, "speech_spans", lambda a: [])
    ring = FakeRing([])  # no scripted reads: total_written stays at 0 forever
    with caplog.at_level(logging.WARNING):
        tm.gate_loop(ring, "them", Transcript(max_turns=10), StopAfter(6))
    assert caplog.text.count("delivered nothing for 30s") == 1, (
        "must warn once at the threshold, not on every tick after it"
    )




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


def test_only_the_system_block_is_cached():
    # The transcript grows every press, so a breakpoint on it would never see
    # a repeated prefix - it would write a fresh 1h-TTL entry (2x price) on
    # nearly every press and get read back essentially never. Only the
    # system block is byte-identical across presses.
    req = answer.build_request("transcript", "tail")
    system_block = req["system"][0]
    assert "cache_control" in system_block, "system block lost its breakpoint"
    assert system_block["cache_control"]["ttl"] == "1h"
    marked = [b for b in req["messages"][0]["content"] if "cache_control" in b]
    assert marked == [], "the transcript and tail blocks must carry no cache_control"


def test_transcript_is_a_single_block():
    # One block per turn would exhaust the 20-block cache lookback in about ten
    # exchanges and stop hitting, silently.
    req = answer.build_request("[them] a\n[me] b\n[them] c", "tail")
    assert len(req["messages"][0]["content"]) == 2


def test_request_omits_the_parameters_that_400_on_opus_5():
    req = answer.build_request("t", "tail")
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in req


def test_request_pins_the_model_and_effort():
    req = answer.build_request("t", "tail")
    assert req["model"] == "claude-opus-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["max_tokens"] == 2000


def test_fast_mode_is_off_by_default_and_absent_from_the_request(monkeypatch):
    # Measured against the live API: without the org entitlement every request
    # 429s with "rate limit of 0 fast mode input tokens per minute", so the app
    # is unusable rather than merely slower. Off unless explicitly enabled.
    monkeypatch.setattr(config, "FAST_MODE", False)
    req = answer.build_request("t", "tail")
    assert "speed" not in req
    assert "betas" not in req


def test_fast_mode_when_enabled_sends_the_speed_and_the_beta_together(monkeypatch):
    # Both are required together: speed="fast" without the beta flag errors,
    # and the flag alone does nothing.
    monkeypatch.setattr(config, "FAST_MODE", True)
    req = answer.build_request("t", "tail")
    assert req["speed"] == "fast"
    assert req["betas"] == ["fast-mode-2026-02-01"]


def test_query_tail_embeds_the_typed_question():
    assert "how do you handle migrations" in prompts.query_tail(
        "how do you handle migrations"
    )


#
# Screen mode. The third hotkey sends screenshots alongside the transcript,
# under its own system prompt: the speaking prompt hard-enforces 3-5 bullets
# of 12 words, which is exactly wrong for a coding question that wants real
# code. Two prompts means two cached prefixes, deliberately.

import base64  # noqa: E402

import assistant.screen as screen_mod  # noqa: E402


def _png_blocks(req):
    return [b for b in req["messages"][0]["content"] if b["type"] == "image"]


def test_screen_request_puts_images_before_the_text():
    # Documented ordering for multi-modal accuracy: image first, then the
    # question about it. Reversed, the model reads the ask before it has
    # anything to look at.
    req = answer.build_request("[them] hi", "tail", images=[b"\x89PNGfake"])
    kinds = [b["type"] for b in req["messages"][0]["content"]]
    assert kinds == ["image", "text", "text"]


def test_screen_request_base64s_each_png_as_its_own_block():
    req = answer.build_request(
        "t", "tail", images=[b"\x89PNGone", b"\x89PNGtwo"]
    )
    blocks = _png_blocks(req)
    assert len(blocks) == 2, "one block per monitor, never stitched into one"
    assert [b["source"]["data"] for b in blocks] == [
        base64.b64encode(b"\x89PNGone").decode(),
        base64.b64encode(b"\x89PNGtwo").decode(),
    ]
    assert all(b["source"]["media_type"] == "image/png" for b in blocks)
    assert all(b["source"]["type"] == "base64" for b in blocks)


def test_audio_only_requests_carry_no_image_block():
    # Regression guard on the default argument. A mutable default or a stray
    # image block would add ~1800 tokens to every Ctrl+Space press, invisibly.
    assert _png_blocks(answer.build_request("t", "tail")) == []
    assert len(answer.build_request("t", "tail")["messages"][0]["content"]) == 2


def test_no_empty_text_block_when_the_transcript_is_empty():
    # Live 400: "messages: text content blocks must be non-empty". Pressing
    # Ctrl+Alt+Space before anyone has spoken renders an empty transcript,
    # which became an empty text block and the whole request was rejected.
    # on_last_question guards len(transcript)==0; on_screen and on_query do
    # not, and should not - a screenshot alone is a legitimate question. So
    # the guard belongs here, where every caller passes through.
    for req in (
        answer.build_request("", "tail"),
        answer.build_request("", "tail", images=[b"\x89PNG"]),
        answer.build_request("   \n  ", "tail"),
    ):
        blocks = req["messages"][0]["content"]
        empties = [
            b for b in blocks
            if b["type"] == "text" and not b["text"].strip()
        ]
        assert empties == [], f"empty text block would 400: {empties}"
        assert blocks, "content must never be an empty list either"
        assert blocks[-1]["text"] == "tail", "the tail must survive"


def test_a_present_transcript_is_still_sent():
    # The obvious way to break the fix above is to drop the transcript block
    # unconditionally, which passes that test and silently removes all context.
    req = answer.build_request("[them] hello", "tail")
    texts = [b["text"] for b in req["messages"][0]["content"] if b["type"] == "text"]
    assert texts == ["[them] hello", "tail"]


def test_screen_mode_uses_the_code_prompt_not_the_speaking_one():
    # The whole reason screen mode exists. If these ever collapse to one
    # prompt, screen answers come back as 12-word bullets and the code is gone.
    spoken = answer.build_request("t", prompts.last_question_tail())
    coded = answer.build_request(
        "t", prompts.screen_tail(), images=[b"x"], system=prompts.CODE_SYSTEM
    )
    assert spoken["system"][0]["text"] is prompts.SYSTEM
    assert coded["system"][0]["text"] is prompts.CODE_SYSTEM
    assert prompts.CODE_SYSTEM != prompts.SYSTEM


def test_code_prompt_keeps_its_own_cache_breakpoint():
    req = answer.build_request(
        "t", "tail", images=[b"x"], system=prompts.CODE_SYSTEM
    )
    assert req["system"][0]["cache_control"]["ttl"] == "1h"
    # Images are ~1800 tokens and byte-identical only if the screen never
    # changed. A breakpoint on them would write a fresh 1h entry per press.
    marked = [b for b in req["messages"][0]["content"] if "cache_control" in b]
    assert marked == [], "no cache_control below the system block"


def test_code_prompt_carries_nothing_volatile():
    # Not the blanket brace ban SYSTEM gets: this prompt contains a Python
    # example and `seen = {}` is legitimate. The real invariant is that the
    # placeholder was substituted, so assert that directly.
    assert str(datetime.now().year) not in prompts.CODE_SYSTEM
    assert "{NAME}" not in prompts.CODE_SYSTEM
    assert config.NAME in prompts.CODE_SYSTEM, "substitution never happened"


def test_code_prompt_still_bans_the_machine_vocabulary():
    # Screen mode swaps the output shape, not the voice. Losing the humanizer
    # block here would make every code explanation read as generated.
    assert "leverage" in prompts.CODE_SYSTEM
    assert "Great question" in prompts.CODE_SYSTEM


def test_screen_tail_names_both_inputs():
    tail = prompts.screen_tail()
    assert "creen" in tail, "must tell the model the images are the screen"
    assert "ranscript" in tail, "must tell it the transcript is what was said"


class _FakeShot:
    def __init__(self, tag):
        self.rgb = tag
        self.size = (1536, 864)


class _FakeSct:
    """Stand-in for mss.mss(). monitors[0] is mss's all-in-one union."""

    def __init__(self, count):
        self.monitors = [{"union": True}] + [{"i": i} for i in range(count)]
        self.grabbed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def grab(self, mon):
        self.grabbed.append(mon)
        return _FakeShot(f"rgb{mon['i']}".encode())


def test_grab_returns_one_png_per_monitor(monkeypatch):
    sct = _FakeSct(count=2)
    monkeypatch.setattr(screen_mod.mss, "MSS", lambda: sct)
    monkeypatch.setattr(
        screen_mod.mss.tools, "to_png", lambda rgb, size: b"PNG:" + rgb
    )

    out = screen_mod.grab()

    assert out == [b"PNG:rgb0", b"PNG:rgb1"]
    # monitors[0] IS the stitched union of every screen - grabbing it would
    # hand back the squashed image this design exists to avoid, plus a
    # duplicate of every monitor already captured.
    assert sct.grabbed == [{"i": 0}, {"i": 1}]


def test_grab_on_a_single_monitor_returns_one_image(monkeypatch):
    sct = _FakeSct(count=1)
    monkeypatch.setattr(screen_mod.mss, "MSS", lambda: sct)
    monkeypatch.setattr(
        screen_mod.mss.tools, "to_png", lambda rgb, size: b"PNG:" + rgb
    )
    assert screen_mod.grab() == [b"PNG:rgb0"]


def test_code_prompt_clears_the_512_token_cache_floor():
    # The silent failure this whole file already documents once: below Opus 5's
    # 512-token minimum the block stops caching with no error and
    # cache_creation_input_tokens: 0. A characters/3.6 estimate was measured
    # 45% wrong on SYSTEM, so this counts for real rather than estimating.
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("needs a live key to count tokens")

    import anthropic

    client = anthropic.Anthropic(api_key=config.api_key())
    count = client.messages.count_tokens(
        model="claude-opus-5",
        system=[{"type": "text", "text": prompts.CODE_SYSTEM}],
        messages=[{"role": "user", "content": "x"}],
    ).input_tokens
    assert count >= 512, f"CODE_SYSTEM is {count} tokens - will not cache"


#
# No GUI, no hotkeys, no audio device needed here: _AnswerGate is plain
# counter-and-lock logic, and a fake queue exercises it exactly as
# `answer.stream_answer` would.

import assistant.main as main_mod  # noqa: E402
from assistant.main import _AnswerGate  # noqa: E402


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_api_key_never_reaches_the_log_or_the_ui(monkeypatch, caplog):
    # The exception message DOES carry the canary - that's what makes this
    # test able to fail. A canary the exception never contains can never
    # trip, regardless of what stream_answer does with it (see the fix
    # report for the earlier version of this test, which had that flaw).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-canary-value")

    class Boom:
        class beta:
            class messages:
                @staticmethod
                def stream(**kw):
                    raise RuntimeError("401 for key sk-ant-canary-value")

    q = _FakeQueue()
    with caplog.at_level(logging.ERROR):
        answer.stream_answer(Boom(), "[them] hi", "tail", q)

    # The real requirement: nothing key-shaped reaches the ui_queue, which
    # feeds the screen-shared overlay.
    assert not any("sk-ant-canary-value" in str(i) for i in q.items)

    # log.exception() legitimately writes a traceback containing str(exc) -
    # that's normal local debugging output, not a leak, so asserting against
    # caplog.text (which includes the traceback) is a false positive by
    # construction. getMessage() excludes the traceback and only reflects
    # what the log call itself formatted, so this still catches a real
    # regression like log.exception("answer failed: %s", exc).
    assert not any(
        "sk-ant-canary-value" in r.getMessage() for r in caplog.records
    )


def test_answer_gate_drops_puts_from_a_superseded_generation():
    gate = _AnswerGate()
    fake = _FakeQueue()
    first = gate.start(fake)
    second = gate.start(fake)  # a second press supersedes the first

    first.put(("token", "stale"))
    second.put(("token", "live"))

    assert fake.items == [("token", "live")]


def test_answer_gate_does_not_discriminate_by_message_kind():
    # The superseded generation's "clear" and "done" must be dropped too,
    # not just its tokens - otherwise it could still blank or end the live
    # answer's display.
    gate = _AnswerGate()
    fake = _FakeQueue()
    first = gate.start(fake)
    gate.start(fake)  # supersede

    first.put(("clear", None))
    first.put(("token", "stale"))
    first.put(("done", None))

    assert fake.items == []


def test_answer_gate_lets_puts_through_before_any_later_press():
    gate = _AnswerGate()
    fake = _FakeQueue()
    only = gate.start(fake)

    only.put(("token", "a"))
    only.put(("token", "b"))

    assert fake.items == [("token", "a"), ("token", "b")]


def test_guarded_hotkey_survives_a_raising_callback(caplog):
    # pynput kills its entire listener thread on an uncaught callback
    # exception - every hotkey dead at once, silently, app still running.
    def boom() -> None:
        raise RuntimeError("callback exploded")

    wrapped = main_mod._guarded("boom", boom)
    with caplog.at_level(logging.ERROR):
        wrapped()  # must not raise
    assert caplog.records, "the failure was swallowed without a log"


#
# Overlay.__init__ creates a real Tk window and calls SetWindowDisplayAffinity
# - not something to do under pytest. __new__ bypasses __init__ entirely, and
# tk.Entry is replaced by a fake that records what the widget code does
# instead of touching a real window. This covers the wiring; actual on-screen
# behaviour (does the entry really sit inside the excluded HWND, does typing
# work) needs a human at a real screen share - see the fix report.

import assistant.overlay as overlay_mod  # noqa: E402


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
    ov._font_size = overlay_mod._FONT[1]
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


def test_degenerate_decode_is_dropped():
    # Observed live: a 0.8s low-energy segment decoded as the same short
    # sentence twenty-nine times. Left in, it pollutes the transcript, the
    # prompt built from it, and the cached prefix.
    loop = " ".join(["There it goes."] * 29)
    assert tm._is_degenerate(loop)


def test_real_speech_is_not_mistaken_for_a_degenerate_decode():
    real = (
        "So tell me, how would you scale the write path on this system? "
        "We're seeing about ten thousand writes a second. "
        "The read side is fine. "
        "What would you change first?"
    )
    assert not tm._is_degenerate(real)
    assert not tm._is_degenerate("Yeah.")
    assert not tm._is_degenerate("")


def test_query_entry_is_actually_displayed_inside_the_root_window(monkeypatch):
    # Regression. The entry was packed after a widget with expand=True and
    # fill="both", which claims the entire cavity - pack() succeeded, the
    # entry stayed unmapped at 1x1, and Ctrl+Shift+Space silently did
    # nothing. Nothing in the suite checked that a widget was displayed, only
    # that it existed, so this shipped through every review.
    import queue as _queue
    import tkinter as _tk

    from assistant import config as _config

    monkeypatch.setattr(_config, "EXCLUDE_FROM_CAPTURE", False)
    from assistant.overlay import Overlay

    try:
        ov = Overlay(_queue.Queue())
    except _tk.TclError as exc:            # no display available
        pytest.skip(f"tk unavailable: {exc}")

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
    import tkinter as _tk

    from assistant import config as _config

    monkeypatch.setattr(_config, "EXCLUDE_FROM_CAPTURE", False)
    from assistant.overlay import Overlay

    try:
        ov = Overlay(_queue.Queue())
    except _tk.TclError as exc:
        pytest.skip(f"tk unavailable: {exc}")
    ov.root.attributes("-alpha", 0.0)
    return ov


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


def test_dragging_the_body_still_moves_and_does_not_resize(monkeypatch):
    # Regression: adding the grip must not turn a body drag into a resize.
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.geometry("460x300+100+100")
        ov.root.update()
        ov.text.event_generate("<Button-1>", rootx=150, rooty=150)
        ov.root.update()
        ov.text.event_generate("<B1-Motion>", rootx=210, rooty=190)
        ov.root.update()

        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (160, 140)
        assert (ov.root.winfo_width(), ov.root.winfo_height()) == (460, 300)
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
        assert _font_size_of(ov._entry) != overlay_mod._FONT[1], (
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


def test_font_scaling_keeps_the_hwnd(monkeypatch):
    ov = _live_overlay(monkeypatch)
    try:
        ov.root.update()
        before = ov.root.winfo_id()
        for _ in range(6):
            ov.text.event_generate("<Control-MouseWheel>", delta=120)
        ov.root.update()
        assert ov.root.winfo_id() == before, "HWND changed - exclusion would drop"
    finally:
        ov.root.destroy()


def test_overlay_can_be_dragged_and_keeps_its_hwnd(monkeypatch):
    # overrideredirect(True) means no titlebar and no window manager to move
    # the window, so dragging is hand-rolled. It must move by the drag delta -
    # not jump so the grab point lands under the cursor - and it must do so
    # without recreating the toplevel: display affinity lives on the HWND, so
    # a new HWND would silently drop capture exclusion mid-drag.
    import queue as _queue
    import tkinter as _tk

    from assistant import config as _config

    monkeypatch.setattr(_config, "EXCLUDE_FROM_CAPTURE", False)
    from assistant.overlay import Overlay

    try:
        ov = Overlay(_queue.Queue())
    except _tk.TclError as exc:
        pytest.skip(f"tk unavailable: {exc}")

    try:
        ov.root.attributes("-alpha", 0.0)
        ov.root.geometry("+100+100")
        ov.root.update()
        hwnd_before = ov.root.winfo_id()

        # Drive the real handlers rather than setting geometry ourselves -
        # otherwise this passes with no drag code at all.
        ov.text.event_generate("<Button-1>", rootx=150, rooty=150)
        ov.root.update()
        assert ov._drag_from == (50, 50), "grab did not record the offset"

        ov.text.event_generate("<B1-Motion>", rootx=210, rooty=190)
        ov.root.update()

        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (160, 140), (
            f"moved to {ov.root.winfo_x()},{ov.root.winfo_y()} - expected 160,140"
        )
        assert ov.root.winfo_id() == hwnd_before, "HWND changed - exclusion would drop"

        ov.text.event_generate("<ButtonRelease-1>", rootx=210, rooty=190)
        ov.root.update()
        assert ov._drag_from is None, "release did not clear the drag state"

        # A motion event with no grab must not move anything.
        ov.text.event_generate("<B1-Motion>", rootx=400, rooty=400)
        ov.root.update()
        assert (ov.root.winfo_x(), ov.root.winfo_y()) == (160, 140)
    finally:
        ov.root.destroy()
