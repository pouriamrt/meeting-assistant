import logging
from datetime import datetime
from typing import NamedTuple

import pytest

from assistant.transcribe import Transcript
from assistant import answer, config, prompts


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












































#
# The loop carries two positions at once: `cursor` into the ring and `pending`
# as a local accumulator. That interaction is the hardest logic in the project
# and needs no hardware to test — a scripted ring and a stubbed VAD reach every
# branch.































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


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Width and height out of the PNG IHDR. No Pillow in this project, and
    trusting the value we passed in would test nothing."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


class _Size(NamedTuple):
    width: int
    height: int


class _RealShot:
    """Like an mss screenshot: rgb is width*height*3 bytes."""

    def __init__(self, width, height):
        self.size = _Size(width, height)
        self.rgb = bytes(range(256)) * ((width * height * 3) // 256 + 1)
        self.rgb = self.rgb[: width * height * 3]


class _RealSct:
    def __init__(self, width=8, height=4):
        self.monitors = [{"union": True}, {"i": 0}]
        self._w, self._h = width, height

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def grab(self, _mon):
        return _RealShot(self._w, self._h)


def test_half_res_halves_both_dimensions(monkeypatch):
    # Measured payoff: input 2859 -> 868 tokens, total 9.24s -> 8.09s. Read
    # back out of the PNG header rather than from the size we handed in.
    monkeypatch.setattr(config, "SCREEN_HALF_RES", True)
    monkeypatch.setattr(screen_mod.mss, "MSS", lambda: _RealSct(8, 4))
    (png,) = screen_mod.grab()
    assert _png_dimensions(png) == (4, 2)


def test_full_res_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(config, "SCREEN_HALF_RES", False)
    monkeypatch.setattr(screen_mod.mss, "MSS", lambda: _RealSct(8, 4))
    (png,) = screen_mod.grab()
    assert _png_dimensions(png) == (8, 4)


def test_halving_keeps_the_pixels_aligned_not_just_the_size(monkeypatch):
    # A wrong reshape or a non-contiguous buffer still produces a PNG of the
    # right dimensions, full of garbage. Check actual pixel values: halving
    # must keep every other pixel of every other row, in order.
    width, height = 8, 4
    shot = _RealShot(width, height)
    data, size = screen_mod._halve(shot.rgb, width, height)
    assert size == (4, 2)

    rows = [shot.rgb[r * width * 3:(r + 1) * width * 3] for r in range(height)]
    expected = b"".join(
        b"".join(rows[r][c * 3:c * 3 + 3] for c in range(0, width, 2))
        for r in range(0, height, 2)
    )
    assert data == expected


def test_grab_returns_one_png_per_monitor(monkeypatch):
    # About which monitors get grabbed, not about resolution - the half-res
    # path has its own tests and needs a fake with real pixel data.
    monkeypatch.setattr(config, "SCREEN_HALF_RES", False)
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
    monkeypatch.setattr(config, "SCREEN_HALF_RES", False)
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


def test_screen_press_logs_a_digest_of_what_it_captured(monkeypatch, caplog):
    # Reported as "it still answers the previous question". Fresh grabs and
    # the real callback path both check out, so the log has to be able to
    # settle it: same digest across two presses means the pixels never
    # changed, different digests move the search to the request.
    import assistant.screen as screen_mod

    # Shaped like the real failure rather than like a convenient fixture:
    # two monitors, the FIRST byte-identical across presses, and the change
    # confined to the END of the SECOND. That kills three ways of getting the
    # digest wrong at once - a constant, a prefix-only hash, and hashing only
    # monitors[0] - each of which would report "same pixels" for two
    # genuinely different screens, which is the one thing this log exists to
    # rule out.
    unchanged = b"\x89PNG" + b"identical" * 400
    grabs = [
        [unchanged, b"\x89PNG" + b"second" * 400 + b"QUESTION-ONE"],
        [unchanged, b"\x89PNG" + b"second" * 400 + b"QUESTION-TWO"],
    ]
    monkeypatch.setattr(screen_mod, "grab", lambda: grabs.pop(0))
    monkeypatch.setattr(main_mod.answer, "stream_answer", lambda *a, **k: None)

    class _Hotkeys:
        captured: dict = {}

        def __init__(self, mapping):
            _Hotkeys.captured = mapping

        def start(self):
            pass

    from pynput import keyboard as kb

    monkeypatch.setattr(kb, "GlobalHotKeys", _Hotkeys)
    main_mod._install_hotkeys(
        object(), Transcript(max_turns=10), _FakeQueue(), main_mod._AnswerGate()
    )

    with caplog.at_level(logging.INFO):
        _Hotkeys.captured["<ctrl>+<alt>+<space>"]()
        _Hotkeys.captured["<ctrl>+<alt>+<space>"]()

    lines = [m for m in (r.getMessage() for r in caplog.records) if "screen:" in m]
    assert len(lines) == 2, f"expected one log per press, got {lines}"
    assert all("2 monitor(s)" in m for m in lines)
    digests = [m.split("sha=")[1] for m in lines]
    assert digests[0] != digests[1], (
        "two different screens logged the same digest - the log cannot "
        "distinguish a stale capture from a fresh one, which is its whole job"
    )
