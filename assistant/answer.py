"""Request assembly and token streaming."""

import base64
import logging
from collections.abc import Sequence
from typing import Protocol

import anthropic

from . import config, prompts

log = logging.getLogger(__name__)


class Emitter(Protocol):
    """Anything this can stream tokens into.

    Narrower than `queue.Queue` on purpose: main.py wraps the real queue in a
    generation-scoped view that drops superseded answers, and that view is not
    a Queue. Naming the one method actually used lets the wrapper be passed
    honestly instead of cast to a type it is not.
    """

    def put(self, item: tuple) -> None: ...

_CACHE = {"type": "ephemeral", "ttl": "1h"}
_FAST_BETA = "fast-mode-2026-02-01"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"  # "default" form; -06-01 400s

# Set once the fast pool answers with a 4xx, then every press skips straight to
# standard speed. Written from answer threads without a lock: the worst a race
# costs is one extra fast attempt.
_fast_refused = False


def build_request(
    transcript_text: str,
    tail: str,
    images: Sequence[bytes] = (),
    system: str | None = None,
) -> dict:
    """Assemble the request. One cache breakpoint, on the system block only.

    The transcript grows on every press, so it never repeats a prior prefix
    byte-for-byte - a breakpoint on it would write a fresh 1h-TTL entry (2x
    base price) on nearly every press and get read back essentially never.
    Only the system block is genuinely identical across presses, so only it
    is marked. Screenshots are worse still: ~1800 tokens that differ every
    press, so they carry no breakpoint either.

    `images` are PNG bytes, one per monitor, and go ahead of the text - the
    documented ordering for multi-modal accuracy is image first, then the
    question about it. `system` selects the prefix: SYSTEM speaks in bullets,
    CODE_SYSTEM writes code. It defaults inside the body rather than in the
    signature so the default stays a live reference to the module constant.

    No temperature, top_p, top_k, or thinking key: the first three 400 on every
    model since 2026-10-21, and Opus 5.5 rejects disabled thinking outright.
    max_tokens covers thinking plus text together, so it is sized well above
    the four bullets the prompt asks for.
    """
    content: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png).decode(),
            },
        }
        for png in images
    ]
    # Only when there is something to send. The API rejects the whole request
    # with "text content blocks must be non-empty", and an empty transcript is
    # ordinary: on_screen and on_query both fire happily before anyone has
    # spoken, and a screenshot on its own is a legitimate question.
    # on_last_question has its own len()==0 guard, so this is not that check
    # moved - it covers the two callers that deliberately do not want one.
    if transcript_text.strip():
        content.append({"type": "text", "text": transcript_text})
    content.append({"type": "text", "text": tail})

    req: dict = {
        # Plain Opus 5.5 is SLOWER here: thinking cannot be turned off, so even
        # at low effort it reasons before the first word (TTFT 2.4s vs Opus 5's
        # 0.5s, measured). Fast mode is what makes 5.5 the faster choice.
        "model": "claude-opus-5-5",
        "max_tokens": 2000,
        "output_config": {"effort": "low"},
        "system": [
            {
                "type": "text",
                "text": prompts.SYSTEM if system is None else system,
                "cache_control": _CACHE,
            }
        ],
        "messages": [{"role": "user", "content": content}],
        # A classifier decline is otherwise an empty card mid-interview; this
        # re-runs it server-side on the model Anthropic routes that category to.
        "fallbacks": "default",
        "betas": [_FALLBACK_BETA],
    }
    if config.FAST_MODE:
        req["speed"] = "fast"
        req["betas"] = [_FAST_BETA, _FALLBACK_BETA]
    return req


def _standard_speed(req: dict) -> dict:
    """The same request without fast mode, refusal fallback kept."""
    slow = {k: v for k, v in req.items() if k != "speed"}
    slow["betas"] = [b for b in req["betas"] if b != _FAST_BETA]
    return slow


def _stream(client: "anthropic.Anthropic", req: dict, ui_queue: Emitter):
    with client.beta.messages.stream(**req) as stream:
        for text in stream.text_stream:
            ui_queue.put(("token", text))
        return stream.get_final_message()


def _answer(client: "anthropic.Anthropic", req: dict, ui_queue: Emitter):
    """Stream fast when asked, at standard speed if the fast attempt fails."""
    global _fast_refused
    if "speed" not in req or _fast_refused:
        return _stream(client, _standard_speed(req), ui_queue)
    try:
        # No SDK retries on the fast pool: a zero quota would sit out its
        # retry-after before the standard-speed answer could start.
        return _stream(client.with_options(max_retries=0), req, ui_queue)
    except anthropic.APIError as exc:
        # Any failure, not just a 429: with retries off, an overload or a
        # dropped connection would otherwise go straight to an error card.
        # Only a 4xx sticks - no entitlement or no quota will not fix itself
        # mid-session. A mid-stream error event arrives on an HTTP 200, so the
        # range is bounded below too.
        if isinstance(exc, anthropic.APIStatusError) and 400 <= exc.status_code < 500:
            _fast_refused = True
        log.warning("fast mode failed (%s), answering at standard speed",
                    type(exc).__name__)
        ui_queue.put(("clear", None))
        return _stream(client, _standard_speed(req), ui_queue)


def stream_answer(
    client: "anthropic.Anthropic",
    transcript_text: str,
    tail: str,
    ui_queue: Emitter,
    images: Sequence[bytes] = (),
    system: str | None = None,
) -> None:
    """Stream one answer onto the UI queue. Runs on its own thread per press."""
    req = build_request(transcript_text, tail, images, system)
    ui_queue.put(("clear", None))
    try:
        final = _answer(client, req, ui_queue)
    except Exception as exc:                     # surface it, never swallow it
        log.exception("answer failed")
        ui_queue.put(("token", f"\n[error: {type(exc).__name__}]"))
        ui_queue.put(("done", None))
        return

    if final.stop_reason == "refusal":
        # Every model in the fallback chain declined. A blank card reads as a
        # dead app, so say so.
        log.warning("answer declined: %s", final.stop_details)
        ui_queue.put(("token", "\n[declined]"))
    ui_queue.put(("done", None))

    # A cache miss is invisible in the output and costs 10x. Only the system
    # block is cached now, so zero cache reads across consecutive presses is
    # a real signal again: it means an invalidator got into that prefix.
    usage = final.usage
    log.info(
        "cache_read=%s cache_write=%s in=%s out=%s stop=%s",
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        usage.input_tokens,
        usage.output_tokens,
        final.stop_reason,
    )
    if final.stop_reason == "max_tokens":
        log.warning("answer truncated at max_tokens - raise it in answer.py")
