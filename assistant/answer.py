"""Request assembly and token streaming."""

import logging
import queue

from . import prompts

log = logging.getLogger(__name__)

_CACHE = {"type": "ephemeral", "ttl": "1h"}


def build_request(transcript_text: str, tail: str) -> dict:
    """Assemble the request. One cache breakpoint, on the system block only.

    The transcript grows on every press, so it never repeats a prior prefix
    byte-for-byte - a breakpoint on it would write a fresh 1h-TTL entry (2x
    base price) on nearly every press and get read back essentially never.
    Only the system block is genuinely identical across presses, so only it
    is marked.

    No temperature, top_p, top_k, or thinking key: the first three are a 400 on
    Opus 5, and disabling thinking leaks <thinking> tags into visible output.
    max_tokens covers thinking plus text together, so it is sized well above
    the four bullets the prompt asks for.
    """
    return {
        "model": "claude-opus-5",
        "max_tokens": 2000,
        "speed": "fast",
        "betas": ["fast-mode-2026-02-01"],
        "output_config": {"effort": "low"},
        "system": [
            {"type": "text", "text": prompts.SYSTEM, "cache_control": _CACHE}
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": transcript_text},
                    {"type": "text", "text": tail},
                ],
            }
        ],
    }


def stream_answer(
    client,
    transcript_text: str,
    tail: str,
    ui_queue: "queue.Queue",
) -> None:
    """Stream one answer onto the UI queue. Runs on its own thread per press."""
    ui_queue.put(("clear", None))
    try:
        with client.beta.messages.stream(
            **build_request(transcript_text, tail)
        ) as stream:
            for text in stream.text_stream:
                ui_queue.put(("token", text))
            final = stream.get_final_message()
    except Exception as exc:                     # surface it, never swallow it
        log.exception("answer failed")
        ui_queue.put(("token", f"\n[error: {type(exc).__name__}]"))
        ui_queue.put(("done", None))
        return

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
