"""Follow-up coverage for #144896: the SSE media type and the session-level latch.

The word-match fix keeps "upstream ... not supported" and "stream-json" from reading as a
stream refusal. Two things it does not pin: a gateway that refuses ``text/event-stream`` names
streaming without ever saying the word "stream" (so a word-only rule turns a genuine refusal
into a terminal error), and the latch is read on every later request, so the flag has to be
right *after* the failed turn — not only inside the predicate.
"""

from types import SimpleNamespace

import pytest

from agent.anthropic_adapter import _is_stream_unavailable_error, create_anthropic_message
from agent.turn_api_call import _should_stream

_UPSTREAM_THINKING = (
    "Incomplete upstream response (first attempt: status 400, upstream said: "
    "adaptive thinking is not supported on this model)"
)
_UPSTREAM_INTEGER = (
    "Incomplete upstream response (first attempt: status 400, upstream said: "
    "For 'integer' type, properties maximum, minimum are not supported)"
)
_STREAM_JSON = (
    "Native history replay not supported: --resume rejected\n"
    "native stderr: Error: --input-format stream-json is not supported with --resume"
)


class _BrokenStream:
    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        raise self._exc

    def __exit__(self, *a):
        return False


class _Ctx:
    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        return self._inner

    def __exit__(self, *a):
        return False


def _valid_sdk_stream():
    """A well-formed Anthropic MessageStream carrying one ``hi`` text block."""
    from anthropic import NOT_GIVEN
    from anthropic._models import construct_type_unchecked
    from anthropic.lib.streaming import MessageStream
    from anthropic.types import RawMessageStreamEvent

    raw = [construct_type_unchecked(type_=RawMessageStreamEvent, value=v) for v in (
        {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
                                              "model": "m", "content": [],
                                              "usage": {"input_tokens": 5, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )]
    return MessageStream(iter(raw), output_format=NOT_GIVEN)


def _streaming_agent():
    """anthropic_messages agent whose ``messages.stream`` is driven by a swappable factory."""
    from run_agent import AIAgent

    agent = AIAgent(api_key="k", base_url="https://api.minimax.io/anthropic", model="MiniMax-M2",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode, agent.provider, agent._interrupt_requested = "anthropic_messages", "custom", False
    factory = {"fn": lambda: _BrokenStream(RuntimeError("unset"))}
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: factory["fn"]()))
    agent._anthropic_client = client
    agent._create_request_anthropic_client = lambda *args, **kwargs: client
    printed = []
    agent._print_fn = lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    return agent, factory, printed


def _streaming_turn(agent):
    """Run one real streaming turn; return the exception it raised (or None)."""
    try:
        agent._interruptible_streaming_api_call(
            {"model": "MiniMax-M2", "messages": [], "max_tokens": 8})
    except BaseException as exc:  # the point is what escapes the turn
        return exc
    return None


@pytest.mark.parametrize("text", [
    _UPSTREAM_THINKING,
    _UPSTREAM_INTEGER,
    _STREAM_JSON,
    "non-streaming requests are not supported on this endpoint",
    "502 from downstream service: parameter 'top_k' is not supported",
    "prompt is too long: 218000 tokens > 200000 maximum",
])
def test_an_unrelated_error_is_not_a_stream_refusal(text):
    assert _is_stream_unavailable_error(RuntimeError(text)) is False


@pytest.mark.parametrize("text", [
    "streaming is not supported for this model",
    "stream not supported for this model",
    "streams are not supported by this endpoint",
    "Stream mode is not supported",
    "streaming not supported",
    "the 'stream' parameter is not supported when tools are provided",
    "text/event-stream is not supported by this gateway",
    "event-stream not supported, retry without stream",
])
def test_a_genuine_stream_refusal_is_still_detected(text):
    assert _is_stream_unavailable_error(RuntimeError(text)) is True


@pytest.mark.parametrize("text", [
    "text/event-stream is not supported by this gateway",
    "the gateway answers text/event-stream with 415: not supported",
])
def test_sse_media_type_refusal_is_a_stream_refusal(text):
    """The media type names streaming without the word, so a word-only rule loses it."""
    assert _is_stream_unavailable_error(RuntimeError(text)) is True


@pytest.mark.parametrize("text", [_UPSTREAM_THINKING, _UPSTREAM_INTEGER, _STREAM_JSON,
                                  "non-streaming requests are not supported on this endpoint"])
def test_false_positive_keeps_streaming_on_for_the_session(text):
    """The whole turn path: the error surfaces, but streaming is not switched off."""
    agent, factory, printed = _streaming_agent()
    factory["fn"] = lambda: _BrokenStream(RuntimeError(text))

    assert _streaming_turn(agent) is not None
    assert bool(agent._disable_streaming) is False
    assert not [line for line in printed if "Streaming is not supported" in line]
    assert _should_stream(agent) is True


@pytest.mark.parametrize("text", [
    "streaming is not supported for this model",
    "text/event-stream is not supported by this gateway",
])
def test_genuine_refusal_disables_streaming_and_says_so(text):
    agent, factory, printed = _streaming_agent()
    factory["fn"] = lambda: _BrokenStream(RuntimeError(text))

    assert _streaming_turn(agent) is not None
    assert bool(agent._disable_streaming) is True
    assert [line for line in printed if "Streaming is not supported for this model/provider" in line]
    assert _should_stream(agent) is False


def test_next_turn_still_streams_after_a_false_positive():
    agent, factory, _printed = _streaming_agent()
    factory["fn"] = lambda: _BrokenStream(RuntimeError(_UPSTREAM_THINKING))
    assert _streaming_turn(agent) is not None

    factory["fn"] = lambda: _Ctx(_valid_sdk_stream())
    final = agent._interruptible_streaming_api_call(
        {"model": "MiniMax-M2", "messages": [], "max_tokens": 8})
    assert final.content[0].text == "hi" and final.stop_reason == "end_turn"
    assert bool(agent._disable_streaming) is False


@pytest.mark.parametrize("text,prefers_stream", [
    (_UPSTREAM_THINKING, True),
    (_STREAM_JSON, True),
    ("streaming is not supported for this model", False),
    ("text/event-stream is not supported by this gateway", False),
])
def test_next_request_prefers_the_channel_the_predicate_chose(text, prefers_stream, monkeypatch):
    """The latch is read by the Anthropic create path on every later request."""
    agent, factory, _printed = _streaming_agent()
    factory["fn"] = lambda: _BrokenStream(RuntimeError(text))
    assert _streaming_turn(agent) is not None

    captured = {}
    monkeypatch.setattr(
        "agent.anthropic_adapter.create_anthropic_message",
        lambda client, kwargs, **kw: captured.setdefault("prefer_stream", kw.get("prefer_stream")),
    )
    agent._anthropic_messages_create({"model": "MiniMax-M2", "messages": []}, client=SimpleNamespace())
    assert captured["prefer_stream"] is prefers_stream


def test_aux_create_still_falls_back_for_a_genuine_stream_refusal():
    created = []
    messages = SimpleNamespace(
        stream=lambda **kw: _BrokenStream(RuntimeError("stream not supported for this model")),
        create=lambda **kw: created.append(kw) or SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    create_anthropic_message(SimpleNamespace(messages=messages),
                             {"model": "m", "messages": [], "max_tokens": 8})
    assert len(created) == 1 and "stream" not in created[0]
