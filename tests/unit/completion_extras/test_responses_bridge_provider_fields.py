"""Regression: provider-specific fields on a streamed tool call must not crash.

The bridge built `tool_call_chunk` from `ChatCompletionToolCallChunk`, which is a
TypedDict (a plain dict at runtime), then assigned the provider metadata as an
attribute (`tool_call_chunk.provider_specific_fields = ...`). Attribute assignment
on a dict raises, so every streamed tool call from a provider that supplies
provider-specific fields failed with:

    AttributeError: 'dict' object has no attribute 'provider_specific_fields'

Only reached when the upstream item carries that metadata, which is why it looked
model-specific: gemini-3.8/3.5/3.7-flash supply it, gemini-3.1-flash-lite does not.

These tests drive the streaming bridge with an upstream `output_item.added` that
carries `provider_specific_fields`, and assert the metadata is delivered on the
emitted tool call. They fail on the pre-fix code (AttributeError) and pass on it.
"""

import json

import pytest
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    OpenAiResponsesToChatCompletionStreamIterator,
)


def _drive(frames):
    """Feed frames through the iterator; return (chunks, error)."""
    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=iter(["data: " + json.dumps(f) + "\n\n" for f in frames]),
        sync_stream=True,
    )
    chunks, error = [], None
    try:
        for chunk in iterator:
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001 - the failure mode under test
        error = exc
    return chunks, error


def _tool_call_frames(provider_fields):
    """One function_call whose output item carries provider-specific fields."""
    item = {
        "type": "function_call",
        "id": "fc_1",
        "name": "act_0",
        "call_id": "call_1",
        "arguments": "",
    }
    if provider_fields is not None:
        item["provider_specific_fields"] = provider_fields
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_1",
            "delta": '{"x": "1"}',
        },
    ]


def _emitted_tool_calls(chunks):
    out = []
    for chunk in chunks:
        for choice in chunk.choices:
            for tool_call in getattr(choice.delta, "tool_calls", None) or []:
                out.append(tool_call)
    return out


def test_streamed_tool_call_with_provider_specific_fields_does_not_raise():
    """The AttributeError regression: a dict-backed tool call must accept the field."""
    chunks, error = _drive(_tool_call_frames({"foo": "bar"}))

    assert error is None, f"streaming bridge raised on provider metadata: {error!r}"
    assert _emitted_tool_calls(chunks), "expected an emitted tool call"


def test_provider_specific_fields_are_delivered_on_the_tool_call():
    """The metadata must be carried through, not dropped."""
    chunks, error = _drive(_tool_call_frames({"foo": "bar"}))

    assert error is None
    calls = _emitted_tool_calls(chunks)
    assert calls, "expected an emitted tool call"
    assert calls[0]["provider_specific_fields"] == {"foo": "bar"}


def test_streamed_tool_call_without_provider_fields_still_works():
    """No over-correction: the no-metadata path keeps working."""
    chunks, error = _drive(_tool_call_frames(None))

    assert error is None
    calls = _emitted_tool_calls(chunks)
    assert calls
    assert "provider_specific_fields" not in calls[0]


@pytest.mark.parametrize("fields", [{"a": 1}, {"nested": {"k": [1, 2]}}, {"s": "x"}])
def test_provider_specific_field_shapes_round_trip(fields):
    chunks, error = _drive(_tool_call_frames(fields))

    assert error is None
    calls = _emitted_tool_calls(chunks)
    assert calls and calls[0]["provider_specific_fields"] == fields
