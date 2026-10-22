"""
Regression tests for the Responses -> Chat stream bridge on boundary deltas.

A stream provider may emit a ``*.delta`` frame whose payload is an empty string,
or omit the ``delta`` key entirely, when a tool call or message is split across
deltas. Both are legitimate boundary/placeholder frames, not malformed chunks.

These previously raised ``ValueError("Chat provider: Invalid ... delta ...")``,
which ``base_model_iterator`` re-raises as ``RuntimeError("Error parsing chunk:
...")`` and which aborts the whole stream -- discarding an in-flight tool call
even though the surrounding argument text was valid and complete.
"""

import json

import pytest
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    OpenAiResponsesToChatCompletionStreamIterator,
)


def _sse(*frames):
    return ["data: " + json.dumps(f) + "\n\n" for f in frames]


def _drive(frames):
    """Feeds frames through the iterator; returns (chunks, error)."""
    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=iter(_sse(*frames)), sync_stream=True
    )
    chunks, error = [], None
    try:
        for chunk in iterator:
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001 - the failure mode under test
        error = exc
    return chunks, error


def _arguments(chunks):
    args = ""
    for chunk in chunks:
        for choice in chunk.choices:
            for tool_call in getattr(choice.delta, "tool_calls", None) or []:
                if getattr(tool_call.function, "arguments", None):
                    args += tool_call.function.arguments
    return args


def _content(chunks):
    text = ""
    for chunk in chunks:
        for choice in chunk.choices:
            if getattr(choice.delta, "content", None):
                text += choice.delta.content
    return text


# --- the reported production stall ------------------------------------------


def test_empty_function_call_argument_delta_is_a_no_op():
    """The production stall: an empty argument delta must not abort the stream."""
    chunks, error = _drive(
        [
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "",
            }
        ]
    )

    assert error is None, f"empty argument delta aborted the stream: {error!r}"
    assert len(chunks) == 1


def test_missing_function_call_argument_delta_key_is_a_no_op():
    """Same branch, ``delta`` key absent entirely rather than empty."""
    chunks, error = _drive(
        [
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
            }
        ]
    )

    assert error is None, f"missing argument delta key aborted the stream: {error!r}"
    assert len(chunks) == 1


def test_missing_output_text_delta_key_is_a_no_op():
    """The text branch raised on an absent key (it already tolerated an empty one)."""
    chunks, error = _drive(
        [{"type": "response.output_text.delta", "output_index": 0}]
    )

    assert error is None, f"missing text delta key aborted the stream: {error!r}"
    assert len(chunks) == 1


# --- the surrounding payload must survive intact -----------------------------


def test_tool_call_survives_an_empty_boundary_delta():
    """A tool call split across deltas with an empty frame in the middle must
    still deliver its full argument payload."""
    chunks, error = _drive(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "read_file",
                    "call_id": "call_1",
                    "arguments": "",
                },
                "sequence_number": 2,
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '{"path": "a.py"',
                "sequence_number": 1182,
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "",
                "sequence_number": 1183,
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "}",
                "sequence_number": 1184,
            },
        ]
    )

    assert error is None, f"stream aborted across the empty delta: {error!r}"
    assert _arguments(chunks) == '{"path": "a.py"}'
    assert not any(getattr(c, "id", None) is None for c in chunks)


def test_text_content_streams_across_an_empty_delta():
    """No over-correction: real text content still arrives, empty frames ignored."""
    chunks, error = _drive(
        [
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"},
            {"type": "response.output_text.delta", "output_index": 0, "delta": ""},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "lo"},
        ]
    )

    assert error is None
    assert _content(chunks) == "Hello"


# --- guards against re-introducing the raise --------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": None,
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": "",
        },
    ],
)
def test_function_call_argument_boundary_frames_never_raise(frame):
    """Neither an empty nor a ``None`` argument delta may abort the stream."""
    _, error = _drive([frame])
    assert error is None, f"boundary frame aborted the stream: {error!r}"
