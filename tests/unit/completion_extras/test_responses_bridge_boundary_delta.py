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


# --- the upstream repeats the payload: arguments must arrive exactly once -----


def _arguments_by_index(chunks):
    """Assemble streamed arguments per ``tool_call.index``, as a client does."""
    assembled = {}
    for chunk in chunks:
        for choice in chunk.choices:
            for tool_call in getattr(choice.delta, "tool_calls", None) or []:
                assembled[tool_call.index] = assembled.get(tool_call.index, "") + (
                    getattr(tool_call.function, "arguments", None) or ""
                )
    return assembled


def _added_frame(output_index, name, call_id, arguments):
    return {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "name": name,
            "call_id": call_id,
            "arguments": arguments,
        },
    }


def _argument_delta(output_index, call_id, delta):
    return {
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "item_id": f"fc_{call_id}",
        "delta": delta,
    }


def _two_parallel_calls_repeating_arguments():
    """Two tool calls, with the complete arguments on the added event *and* the deltas.

    This is the shape the upstream gateway actually sent on most sampled
    requests: ``output_item.added`` carries the finished payload, then the same
    payload streams again as deltas.
    """
    payloads = (
        ("call_1", "read_alpha", '{"path": "alpha"}'),
        ("call_2", "read_beta", '{"path": "beta"}'),
    )
    frames = []
    for output_index, (call_id, name, complete) in enumerate(payloads):
        frames.append(_added_frame(output_index, name, call_id, complete))
        head, tail = complete.split(": ", 1)
        frames.append(_argument_delta(output_index, call_id, head + ": "))
        frames.append(_argument_delta(output_index, call_id, tail))
    return frames


def test_two_parallel_calls_with_repeated_arguments_assemble_valid_json():
    """Regression: repeated argument payloads must not be concatenated.

    Forwarding both the ``output_item.added`` payload and the identical deltas
    made a client assembling per the chat-completions contract receive two
    copies, which is invalid JSON. Only a turn with two or more tool calls can
    show this, so a single-tool-call probe reported a false pass.
    """
    chunks, error = _drive(_two_parallel_calls_repeating_arguments())

    assert error is None, f"stream aborted: {error!r}"

    assembled = _arguments_by_index(chunks)
    assert sorted(assembled) == [0, 1], f"expected two tool calls, got {assembled}"
    assert json.loads(assembled[0]) == {"path": "alpha"}
    assert json.loads(assembled[1]) == {"path": "beta"}


def test_first_call_arguments_are_not_duplicated_by_later_calls():
    """The duplication appended a later call's payload onto the first call.

    Guard the assembled payloads against containing each other, which is what a
    naive concatenation of two identical argument strings produces.
    """
    chunks, error = _drive(_two_parallel_calls_repeating_arguments())

    assert error is None, f"stream aborted: {error!r}"

    assembled = _arguments_by_index(chunks)
    for index, payload in assembled.items():
        assert payload.count("{") == 1, f"tool call {index} assembled a repeated payload: {payload!r}"
        assert payload.count("}") == 1, f"tool call {index} assembled a repeated payload: {payload!r}"


def test_arguments_only_from_deltas_still_assemble_fully():
    """No over-correction: the other upstream shape must keep working."""
    frames = [
        _added_frame(0, "read_alpha", "call_1", ""),
        _argument_delta(0, "call_1", '{"path": '),
        _argument_delta(0, "call_1", '"alpha"}'),
    ]

    chunks, error = _drive(frames)

    assert error is None, f"stream aborted: {error!r}"
    assert json.loads(_arguments_by_index(chunks)[0]) == {"path": "alpha"}
