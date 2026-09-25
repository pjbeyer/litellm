import asyncio
import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Final

import httpx
import pytest
import pytest_asyncio
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

SAIL_API_BASE: Final = "https://api.sailresearch.com/v1"
MODEL: Final = "sail/zai-org/GLM-5.3"
PROMPT_TOKENS: Final = 1000
CACHED_TOKENS: Final = 200
COMPLETION_TOKENS: Final = 500


def cost_at(column_suffix: str) -> float:
    prices: Final[Mapping[str, object]] = litellm.model_cost[MODEL]
    return (
        (PROMPT_TOKENS - CACHED_TOKENS) * float(prices[f"input_cost_per_token{column_suffix}"])
        + CACHED_TOKENS * float(prices[f"cache_read_input_token_cost{column_suffix}"])
        + COMPLETION_TOKENS * float(prices[f"output_cost_per_token{column_suffix}"])
    )


def sent_body(route: respx.Route) -> dict[str, object]:
    return json.loads(route.calls.last.request.content)


def chat_completion_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-sail",
        "object": "chat.completion",
        "created": 0,
        "model": "zai-org/GLM-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            "prompt_tokens_details": {"cached_tokens": CACHED_TOKENS},
        },
    }


def chat_completion_stream() -> bytes:
    chunk: Final = {"id": "chatcmpl-sail", "object": "chat.completion.chunk", "created": 0, "model": "zai-org/GLM-5.3"}
    events: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": chat_completion_body()["usage"]},
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"


def responses_body() -> dict[str, object]:
    return {
        "id": "resp_sail",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "zai-org/GLM-5.3",
        "output": [
            {
                "type": "message",
                "id": "msg_sail",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": PROMPT_TOKENS,
            "input_tokens_details": {"cached_tokens": CACHED_TOKENS},
            "output_tokens": COMPLETION_TOKENS,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
    }


def messages_body() -> dict[str, object]:
    return {
        "id": "msg_sail",
        "type": "message",
        "role": "assistant",
        "model": "zai-org/GLM-5.3",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": PROMPT_TOKENS - CACHED_TOKENS,
            "cache_read_input_tokens": CACHED_TOKENS,
            "output_tokens": COMPLETION_TOKENS,
        },
    }


class SpendCapture(CustomLogger):
    """Records the cost the spend logs would store for one call, matched by its call id."""

    def __init__(self, call_id: str) -> None:
        super().__init__()
        self.call_id = call_id
        self.costs: tuple[object, ...] = ()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        if kwargs.get("litellm_call_id") == self.call_id:
            payload: Final = kwargs.get("standard_logging_object")
            self.costs = (*self.costs, payload.get("response_cost") if isinstance(payload, dict) else None)

    async def settled_cost(self) -> object:
        await asyncio.sleep(0)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
        assert len(self.costs) == 1, self.costs
        return self.costs[0]


@pytest.fixture
def sail_env(local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SAIL_API_KEY", "sail-test-key")
    monkeypatch.delenv("SAIL_API_BASE", raising=False)
    monkeypatch.setattr(
        litellm,
        "disable_aiohttp_transport",
        True,
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest_asyncio.fixture
async def spend_capture(monkeypatch: pytest.MonkeyPatch) -> SpendCapture:
    GLOBAL_LOGGING_WORKER.start()
    capture: Final = SpendCapture(call_id=f"sail-{uuid.uuid4()}")
    monkeypatch.setattr(litellm, "callbacks", [capture])
    return capture


@pytest.fixture
def chat_route(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{SAIL_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_completion_body())
    )
