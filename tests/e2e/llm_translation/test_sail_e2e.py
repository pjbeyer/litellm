"""Live e2e: Sail through the gateway, where LiteLLM turns ``service_tier`` into Sail's
``metadata.completion_window`` and bills the price columns of the window it sent.

The backend is the cheapest ``sail/`` chat row in the proxy's own cost map whose base,
balanced and flex rates all differ, so a bill at the wrong tier cannot pass. Requires
SAIL_API_KEY on the proxy; no skip gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import openai
import pytest
from e2e_config import SLOW_PROVIDER_TIMEOUT_SECONDS, unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, SpendLogRow
from openai import OpenAI
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict, RootModel
from sdk_clients import NO_PROXY_CACHE, SdkClients, response_header

pytestmark = pytest.mark.e2e

PricedTier = Literal["base", "balanced", "flex"]
PRICED_TIERS: Final[tuple[PricedTier, ...]] = ("base", "balanced", "flex")
PROMPT: Final = "Reply with one word."
MAX_TOKENS: Final = 512


@dataclass(frozen=True, slots=True)
class _Rates:
    input: float
    output: float
    cache_read: float


@dataclass(frozen=True, slots=True)
class _Tokens:
    prompt: int
    cached: int
    completion: int


class _SailPrice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    litellm_provider: str | None = None
    mode: str | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    input_cost_per_token_balanced: float | None = None
    output_cost_per_token_balanced: float | None = None
    cache_read_input_token_cost_balanced: float | None = None
    input_cost_per_token_flex: float | None = None
    output_cost_per_token_flex: float | None = None
    cache_read_input_token_cost_flex: float | None = None

    def _columns(self, tier: PricedTier) -> tuple[float | None, float | None, float | None]:
        match tier:
            case "base":
                return (self.input_cost_per_token, self.output_cost_per_token, self.cache_read_input_token_cost)
            case "balanced":
                return (
                    self.input_cost_per_token_balanced,
                    self.output_cost_per_token_balanced,
                    self.cache_read_input_token_cost_balanced,
                )
            case "flex":
                return (
                    self.input_cost_per_token_flex,
                    self.output_cost_per_token_flex,
                    self.cache_read_input_token_cost_flex,
                )

    def rates(self, tier: PricedTier) -> _Rates | None:
        match self._columns(tier):
            case (float() as input_rate, float() as output_rate, float() as cache_read_rate):
                return _Rates(input=input_rate, output=output_rate, cache_read=cache_read_rate)
            case _:
                return None


class _SailCostMap(RootModel[dict[str, _SailPrice]]):
    pass


@dataclass(frozen=True, slots=True)
class _Backend:
    name: str
    rates: Mapping[PricedTier, _Rates]


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-12, abs(expected) * 1e-2)


def _cost(rates: _Rates, tokens: _Tokens) -> float:
    return (
        (tokens.prompt - tokens.cached) * rates.input
        + tokens.cached * rates.cache_read
        + tokens.completion * rates.output
    )


def _tiered(name: str, price: _SailPrice) -> _Backend | None:
    rates: Final[Mapping[PricedTier, _Rates | None]] = {tier: price.rates(tier) for tier in PRICED_TIERS}
    complete: Final[Mapping[PricedTier, _Rates]] = {tier: rate for tier, rate in rates.items() if rate is not None}
    if len(complete) != len(PRICED_TIERS) or len({rate.input for rate in complete.values()}) != len(PRICED_TIERS):
        return None
    return _Backend(name=name, rates=complete)


@pytest.fixture(scope="module")
def backend(proxy: ProxyClient) -> _Backend:
    cost_map: Final = unwrap(
        proxy.transport.get(
            "/public/litellm_model_cost_map",
            headers=proxy.management_headers(),
            params=NoBody(),
            response_type=_SailCostMap,
        )
    ).root
    sail_chat_rows: Final = {
        name: price for name, price in cost_map.items() if price.litellm_provider == "sail" and price.mode == "chat"
    }
    tiered: Final = tuple(
        backend for backend in (_tiered(name, price) for name, price in sail_chat_rows.items()) if backend is not None
    )
    assert tiered, f"no sail/ chat row in the proxy's cost map prices base, balanced and flex apart: {sail_chat_rows}"
    return min(tiered, key=lambda backend: (backend.rates["base"].input, backend.rates["base"].output, backend.name))


def _register(proxy: ProxyClient, resources: ResourceManager, backend: _Backend) -> tuple[str, str]:
    model: Final = f"e2e-sail-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model, LiteLLMParamsBody(model=backend.name, api_key="os.environ/SAIL_API_KEY")
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _openai(sdk: SdkClients, key: str) -> OpenAI:
    return sdk.openai(key).with_options(timeout=SLOW_PROVIDER_TIMEOUT_SECONDS)


def _assert_billed_at(backend: _Backend, tier: PricedTier, tokens: _Tokens, header_cost: str | None) -> float:
    assert tokens.prompt > 0 and tokens.completion > 0, f"Sail reported no usage, so no cost is real: {tokens}"
    assert header_cost is not None, "x-litellm-response-cost header missing"
    costs: Final = {priced: _cost(rates, tokens) for priced, rates in backend.rates.items()}
    assert not any(_approx_equal(costs[other], costs[tier]) for other in PRICED_TIERS if other != tier), (
        f"{backend.name} prices tiers too close together to tell {tier} apart at {tokens}: {costs}"
    )
    assert _approx_equal(float(header_cost), costs[tier]), (
        f"header cost {header_cost} is not {backend.name}'s {tier} price at {tokens}: expected {costs[tier]}, "
        f"all tiers {costs}"
    )
    return float(header_cost)


def _assert_spend_row_matches(proxy: ProxyClient, key: str, header_cost: float) -> None:
    def priced(rows: list[SpendLogRow]) -> bool:
        return any((row.spend or 0) > 0 for row in rows)

    rows: Final = [row for row in proxy.poll_logs_for_key(key, predicate=priced) if (row.spend or 0) > 0]
    assert rows, f"no priced spend row landed for key {key}"
    assert rows[0].custom_llm_provider == "sail", f"spend row misattributed: {rows[0]}"
    assert rows[0].spend is not None and _approx_equal(rows[0].spend, header_cost), (
        f"logged spend {rows[0].spend} disagrees with the x-litellm-response-cost header {header_cost}"
    )


class TestSailChatCompletions:
    @pytest.mark.covers("llm.chat_completions.sail.service_tier.nonstream.cost_logged")
    @pytest.mark.parametrize(
        ("service_tier", "billed_tier"), [("flex", "flex"), ("balanced", "balanced"), ("auto", "base")]
    )
    def test_service_tier_bills_the_matching_completion_window(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        backend: _Backend,
        service_tier: str,
        billed_tier: PricedTier,
    ) -> None:
        model, key = _register(proxy, resources, backend)

        raw: Final = _openai(sdk, key).chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": f"{PROMPT} {unique_marker()}"}],
            max_completion_tokens=MAX_TOKENS,
            extra_body={**NO_PROXY_CACHE, "service_tier": service_tier},
        )
        usage: Final = raw.parse().usage
        assert usage is not None, "chat response carries no usage"
        details: Final = usage.prompt_tokens_details
        tokens: Final = _Tokens(
            prompt=usage.prompt_tokens,
            cached=(details.cached_tokens or 0) if details else 0,
            completion=usage.completion_tokens,
        )

        header_cost: Final = _assert_billed_at(
            backend, billed_tier, tokens, response_header(raw.headers, "x-litellm-response-cost")
        )
        _assert_spend_row_matches(proxy, key, header_cost)

    @pytest.mark.covers("llm.chat_completions.sail.service_tier.nonstream.rejects_unknown_tier")
    def test_unknown_service_tier_is_rejected(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: _Backend
    ) -> None:
        model, key = _register(proxy, resources, backend)

        with pytest.raises(openai.BadRequestError) as raised:
            _ = _openai(sdk, key).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                max_completion_tokens=MAX_TOKENS,
                extra_body={**NO_PROXY_CACHE, "service_tier": "bogus"},
            )
        assert "service_tier" in raised.value.message, f"400 does not name service_tier: {raised.value.message}"


class TestSailResponses:
    @pytest.mark.covers("llm.responses.sail.service_tier.nonstream.cost_logged")
    def test_flex_completion_window_bills_flex_rates(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: _Backend
    ) -> None:
        model, key = _register(proxy, resources, backend)

        raw: Final = _openai(sdk, key).responses.with_raw_response.create(
            model=model,
            input=f"{PROMPT} {unique_marker()}",
            max_output_tokens=MAX_TOKENS,
            metadata={"completion_window": "flex"},
            extra_body=NO_PROXY_CACHE,
        )
        usage: Final = raw.parse().usage
        assert usage is not None, "responses answer carries no usage"
        tokens: Final = _Tokens(
            prompt=usage.input_tokens,
            cached=usage.input_tokens_details.cached_tokens,
            completion=usage.output_tokens,
        )

        header_cost: Final = _assert_billed_at(
            backend, "flex", tokens, response_header(raw.headers, "x-litellm-response-cost")
        )
        _assert_spend_row_matches(proxy, key, header_cost)


class TestSailMessages:
    @pytest.mark.covers("llm.messages.sail.basic.nonstream.works")
    def test_plain_call_returns_a_message(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: _Backend
    ) -> None:
        model, key = _register(proxy, resources, backend)

        message: Final = sdk.anthropic(key).messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": PROMPT}],
            extra_body=NO_PROXY_CACHE,
        )
        assert message.role == "assistant" and message.content, f"/v1/messages returned no content: {message}"
        assert message.usage.output_tokens > 0, f"/v1/messages reported no output usage: {message.usage}"
