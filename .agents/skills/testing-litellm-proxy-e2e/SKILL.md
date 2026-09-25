---
name: testing-litellm-proxy-e2e
description: Boot a real litellm proxy from a worktree and drive E2E tests (deployment registration, count/token endpoints) without mocks. Use when verifying proxy-facing changes end-to-end.
---

## Booting the proxy from a worktree

- Use the worktree's own venv: `uv run --no-sync python litellm/proxy/proxy_cli.py --config <yaml> --port 4000 --detailed_debug --num_workers 1`, run from the worktree root. `uv run --no-sync` works when the worktree env is already synced.
- `master_key` cannot be a publicly known default (`sk-1234` etc.): the proxy refuses to boot with `UnsafeMasterKeyError`. Use `master_key: os.environ/LITELLM_MASTER_KEY` in a minimal config and export a generated key (`sk-$(openssl rand -hex 16)`), or set `LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true`.
- Minimal config is enough for DB-backed testing: `model_list: []` + `general_settings.master_key` — deployments can be added at runtime via `POST /model/new` when `STORE_MODEL_IN_DB=True` and `DATABASE_URL` point at a synced Postgres (run `uv run --no-sync prisma db push --accept-data-loss --skip-generate` from `litellm/proxy` first if the DB is empty).
- Readiness check: `GET /health/liveliness` -> 200.

## Registering deployments at runtime

- `POST /model/new` with `Authorization: Bearer <master_key>` and `{"model_name":"<name>","litellm_params":{"model":"gemini/gemini-2.5-flash","api_key":"os.environ/GEMINI_API_KEY"}}`. It persists to `LiteLLM_ProxyModelTable` and hot-loads into the router; no restart needed. Bad-key deployments register fine (no health check on add) — useful for error-path tests.
- Google-route URLs like `/v1beta/models/{model_name}:countTokens` resolve `{model_name}` against registered `model_name`s, so register a deployment literally named e.g. `gemini-2.5-flash` for those tests.

## Count-tokens endpoints (all take `Authorization: Bearer <key>`)

- `POST /v1/messages/count_tokens` (Anthropic shape): `{model, messages, system?, tools?}` -> `{"input_tokens": N}`. Routing: anthropic_endpoints -> internal `token_counter(call_endpoint=True)` -> provider counter -> local-tokenizer fallback.
- `POST /v1beta/models/{model_name}:countTokens` and `/models/{model_name}:countTokens` (Gemini shape): `{contents, tools?, systemInstruction?}` -> `{totalTokens, promptTokensDetails}`.
- `POST /v1/responses/input_tokens` (Responses shape): `{model, input, tools?}` -> `{"object":"response.input_tokens","input_tokens":N}`.
- `POST /utils/token_counter` -> `TokenCountResponse`.

## Distinguishing provider counting from local-tokenizer fallback

- Provider failures return `TokenCountResponse(error=True)` and the proxy logs `Provider token counting failed (<status>): ... Falling back to local tokenizer` at WARNING; absence of that warning means the count came from the provider. `--detailed_debug` does NOT log the outbound httpx URL/body for these calls — use the warning absence + a deliberately-broken deployment (bad api_key -> real provider error JSON in log) to prove the upstream path is live.
- The local fallback (tiktoken) produces different counts than the provider for the same input, so a count change across a bad-key deployment is expected, not a bug.
- Anthropic `web_search_*`/hosted tools mixed with function tools trigger the shared Vertex drop rule (`Dropping search tools and keeping function declarations` warning) — by design, not a failure.

## Devin Secrets Needed

- `GEMINI_API_KEY` (org secret) for gemini/ deployments; `LITELLM_MASTER_KEY` can be generated ad hoc.
