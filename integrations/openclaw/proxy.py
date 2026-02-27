"""
Orkestra proxy for OpenClaw.

Exposes an OpenAI-compatible /v1/chat/completions endpoint backed by
orkestra-router's intelligent cost-based model routing.

Environment variables
---------------------
Single-provider mode:
    ORKESTRA_PROVIDER   google | anthropic | openai
    GEMINI_API_KEY      (when ORKESTRA_PROVIDER=google)
    ANTHROPIC_API_KEY   (when ORKESTRA_PROVIDER=anthropic)
    OPENAI_API_KEY      (when ORKESTRA_PROVIDER=openai)

Multi-provider mode:
    ORKESTRA_PROVIDERS  JSON array of {"name": "...", "key_env": "..."}
                        e.g. '[{"name":"anthropic","key_env":"ANTHROPIC_API_KEY"},
                                {"name":"google","key_env":"GEMINI_API_KEY"}]'
    ORKESTRA_STRATEGY   cheapest | balanced | smartest  (default: cheapest)
    ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY  as needed

Network:
    ORKESTRA_HOST       default: 127.0.0.1
    ORKESTRA_PORT       default: 8765
"""

import json
import os
import sys
import uuid
from typing import List, Optional

import orkestra as o
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = os.environ.get("ORKESTRA_HOST", "127.0.0.1")
PORT = int(os.environ.get("ORKESTRA_PORT", "8765"))
DEFAULT_STRATEGY = os.environ.get("ORKESTRA_STRATEGY", "cheapest")

# Map provider name → conventional env var for the API key
_PROVIDER_KEY_ENV = {
    "google": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _resolve_key(provider_name: str, key_env: Optional[str] = None) -> str:
    """Read the appropriate API key from env, exiting with a helpful message if missing."""
    env_var = key_env or _PROVIDER_KEY_ENV.get(provider_name)
    if not env_var:
        print(
            f"[orkestra-proxy] ERROR: Unknown provider '{provider_name}'. "
            f"Must be one of: google, anthropic, openai.",
            file=sys.stderr,
        )
        sys.exit(1)
    value = os.environ.get(env_var, "").strip()
    if not value:
        print(
            f"[orkestra-proxy] ERROR: Environment variable '{env_var}' is required for "
            f"provider '{provider_name}' but is not set.",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


# ---------------------------------------------------------------------------
# Build provider(s) at startup
# ---------------------------------------------------------------------------

_multi_provider: Optional[o.MultiProvider] = None
_single_provider: Optional[o.Provider] = None
_is_multi = False
_active_provider_name = "unknown"

_providers_json = os.environ.get("ORKESTRA_PROVIDERS", "").strip()

if _providers_json:
    # ---- Multi-provider mode ----
    try:
        _provider_defs = json.loads(_providers_json)
    except json.JSONDecodeError as exc:
        print(
            f"[orkestra-proxy] ERROR: ORKESTRA_PROVIDERS is not valid JSON: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    _built_providers = []
    for entry in _provider_defs:
        name = entry.get("name", "").strip().lower()
        key_env = entry.get("key_env", "").strip() or None
        api_key = _resolve_key(name, key_env)
        _built_providers.append(o.Provider(name, api_key))

    if not _built_providers:
        print(
            "[orkestra-proxy] ERROR: ORKESTRA_PROVIDERS parsed to an empty list.",
            file=sys.stderr,
        )
        sys.exit(1)

    _multi_provider = o.MultiProvider(_built_providers)
    _is_multi = True
    _active_provider_name = "multi"
    print(
        f"[orkestra-proxy] Multi-provider mode with "
        f"{[p['name'] for p in _provider_defs]}. Default strategy: {DEFAULT_STRATEGY}",
        file=sys.stderr,
    )

else:
    # ---- Single-provider mode ----
    _provider_name = os.environ.get("ORKESTRA_PROVIDER", "").strip().lower()
    if not _provider_name:
        print(
            "[orkestra-proxy] ERROR: Set either ORKESTRA_PROVIDERS (multi) or "
            "ORKESTRA_PROVIDER (single) in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    _api_key = _resolve_key(_provider_name)
    _single_provider = o.Provider(_provider_name, _api_key)
    _active_provider_name = _provider_name
    print(
        f"[orkestra-proxy] Single-provider mode: {_provider_name}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Orkestra OpenClaw Proxy", version="1.0.0")


# -- Request / response models -----------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None           # ignored — orkestra picks the model
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 1.0
    strategy: Optional[str] = None        # overrides DEFAULT_STRATEGY for this call


# -- Routes ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "provider": _active_provider_name, "multi": _is_multi}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Orkestra takes a single prompt string; use the last message in the array.
    if not req.messages:
        return JSONResponse(
            status_code=422,
            content={"error": "messages array must not be empty"},
        )
    prompt = req.messages[-1].content

    try:
        if _is_multi:
            strategy = req.strategy or DEFAULT_STRATEGY
            response = _multi_provider.chat(
                prompt,
                strategy=strategy,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        else:
            response = _single_provider.chat(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"Orkestra call failed: {exc}"},
        )

    # Build OpenAI-compatible response shape
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
        },
        "_orkestra": {
            "model": response.model,
            "provider": response.provider,
            "cost": response.cost,
            "savings": response.savings,
            "savings_percent": response.savings_percent,
            "base_model": response.base_model,
            "base_cost": response.base_cost,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
