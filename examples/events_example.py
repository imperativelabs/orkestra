"""events_example.py — walkthrough of orkestra's event system.

No API keys required. The LLM backend is mocked so you can run this
offline and see every lifecycle event fire in real time.

Run:
    python examples/events_example.py
"""

import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Set up a mocked backend + router (no real API calls)
# ---------------------------------------------------------------------------

_router = MagicMock()
_router.route.return_value = "gemini-3-flash-preview"

_mock_knn_module = MagicMock()
_mock_knn_module.KNNRouter = MagicMock(return_value=_router)

_backend = MagicMock()
_backend.name = "google"
_backend.call.return_value = {
    "text": "The mitochondria is the powerhouse of the cell.",
    "input_tokens": 12,
    "output_tokens": 9,
}
_backend.stream.return_value = iter(["The ", "sky ", "is ", "blue."])

_patches = [
    patch("orkestra.provider.create_backend", return_value=_backend),
    patch.dict(sys.modules, {"orkestra.router.knn": _mock_knn_module}),
]
for p in _patches:
    p.start()

# ---------------------------------------------------------------------------
# Now import orkestra — patching is in effect
# ---------------------------------------------------------------------------

import orkestra as o
from orkestra import register_event, EventData

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

DIVIDER = "─" * 60

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ===========================================================================
# 1. GLOBAL EVENTS — fire for every provider
# ===========================================================================

section("1. GLOBAL EVENTS — fire for every provider")

@register_event("on_request")
def on_request(data: EventData):
    print(f"  [on_request]  provider={data.provider!r}  prompt={data.prompt!r}")

@register_event("on_chat")
def on_chat(data: EventData):
    print(f"  [on_chat]     prompt={data.prompt!r}")

@register_event("on_route")
def on_route(data: EventData):
    print(f"  [on_route]    model={data.model!r}")

@register_event("on_response")
def on_response(data: EventData):
    resp = data.response
    print(
        f"  [on_response] model={resp.model!r}  "
        f"cost=${resp.cost:.8f}  "
        f"tokens={resp.input_tokens}in/{resp.output_tokens}out"
    )

provider = o.Provider("google", "FAKE_KEY")
print("\n  >>> provider.chat('What is the powerhouse of the cell?')\n")
response = provider.chat("What is the powerhouse of the cell?")
print(f"\n  Response text: {response.text!r}")


# ===========================================================================
# 2. STREAMING EVENTS — on_stream, on_chunk, on_stream_complete
# ===========================================================================

section("2. STREAMING EVENTS")

# Reset mock stream for each call
_backend.stream.return_value = iter(["The ", "sky ", "is ", "blue."])

@register_event("on_stream")
def on_stream(data: EventData):
    print(f"  [on_stream]         provider={data.provider!r}  prompt={data.prompt!r}")

@register_event("on_chunk")
def on_chunk(data: EventData):
    chunk = data.metadata["chunk"]
    print(f"  [on_chunk]          chunk={chunk!r}")

@register_event("on_stream_complete")
def on_stream_complete(data: EventData):
    print(f"  [on_stream_complete] model={data.model!r}")

print("\n  >>> list(provider.stream_text('What color is the sky?'))\n")
chunks = list(provider.stream_text("What color is the sky?"))
print(f"\n  Assembled: {''.join(chunks)!r}")


# ===========================================================================
# 3. PROVIDER-LEVEL EVENTS — scoped to one provider instance
# ===========================================================================

section("3. PROVIDER-LEVEL EVENTS — scoped to one instance")

# New isolated provider so we can show provider-scoped events clearly
_backend_b = MagicMock()
_backend_b.name = "anthropic"
_backend_b.call.return_value = {
    "text": "42.",
    "input_tokens": 5,
    "output_tokens": 2,
}

with patch("orkestra.provider.create_backend", return_value=_backend_b):
    anthropic = o.Provider("anthropic", "FAKE_KEY", smart_routing=False, default_model="claude-sonnet-4-5")

scoped_log = []

@anthropic.event("on_response")
def anthropic_only(data: EventData):
    scoped_log.append(data.response.text)
    print(f"  [provider event: on_response] text={data.response.text!r}  (anthropic only)")

print("\n  >>> anthropic.chat('What is 6 x 7?')  — provider event fires\n")
anthropic.chat("What is 6 x 7?")

print("\n  >>> provider.chat('...')  — provider event does NOT fire for google\n")
provider.chat("Does the anthropic handler fire here?")

print(f"\n  Anthropic handler fired {len(scoped_log)} time(s): {scoped_log}")
print(f"  (Global on_response fired for both calls — see output above)")


# ===========================================================================
# 4. USING EventData FIELDS
# ===========================================================================

section("4. EventData FIELDS REFERENCE")

print("""
  Every handler receives an EventData object:

    data.event      — event name ("on_response", "on_chunk", etc.)
    data.provider   — "google" / "anthropic" / "openai"
    data.prompt     — the original prompt string
    data.model      — resolved model name (None before routing)
    data.response   — orkestra.Response (None for pre-call events)
    data.metadata   — dict; "chunk" key available in on_chunk events

  Example handler pattern:

    @register_event("on_response")
    def log(data: EventData):
        print(f"[{data.provider}] {data.model} — ${data.response.cost:.6f}")
        print(f"Saved {data.response.savings_percent:.1f}% vs premium model")
""")


# ===========================================================================
# Teardown patches
# ===========================================================================

for p in _patches:
    p.stop()

print(DIVIDER)
print("  Done.")
print(DIVIDER)
