---
name: orkestra
description: Route LLM tasks to the cheapest model that can handle them. Use for any standalone LLM task (summarization, Q&A, classification, code generation, translation, drafting) where cost matters. Simple prompts go to budget models, complex ones to premium — automatically. Typical savings 70-80%.
user-invocable: true
metadata: {"openclaw": {"emoji": "🎼", "requires": {"bins": ["python3", "uvicorn"]}, "homepage": "https://github.com/imperativelabs/orkestra"}}
---

# Orkestra Skill

Orkestra is a local routing proxy that sits between you and the LLM providers you already have keys for (Google, Anthropic, OpenAI). It classifies every prompt by complexity using a KNN router and automatically sends it to the cheapest model that can handle it.

**Typical savings: 70–80%** on mixed workloads.

---

## When to Use This Skill

**✅ Use Orkestra for:**
- Summarization, Q&A, classification, drafting, translation, code generation
- Any standalone LLM task where cost matters
- Tasks where you want to delegate model selection automatically

**❌ Do NOT use Orkestra for:**
- Multi-turn conversations — the proxy only sees the last message; earlier context is lost
- Tasks that need OpenClaw tools (exec, browser, canvas) — call those tools directly
- Streaming output back to the user in real time — the proxy returns full responses only

---

## Checking the Proxy is Running

```bash
curl -s http://127.0.0.1:8765/health
```

Expected response:
```json
{"status": "ok", "provider": "anthropic", "multi": false}
```

If the proxy is not running, start it:

```bash
python3 {baseDir}/proxy.py &
sleep 2
# Then verify:
curl -s http://127.0.0.1:8765/health
```

---

## Calling Orkestra

Send a POST to `/v1/chat/completions` using the standard messages format:

```bash
RESULT=$(curl -s -X POST http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Summarize the following text in 3 bullet points: ..."}],
    "max_tokens": 2048
  }')
```

Extract the response text:

```bash
echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
```

Optionally log cost transparency:

```bash
echo "$RESULT" | python3 -c "
import sys, json
r = json.load(sys.stdin)
o = r['_orkestra']
print(f\"Model: {o['model']} | Cost: \${o['cost']:.6f} | Saved: {o['savings_percent']:.1f}%\")
"
```

---

## Strategy Override

By default the strategy is whatever `ORKESTRA_STRATEGY` is set to in your config (usually `cheapest`). You can override per-request:

```bash
# Force premium model for a hard problem
curl -s -X POST http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Prove the Riemann hypothesis"}], "strategy": "smartest"}'

# Force budget model
curl -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "strategy": "cheapest"}' \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8765/v1/chat/completions

# Let Orkestra balance cost and capability
curl -d '{"messages": [{"role": "user", "content": "Write a Python quicksort"}], "strategy": "balanced"}' \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8765/v1/chat/completions
```

| Strategy | Behaviour |
|----------|-----------|
| `cheapest` | Cheapest model that fits the prompt complexity |
| `balanced` | Mid-tier preference, ties broken by cost |
| `smartest` | Most capable available model |

---

## Provider Configuration

Orkestra has **no API key of its own**. It routes requests through the provider credentials you already have. The proxy reads them at startup from env vars you set in your OpenClaw skills config:

```json
// ~/.openclaw/openclaw.json — single provider (Anthropic example)
{
  "skills": {
    "entries": {
      "orkestra": {
        "enabled": true,
        "env": {
          "ORKESTRA_PROVIDER": "anthropic",
          "ANTHROPIC_API_KEY": "sk-ant-..."
        }
      }
    }
  }
}
```

See `integrations/openclaw/README.md` for Google, OpenAI, and multi-provider config examples.
