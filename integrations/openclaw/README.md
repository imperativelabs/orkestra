# OpenClaw × Orkestra Integration

Use Orkestra as a cost-routing skill inside [OpenClaw](https://github.com/openclaw/openclaw). Every LLM call from your agent is automatically routed to the cheapest model capable of handling it — using the API keys you already have.

**No Orkestra account. No Orkestra API key. Just your existing provider credentials.**

---

## Architecture

```
OpenClaw agent
    │  (bash tool)
    ▼
Orkestra proxy  ──  http://127.0.0.1:8765
    │
    ├─ budget prompt   →  gemini-2.5-flash-lite / claude-haiku-4 / gpt-4o-mini
    ├─ balanced prompt →  gemini-3-flash-preview / claude-sonnet-4-5 / gpt-4o
    └─ complex prompt  →  gemini-3-pro-preview / claude-opus-4 / o3
```

---

## Installation

### 1. Copy files

```bash
mkdir -p ~/.openclaw/skills/orkestra
cp proxy.py SKILL.md install.sh README.md ~/.openclaw/skills/orkestra/
```

### 2. Install dependencies

```bash
cd ~/.openclaw/skills/orkestra
bash install.sh
```

### 3. Configure `~/.openclaw/openclaw.json`

See configuration examples below, then save and close the file.

### 4. Start the proxy

```bash
python3 ~/.openclaw/skills/orkestra/proxy.py &
```

### 5. Verify

```bash
curl http://127.0.0.1:8765/health
# → {"status":"ok","provider":"anthropic","multi":false}
```

### 6. Restart the OpenClaw gateway

---

## Configuration Examples

### Single Provider — Anthropic

```json
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

### Single Provider — Google

```json
{
  "skills": {
    "entries": {
      "orkestra": {
        "enabled": true,
        "env": {
          "ORKESTRA_PROVIDER": "google",
          "GEMINI_API_KEY": "AIza..."
        }
      }
    }
  }
}
```

### Single Provider — OpenAI

```json
{
  "skills": {
    "entries": {
      "orkestra": {
        "enabled": true,
        "env": {
          "ORKESTRA_PROVIDER": "openai",
          "OPENAI_API_KEY": "sk-..."
        }
      }
    }
  }
}
```

### Multi-Provider (route across all three)

```json
{
  "skills": {
    "entries": {
      "orkestra": {
        "enabled": true,
        "env": {
          "ORKESTRA_PROVIDERS": "[{\"name\":\"anthropic\",\"key_env\":\"ANTHROPIC_API_KEY\"},{\"name\":\"google\",\"key_env\":\"GEMINI_API_KEY\"},{\"name\":\"openai\",\"key_env\":\"OPENAI_API_KEY\"}]",
          "ORKESTRA_STRATEGY": "cheapest",
          "ANTHROPIC_API_KEY": "sk-ant-...",
          "GEMINI_API_KEY": "AIza...",
          "OPENAI_API_KEY": "sk-..."
        }
      }
    }
  }
}
```

`ORKESTRA_STRATEGY` sets the default strategy. Options: `cheapest` · `balanced` · `smartest`.

---

## Supported Models

| Provider | Budget | Balanced | Premium |
|----------|--------|----------|---------|
| **Google** | `gemini-2.5-flash-lite` | `gemini-3-flash-preview` | `gemini-3-pro-preview` |
| **Anthropic** | `claude-haiku-4` | `claude-sonnet-4-5` | `claude-opus-4` |
| **OpenAI** | `gpt-4o-mini` | `gpt-4o` | `o3` |

Pricing (per 1M tokens):

| Provider | Budget | Balanced | Premium |
|----------|--------|----------|---------|
| Google | $0.10 / $0.40 | $0.50 / $3.00 | $2.00 / $12.00 |
| Anthropic | $0.80 / $4.00 | $3.00 / $15.00 | $15.00 / $75.00 |
| OpenAI | $0.15 / $0.60 | $2.50 / $10.00 | $10.00 / $40.00 |

---

## Auto-Start

### macOS (launchd)

Create `~/Library/LaunchAgents/com.orkestra.proxy.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.orkestra.proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/Users/YOUR_USER/.openclaw/skills/orkestra/proxy.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>ORKESTRA_PROVIDER</key>
    <string>anthropic</string>
    <key>ANTHROPIC_API_KEY</key>
    <string>sk-ant-...</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/orkestra-proxy.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/orkestra-proxy.err</string>
</dict>
</plist>
```

Load it:

```bash
launchctl load ~/Library/LaunchAgents/com.orkestra.proxy.plist
```

### Linux (systemd user service)

Create `~/.config/systemd/user/orkestra-proxy.service`:

```ini
[Unit]
Description=Orkestra OpenClaw Proxy
After=network.target

[Service]
ExecStart=/usr/bin/python3 %h/.openclaw/skills/orkestra/proxy.py
Environment=ORKESTRA_PROVIDER=anthropic
Environment=ANTHROPIC_API_KEY=sk-ant-...
Restart=on-failure

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user enable --now orkestra-proxy
```

---

## Troubleshooting

### Skill not loading in OpenClaw

```bash
openclaw doctor
```

- Confirm `python3` and `uvicorn` are on your `PATH`:
  ```bash
  which python3 uvicorn
  ```
- Ensure `SKILL.md` is present in `~/.openclaw/skills/orkestra/`.

### Proxy not responding

Start it manually and watch for errors:

```bash
python3 ~/.openclaw/skills/orkestra/proxy.py
```

Common causes:
- `orkestra-router` not installed → run `bash install.sh`
- Missing API key env var → check your `openclaw.json` config
- Port already in use → set `ORKESTRA_PORT=8766` and update calls accordingly

### Port conflict

```bash
ORKESTRA_PORT=8766 python3 ~/.openclaw/skills/orkestra/proxy.py &
```

Update any hardcoded port references in `SKILL.md` to match.
