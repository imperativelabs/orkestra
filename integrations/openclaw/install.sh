#!/usr/bin/env bash
set -e

echo "Installing Orkestra proxy dependencies..."
pip install orkestra-router fastapi uvicorn pydantic

echo ""
echo "Done. Next steps:"
echo ""
echo "  1. Configure your provider in ~/.openclaw/openclaw.json"
echo "     See integrations/openclaw/README.md for full config examples."
echo ""
echo "  2. Start the proxy:"
echo "     python3 ~/.openclaw/skills/orkestra/proxy.py &"
echo ""
echo "  3. Verify it's running:"
echo "     curl http://127.0.0.1:8765/health"
echo ""
echo "  4. Restart the OpenClaw gateway."
echo ""
