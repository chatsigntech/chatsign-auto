#!/bin/bash
# 停止 Cloudflare Tunnel 服务（ChatSign Orchestrator 版）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_FILE="$PROJECT_DIR/temp/cloudflared.pid"
OS_TYPE="$(uname)"
LAUNCHD_LABEL="com.chatsign.cloudflared"
LAUNCHD_FILE="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"

echo "=========================================="
echo "停止 Cloudflare Tunnel 服务"
echo "=========================================="

# macOS: 卸载 LaunchAgent
if [[ "$OS_TYPE" == "Darwin" ]] && [ -f "$LAUNCHD_FILE" ]; then
    launchctl unload "$LAUNCHD_FILE" 2>/dev/null && echo "[OK] LaunchAgent 已卸载" || echo "[WARN] LaunchAgent 未运行"
    rm -f "$LAUNCHD_FILE"
fi

# 通过 PID 文件停止
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "[OK] 进程已终止 (PID: $PID)"
    fi
    rm -f "$PID_FILE"
fi

# 清理残留进程
pkill -f "cloudflared.*tunnel run.*chatsign" 2>/dev/null && echo "[OK] 清理残留进程" || true

echo "[OK] Cloudflare Tunnel 已停止"
echo "=========================================="
