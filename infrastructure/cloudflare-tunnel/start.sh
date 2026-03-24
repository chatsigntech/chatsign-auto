#!/bin/bash
# 启动 Cloudflare Tunnel 服务（ChatSign Orchestrator 版）
# 关键差异: 使用 --config 指定项目配置，不覆盖全局 ~/.cloudflared/config.yml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_CONFIG="$SCRIPT_DIR/config.yml"
LOG_FILE="$PROJECT_DIR/temp/cloudflared.log"
PID_FILE="$PROJECT_DIR/temp/cloudflared.pid"
OS_TYPE="$(uname)"
LAUNCHD_LABEL="com.chatsign.cloudflared"

echo "=========================================="
echo "启动 Cloudflare Tunnel 服务"
echo "=========================================="

# 检查 cloudflared
if ! command -v cloudflared &> /dev/null; then
    echo "[ERROR] cloudflared 未安装"
    echo "macOS: brew install cloudflare/cloudflare/cloudflared"
    echo "Linux: 参见 install.sh"
    exit 1
fi

# 检查配置文件
if [ ! -f "$PROJECT_CONFIG" ]; then
    echo "[ERROR] 配置文件不存在: $PROJECT_CONFIG"
    echo "请先运行: $SCRIPT_DIR/install.sh"
    exit 1
fi

# 检查是否已在运行
EXISTING_PID=$(pgrep -f "cloudflared.*tunnel run.*chatsign" || echo "")
if [ -n "$EXISTING_PID" ]; then
    echo "[OK] Cloudflare Tunnel 已在运行 (PID: $EXISTING_PID)"
    exit 0
fi

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS: 使用 launchd
    PLIST_FILE="$SCRIPT_DIR/launchd.plist"
    LAUNCHD_DIR="$HOME/Library/LaunchAgents"
    LAUNCHD_DEST="$LAUNCHD_DIR/${LAUNCHD_LABEL}.plist"

    if [ ! -f "$PLIST_FILE" ]; then
        echo "[ERROR] launchd.plist 不存在，请先运行: $SCRIPT_DIR/install.sh"
        exit 1
    fi

    mkdir -p "$LAUNCHD_DIR"

    # 仅在内容变化时更新，避免无谓的 unload/load
    if ! cmp -s "$PLIST_FILE" "$LAUNCHD_DEST" 2>/dev/null; then
        # 先卸载旧版本（如果存在）
        launchctl unload "$LAUNCHD_DEST" 2>/dev/null || true
        cp "$PLIST_FILE" "$LAUNCHD_DEST"
    fi

    launchctl load "$LAUNCHD_DEST" 2>/dev/null || {
        echo "[WARN] 服务已加载，正在重启..."
        launchctl unload "$LAUNCHD_DEST" 2>/dev/null
        launchctl load "$LAUNCHD_DEST"
    }
else
    # Linux: 直接启动（生产环境建议用 systemd）
    CLOUDFLARED_BIN=$(command -v cloudflared)
    nohup "$CLOUDFLARED_BIN" --config "$PROJECT_CONFIG" tunnel run \
        >> "$LOG_FILE" 2>&1 &
fi

# 等待启动
sleep 2

# 验证
PID=$(pgrep -f "cloudflared.*tunnel run.*chatsign" || echo "")
if [ -n "$PID" ]; then
    echo "$PID" > "$PID_FILE"
    echo "[OK] Cloudflare Tunnel 已启动"
    echo "     PID:  $PID"
    echo "     日志: $LOG_FILE"
    echo "     配置: $PROJECT_CONFIG（未覆盖 ~/.cloudflared/）"
else
    echo "[ERROR] 启动失败，请查看日志: $LOG_FILE"
    exit 1
fi

echo "=========================================="
