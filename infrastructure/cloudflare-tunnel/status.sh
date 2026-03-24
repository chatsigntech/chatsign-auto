#!/bin/bash
# 检查 Cloudflare Tunnel 服务状态（ChatSign Orchestrator 版）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_CONFIG="$SCRIPT_DIR/config.yml"
LOG_FILE="$PROJECT_DIR/temp/cloudflared.log"

echo "=========================================="
echo "Cloudflare Tunnel 服务状态"
echo "=========================================="

# 检查进程
PID=$(pgrep -f "cloudflared.*tunnel run.*chatsign" || echo "")

if [ -n "$PID" ]; then
    echo "[OK] 服务运行中"
    echo "     PID:  $PID"

    # 从配置文件读取域名
    if [ -f "$PROJECT_CONFIG" ]; then
        HOSTNAME=$(grep 'hostname:' "$PROJECT_CONFIG" | head -1 | awk '{print $2}')
        echo "     域名: https://$HOSTNAME -> localhost:8000"
    fi

    # 显示最近日志
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "最近日志（最后 5 行）:"
        echo "----------------------------------------"
        tail -5 "$LOG_FILE"
    fi
else
    echo "[NOT RUNNING] 服务未运行"
fi

echo ""
echo "=========================================="
echo "管理命令:"
echo "  启动: $SCRIPT_DIR/start.sh"
echo "  停止: $SCRIPT_DIR/stop.sh"
echo "  日志: tail -f $LOG_FILE"
echo "=========================================="
