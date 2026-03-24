#!/bin/bash
# Cloudflare Tunnel 安装向导（ChatSign Orchestrator 版）
# 来源: myAgent/infrastructure/cloudflare-tunnel/install.sh
#
# 与 myAgent 版本的关键差异：
# - 支持 Linux（dpkg）+ macOS（brew）
# - 单端口 ingress（FastAPI 8000）
# - 不覆盖 ~/.cloudflared/config.yml（多项目可共存）
# - 使用 uname 检测系统来处理 sed 差异

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OS_TYPE="$(uname)"

DEFAULT_TUNNEL_NAME="chatsign"
DEFAULT_SUBDOMAIN="sign"

# 从 infrastructure.yaml 读取端口（单一 source of truth），回退默认 8000
INFRA_CONFIG="$PROJECT_DIR/config/infrastructure.yaml"
LOCAL_PORT=$(grep 'local_port:' "$INFRA_CONFIG" 2>/dev/null | awk '{print $2}' || echo "")
LOCAL_PORT="${LOCAL_PORT:-8000}"

echo "=========================================="
echo "ChatSign Orchestrator - Cloudflare Tunnel 安装向导"
echo "=========================================="

# --- 检查/安装 cloudflared ---
if command -v cloudflared &> /dev/null; then
    echo "[OK] cloudflared 已安装 ($(cloudflared --version 2>&1 | head -1))"
else
    echo "正在安装 cloudflared..."
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        brew install cloudflare/cloudflare/cloudflared
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        ARCH="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
        curl -fsSL --output /tmp/cloudflared.deb \
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}.deb"
        sudo dpkg -i /tmp/cloudflared.deb
        rm -f /tmp/cloudflared.deb
    else
        echo "[ERROR] 不支持的操作系统: $OS_TYPE"
        exit 1
    fi
    echo "[OK] cloudflared 安装完成"
fi

# --- 登录 ---
if [ -f "$HOME/.cloudflared/cert.pem" ]; then
    echo "[OK] 已登录 Cloudflare 账号"
else
    echo "请登录 Cloudflare 账号（浏览器将自动打开）..."
    cloudflared tunnel login
fi

# --- 创建/检查 Tunnel ---
read -p "Tunnel 名称 [${DEFAULT_TUNNEL_NAME}]: " TUNNEL_NAME
TUNNEL_NAME="${TUNNEL_NAME:-$DEFAULT_TUNNEL_NAME}"

TUNNEL_LIST=$(cloudflared tunnel list 2>/dev/null || echo "")
TUNNEL_ID=$(echo "$TUNNEL_LIST" | grep -w "$TUNNEL_NAME" | head -1 | awk '{print $1}')

if [ -n "$TUNNEL_ID" ]; then
    echo "[OK] Tunnel '$TUNNEL_NAME' 已存在"
else
    echo "创建 Tunnel: $TUNNEL_NAME"
    cloudflared tunnel create "$TUNNEL_NAME"
    TUNNEL_ID=$(cloudflared tunnel list 2>/dev/null | grep -w "$TUNNEL_NAME" | head -1 | awk '{print $1}')
fi
echo "   Tunnel ID: $TUNNEL_ID"

# --- 配置 DNS ---
read -p "请输入你的域名（例如: hhkej.com）: " DOMAIN
read -p "二级域名前缀 [${DEFAULT_SUBDOMAIN}]: " SUBDOMAIN
SUBDOMAIN="${SUBDOMAIN:-$DEFAULT_SUBDOMAIN}"
FULL_DOMAIN="${SUBDOMAIN}.${DOMAIN}"

echo "配置 DNS 记录: $FULL_DOMAIN -> Tunnel"
cloudflared tunnel route dns "$TUNNEL_NAME" "$FULL_DOMAIN" || {
    echo "[WARN] DNS 记录可能已存在"
}

# --- 生成 config.yml（项目目录，不覆盖全局 ~/.cloudflared/） ---
PROJECT_CONFIG="$SCRIPT_DIR/config.yml"
cat > "$PROJECT_CONFIG" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $HOME/.cloudflared/$TUNNEL_ID.json

ingress:
  # 所有请求转发到 FastAPI
  - hostname: $FULL_DOMAIN
    service: http://localhost:$LOCAL_PORT
    originRequest:
      noTLSVerify: true
      connectTimeout: 30s
  - service: http_status:404
EOF
echo "[OK] 配置文件已生成: $PROJECT_CONFIG"
echo "     注意: 不覆盖 ~/.cloudflared/config.yml（使用 --config 指定）"

# --- 更新 infrastructure.yaml ---
# 跨平台 sed in-place
sed_inplace() {
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

INFRA_CONFIG="$PROJECT_DIR/config/infrastructure.yaml"
if [ -f "$INFRA_CONFIG" ]; then
    # 按 key 匹配整行（支持重复运行 install.sh 覆盖旧值）
    sed_inplace \
        -e "s/^  enabled: .*/  enabled: true/" \
        -e "s/^  tunnel_name: .*/  tunnel_name: $TUNNEL_NAME/" \
        -e "s/^  domain: .*/  domain: $FULL_DOMAIN/" \
        "$INFRA_CONFIG"
    echo "[OK] infrastructure.yaml 已更新"
fi

# --- 生成 launchd plist / systemd service（从模板替换占位符） ---
CLOUDFLARED_BIN=$(command -v cloudflared)

if [[ "$OS_TYPE" == "Darwin" ]]; then
    PLIST_TEMPLATE="$SCRIPT_DIR/launchd.plist.template"
    PLIST_OUTPUT="$SCRIPT_DIR/launchd.plist"
    if [ -f "$PLIST_TEMPLATE" ]; then
        sed -e "s|__CLOUDFLARED_BIN__|$CLOUDFLARED_BIN|g" \
            -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
            -e "s|__TUNNEL_NAME__|$TUNNEL_NAME|g" \
            -e "s|__HOME__|$HOME|g" \
            "$PLIST_TEMPLATE" > "$PLIST_OUTPUT"
        echo "[OK] launchd.plist 已生成: $PLIST_OUTPUT"
    fi
else
    SERVICE_TEMPLATE="$SCRIPT_DIR/systemd/chatsign-cloudflared.service.template"
    SERVICE_OUTPUT="$SCRIPT_DIR/systemd/chatsign-cloudflared.service"
    if [ -f "$SERVICE_TEMPLATE" ]; then
        sed -e "s|__CLOUDFLARED_BIN__|$CLOUDFLARED_BIN|g" \
            -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
            -e "s|__USER__|$(whoami)|g" \
            "$SERVICE_TEMPLATE" > "$SERVICE_OUTPUT"
        echo "[OK] systemd service 已生成: $SERVICE_OUTPUT"
        echo "     安装命令: sudo cp $SERVICE_OUTPUT /etc/systemd/system/"
    fi
fi

# --- 确保 temp/ 目录存在 ---
mkdir -p "$PROJECT_DIR/temp"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  域名:      https://$FULL_DOMAIN"
echo "  WebSocket:  wss://$FULL_DOMAIN/ws/"
echo "  API 文档:   https://$FULL_DOMAIN/docs"
echo "  本地端口:   $LOCAL_PORT (FastAPI)"
echo "  配置文件:   $PROJECT_CONFIG"
echo ""
echo "下一步:"
echo "  1. 启动 Tunnel:    $SCRIPT_DIR/start.sh"
echo "  2. 启动编排器:      python -m uvicorn backend.main:app --port $LOCAL_PORT"
echo "  3. 验证:            curl https://$FULL_DOMAIN/docs"
echo ""
echo "=========================================="
