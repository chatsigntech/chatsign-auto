# Cloudflare Tunnel 集成方案

**来源**: myAgent 项目 (`/Users/chatsign/Projects/myAgent`)
**目标**: chatsign-auto 编排器
**目标服务器**: `chatsign@10.228.128.11`（GPU 训练服务器，内网无公网 IP）
**日期**: 2026-03-24

---

## 0. 前提条件

> **重要**: 当前 chatsign-auto 仓库是文档+配置+子模块的集合，`backend/` 目录及其 Python 代码尚未创建。
> Tunnel 集成分两部分：
>
> - **基础设施部分**（脚本、配置）：**可立即实施**，不依赖 backend 代码
> - **FastAPI 集成部分**（lifespan、状态 API）：**需等 backend 代码就绪后实施**
>
> 下文会在每个章节标注 `[立即]` 或 `[等 backend 就绪]`。

---

## 1. 集成目标

将 myAgent 的 Cloudflare Tunnel 能力集成到 chatsign-auto，使编排器在 GPU 服务器（`10.228.128.11`，内网）上也能被远程访问。

**集成后的能力**：
- 远程访问 WebUI 查看 Phase 1-6 进度
- 远程调用 REST API 创建/管理任务
- WebSocket 实时接收 GPU 训练状态推送
- 自动 HTTPS/WSS，无需额外 SSL 配置

---

## 2. 架构对比与设计决策

### 2.1 myAgent 架构（当前）

```
客户端
  ↓ wss://bot.hhkej.com
  ↓
Cloudflare CDN
  ↓ 隧道
  ↓
cloudflared
  ├─→ /download/*  → localhost:8080 (PublicServerManager, aiohttp)
  ├─→ /web/*       → localhost:8080 (PublicServerManager, aiohttp)
  ├─→ /ws          → localhost:8767 (AppChannel WebSocket)
  └─→ 其他         → localhost:8766 (Gateway WebSocket)
```

**特点**: 多端口、多服务、路径分流

### 2.2 chatsign-auto 架构（集成后）

```
客户端 (浏览器/curl)
  ↓ https://sign.hhkej.com
  ↓
Cloudflare CDN
  ↓ 隧道
  ↓
cloudflared --config infrastructure/cloudflare-tunnel/config.yml
  └─→ 所有请求 → localhost:8000 (FastAPI 统一入口)
        ├─→ /docs        → Swagger API 文档
        ├─→ /api/*       → REST API (任务管理、认证)
        ├─→ /ws/*        → WebSocket (进度推送)
        └─→ /            → WebUI 静态页面
```

**特点**: 单端口、单服务、FastAPI 统一处理

### 2.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| PublicServerManager | **不集成** | FastAPI 已提供 HTTP + WebSocket + 静态文件 |
| CloudflareTunnelManager | **集成并适配** | 核心隧道管理能力，代码自包含 |
| Tunnel 路由 | **单一 ingress** | chatsign-auto 只有一个 FastAPI 端口 |
| 生命周期管理 | **FastAPI lifespan** | 替代 myAgent 的手动 main.py 管理 |
| 配置文件隔离 | **`--config` 指定路径** | 不覆盖 `~/.cloudflared/config.yml`，多项目可共存 |
| 开机自启 | **macOS: launchd / Linux: systemd** | 双平台支持 |
| 域名 | **新建子域名** | 与 myAgent 的 bot.hhkej.com 隔离 |

---

## 3. 文件结构

### 3.1 需要新增的文件

```
chatsign-auto/
├── backend/                                       [等 backend 就绪]
│   └── infrastructure/
│       ├── __init__.py                            # 空文件，Python 包标识
│       └── cloudflare_tunnel.py                   # 适配后的 TunnelManager
├── infrastructure/                                [立即]
│   └── cloudflare-tunnel/
│       ├── README.md                              # 文档
│       ├── config.yml.template                    # 配置模板（不含敏感信息）
│       ├── install.sh                             # 安装脚本（适配 chatsign-auto）
│       ├── start.sh                               # 启动脚本
│       ├── stop.sh                                # 停止脚本
│       ├── status.sh                              # 状态检查
│       ├── launchd.plist                          # macOS 自启配置
│       └── systemd/
│           └── chatsign-cloudflared.service       # Linux systemd 服务
└── config/                                        [立即]
    └── infrastructure.yaml                        # 基础设施配置
```

### 3.2 需要修改的文件

| 文件 | 修改内容 | 时机 |
|------|---------|------|
| `backend/main.py` | 在 FastAPI lifespan 中启动/关闭 Tunnel | 等 backend 就绪 |
| `.gitignore` | 添加 `infrastructure/cloudflare-tunnel/config.yml` | 立即 |

---

## 4. 核心代码

### 4.1 CloudflareTunnelManager（适配版） `[等 backend 就绪]`

**文件**: `backend/infrastructure/cloudflare_tunnel.py`

基于 myAgent 的 `src/infrastructure/cloudflare_tunnel.py`（227 行），做以下适配：

```python
"""Cloudflare Tunnel 管理模块

随 ChatSign Orchestrator 启动/停止 Cloudflare Tunnel 服务。
来源: myAgent 项目，适配 FastAPI lifespan 生命周期。
"""

import asyncio
import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CloudflareTunnelManager:
    """Cloudflare Tunnel 管理器

    与 myAgent 版本的关键差异：
    - 使用 --config 指定项目配置文件，不覆盖 ~/.cloudflared/config.yml
    - 支持 monitor 模式（只检测状态，不管理进程）
    - tunnel_name 默认值从 "nanobot" 改为 "chatsign"
    """

    def __init__(
        self,
        tunnel_name: str = "chatsign",
        auto_start: bool = True,
        mode: str = "manage",
    ):
        """
        Args:
            tunnel_name: Tunnel 名称
            auto_start: 是否自动启动（仅 manage 模式有效）
            mode: "manage" = 启动/停止进程; "monitor" = 只检测状态
        """
        self.tunnel_name = tunnel_name
        self.auto_start = auto_start
        self.mode = mode  # "manage" | "monitor"
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None

        # 项目根目录（backend/infrastructure/ → 项目根）
        self.project_dir = Path(__file__).parent.parent.parent
        self.log_file = self.project_dir / "temp" / "cloudflared.log"
        self.pid_file = self.project_dir / "temp" / "cloudflared.pid"

        # 配置文件路径（项目目录，不覆盖全局 ~/.cloudflared/config.yml）
        self.project_config = (
            self.project_dir / "infrastructure" / "cloudflare-tunnel" / "config.yml"
        )

    def is_running(self) -> bool:
        """检查 Cloudflare Tunnel 是否正在运行"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"cloudflared.*tunnel run {self.tunnel_name}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.pid = int(result.stdout.strip().split("\n")[0])
                return True
        except Exception as e:
            logger.debug(f"检查 Cloudflare Tunnel 状态失败: {e}")
        return False

    def _find_cloudflared(self) -> Optional[str]:
        """查找 cloudflared 可执行文件路径"""
        path = shutil.which("cloudflared")
        if path:
            return path

        common_paths = [
            "/opt/homebrew/bin/cloudflared",
            "/usr/local/bin/cloudflared",
            "/usr/bin/cloudflared",
            str(Path.home() / ".cloudflared" / "bin" / "cloudflared"),
        ]
        for p in common_paths:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p

        return None

    def check_cloudflared_installed(self) -> bool:
        return self._find_cloudflared() is not None

    def check_config_exists(self) -> bool:
        return self.project_config.exists()

    async def start(self) -> bool:
        """启动 Cloudflare Tunnel（仅 manage 模式）"""
        # monitor 模式下不管理进程
        if self.mode == "monitor":
            running = self.is_running()
            if running:
                logger.info(f"Cloudflare Tunnel 运行中 (PID: {self.pid}, monitor 模式)")
            else:
                logger.warning("Cloudflare Tunnel 未运行 (monitor 模式，不自动启动)")
            return running

        if self.is_running():
            logger.info(f"Cloudflare Tunnel 已在运行 (PID: {self.pid})")
            return True

        if not self.auto_start:
            logger.warning("Cloudflare Tunnel 未运行且未启用自动启动")
            return False

        if not self.check_cloudflared_installed():
            logger.error(
                "cloudflared 未安装。"
                "macOS: brew install cloudflare/cloudflare/cloudflared | "
                "Linux: 参见 https://developers.cloudflare.com/cloudflare-one/"
            )
            return False

        if not self.check_config_exists():
            logger.error(f"配置文件不存在: {self.project_config}")
            logger.error("请先运行: infrastructure/cloudflare-tunnel/install.sh")
            return False

        log_fd = None
        try:
            logger.info(f"启动 Cloudflare Tunnel: {self.tunnel_name}")

            cloudflared_bin = self._find_cloudflared()
            if not cloudflared_bin:
                logger.error("cloudflared 未找到")
                return False

            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fd = open(self.log_file, "a")

            # 使用 --config 指定项目配置文件，不覆盖全局 ~/.cloudflared/config.yml
            self.process = subprocess.Popen(
                [
                    cloudflared_bin,
                    "--config", str(self.project_config),
                    "tunnel", "run", self.tunnel_name,
                ],
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            self.pid = self.process.pid
            log_fd.close()
            log_fd = None

            await asyncio.sleep(2)

            if self.is_running():
                self.pid_file.write_text(str(self.pid))
                logger.info(f"Cloudflare Tunnel 启动成功 (PID: {self.pid})")
                logger.info(f"日志: {self.log_file}")
                return True
            else:
                logger.error("Cloudflare Tunnel 启动失败，请查看日志")
                return False

        except Exception as e:
            logger.error(f"启动 Cloudflare Tunnel 失败: {e}")
            return False
        finally:
            if log_fd is not None:
                log_fd.close()

    async def stop(self):
        """停止 Cloudflare Tunnel（仅 manage 模式）"""
        if self.mode == "monitor":
            logger.debug("monitor 模式，跳过 stop")
            return

        if not self.is_running():
            logger.info("Cloudflare Tunnel 未运行")
            return

        try:
            logger.info(f"停止 Cloudflare Tunnel (PID: {self.pid})...")
            os.kill(self.pid, signal.SIGTERM)

            for _ in range(50):
                if not self.is_running():
                    break
                await asyncio.sleep(0.1)

            if self.is_running():
                logger.warning("进程未响应 SIGTERM，发送 SIGKILL")
                os.kill(self.pid, signal.SIGKILL)
                await asyncio.sleep(0.5)

            if self.pid_file.exists():
                self.pid_file.unlink()

            logger.info("Cloudflare Tunnel 已停止")

        except ProcessLookupError:
            logger.debug("进程已不存在")
        except Exception as e:
            logger.error(f"停止 Cloudflare Tunnel 失败: {e}")

    def get_status(self) -> dict:
        return {
            "running": self.is_running(),
            "pid": self.pid,
            "tunnel_name": self.tunnel_name,
            "mode": self.mode,
            "log_file": str(self.log_file),
        }
```

**与 myAgent 原版的差异**（共 5 处）：

| # | 差异 | myAgent 原版 | 适配版 | 理由 |
|---|------|-------------|--------|------|
| 1 | 配置文件策略 | `sync_config()` 覆盖 `~/.cloudflared/config.yml` | `--config` 指定项目路径 | **避免覆盖全局配置**，多项目可共存 |
| 2 | 运行模式 | 只有 manage 模式 | 支持 `manage` / `monitor` | 生产环境 systemd 管理时仍可监测状态 |
| 3 | 默认 tunnel_name | `"nanobot"` | `"chatsign"` | 项目隔离 |
| 4 | pgrep 匹配 | `pgrep -f "cloudflared tunnel run {name}"` | `pgrep -f "cloudflared.*tunnel run {name}"` | 兼容 `--config` 参数插入的情况 |
| 5 | 日志 emoji | 保留 | 移除 | 遵循编排器日志风格 |

---

### 4.2 FastAPI lifespan 集成 `[等 backend 就绪]`

**文件**: `backend/main.py`（修改）

```python
# backend/main.py 中添加

import yaml
from contextlib import asynccontextmanager
from pathlib import Path
from backend.infrastructure.cloudflare_tunnel import CloudflareTunnelManager


def _load_infrastructure_config() -> dict:
    """加载基础设施配置"""
    config_path = Path(__file__).parent.parent / "config" / "infrastructure.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@asynccontextmanager
async def lifespan(app):
    """FastAPI 生命周期管理"""

    # === 启动阶段 ===
    infra_config = _load_infrastructure_config()
    tunnel_config = infra_config.get("cloudflare_tunnel", {})

    if tunnel_config.get("enabled", False):
        tunnel = CloudflareTunnelManager(
            tunnel_name=tunnel_config.get("tunnel_name", "chatsign"),
            auto_start=tunnel_config.get("auto_start", True),
            mode=tunnel_config.get("mode", "manage"),
        )
        await tunnel.start()
        # 通过 app.state 传递，避免循环导入
        app.state.cloudflare_tunnel = tunnel
    else:
        app.state.cloudflare_tunnel = None

    yield  # FastAPI 运行中...

    # === 关闭阶段 ===
    tunnel = getattr(app.state, "cloudflare_tunnel", None)
    if tunnel:
        await tunnel.stop()


# FastAPI app 初始化时使用
app = FastAPI(
    title="ChatSign Orchestrator",
    lifespan=lifespan,
)
```

**与原方案的差异**：用 `app.state` 替代模块级全局变量 `_cloudflare_tunnel`，避免循环导入。

### 4.3 Tunnel 状态 API `[等 backend 就绪]`

**文件**: `backend/api/infrastructure.py`（新增）

```python
"""基础设施状态 API"""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/infrastructure", tags=["infrastructure"])


@router.get("/tunnel/status")
async def get_tunnel_status(request: Request):
    """获取 Cloudflare Tunnel 运行状态"""
    tunnel = getattr(request.app.state, "cloudflare_tunnel", None)
    if tunnel is None:
        return {"enabled": False}
    return {"enabled": True, **tunnel.get_status()}
```

**与原方案的差异**：通过 `request.app.state` 获取实例，无 import 依赖。

---

## 5. 配置文件 `[立即]`

### 5.1 infrastructure.yaml

**文件**: `config/infrastructure.yaml`

```yaml
# ChatSign Orchestrator 基础设施配置

# Cloudflare Tunnel - 公网访问能力
cloudflare_tunnel:
  # 是否启用（默认关闭，需要先运行 install.sh）
  enabled: false

  # Tunnel 名称（需与 cloudflared tunnel create 时的名称一致）
  tunnel_name: chatsign

  # 是否自动启动（仅 manage 模式有效）
  auto_start: true

  # 运行模式:
  #   manage  — FastAPI 启动时自动启动/停止 Tunnel 进程（开发环境）
  #   monitor — 只检测 Tunnel 状态，不管理进程（生产环境由 systemd 管理）
  mode: manage

  # 公网域名（仅用于日志和文档）
  domain: sign.hhkej.com

  # FastAPI 本地端口
  local_port: 8000
```

### 5.2 Tunnel config.yml 模板

**文件**: `infrastructure/cloudflare-tunnel/config.yml.template`

```yaml
# Cloudflare Tunnel 配置
# 由 install.sh 生成实际的 config.yml（包含 tunnel ID）
# 请勿直接编辑此模板
#
# 运行方式（使用 --config 指定项目配置，不覆盖全局 ~/.cloudflared/）：
#   cloudflared --config infrastructure/cloudflare-tunnel/config.yml tunnel run chatsign

tunnel: <TUNNEL_ID>
credentials-file: <HOME>/.cloudflared/<TUNNEL_ID>.json

ingress:
  # 所有请求转发到 FastAPI（HTTP + WebSocket 统一处理）
  #
  # 注意：这里使用 http:// 而非 ws://，因为 Cloudflare Tunnel 会自动处理
  # HTTP → WebSocket 的 Upgrade 协商，FastAPI 内置的 WebSocket 端点
  # （如 /ws/progress）无需额外配置即可正常工作。
  - hostname: <DOMAIN>
    service: http://localhost:8000
    originRequest:
      noTLSVerify: true
      connectTimeout: 30s
  - service: http_status:404
```

---

## 6. 运维脚本 `[立即]`

### 6.1 install.sh

**文件**: `infrastructure/cloudflare-tunnel/install.sh`

```bash
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
LOCAL_PORT=8000

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

if cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
    echo "[OK] Tunnel '$TUNNEL_NAME' 已存在"
    TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
else
    echo "创建 Tunnel: $TUNNEL_NAME"
    cloudflared tunnel create "$TUNNEL_NAME"
    TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
fi
echo "   Tunnel ID: $TUNNEL_ID"

# --- 配置 DNS ---
read -p "请输入你的域名（例如: hhkej.com）: " DOMAIN
read -p "二级域名前缀 [${DEFAULT_SUBDOMAIN}]: " SUBDOMAIN
SUBDOMAIN="${SUBDOMAIN:-$DEFAULT_SUBDOMAIN}"
FULL_DOMAIN="${SUBDOMAIN}.${DOMAIN}"

echo "配置 DNS 记录: $FULL_DOMAIN → Tunnel"
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
INFRA_CONFIG="$PROJECT_DIR/config/infrastructure.yaml"
if [ -f "$INFRA_CONFIG" ]; then
    # 使用 uname 区分 sed 语法（修复 myAgent 原版的跨平台问题）
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        sed -i '' "s/enabled: false/enabled: true/" "$INFRA_CONFIG"
        sed -i '' "s/tunnel_name: chatsign/tunnel_name: $TUNNEL_NAME/" "$INFRA_CONFIG"
        sed -i '' "s/domain: sign.hhkej.com/domain: $FULL_DOMAIN/" "$INFRA_CONFIG"
    else
        sed -i "s/enabled: false/enabled: true/" "$INFRA_CONFIG"
        sed -i "s/tunnel_name: chatsign/tunnel_name: $TUNNEL_NAME/" "$INFRA_CONFIG"
        sed -i "s/domain: sign.hhkej.com/domain: $FULL_DOMAIN/" "$INFRA_CONFIG"
    fi
    echo "[OK] infrastructure.yaml 已更新"
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
```

### 6.2 start.sh

**文件**: `infrastructure/cloudflare-tunnel/start.sh`

```bash
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

# 从 config.yml 中提取 tunnel 名称（兼容未安装 yq 的环境）
TUNNEL_NAME=$(grep '^tunnel:' "$PROJECT_CONFIG" 2>/dev/null | awk '{print $2}')
if [ -z "$TUNNEL_NAME" ]; then
    TUNNEL_NAME="chatsign"
fi

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
EXISTING_PID=$(pgrep -f "cloudflared.*tunnel run" || echo "")
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
    LAUNCHD_LABEL="com.chatsign.cloudflared"

    mkdir -p "$LAUNCHD_DIR"
    cp "$PLIST_FILE" "$LAUNCHD_DIR/${LAUNCHD_LABEL}.plist"

    launchctl load "$LAUNCHD_DIR/${LAUNCHD_LABEL}.plist" 2>/dev/null || {
        echo "[WARN] 服务已加载，正在重启..."
        launchctl unload "$LAUNCHD_DIR/${LAUNCHD_LABEL}.plist" 2>/dev/null
        launchctl load "$LAUNCHD_DIR/${LAUNCHD_LABEL}.plist"
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
PID=$(pgrep -f "cloudflared.*tunnel run" || echo "")
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
```

### 6.3 stop.sh

**文件**: `infrastructure/cloudflare-tunnel/stop.sh`

```bash
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
pkill -f "cloudflared.*tunnel run" 2>/dev/null && echo "[OK] 清理残留进程" || true

echo "[OK] Cloudflare Tunnel 已停止"
echo "=========================================="
```

### 6.4 status.sh

**文件**: `infrastructure/cloudflare-tunnel/status.sh`

```bash
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
PID=$(pgrep -f "cloudflared.*tunnel run" || echo "")

if [ -n "$PID" ]; then
    echo "[OK] 服务运行中"
    echo "     PID:  $PID"

    # 从配置文件读取域名
    if [ -f "$PROJECT_CONFIG" ]; then
        HOSTNAME=$(grep 'hostname:' "$PROJECT_CONFIG" | head -1 | awk '{print $2}')
        echo "     域名: https://$HOSTNAME → localhost:8000"
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
```

### 6.5 systemd 服务（Linux 生产环境）

**文件**: `infrastructure/cloudflare-tunnel/systemd/chatsign-cloudflared.service`

```ini
[Unit]
Description=ChatSign Cloudflare Tunnel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=chatsign

# 使用 --config 指定项目配置文件，不覆盖全局 ~/.cloudflared/config.yml
ExecStart=/usr/local/bin/cloudflared \
    --config /opt/chatsign/orchestrator/infrastructure/cloudflare-tunnel/config.yml \
    tunnel run

Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/chatsign/cloudflared.log
StandardError=append:/var/log/chatsign/cloudflared.log

# 安全加固
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/log/chatsign

[Install]
WantedBy=multi-user.target
```

### 6.6 launchd.plist（macOS 开发环境）

**文件**: `infrastructure/cloudflare-tunnel/launchd.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.chatsign.cloudflared</string>

    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/cloudflared</string>
        <string>--config</string>
        <string>/Users/chatsign/Projects/chatsign-auto/infrastructure/cloudflare-tunnel/config.yml</string>
        <string>tunnel</string>
        <string>run</string>
        <string>chatsign</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/chatsign/Projects/chatsign-auto</string>

    <key>StandardOutPath</key>
    <string>/Users/chatsign/Projects/chatsign-auto/temp/cloudflared.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/chatsign/Projects/chatsign-auto/temp/cloudflared.log</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>/Users/chatsign</string>
    </dict>
</dict>
</plist>
```

---

## 7. .gitignore 补充 `[立即]`

```gitignore
# Cloudflare Tunnel（config.yml 包含 Tunnel ID，由 install.sh 生成）
infrastructure/cloudflare-tunnel/config.yml
```

---

## 8. 依赖项

### 8.1 Python 依赖

**无新增**。CloudflareTunnelManager 仅使用标准库（asyncio, subprocess, shutil, signal, pathlib）。

加载 infrastructure.yaml 需要 `PyYAML`，FastAPI 项目通常已包含。

### 8.2 系统依赖

| 依赖 | macOS | Ubuntu (10.228.128.11) |
|------|-------|------------------------|
| cloudflared | `brew install cloudflare/cloudflare/cloudflared` | `dpkg -i cloudflared-linux-amd64.deb` |
| pgrep | 内置 | 内置（procps） |

---

## 9. 实施步骤

### Phase A: 基础设施（立即可执行）

```bash
# Step 1: 创建目录
cd /Users/chatsign/Projects/chatsign-auto
mkdir -p infrastructure/cloudflare-tunnel/systemd
mkdir -p config
mkdir -p temp

# Step 2: 创建配置和脚本文件（按本文档 5-6 节内容）

# Step 3: 在 GPU 服务器上安装
ssh chatsign@10.228.128.11
cd /opt/chatsign/orchestrator
infrastructure/cloudflare-tunnel/install.sh

# Step 4: 启动 Tunnel（systemd 方式）
sudo cp infrastructure/cloudflare-tunnel/systemd/chatsign-cloudflared.service \
    /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable chatsign-cloudflared
sudo systemctl start chatsign-cloudflared

# Step 5: 验证 Tunnel 连通性
infrastructure/cloudflare-tunnel/status.sh
curl https://sign.hhkej.com  # 预期: 502（FastAPI 尚未运行）
```

### Phase B: FastAPI 集成（等 backend 就绪）

```bash
# Step 6: 创建 Python 模块
mkdir -p backend/infrastructure
touch backend/infrastructure/__init__.py
# 创建 cloudflare_tunnel.py（按 4.1 节）

# Step 7: 修改 backend/main.py（按 4.2 节）
# 添加 lifespan + app.state

# Step 8: 添加状态 API（按 4.3 节，可选）

# Step 9: 配置 infrastructure.yaml
# 生产环境: mode: monitor（systemd 管理进程）
# 开发环境: mode: manage（FastAPI 管理进程）

# Step 10: 验证完整链路
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
curl https://sign.hhkej.com/docs
curl https://sign.hhkej.com/api/infrastructure/tunnel/status
```

---

## 10. 部署场景

### 10.1 开发环境（macOS 本机）

```yaml
# config/infrastructure.yaml
cloudflare_tunnel:
  enabled: true
  mode: manage      # FastAPI 自动启动/停止 Tunnel
  auto_start: true
```

### 10.2 生产环境（GPU 服务器 10.228.128.11）

```yaml
# config/infrastructure.yaml
cloudflare_tunnel:
  enabled: true
  mode: monitor     # 只监测，systemd 管理进程
```

```bash
# systemd 管理 cloudflared 生命周期（独立于 FastAPI）
sudo systemctl start chatsign-cloudflared    # Tunnel
sudo systemctl start chatsign-orchestrator   # FastAPI

# 两个服务独立重启互不影响
```

### 10.3 Tunnel 与现有 Nginx 方案的关系

现有部署指南（`deployment/DEPLOYMENT_GUIDE.md` Step 7）已配置了 Nginx 反向代理。
两种方案**不冲突**：

| 方案 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **Nginx + SSL** | 有公网 IP 的服务器 | 性能更好、完全自控 | 需要公网 IP、SSL 证书管理 |
| **Cloudflare Tunnel** | 无公网 IP（如 10.228.128.11） | 无需公网 IP、自动 SSL、CDN | 依赖 Cloudflare |
| **两者并用** | 内网 Nginx + 外网 Tunnel | 内外网双通道 | 维护两套配置 |

GPU 服务器 `10.228.128.11` 在内网，**优先使用 Cloudflare Tunnel**。

---

## 11. 安全考虑

### 11.1 已有的安全措施

- Cloudflare 自动 DDoS 防护
- TLS 端到端加密
- 进程隔离（`start_new_session=True`）
- JWT 认证（编排器已有，Tunnel 自动继承）

### 11.2 额外建议

| 措施 | 说明 | 优先级 |
|------|------|--------|
| Cloudflare Access | 在 Cloudflare 侧添加 Zero Trust 策略 | 推荐 |
| IP 白名单 | 通过 Cloudflare WAF 限制来源 IP | 可选 |
| 速率限制 | 通过 Cloudflare Rate Limiting | 可选 |

### 11.3 敏感文件保护

以下文件**不应提交到 Git**：

```
infrastructure/cloudflare-tunnel/config.yml    # 包含 Tunnel ID（已加入 .gitignore）
~/.cloudflared/*.json                          # Tunnel 凭证
~/.cloudflared/cert.pem                        # Cloudflare 登录证书
```

---

## 12. 回滚方案

### 12.1 卸载 Tunnel

```bash
# 停止服务
infrastructure/cloudflare-tunnel/stop.sh

# Linux: 移除 systemd 服务
sudo systemctl stop chatsign-cloudflared
sudo systemctl disable chatsign-cloudflared
sudo rm /etc/systemd/system/chatsign-cloudflared.service
sudo systemctl daemon-reload

# macOS: 移除 LaunchAgent
launchctl unload ~/Library/LaunchAgents/com.chatsign.cloudflared.plist
rm ~/Library/LaunchAgents/com.chatsign.cloudflared.plist

# 删除 Tunnel（可选，会删除 DNS 记录）
cloudflared tunnel delete chatsign
```

### 12.2 禁用但保留

```yaml
# config/infrastructure.yaml
cloudflare_tunnel:
  enabled: false    # 禁用，不删除任何文件
```

---

## 13. 复用评估总结

### 13.1 可直接复用

| 组件 | 复用度 | 说明 |
|------|--------|------|
| Tunnel 管理核心逻辑 | 90% | 去掉 sync_config，加 --config 和 monitor 模式 |

### 13.2 需要适配

| 组件 | 修改内容 |
|------|---------|
| install.sh | Linux 支持、uname sed、单端口 ingress、不覆盖全局配置 |
| start.sh | 全部重写（--config、双平台） |
| stop.sh | 改 Label 名，其余复用 |
| status.sh | 改域名和端口显示 |
| launchd.plist | 加 --config 参数、改 Label/路径 |

### 13.3 不集成

| 组件 | 理由 |
|------|------|
| `public_server.py` | FastAPI 已覆盖全部能力 |
| gateway.yaml CORS | chatsign-auto 没有独立 Gateway 层 |

### 13.4 新增

| 组件 | 理由 |
|------|------|
| systemd service | GPU 服务器是 Linux |
| monitor 模式 | 生产环境 systemd 管理时保留状态可观测性 |
| `--config` 隔离 | 多项目共存（myAgent + chatsign-auto 同机） |
| 回滚方案 | 完整的卸载和禁用流程 |
| `/api/infrastructure/tunnel/status` | 远程监控 |

---

## 附录: myAgent 源文件参考

| 文件 | 路径 | 行数 |
|------|------|------|
| CloudflareTunnelManager | `/Users/chatsign/Projects/myAgent/src/infrastructure/cloudflare_tunnel.py` | 227 |
| PublicServerManager | `/Users/chatsign/Projects/myAgent/src/infrastructure/public_server.py` | 157 |
| Tunnel 配置 | `/Users/chatsign/Projects/myAgent/infrastructure/cloudflare-tunnel/config.yml` | 28 |
| 基础设施配置 | `/Users/chatsign/Projects/myAgent/config/infrastructure.yaml` | 53 |
| 安装脚本 | `/Users/chatsign/Projects/myAgent/infrastructure/cloudflare-tunnel/install.sh` | 131 |
| 启动脚本 | `/Users/chatsign/Projects/myAgent/infrastructure/cloudflare-tunnel/start.sh` | 68 |
| 停止脚本 | `/Users/chatsign/Projects/myAgent/infrastructure/cloudflare-tunnel/stop.sh` | 38 |
| 状态检查 | `/Users/chatsign/Projects/myAgent/infrastructure/cloudflare-tunnel/status.sh` | 44 |
| launchd 配置 | `/Users/chatsign/Projects/myAgent/infrastructure/cloudflare-tunnel/launchd.plist` | 46 |
| 启动集成逻辑 | `/Users/chatsign/Projects/myAgent/src/main.py:3240-3331` | 91 |
