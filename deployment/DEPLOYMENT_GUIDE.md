# 生产环境部署指南

**版本**: 1.0
**最后更新**: 2026-03-24
**用途**: 部署 ChatSign Orchestrator 到生产环境

---

## 📋 前置要求

### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|--------|
| **OS** | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| **Python** | 3.10 | 3.11+ |
| **CPU** | 4 核 | 8+ 核 |
| **RAM** | 16GB | 32GB+ |
| **存储** | 500GB | 2TB+ (RAID-6) |
| **GPU** | 1×GPU (可选) | 8×NVIDIA A100/H100 |

### 外部依赖

```bash
# 系统库
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-venv
sudo apt install -y git curl wget
sudo apt install -y libssl-dev libffi-dev

# GPU 支持（可选）
cuda-11.8 或更高版本
cuDNN 8.x
nvidia-container-toolkit
```

### 外部项目

以下项目必须部署在同一机器或可访问的位置：

| 项目 | 作用 | 路径 | 说明 |
|------|------|------|------|
| **UniSignMimicTurbo** | Phase 4/5 脚本 | `/opt/chatsign/unisign` | 视频处理和增广 |
| **GlossAware** | Phase 6 脚本 | `/opt/chatsign/gloss-aware` | 模型训练 |
| **chatsign-accuracy** | Phase 1-3 采集 | 同一服务器或网络 | 数据采集平台 |

---

## 🚀 部署步骤

### Step 1: 准备服务器

```bash
# 1.1 创建应用用户
sudo useradd -m -s /bin/bash chatsign
sudo usermod -aG sudo chatsign

# 1.2 创建应用目录
sudo mkdir -p /opt/chatsign
sudo chown -R chatsign:chatsign /opt/chatsign
sudo chmod 755 /opt/chatsign

# 1.3 创建数据目录
sudo mkdir -p /data/chatsign/shared
sudo mkdir -p /data/chatsign/tasks
sudo mkdir -p /data/chatsign/cache
sudo chown -R chatsign:chatsign /data/chatsign
sudo chmod 755 /data/chatsign

# 1.4 创建日志目录
sudo mkdir -p /var/log/chatsign
sudo chown -R chatsign:chatsign /var/log/chatsign
```

### Step 2: 克隆项目

```bash
# 2.1 切换到应用用户
sudo su - chatsign

# 2.2 克隆 chatsign-auto
cd /opt/chatsign
git clone <repo-url> orchestrator
cd orchestrator

# 2.3 检查项目结构
ls -la backend/
# 应该看到: main.py, config.py, database.py, models/, api/, core/, workers/
```

### Step 3: 配置环境

```bash
# 3.1 创建虚拟环境
python3.11 -m venv /opt/chatsign/venv
source /opt/chatsign/venv/bin/activate

# 3.2 升级 pip
pip install --upgrade pip setuptools wheel

# 3.3 安装依赖
cd /opt/chatsign/orchestrator
pip install -r requirements.txt

# 3.4 验证安装
python -c "import fastapi, sqlmodel, torch; print('All imports OK')"
```

### Step 4: 配置应用

```bash
# 4.1 创建 .env 文件
cp backend/.env.example .env

# 4.2 编辑关键配置
cat > .env << 'EOF'
# 数据库
DATABASE_URL=sqlite:////data/chatsign/orchestrator/tasks.db

# 安全性（生产环境必须修改）
SECRET_KEY=your-super-secret-key-change-this-in-production-12345678
DEFAULT_ADMIN_PASSWORD=change-this-password-in-production

# 外部项目路径
UNISIGN_PATH=/opt/chatsign/unisign
GLOSS_AWARE_PATH=/opt/chatsign/gloss-aware
PSEUDO_GLOSS_PATH=/opt/chatsign/pseudo-gloss-English

# 共享数据目录
SHARED_DATA_ROOT=/data/chatsign/shared

# GPU 配置
MAX_GPUS=8                    # 限制最多使用 8 个 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 日志
LOG_LEVEL=INFO
LOG_FILE=/var/log/chatsign/orchestrator.log

# API 配置
API_WORKERS=4                 # 并发工作进程
API_HOST=0.0.0.0
API_PORT=8000
EOF

# 4.3 生成强随机密钥（使用工具生成）
python -c "import secrets; print(secrets.token_urlsafe(32))"
# 输出示例: "FP4kqNwRz-8XGYp9LmJvXqL1Z2a3B4c5D6e7"
# 替换到 SECRET_KEY 中
```

### Step 5: 初始化数据库

```bash
# 5.1 创建数据库目录
mkdir -p /data/chatsign/orchestrator

# 5.2 初始化数据库（自动创建表和 admin 用户）
cd /opt/chatsign/orchestrator
python -c "
import asyncio
from backend.database import init_db, AsyncSessionLocal
from backend.models.user import User
from backend.core.security import hash_password
from sqlmodel import Session, select

async def setup():
    await init_db()
    print('Database initialized successfully')

asyncio.run(setup())
"

# 5.3 验证数据库
sqlite3 /data/chatsign/orchestrator/tasks.db ".tables"
# 应该看到: phase_states, pipeline_tasks, users 等表
```

### Step 6: 配置 Systemd 服务

```bash
# 6.1 创建 systemd 服务文件
sudo tee /etc/systemd/system/chatsign-orchestrator.service << 'EOF'
[Unit]
Description=ChatSign Orchestrator Service
After=network.target
Wants=network-online.target

[Service]
Type=notify
User=chatsign
Group=chatsign
WorkingDirectory=/opt/chatsign/orchestrator
Environment="PATH=/opt/chatsign/venv/bin"
Environment="PYTHONUNBUFFERED=1"

# 启动命令
ExecStart=/opt/chatsign/venv/bin/python -m uvicorn \
  backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --access-log \
  --log-config logging_config.yaml

# 重启策略
Restart=always
RestartSec=10
StartLimitInterval=600
StartLimitBurst=3

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=chatsign-orchestrator

# 资源限制
LimitNOFILE=65535
LimitNPROC=65535

[Install]
WantedBy=multi-user.target
EOF

# 6.2 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable chatsign-orchestrator
sudo systemctl start chatsign-orchestrator

# 6.3 验证服务状态
sudo systemctl status chatsign-orchestrator
journalctl -u chatsign-orchestrator -f
```

### Step 7: 配置 Nginx 反向代理

```bash
# 7.1 创建 Nginx 配置
sudo tee /etc/nginx/sites-available/chatsign-orchestrator << 'EOF'
upstream orchestrator {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name orchestrator.example.com;

    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name orchestrator.example.com;

    # SSL 证书（使用 Let's Encrypt 或自签名）
    ssl_certificate /etc/letsencrypt/live/orchestrator.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/orchestrator.example.com/privkey.pem;

    # SSL 配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # 日志
    access_log /var/log/nginx/orchestrator_access.log;
    error_log /var/log/nginx/orchestrator_error.log;

    # 代理设置
    location / {
        proxy_pass http://orchestrator;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket 支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # 超时配置
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # 大文件上传限制
    client_max_body_size 2G;
}
EOF

# 7.2 启用配置
sudo ln -s /etc/nginx/sites-available/chatsign-orchestrator /etc/nginx/sites-enabled/

# 7.3 测试配置
sudo nginx -t

# 7.4 重启 Nginx
sudo systemctl restart nginx
```

### Step 8: 验证部署

```bash
# 8.1 检查服务状态
sudo systemctl status chatsign-orchestrator

# 8.2 访问 API
curl -k https://orchestrator.example.com/docs

# 8.3 登录测试
curl -X POST https://orchestrator.example.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "your-configured-password"
  }'

# 输出应该包含 JWT token

# 8.4 创建测试任务
curl -X POST https://orchestrator.example.com/api/tasks \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Task",
    "config": {"augmentation_preset": "light"}
  }'
```

---

## 📊 生产环境监控

### 1. 日志监控

```bash
# 实时日志
sudo journalctl -u chatsign-orchestrator -f

# 按日期查看
sudo journalctl -u chatsign-orchestrator --since "2026-03-24 10:00" --until "2026-03-24 12:00"

# 错误日志
sudo journalctl -u chatsign-orchestrator -p err -n 100
```

### 2. 系统资源监控

```bash
# 使用 top 查看 CPU/内存
top -p $(pgrep -f "uvicorn")

# 使用 nvidia-smi 查看 GPU
nvidia-smi
watch -n 1 nvidia-smi

# 磁盘使用
df -h /data/chatsign/
```

### 3. 数据库监控

```bash
# 检查数据库大小
du -sh /data/chatsign/orchestrator/tasks.db

# 检查未处理的任务
sqlite3 /data/chatsign/orchestrator/tasks.db \
  "SELECT task_id, name, status FROM pipeline_tasks WHERE status = 'pending' LIMIT 10;"

# 检查失败的 Phase
sqlite3 /data/chatsign/orchestrator/tasks.db \
  "SELECT task_id, phase_num, status, error_message FROM phase_states WHERE status = 'failed';"
```

### 4. API 监控

```bash
# 检查 API 响应
curl -s https://orchestrator.example.com/health || echo "API 不可用"

# 列出运行中的任务
curl -s -H "Authorization: Bearer {token}" \
  https://orchestrator.example.com/api/tasks?status=running | jq '.tasks[] | {task_id, name, progress}'
```

---

## 🔧 常见维护操作

### 重启服务

```bash
sudo systemctl restart chatsign-orchestrator
# 或平滑重启（不中断连接）
sudo systemctl reload chatsign-orchestrator
```

### 查看配置

```bash
# 查看当前配置
cat /opt/chatsign/orchestrator/.env

# 修改配置后需要重启
sudo systemctl restart chatsign-orchestrator
```

### 清理缓存

```bash
# 清理过期的缓存文件
find /data/chatsign/cache -type f -mtime +7 -delete

# 清理中间帧（Phase 4 完成后）
find /data/chatsign/tasks -name "*frames*" -type d -mtime +3 -exec rm -rf {} \;
```

### 数据库备份

```bash
# 全量备份
sqlite3 /data/chatsign/orchestrator/tasks.db ".backup /backup/tasks_$(date +%Y%m%d_%H%M%S).db"

# 定期备份（Cron）
0 2 * * * sqlite3 /data/chatsign/orchestrator/tasks.db ".backup /backup/tasks_$(date +\%Y\%m\%d).db"
```

### 日志轮转

```bash
# 创建 logrotate 配置
sudo tee /etc/logrotate.d/chatsign-orchestrator << 'EOF'
/var/log/chatsign/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 chatsign chatsign
    sharedscripts
    postrotate
        systemctl reload chatsign-orchestrator > /dev/null 2>&1 || true
    endscript
}
EOF
```

---

## 🚨 故障排除

### 问题 1: 服务无法启动

```bash
# 查看详细错误
sudo systemctl status chatsign-orchestrator
journalctl -u chatsign-orchestrator -n 50

# 常见原因和解决
# 1. 端口被占用
sudo lsof -i :8000

# 2. 权限问题
sudo chown -R chatsign:chatsign /opt/chatsign /data/chatsign

# 3. 依赖缺失
source /opt/chatsign/venv/bin/activate
pip install -r requirements.txt
```

### 问题 2: GPU 无法访问

```bash
# 检查 GPU
nvidia-smi

# 检查权限
sudo usermod -aG video chatsign
sudo systemctl restart chatsign-orchestrator

# 检查 CUDA 环境
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题 3: 数据库锁定

```bash
# 检查是否有进程锁定数据库
lsof | grep tasks.db

# 如果是 stale lock，清理
rm /data/chatsign/orchestrator/tasks.db.lock

# 验证数据库完整性
sqlite3 /data/chatsign/orchestrator/tasks.db "PRAGMA integrity_check;"
```

### 问题 4: 磁盘空间不足

```bash
# 检查磁盘使用
df -h /data/chatsign/

# 清理旧任务的中间文件
# 但不删除最终输出！
find /data/chatsign/tasks -name "*frames*" -type d -delete
find /data/chatsign/cache -type f -mtime +7 -delete
```

---

## 📈 性能优化

### 1. 增加 Worker 数

```bash
# 编辑 systemd 配置
sudo systemctl edit chatsign-orchestrator

# 修改 ExecStart 中的 --workers 参数
--workers 8  # 从 4 改为 8
```

### 2. 数据库优化

```bash
# 定期 VACUUM 整理数据库
sqlite3 /data/chatsign/orchestrator/tasks.db "VACUUM;"

# 添加到 cron（每周）
0 3 * * 0 sqlite3 /data/chatsign/orchestrator/tasks.db "VACUUM;"
```

### 3. GPU 内存优化

在 `.env` 中调整：
```bash
# Phase 5 batch size
PHASE5_BATCH_SIZE=4  # 减小可降低内存占用
```

---

## 🔐 安全建议

### 1. 更改默认密码

```bash
# 立即更改 admin 密码
curl -X POST https://orchestrator.example.com/api/auth/change-password \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "old_password": "default-password",
    "new_password": "secure-new-password"
  }'
```

### 2. 启用 HTTPS

```bash
# 使用 Let's Encrypt 获取免费证书
sudo apt install certbot python3-certbot-nginx
sudo certbot certonly --nginx -d orchestrator.example.com
```

### 3. 限制访问

```bash
# 在 Nginx 配置中限制 IP
location /api/admin {
    allow 10.0.0.0/8;      # 内部网络
    deny all;
}
```

### 4. 定期更新

```bash
# 检查 Python 包更新
pip list --outdated

# 更新所有包
pip install --upgrade -r requirements.txt

# 重启服务
sudo systemctl restart chatsign-orchestrator
```

---

## 📚 相关文档

- [README.md](../README.md) - 项目概览
- [PROJECT_STATUS.md](../PROJECT_STATUS.md) - Phase 完成情况
- [.env.example](../backend/.env.example) - 配置示例

---

**版本**: 1.0
**最后更新**: 2026-03-24
**维护者**: ChatSign Team
**状态**: ✅ 完成
