import logging
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from sqlmodel import Session, select

from backend.config import settings
from backend.database import engine, init_db
from backend.models.user import User
from backend.api import auth, tasks, phases, config, accuracy

logger = logging.getLogger("orchestrator")


def _ensure_admin():
    """Create default admin user if not exists."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    with Session(engine) as session:
        existing = session.exec(select(User).where(User.username == "admin")).first()
        if not existing:
            admin = User(
                username="admin",
                hashed_password=pwd_context.hash(settings.DEFAULT_ADMIN_PASSWORD),
                is_admin=True,
            )
            session.add(admin)
            session.commit()
            logger.info("Default admin user created")


def _ensure_directories():
    """Create required data directories."""
    for d in [
        settings.SHARED_DATA_ROOT,
        Path(settings.LOG_FILE).parent,
        Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent,
    ]:
        d.mkdir(parents=True, exist_ok=True)


_accuracy_proc = None


def _start_accuracy_service():
    """Start chatsign-accuracy Node.js service if not already running."""
    global _accuracy_proc
    from backend.config import BASE_DIR
    accuracy_dir = BASE_DIR / "chatsign-accuracy"
    if not (accuracy_dir / "backend" / "server.js").exists():
        logger.warning("Accuracy service not found, skipping auto-start")
        return

    # Check if already running on port 5443
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        sock.connect(("127.0.0.1", 5443))
        sock.close()
        logger.info("Accuracy service already running on port 5443")
        return
    except (ConnectionRefusedError, OSError):
        pass

    logger.info("Starting accuracy service (Node.js)...")
    env = {**__import__("os").environ, "NODE_ENV": "production", "USE_GOOGLE_DRIVE": "false"}
    _accuracy_proc = subprocess.Popen(
        ["node", "backend/server.js"],
        cwd=str(accuracy_dir),
        env=env,
        stdout=open(str(BASE_DIR / "logs" / "accuracy.log"), "a"),
        stderr=subprocess.STDOUT,
    )
    logger.info(f"Accuracy service started (PID {_accuracy_proc.pid})")


def _stop_accuracy_service():
    """Stop the accuracy service if we started it."""
    global _accuracy_proc
    if _accuracy_proc and _accuracy_proc.poll() is None:
        _accuracy_proc.terminate()
        _accuracy_proc.wait(timeout=5)
        logger.info("Accuracy service stopped")
        _accuracy_proc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _ensure_directories()
    init_db()
    _ensure_admin()
    _start_accuracy_service()
    logger.info(f"Orchestrator started on {settings.API_HOST}:{settings.API_PORT}")
    yield
    # Shutdown
    _stop_accuracy_service()
    logger.info("Orchestrator shutting down")


app = FastAPI(
    title="ChatSign Orchestrator",
    description="6-phase sign language video processing pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(phases.router)
app.include_router(config.router)
app.include_router(accuracy.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# Serve frontend static files
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve the Vue SPA for all non-API routes."""
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "service": "ChatSign Orchestrator",
            "docs": "/docs",
            "health": "/health",
        }
