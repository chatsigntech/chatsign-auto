import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from sqlmodel import Session, select

from backend.config import settings
from backend.database import engine, init_db
from backend.models.user import User
from backend.api import auth, tasks, phases

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _ensure_directories()
    init_db()
    _ensure_admin()
    logger.info(f"Orchestrator started on {settings.API_HOST}:{settings.API_PORT}")
    yield
    # Shutdown
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


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
def root():
    return {
        "service": "ChatSign Orchestrator",
        "docs": "/docs",
        "health": "/health",
    }
