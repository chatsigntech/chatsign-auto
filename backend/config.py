import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/tasks.db")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    DEFAULT_ADMIN_PASSWORD: str = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # External project paths
    UNISIGN_PATH: Path = Path(os.getenv("UNISIGN_PATH", str(BASE_DIR / "UniSignMimicTurbo")))
    GLOSS_AWARE_PATH: Path = Path(os.getenv("GLOSS_AWARE_PATH", str(BASE_DIR / "gloss_aware")))
    PSEUDO_GLOSS_PATH: Path = Path(os.getenv("PSEUDO_GLOSS_PATH", str(BASE_DIR / "pseudo-gloss-English")))
    SPAMO_SEGMENT_PATH: Path = Path(os.getenv("SPAMO_SEGMENT_PATH", str(BASE_DIR / "spamo_segement")))
    GUAVA_AUG_PATH: Path = Path(os.getenv("GUAVA_AUG_PATH", str(BASE_DIR / "guava-aug")))
    GLOSS_CSV_PATH: Path = Path(os.getenv("GLOSS_CSV_PATH", str(BASE_DIR / "data" / "gloss.csv")))
    CHATSIGN_ACCURACY_URL: str = os.getenv("CHATSIGN_ACCURACY_URL", "https://localhost:5443")
    CHATSIGN_ACCURACY_DATA: Path = Path(os.getenv(
        "CHATSIGN_ACCURACY_DATA",
        str(BASE_DIR / "chatsign-accuracy" / "backend" / "data")
    ))

    # Data directories
    SHARED_DATA_ROOT: Path = Path(os.getenv("SHARED_DATA_ROOT", str(BASE_DIR / "data" / "shared")))

    # GPU
    MAX_GPUS: int = int(os.getenv("MAX_GPUS", "1"))
    CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "orchestrator.log"))

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Accuracy system
    ACCURACY_API_URL: str = os.getenv("ACCURACY_API_URL", "https://localhost:5443")

    # Augmentation config (JSON, managed via /api/config/augmentation)
    AUGMENTATION_CONFIG_PATH: Path = Path(os.getenv("AUGMENTATION_CONFIG_PATH", str(BASE_DIR / "data" / "augmentation_config.json")))

    @property
    def cuda_device_ids(self) -> list[int]:
        return [int(d) for d in self.CUDA_VISIBLE_DEVICES.split(",") if d.strip()]


settings = Settings()
