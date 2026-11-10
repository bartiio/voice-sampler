"""
backend/app/config.py
Centralized configuration management for Voice Sampler backend.
Supports environment variable overrides and provides structured path helpers.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application configuration and path management settings."""

    # Application Metadata
    app_name: str = "Voice Sampler API"
    app_version: str = "0.1.0"
    app_description: str = "Backend API for Voice Sampler desktop application"

    # Server Bindings
    host: str = Field(
        default_factory=lambda: os.getenv("VOICE_SAMPLER_HOST", "127.0.0.1")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("VOICE_SAMPLER_PORT", "8000"))
    )

    # Environment & Operational Modes
    dev_mode: bool = Field(
        default_factory=lambda: os.getenv("VOICE_SAMPLER_DEV_MODE", "0").lower()
        in ("1", "true", "yes")
    )
    simulation_mode: bool = Field(
        default_factory=lambda: os.getenv("VOICE_SAMPLER_SIMULATION_MODE", "0").lower()
        in ("1", "true", "yes")
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("VOICE_SAMPLER_LOG_LEVEL", "info").lower()
    )

    # Root Directory
    # Relative path from backend/app/config.py: config.py -> app -> backend -> project_root
    project_root: Path = Field(
        default_factory=lambda: Path(
            os.getenv("VOICE_SAMPLER_ROOT", str(Path(__file__).resolve().parent.parent.parent))
        ).resolve()
    )

    # Runtime Data Directories (Resolved dynamically in model_post_init)
    data_dir: Optional[Path] = None
    db_dir: Optional[Path] = None
    db_path: Optional[Path] = None
    samples_dir: Optional[Path] = None
    samples_raw_dir: Optional[Path] = None
    samples_preprocessed_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    models_f5_dir: Optional[Path] = None
    models_rvc_dir: Optional[Path] = None
    generated_dir: Optional[Path] = None
    generated_tts_dir: Optional[Path] = None
    generated_sts_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

    # Allowed CORS Origins
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "app://-",
            "*",
        ]
    )

    def model_post_init(self, __context) -> None:
        """Resolve all derived directory paths after project_root is established."""
        if self.data_dir is None:
            custom_data_dir = os.getenv("VOICE_SAMPLER_DATA_DIR")
            self.data_dir = (
                Path(custom_data_dir).resolve()
                if custom_data_dir
                else (self.project_root / "data")
            )

        self.db_dir = self.data_dir / "db"
        custom_db_path = os.getenv("VOICE_SAMPLER_DB_PATH")
        self.db_path = (
            Path(custom_db_path).resolve()
            if custom_db_path
            else (self.db_dir / "voice_sampler.db")
        )

        self.samples_dir = self.data_dir / "samples"
        self.samples_raw_dir = self.samples_dir / "raw"
        self.samples_preprocessed_dir = self.samples_dir / "preprocessed"

        self.models_dir = self.data_dir / "models"
        self.models_f5_dir = self.models_dir / "f5_tts"
        self.models_rvc_dir = self.models_dir / "rvc"

        self.generated_dir = self.data_dir / "generated"
        self.generated_tts_dir = self.generated_dir / "tts"
        self.generated_sts_dir = self.generated_dir / "sts"

        self.cache_dir = self.data_dir / "cache"

    def ensure_directories(self) -> List[Path]:
        """
        Create all required data and model storage directories if they do not exist.
        Returns the list of confirmed directories.
        """
        dirs = [
            self.data_dir,
            self.db_dir,
            self.samples_raw_dir,
            self.samples_preprocessed_dir,
            self.models_dir,
            self.models_f5_dir,
            self.models_rvc_dir,
            self.generated_dir,
            self.generated_tts_dir,
            self.generated_sts_dir,
            self.cache_dir,
        ]
        for directory in dirs:
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
        return [d for d in dirs if d is not None]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()


# Module-level convenience references
_default_settings = get_settings()
PROJECT_ROOT: Path = _default_settings.project_root
DATA_DIR: Path = _default_settings.data_dir or (_default_settings.project_root / "data")
DB_DIR: Path = _default_settings.db_dir or (DATA_DIR / "db")
DB_PATH: Path = _default_settings.db_path or (DB_DIR / "voice_sampler.db")
