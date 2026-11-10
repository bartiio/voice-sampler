"""
Unit tests for backend configuration and directory management.
"""

import os
from pathlib import Path
from unittest.mock import patch

from backend.app.config import Settings, get_settings


def test_default_settings_paths():
    settings = get_settings()
    assert settings.app_name == "Voice Sampler API"
    assert settings.app_version == "0.1.0"
    assert settings.project_root.exists()
    assert settings.data_dir is not None
    assert settings.db_dir is not None
    assert settings.db_path is not None
    assert settings.samples_raw_dir is not None
    assert settings.models_f5_dir is not None
    assert settings.generated_tts_dir is not None


def test_settings_env_overrides(tmp_path):
    custom_root = tmp_path / "custom_root"
    custom_root.mkdir()
    custom_data = tmp_path / "custom_data"

    with patch.dict(
        os.environ,
        {
            "VOICE_SAMPLER_HOST": "0.0.0.0",
            "VOICE_SAMPLER_PORT": "9000",
            "VOICE_SAMPLER_DEV_MODE": "1",
            "VOICE_SAMPLER_ROOT": str(custom_root),
            "VOICE_SAMPLER_DATA_DIR": str(custom_data),
        },
    ):
        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.dev_mode is True
        assert settings.project_root == custom_root
        assert settings.data_dir == custom_data
        assert settings.db_dir == custom_data / "db"


def test_ensure_directories(tmp_path):
    custom_data = tmp_path / "app_data"
    settings = Settings(data_dir=custom_data)
    dirs = settings.ensure_directories()

    assert custom_data.exists()
    assert (custom_data / "db").exists()
    assert (custom_data / "samples" / "raw").exists()
    assert (custom_data / "samples" / "preprocessed").exists()
    assert (custom_data / "models" / "f5_tts").exists()
    assert (custom_data / "models" / "rvc").exists()
    assert (custom_data / "generated" / "tts").exists()
    assert (custom_data / "generated" / "sts").exists()
    assert (custom_data / "cache").exists()
    assert len(dirs) >= 10
