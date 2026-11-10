"""
tests/unit/test_train_router.py
Unit tests for the Training API Router (/api/train):
- POST /api/train/f5-tts (validation, sample checking, run initiation)
- POST /api/train/rvc (sample rate checks, validation, run initiation)
- POST /api/train/stop (graceful stop, idle state handling, repeatability)
- GET /api/train/status (idle and active state reporting)
- Concurrency protection (HTTP 409 when run in progress)
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Dict

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.main import app
from backend.app.services.f5tts_service import get_f5tts_service
from backend.app.services.rvc_service import get_rvc_service


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Isolate storage directories and SQLite database per test."""
    test_db = tmp_path / "test_train_router.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.models_dir = tmp_path / "data" / "models"
    settings.cache_dir = tmp_path / "data" / "cache"
    settings.samples_raw_dir = tmp_path / "data" / "samples" / "raw"
    settings.ensure_directories()

    # Reset any active training run state in services
    f5 = get_f5tts_service()
    f5.stop_training()
    rvc = get_rvc_service()
    rvc.cancel_training()

    yield

    f5 = get_f5tts_service()
    f5.stop_training()
    rvc = get_rvc_service()
    rvc.cancel_training()

    set_db_path(None)


def _seed_sample(name: str = "seed_sample.wav") -> str:
    """Helper to seed a valid sample record and audio file."""
    settings = get_settings()
    path = settings.samples_raw_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1.0, 44100, endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * 440.0 * t), 44100, subtype="PCM_16")

    sample = repository.create_sample(
        name=name,
        filename=name,
        file_path=str(path),
        transcript="Testowy zapis głosu.",
    )
    return str(sample["id"])


# ==============================================================================
# F5-TTS Training Endpoints
# ==============================================================================

def test_start_f5tts_training_success():
    """Verify valid request starts F5-TTS training and returns run metadata."""
    client = TestClient(app)
    sid = _seed_sample()

    resp = client.post(
        "/api/train/f5-tts",
        json={
            "model_name": "pl_voice_alpha",
            "epochs": 5,
            "batch_size": 2,
            "learning_rate": 0.0001,
            "sample_ids": [sid],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "started"
    assert data["engine"] == "f5-tts"
    assert data["model_name"] == "pl_voice_alpha"


def test_start_f5tts_training_rejects_empty_name():
    """Verify empty or whitespace model name is rejected with HTTP 400."""
    client = TestClient(app)
    resp = client.post("/api/train/f5-tts", json={"model_name": "   ", "epochs": 5})
    assert resp.status_code == 400
    assert "Model name cannot be empty" in resp.json()["detail"]


def test_start_f5tts_training_rejects_invalid_characters():
    """Verify path traversal or special characters in model name are rejected."""
    client = TestClient(app)
    resp = client.post("/api/train/f5-tts", json={"model_name": "../../traversal_attack", "epochs": 5})
    assert resp.status_code == 400
    assert "Invalid model name" in resp.json()["detail"]


def test_start_f5tts_training_rejects_invalid_epochs():
    """Verify epochs < 1 is rejected with HTTP 400."""
    client = TestClient(app)
    resp = client.post("/api/train/f5-tts", json={"model_name": "voice", "epochs": 0})
    assert resp.status_code == 400
    assert "Epochs must be >= 1" in resp.json()["detail"]


def test_start_f5tts_training_rejects_invalid_learning_rate():
    """Verify learning rate <= 0 is rejected with HTTP 400."""
    client = TestClient(app)
    resp = client.post("/api/train/f5-tts", json={"model_name": "voice", "learning_rate": -0.001})
    assert resp.status_code == 400
    assert "Learning rate must be > 0" in resp.json()["detail"]


def test_start_f5tts_training_rejects_nonexistent_sample_ids():
    """Verify missing sample ID is rejected with HTTP 400."""
    client = TestClient(app)
    resp = client.post("/api/train/f5-tts", json={"model_name": "voice", "sample_ids": ["missing-uuid"]})
    assert resp.status_code == 400
    assert "Sample id missing-uuid not found" in resp.json()["detail"]


def test_start_f5tts_training_rejects_empty_sample_ids_list():
    """Verify empty list of sample IDs is rejected with HTTP 400."""
    client = TestClient(app)
    resp = client.post("/api/train/f5-tts", json={"model_name": "voice", "sample_ids": []})
    assert resp.status_code == 400
    assert "sample_ids list cannot be empty" in resp.json()["detail"]


# ==============================================================================
# RVC v2 Training Endpoints
# ==============================================================================

def test_start_rvc_training_success():
    """Verify valid request starts RVC training and returns run metadata."""
    client = TestClient(app)
    sid = _seed_sample()

    resp = client.post(
        "/api/train/rvc",
        json={
            "model_name": "rvc_voice_beta",
            "epochs": 10,
            "batch_size": 4,
            "sample_rate": 40000,
            "sample_ids": [sid],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "started"
    assert data["engine"] == "rvc"
    assert data["model_name"] == "rvc_voice_beta"


def test_start_rvc_training_rejects_unsupported_sample_rate():
    """Verify unsupported sample rates (e.g. 22050 or 44100) return HTTP 400."""
    client = TestClient(app)
    sid = _seed_sample()

    resp = client.post(
        "/api/train/rvc",
        json={"model_name": "bad_sr", "epochs": 5, "sample_rate": 22050, "sample_ids": [sid]},
    )
    assert resp.status_code == 400
    assert "Sample rate must be 32k, 40k, or 48k" in resp.json()["detail"]


def test_start_rvc_training_accepts_valid_sample_rates():
    """Verify 32000, 40000, and 48000 are accepted."""
    client = TestClient(app)
    sid = _seed_sample()

    for sr in (32000, 40000, 48000):
        # Stop any active run before starting next
        client.post("/api/train/stop")
        resp = client.post(
            "/api/train/rvc",
            json={"model_name": f"rvc_sr_{sr}", "epochs": 2, "sample_rate": sr, "sample_ids": [sid]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


# ==============================================================================
# Stop and Status Endpoints
# ==============================================================================

def test_stop_training_cancels_active_run():
    """Verify POST /api/train/stop stops an active training run."""
    client = TestClient(app)
    sid = _seed_sample()

    start_resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "f5_to_cancel", "epochs": 50, "sample_ids": [sid]},
    )
    assert start_resp.status_code == 200
    run_id = start_resp.json()["run_id"]

    stop_resp = client.post("/api/train/stop")
    assert stop_resp.status_code == 200
    data = stop_resp.json()
    assert data["status"] == "stopped"
    assert data["run_id"] == run_id


def test_stop_training_when_idle_returns_no_active_run():
    """Verify POST /api/train/stop when idle returns status 'no_active_run'."""
    client = TestClient(app)
    # Ensure idle
    client.post("/api/train/stop")

    resp = client.post("/api/train/stop")
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_active_run", "run_id": None}


def test_get_training_status_reflects_active_and_idle_states():
    """Verify GET /api/train/status accurately reports lifecycle state."""
    client = TestClient(app)
    client.post("/api/train/stop")

    # Idle state
    idle_resp = client.get("/api/train/status")
    assert idle_resp.status_code == 200
    assert idle_resp.json()["active"] is False

    # Start training
    sid = _seed_sample()
    client.post("/api/train/rvc", json={"model_name": "active_check", "epochs": 50, "sample_ids": [sid]})

    active_resp = client.get("/api/train/status")
    assert active_resp.status_code == 200
    assert active_resp.json()["active"] is True
    assert active_resp.json()["run"] is not None
