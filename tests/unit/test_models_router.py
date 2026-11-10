"""
tests/unit/test_models_router.py
Unit tests for the Models Registry API Router (/api/models):
- GET /api/models (empty listing, engine filtering, path formatting)
- GET /api/models/{name} (existing model metadata vs 404)
- DELETE /api/models/{name} (file unlinking, database removal vs 404)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient
import pytest

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.main import app


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Isolate storage directories and SQLite database per test."""
    test_db = tmp_path / "test_models_router.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.models_dir = tmp_path / "data" / "models"
    settings.cache_dir = tmp_path / "data" / "cache"
    settings.ensure_directories()

    yield

    set_db_path(None)


def _seed_model(
    name: str,
    engine: str = "f5-tts",
    create_disk_files: bool = True,
) -> Dict[str, Any]:
    """Helper to seed model in DB and optionally on disk."""
    settings = get_settings()
    model_dir = settings.models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    if engine == "f5-tts":
        ckpt_name = f"{name}_f5tts.pt"
        ckpt_path = model_dir / ckpt_name
        root_ckpt = settings.models_dir / ckpt_name
        index_path = None
        vocab_path = model_dir / "vocab.txt"
    else:
        ckpt_name = f"{name}.pth"
        ckpt_path = model_dir / ckpt_name
        root_ckpt = settings.models_dir / ckpt_name
        index_path = model_dir / f"{name}.index"
        root_index = settings.models_dir / f"{name}.index"
        vocab_path = None

    config_path = model_dir / "config.json"

    if create_disk_files:
        ckpt_path.write_bytes(b"MOCK_CHECKPOINT_CONTENT")
        root_ckpt.write_bytes(b"MOCK_CHECKPOINT_CONTENT")
        config_path.write_text('{"name": "' + name + '"}', encoding="utf-8")
        if vocab_path:
            vocab_path.write_text("a\nb\nc\n", encoding="utf-8")
        if index_path:
            index_path.write_bytes(b"MOCK_INDEX_CONTENT")
            root_index.write_bytes(b"MOCK_INDEX_CONTENT")

    record = repository.create_model(
        name=name,
        engine=engine,
        checkpoint_path=f"data/models/{ckpt_name}",
        index_path=f"data/models/{name}.index" if index_path else None,
        vocab_path=f"data/models/{name}/vocab.txt" if vocab_path else None,
        config_path=f"data/models/{name}/config.json",
        parameters={"epochs": 10},
        sample_count=2,
    )
    return record


# ==============================================================================
# Model Listing Tests
# ==============================================================================

def test_models_listing_empty():
    """Verify empty database returns {'models': []}."""
    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": []}


def test_models_listing_returns_all_registered():
    """Verify all registered models are listed with normalized paths."""
    client = TestClient(app)
    _seed_model("f5_alpha", engine="f5-tts")
    _seed_model("rvc_beta", engine="rvc")

    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) == 2

    names = [m["name"] for m in models]
    assert "f5_alpha" in names
    assert "rvc_beta" in names

    for m in models:
        assert m["checkpoint_path"].startswith("data/models/")


def test_models_listing_filtered_by_engine():
    """Verify ?engine= query parameter filters appropriately."""
    client = TestClient(app)
    _seed_model("f5_first", engine="f5-tts")
    _seed_model("rvc_second", engine="rvc")

    # Filter F5-TTS
    f5_resp = client.get("/api/models?engine=f5-tts")
    assert f5_resp.status_code == 200
    f5_models = f5_resp.json()["models"]
    assert len(f5_models) == 1
    assert f5_models[0]["engine"] == "f5-tts"
    assert f5_models[0]["name"] == "f5_first"

    # Filter RVC
    rvc_resp = client.get("/api/models?engine=rvc")
    assert rvc_resp.status_code == 200
    rvc_models = rvc_resp.json()["models"]
    assert len(rvc_models) == 1
    assert rvc_models[0]["engine"] == "rvc"
    assert rvc_models[0]["name"] == "rvc_second"


# ==============================================================================
# Model Get Details Tests
# ==============================================================================

def test_get_model_by_name_success():
    """Verify model details are retrieved by name."""
    client = TestClient(app)
    _seed_model("target_model", engine="f5-tts")

    resp = client.get("/api/models/target_model")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "target_model"
    assert data["engine"] == "f5-tts"
    assert "parameters" in data
    assert data["checkpoint_path"].startswith("data/models/")


def test_get_model_nonexistent_returns_404():
    """Verify non-existent model name returns HTTP 404."""
    client = TestClient(app)
    resp = client.get("/api/models/missing_model")
    assert resp.status_code == 404
    assert "Model 'missing_model' not found" in resp.json()["detail"]


# ==============================================================================
# Model Delete Tests
# ==============================================================================

def test_delete_model_removes_db_record_and_disk_artifacts():
    """Verify DELETE /api/models/{name} unlinks files and deletes DB record."""
    client = TestClient(app)
    settings = get_settings()
    _seed_model("to_delete", engine="rvc", create_disk_files=True)

    dedicated_dir = settings.models_dir / "to_delete"
    root_ckpt = settings.models_dir / "to_delete.pth"
    assert dedicated_dir.exists()
    assert root_ckpt.exists()

    del_resp = client.delete("/api/models/to_delete")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"
    assert del_resp.json()["name"] == "to_delete"

    # Verify removed from database
    assert repository.get_model_by_name("to_delete") is None

    # Verify removed from filesystem
    assert not dedicated_dir.exists()
    assert not root_ckpt.exists()


def test_delete_model_nonexistent_returns_404():
    """Verify deleting non-existent model returns HTTP 404."""
    client = TestClient(app)
    resp = client.delete("/api/models/ghost_model")
    assert resp.status_code == 404
    assert "Model 'ghost_model' not found" in resp.json()["detail"]
