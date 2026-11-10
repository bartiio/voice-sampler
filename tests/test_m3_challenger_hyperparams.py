"""
tests/test_m3_challenger_hyperparams.py
Challenger M3-2 Empirical Stress Test Harness.

Empirically challenges:
1. Boundary hyperparameter validation for F5-TTS and RVC training:
   - Negative, zero, and floating-point epochs
   - Negative and zero batch sizes
   - Negative and zero learning rates
   - Empty, whitespace, path-traversal, and special-character model names
   - Empty, non-existent, and malformed sample_ids
   - Unsupported RVC sample rates (22050, 44100, 16000, 96000, 8000, 0, -1) strictly returning 400
   - Concurrency lockout (HTTP 409 Conflict)
2. Model Registry CRUD, filtering, non-existent model handling (404),
   and complete deletion lifecycle cleanup:
   - Empty registry listing
   - Listing with ?engine=f5-tts and ?engine=rvc filtering
   - GET non-existent model (HTTP 404)
   - DELETE non-existent model (HTTP 404)
   - Complete deletion unlinking both SQLite record and all on-disk artifacts
   - Prefix collision stress testing: Deleting a base model must NOT destroy sibling models
     sharing the same prefix (e.g. deleting 'v1' must not unlink 'v10.pth' or 'v1_v2.pth').
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import uuid

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
def setup_challenger_test_env(tmp_path: Path):
    """
    Isolates SQLite database and filesystem storage to an ephemeral temporary directory
    for each individual test case, ensuring deterministic and hermetic verification.
    """
    test_db = tmp_path / "test_m3_challenger.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.models_dir = tmp_path / "data" / "models"
    settings.cache_dir = tmp_path / "data" / "cache"
    settings.samples_raw_dir = tmp_path / "data" / "samples" / "raw"
    settings.ensure_directories()

    # Reset any ongoing service background state
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


def _seed_valid_sample(name: str = "valid_sample.wav", duration: float = 1.0) -> str:
    """Helper to seed a valid audio file on disk and sample record in SQLite."""
    settings = get_settings()
    path = settings.samples_raw_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(path), audio, sr, subtype="PCM_16")

    sample = repository.create_sample(
        name=name,
        filename=name,
        file_path=str(path),
        transcript="Testowy polski zapis mowy.",
    )
    return str(sample["id"])


def _seed_model_artifacts(
    name: str,
    engine: str = "f5-tts",
) -> Dict[str, Any]:
    """Helper to create full simulated model artifacts on disk and register in SQLite."""
    settings = get_settings()
    model_dir = settings.models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    if engine == "f5-tts":
        ckpt_name = f"{name}_f5tts.pt"
        ckpt_path = model_dir / ckpt_name
        root_ckpt = settings.models_dir / ckpt_name
        config_path = model_dir / "config.json"
        root_config = settings.models_dir / f"{name}_config.json"
        vocab_path = model_dir / "vocab.txt"

        ckpt_path.write_bytes(b"MOCK_F5TTS_CKPT")
        root_ckpt.write_bytes(b"MOCK_F5TTS_CKPT")
        config_path.write_text(json.dumps({"name": name, "engine": "f5-tts"}), encoding="utf-8")
        root_config.write_text(json.dumps({"name": name, "engine": "f5-tts"}), encoding="utf-8")
        vocab_path.write_text("<pad>\n<unk>\na\nb\nc\n", encoding="utf-8")

        record = repository.create_model(
            name=name,
            engine=engine,
            checkpoint_path=f"data/models/{ckpt_name}",
            vocab_path=f"data/models/{name}/vocab.txt",
            config_path=f"data/models/{name}/config.json",
            parameters={"epochs": 10, "batch_size": 2},
            sample_count=1,
        )
    else:  # rvc
        ckpt_name = f"{name}.pth"
        index_name = f"{name}.index"
        ckpt_path = model_dir / ckpt_name
        root_ckpt = settings.models_dir / ckpt_name
        index_path = model_dir / index_name
        root_index = settings.models_dir / index_name
        config_path = model_dir / "config.json"
        root_config = settings.models_dir / f"{name}_config.json"

        ckpt_path.write_bytes(b"MOCK_RVC_PTH")
        root_ckpt.write_bytes(b"MOCK_RVC_PTH")
        index_path.write_bytes(b"MOCK_RVC_INDEX")
        root_index.write_bytes(b"MOCK_RVC_INDEX")
        config_path.write_text(json.dumps({"name": name, "engine": "rvc"}), encoding="utf-8")
        root_config.write_text(json.dumps({"name": name, "engine": "rvc"}), encoding="utf-8")

        record = repository.create_model(
            name=name,
            engine=engine,
            checkpoint_path=f"data/models/{ckpt_name}",
            index_path=f"data/models/{index_name}",
            config_path=f"data/models/{name}/config.json",
            parameters={"epochs": 20, "sample_rate": 40000},
            sample_count=1,
        )

    return record


# ==============================================================================
# 1. Hyperparameter Validation Extremes (F5-TTS)
# ==============================================================================

@pytest.mark.parametrize("bad_epoch", [-5, -1, 0])
def test_f5tts_rejects_non_positive_epochs(bad_epoch: int):
    """Verify non-positive epochs return HTTP 400 Bad Request."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "test_voice", "epochs": bad_epoch, "sample_ids": [sid]},
    )
    assert resp.status_code == 400
    assert "Epochs must be >= 1" in resp.json()["detail"]


def test_f5tts_rejects_fractional_floating_point_epochs():
    """Verify fractional/floating-point epochs return HTTP 422 Unprocessable Content."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "test_voice", "epochs": 3.5, "sample_ids": [sid]},
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("bad_batch", [-10, -1, 0])
def test_f5tts_rejects_non_positive_batch_size(bad_batch: int):
    """Verify non-positive batch size returns HTTP 400 Bad Request."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "test_voice", "batch_size": bad_batch, "sample_ids": [sid]},
    )
    assert resp.status_code == 400
    assert "Batch size must be >= 1" in resp.json()["detail"]


@pytest.mark.parametrize("bad_lr", [-0.001, -1.0, 0.0, 0])
def test_f5tts_rejects_non_positive_learning_rate(bad_lr: float):
    """Verify non-positive learning rates return HTTP 400 Bad Request."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "test_voice", "learning_rate": bad_lr, "sample_ids": [sid]},
    )
    assert resp.status_code == 400
    assert "Learning rate must be > 0" in resp.json()["detail"]


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "   ",
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "models/../../traversal",
        "voice name with spaces",
        "voice;rm -rf /",
        "voice|malicious",
        "voice@host#1",
        "voice$secret",
        "voice\x00nullbyte",
        "model_żółw",  # Non-ASCII Polish unicode rejected by strict ASCII alphanumeric regex
    ],
)
def test_f5tts_rejects_invalid_or_traversal_model_names(bad_name: str):
    """Verify empty, path-traversal, or illegal character model names return HTTP 400."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": bad_name, "epochs": 5, "sample_ids": [sid]},
    )
    assert resp.status_code == 400


def test_f5tts_rejects_empty_sample_ids_list():
    """Verify empty list of sample_ids strictly returns HTTP 400."""
    client = TestClient(app)
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "voice_alpha", "epochs": 5, "sample_ids": []},
    )
    assert resp.status_code == 400
    assert "sample_ids list cannot be empty" in resp.json()["detail"]


@pytest.mark.parametrize("missing_id", ["999999", "00000000-0000-0000-0000-000000000000", "ghost-uuid"])
def test_f5tts_rejects_nonexistent_sample_ids(missing_id: str):
    """Verify non-existent sample IDs in sample_ids list return HTTP 400."""
    client = TestClient(app)
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "voice_alpha", "epochs": 5, "sample_ids": [missing_id]},
    )
    assert resp.status_code == 400
    assert f"Sample id {missing_id} not found" in resp.json()["detail"]


def test_f5tts_rejects_start_when_no_samples_available_and_none_specified():
    """Verify training initiation fails with 400 if DB has no samples and sample_ids is None."""
    client = TestClient(app)
    # Database is empty (no samples seeded)
    resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "voice_alpha", "epochs": 5},
    )
    assert resp.status_code == 400
    assert "No audio samples available for training" in resp.json()["detail"]


# ==============================================================================
# 2. Hyperparameter Validation Extremes (RVC v2)
# ==============================================================================

@pytest.mark.parametrize("unsupported_sr", [22050, 44100, 16000, 96000, 8000, 11025, 0, -1])
def test_rvc_strictly_rejects_unsupported_sample_rates(unsupported_sr: int):
    """Verify RVC rejects all sample rates other than 32000, 40000, and 48000 with HTTP 400."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/rvc",
        json={"model_name": "bad_sr_rvc", "sample_rate": unsupported_sr, "sample_ids": [sid]},
    )
    assert resp.status_code == 400
    assert "Sample rate must be 32k, 40k, or 48k" in resp.json()["detail"]


@pytest.mark.parametrize("valid_sr", [32000, 40000, 48000])
def test_rvc_accepts_valid_sample_rates(valid_sr: int):
    """Verify RVC accepts 32000, 40000, and 48000 successfully returning 200."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    client.post("/api/train/stop")

    resp = client.post(
        "/api/train/rvc",
        json={"model_name": f"rvc_sr_{valid_sr}", "epochs": 2, "sample_rate": valid_sr, "sample_ids": [sid]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert data["engine"] == "rvc"
    client.post("/api/train/stop")


@pytest.mark.parametrize("bad_epoch", [-5, 0])
def test_rvc_rejects_invalid_epochs(bad_epoch: int):
    """Verify RVC rejects invalid epochs with HTTP 400."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/rvc",
        json={"model_name": "bad_epoch_rvc", "epochs": bad_epoch, "sample_ids": [sid]},
    )
    assert resp.status_code == 400
    assert "Epochs must be >= 1" in resp.json()["detail"]


@pytest.mark.parametrize("bad_batch", [-1, 0])
def test_rvc_rejects_invalid_batch_size(bad_batch: int):
    """Verify RVC rejects invalid batch size with HTTP 400."""
    client = TestClient(app)
    sid = _seed_valid_sample()
    resp = client.post(
        "/api/train/rvc",
        json={"model_name": "bad_batch_rvc", "batch_size": bad_batch, "sample_ids": [sid]},
    )
    assert resp.status_code == 400
    assert "Batch size must be >= 1" in resp.json()["detail"]


def test_training_concurrency_lockout():
    """Verify that starting a training run while one is already active returns HTTP 409 Conflict."""
    client = TestClient(app)
    sid = _seed_valid_sample()

    resp1 = client.post(
        "/api/train/f5-tts",
        json={"model_name": "run_primary", "epochs": 50, "sample_ids": [sid]},
    )
    assert resp1.status_code == 200

    # Second concurrent attempt must fail with 409 Conflict
    resp2 = client.post(
        "/api/train/rvc",
        json={"model_name": "run_secondary", "epochs": 10, "sample_ids": [sid]},
    )
    assert resp2.status_code == 409
    assert "already in progress" in resp2.json()["detail"]

    # Cancel first run and verify second run can now be queued
    client.post("/api/train/stop")
    resp3 = client.post(
        "/api/train/rvc",
        json={"model_name": "run_secondary", "epochs": 2, "sample_ids": [sid]},
    )
    assert resp3.status_code == 200
    client.post("/api/train/stop")


# ==============================================================================
# 3. Model Registry CRUD, Filtering & Non-Existent Models (404)
# ==============================================================================

def test_model_registry_empty_listing():
    """Verify empty model registry returns {'models': []} with HTTP 200."""
    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": []}


def test_model_registry_crud_and_engine_filtering():
    """Verify model listing and engine filtering (?engine=f5-tts and ?engine=rvc)."""
    client = TestClient(app)
    _seed_model_artifacts("model_f5_1", engine="f5-tts")
    _seed_model_artifacts("model_f5_2", engine="f5-tts")
    _seed_model_artifacts("model_rvc_1", engine="rvc")
    _seed_model_artifacts("model_rvc_2", engine="rvc")

    # All models
    all_resp = client.get("/api/models")
    assert all_resp.status_code == 200
    all_models = all_resp.json()["models"]
    assert len(all_models) == 4
    for m in all_models:
        assert m["checkpoint_path"].startswith("data/models/")

    # Filter F5-TTS
    f5_resp = client.get("/api/models?engine=f5-tts")
    assert f5_resp.status_code == 200
    f5_models = f5_resp.json()["models"]
    assert len(f5_models) == 2
    assert all(m["engine"] == "f5-tts" for m in f5_models)

    # Filter RVC
    rvc_resp = client.get("/api/models?engine=rvc")
    assert rvc_resp.status_code == 200
    rvc_models = rvc_resp.json()["models"]
    assert len(rvc_models) == 2
    assert all(m["engine"] == "rvc" for m in rvc_models)

    # Filter non-existent engine returns empty list
    ghost_resp = client.get("/api/models?engine=nonexistent_engine")
    assert ghost_resp.status_code == 200
    assert ghost_resp.json() == {"models": []}


def test_model_get_by_name_and_nonexistent_returns_404():
    """Verify GET /api/models/{name} returns model metadata or 404 if missing."""
    client = TestClient(app)
    _seed_model_artifacts("my_voice_model", engine="f5-tts")

    # Existing model
    resp = client.get("/api/models/my_voice_model")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "my_voice_model"
    assert data["engine"] == "f5-tts"
    assert "parameters" in data
    assert data["checkpoint_path"].startswith("data/models/")

    # Non-existent model
    missing_resp = client.get("/api/models/ghost_model")
    assert missing_resp.status_code == 404
    assert "Model 'ghost_model' not found" in missing_resp.json()["detail"]


def test_model_delete_nonexistent_returns_404():
    """Verify DELETE /api/models/{name} on a non-existent model returns HTTP 404."""
    client = TestClient(app)
    resp = client.delete("/api/models/ghost_model")
    assert resp.status_code == 404
    assert "Model 'ghost_model' not found" in resp.json()["detail"]


# ==============================================================================
# 4. Model Deletion Lifecycle & Prefix-Collision Stress Test
# ==============================================================================

def test_model_deletion_complete_cleanup_f5tts():
    """
    Verify DELETE /api/models/{name} completely removes SQLite records AND
    unlinks all disk artifacts (dedicated directory and root files).
    """
    client = TestClient(app)
    settings = get_settings()
    _seed_model_artifacts("f5_target", engine="f5-tts")

    dedicated_dir = settings.models_dir / "f5_target"
    root_ckpt = settings.models_dir / "f5_target_f5tts.pt"
    root_cfg = settings.models_dir / "f5_target_config.json"

    assert dedicated_dir.exists()
    assert root_ckpt.exists()
    assert root_cfg.exists()
    assert repository.get_model_by_name("f5_target") is not None

    del_resp = client.delete("/api/models/f5_target")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"
    assert del_resp.json()["name"] == "f5_target"

    # SQLite record must be gone
    assert repository.get_model_by_name("f5_target") is None

    # Disk artifacts must be unlinked
    assert not dedicated_dir.exists()
    assert not root_ckpt.exists()
    assert not root_cfg.exists()

    # Immediate retrieval must return 404
    assert client.get("/api/models/f5_target").status_code == 404
    # Immediate second deletion must return 404
    assert client.delete("/api/models/f5_target").status_code == 404


def test_model_deletion_complete_cleanup_rvc():
    """
    Verify DELETE /api/models/{name} completely removes RVC SQLite records AND
    unlinks all disk artifacts (checkpoint, index, config, and directories).
    """
    client = TestClient(app)
    settings = get_settings()
    _seed_model_artifacts("rvc_target", engine="rvc")

    dedicated_dir = settings.models_dir / "rvc_target"
    root_ckpt = settings.models_dir / "rvc_target.pth"
    root_idx = settings.models_dir / "rvc_target.index"
    root_cfg = settings.models_dir / "rvc_target_config.json"

    assert dedicated_dir.exists()
    assert root_ckpt.exists()
    assert root_idx.exists()
    assert root_cfg.exists()
    assert repository.get_model_by_name("rvc_target") is not None

    del_resp = client.delete("/api/models/rvc_target")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # SQLite record must be gone
    assert repository.get_model_by_name("rvc_target") is None

    # Disk artifacts must be unlinked
    assert not dedicated_dir.exists()
    assert not root_ckpt.exists()
    assert not root_idx.exists()
    assert not root_cfg.exists()

    # Immediate retrieval must return 404
    assert client.get("/api/models/rvc_target").status_code == 404
    # Immediate second deletion must return 404
    assert client.delete("/api/models/rvc_target").status_code == 404


def test_model_deletion_prefix_collision_resilience():
    """
    ADVERSARIAL STRESS TEST:
    Verifies that deleting a model with name 'v1' does NOT delete artifacts of
    sibling models whose names start with 'v1' (e.g. 'v10', 'v1_v2').

    Bug signature: backend/app/routers/models.py:171 used `glob(f"{name}*")`,
    causing `v1*` to match and destroy `v10.pth`, `v1_v2.pth`, etc.
    """
    client = TestClient(app)
    settings = get_settings()

    # Seed base model 'v1'
    _seed_model_artifacts("v1", engine="rvc")
    # Seed sibling model 'v10' sharing prefix 'v1'
    _seed_model_artifacts("v10", engine="rvc")
    # Seed sibling model 'v1_v2' sharing prefix 'v1'
    _seed_model_artifacts("v1_v2", engine="rvc")

    v1_ckpt = settings.models_dir / "v1.pth"
    v10_ckpt = settings.models_dir / "v10.pth"
    v1_v2_ckpt = settings.models_dir / "v1_v2.pth"

    assert v1_ckpt.exists(), "v1.pth must exist before deletion"
    assert v10_ckpt.exists(), "v10.pth must exist before deletion"
    assert v1_v2_ckpt.exists(), "v1_v2.pth must exist before deletion"

    # Delete the base model 'v1'
    del_resp = client.delete("/api/models/v1")
    assert del_resp.status_code == 200

    # Base model 'v1' must be deleted
    assert repository.get_model_by_name("v1") is None
    assert not v1_ckpt.exists(), "v1.pth must be deleted"

    # CRITICAL CHALLENGE ASSERTION:
    # Sibling models 'v10' and 'v1_v2' MUST retain their database records AND disk files!
    assert repository.get_model_by_name("v10") is not None, "Model v10 database record must not be deleted"
    assert repository.get_model_by_name("v1_v2") is not None, "Model v1_v2 database record must not be deleted"

    assert v10_ckpt.exists(), (
        "CRITICAL DEFECT: Deleting model 'v1' unlinked checkpoint of model 'v10' "
        "(data/models/v10.pth) due to overbroad glob prefix matching!"
    )
    assert v1_v2_ckpt.exists(), (
        "CRITICAL DEFECT: Deleting model 'v1' unlinked checkpoint of model 'v1_v2' "
        "(data/models/v1_v2.pth) due to overbroad glob prefix matching!"
    )
