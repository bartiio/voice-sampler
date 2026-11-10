"""
tests/test_m3_challenger_m3_4_collision_stress.py
Empirical stress suite for Challenger M3-4:
Verifies model deletion prefix collision resilience and comprehensive artifact preservation.
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.main import app


@pytest.fixture(autouse=True)
def clean_db_and_models():
    """Ensure clean state before and after each test."""
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup any models created during tests
    for m in repository.list_models():
        repository.delete_model(str(m["id"]))


def _seed_complete_rvc_model(name: str):
    """Seed full RVC model on disk (both dedicated dir and root aliases) and DB."""
    settings = get_settings()
    mdir = settings.models_dir / name
    mdir.mkdir(parents=True, exist_ok=True)

    # Root flat convenience files
    root_ckpt = settings.models_dir / f"{name}.pth"
    root_idx = settings.models_dir / f"{name}.index"
    root_cfg = settings.models_dir / f"{name}_config.json"

    root_ckpt.write_bytes(f"CKPT_{name}".encode("utf-8"))
    root_idx.write_bytes(f"INDEX_{name}".encode("utf-8"))
    root_cfg.write_text(json.dumps({"name": name, "engine": "rvc"}), encoding="utf-8")

    # Dedicated dir files
    dir_ckpt = mdir / f"{name}.pth"
    dir_idx = mdir / f"{name}.index"
    dir_cfg = mdir / "config.json"

    dir_ckpt.write_bytes(f"CKPT_{name}".encode("utf-8"))
    dir_idx.write_bytes(f"INDEX_{name}".encode("utf-8"))
    dir_cfg.write_text(json.dumps({"name": name, "engine": "rvc"}), encoding="utf-8")

    record = repository.create_model(
        name=name,
        engine="rvc",
        checkpoint_path=f"data/models/{name}.pth",
        index_path=f"data/models/{name}.index",
        config_path=f"data/models/{name}/config.json",
        parameters={"epochs": 10, "sample_rate": 40000},
        sample_count=1,
    )
    return record


def _seed_complete_f5tts_model(name: str):
    """Seed full F5-TTS model on disk (both dedicated dir and root aliases) and DB."""
    settings = get_settings()
    mdir = settings.models_dir / name
    mdir.mkdir(parents=True, exist_ok=True)

    # Root flat convenience files
    root_ckpt = settings.models_dir / f"{name}_f5tts.pt"
    root_cfg = settings.models_dir / f"{name}_config.json"
    root_vocab = settings.models_dir / f"{name}_vocab.txt"

    root_ckpt.write_bytes(f"F5CKPT_{name}".encode("utf-8"))
    root_cfg.write_text(json.dumps({"name": name, "engine": "f5-tts"}), encoding="utf-8")
    root_vocab.write_text("<pad>\n<unk>\n", encoding="utf-8")

    # Dedicated dir files
    dir_ckpt = mdir / f"{name}_f5tts.pt"
    dir_cfg = mdir / "config.json"
    dir_vocab = mdir / "vocab.txt"

    dir_ckpt.write_bytes(f"F5CKPT_{name}".encode("utf-8"))
    dir_cfg.write_text(json.dumps({"name": name, "engine": "f5-tts"}), encoding="utf-8")
    dir_vocab.write_text("<pad>\n<unk>\n", encoding="utf-8")

    record = repository.create_model(
        name=name,
        engine="f5-tts",
        checkpoint_path=f"data/models/{name}_f5tts.pt",
        vocab_path=f"data/models/{name}/vocab.txt",
        config_path=f"data/models/{name}/config.json",
        parameters={"epochs": 5, "batch_size": 2},
        sample_count=1,
    )
    return record


def test_rvc_prefix_collision_comprehensive_artifact_preservation():
    """
    CRITICAL EMPIRICAL CHALLENGE:
    Deleting model 'v1' must strictly delete all 'v1' artifacts, while preserving:
    - v10.pth, v10.index, v10_config.json, data/models/v10/
    - v1_v2.pth, v1_v2.index, v1_v2_config.json, data/models/v1_v2/
    - v100.pth, v100.index, v100_config.json, data/models/v100/
    - v1_alpha.pth, v1_alpha.index, v1_alpha_config.json, data/models/v1_alpha/
    - v1-beta.pth, v1-beta.index, v1-beta_config.json, data/models/v1-beta/
    """
    client = TestClient(app)
    settings = get_settings()

    target = "v1"
    siblings = ["v10", "v1_v2", "v100", "v1_alpha", "v1-beta"]

    _seed_complete_rvc_model(target)
    for s in siblings:
        _seed_complete_rvc_model(s)

    # 1. Verify all exist before deletion
    for model_name in [target] + siblings:
        assert (settings.models_dir / f"{model_name}.pth").exists()
        assert (settings.models_dir / f"{model_name}.index").exists()
        assert (settings.models_dir / f"{model_name}_config.json").exists()
        assert (settings.models_dir / model_name).is_dir()
        assert (settings.models_dir / model_name / f"{model_name}.pth").exists()
        assert (settings.models_dir / model_name / f"{model_name}.index").exists()
        assert (settings.models_dir / model_name / "config.json").exists()
        assert repository.get_model_by_name(model_name) is not None

    # 2. Delete base model 'v1'
    del_resp = client.delete(f"/api/models/{target}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"status": "deleted", "name": target, "id": del_resp.json()["id"]}

    # 3. Verify 'v1' is completely deleted
    assert repository.get_model_by_name(target) is None
    assert not (settings.models_dir / f"{target}.pth").exists()
    assert not (settings.models_dir / f"{target}.index").exists()
    assert not (settings.models_dir / f"{target}_config.json").exists()
    assert not (settings.models_dir / target).exists()

    # 4. Verify ALL siblings are 100% intact
    for s in siblings:
        # DB Record
        rec = repository.get_model_by_name(s)
        assert rec is not None, f"Sibling model '{s}' record was wrongly deleted from DB!"

        # Root flat convenience files
        root_ckpt = settings.models_dir / f"{s}.pth"
        root_idx = settings.models_dir / f"{s}.index"
        root_cfg = settings.models_dir / f"{s}_config.json"

        assert root_ckpt.exists(), f"Sibling checkpoint '{root_ckpt}' was destroyed by deleting '{target}'!"
        assert root_idx.exists(), f"Sibling index '{root_idx}' was destroyed by deleting '{target}'!"
        assert root_cfg.exists(), f"Sibling config '{root_cfg}' was destroyed by deleting '{target}'!"

        # Dedicated directory & contents
        mdir = settings.models_dir / s
        assert mdir.is_dir(), f"Sibling directory '{mdir}' was destroyed by deleting '{target}'!"
        assert (mdir / f"{s}.pth").exists()
        assert (mdir / f"{s}.index").exists()
        assert (mdir / "config.json").exists()

    # 5. Clean up siblings via API to verify each deletion independently
    for s in siblings:
        res = client.delete(f"/api/models/{s}")
        assert res.status_code == 200
        assert repository.get_model_by_name(s) is None
        assert not (settings.models_dir / f"{s}.pth").exists()


def test_f5tts_prefix_collision_comprehensive_artifact_preservation():
    """
    CRITICAL EMPIRICAL CHALLENGE:
    Deleting model 'f5_v1' must preserve sibling F5-TTS models:
    - f5_v10 (f5_v10_f5tts.pt, f5_v10_config.json, f5_v10_vocab.txt, data/models/f5_v10/)
    - f5_v1_v2 (f5_v1_v2_f5tts.pt, f5_v1_v2_config.json, f5_v1_v2_vocab.txt, data/models/f5_v1_v2/)
    """
    client = TestClient(app)
    settings = get_settings()

    target = "f5_v1"
    siblings = ["f5_v10", "f5_v1_v2"]

    _seed_complete_f5tts_model(target)
    for s in siblings:
        _seed_complete_f5tts_model(s)

    # Delete base model 'f5_v1'
    del_resp = client.delete(f"/api/models/{target}")
    assert del_resp.status_code == 200

    # Target is gone
    assert repository.get_model_by_name(target) is None
    assert not (settings.models_dir / f"{target}_f5tts.pt").exists()
    assert not (settings.models_dir / f"{target}_config.json").exists()
    assert not (settings.models_dir / f"{target}_vocab.txt").exists()
    assert not (settings.models_dir / target).exists()

    # Siblings are intact
    for s in siblings:
        assert repository.get_model_by_name(s) is not None
        assert (settings.models_dir / f"{s}_f5tts.pt").exists(), f"Sibling F5 checkpoint {s} destroyed!"
        assert (settings.models_dir / f"{s}_config.json").exists(), f"Sibling F5 config {s} destroyed!"
        assert (settings.models_dir / f"{s}_vocab.txt").exists(), f"Sibling F5 vocab {s} destroyed!"
        assert (settings.models_dir / s).is_dir(), f"Sibling F5 directory {s} destroyed!"

        # Cleanup
        client.delete(f"/api/models/{s}")


def test_inter_engine_prefix_collision_resilience():
    """
    Test prefix collisions where sibling models use different engines:
    e.g. 'v1' (RVC) vs 'v10' (F5-TTS) vs 'v1_sub' (RVC).
    Deleting 'v1' (RVC) must preserve 'v10' (F5-TTS) and 'v1_sub' (RVC).
    """
    client = TestClient(app)
    settings = get_settings()

    _seed_complete_rvc_model("engine_v1")
    _seed_complete_f5tts_model("engine_v10")
    _seed_complete_rvc_model("engine_v1_sub")

    del_resp = client.delete("/api/models/engine_v1")
    assert del_resp.status_code == 200

    # RVC base deleted
    assert repository.get_model_by_name("engine_v1") is None
    assert not (settings.models_dir / "engine_v1.pth").exists()
    assert not (settings.models_dir / "engine_v1.index").exists()

    # F5-TTS sibling preserved
    assert repository.get_model_by_name("engine_v10") is not None
    assert (settings.models_dir / "engine_v10_f5tts.pt").exists()
    assert (settings.models_dir / "engine_v10_config.json").exists()
    assert (settings.models_dir / "engine_v10_vocab.txt").exists()
    assert (settings.models_dir / "engine_v10").is_dir()

    # RVC sibling preserved
    assert repository.get_model_by_name("engine_v1_sub") is not None
    assert (settings.models_dir / "engine_v1_sub.pth").exists()
    assert (settings.models_dir / "engine_v1_sub.index").exists()
    assert (settings.models_dir / "engine_v1_sub_config.json").exists()
    assert (settings.models_dir / "engine_v1_sub").is_dir()

    # Cleanup
    client.delete("/api/models/engine_v10")
    client.delete("/api/models/engine_v1_sub")
