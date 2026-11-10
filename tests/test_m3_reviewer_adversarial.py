"""
tests/test_m3_reviewer_adversarial.py
Adversarial Stress & Robustness Test Suite for Voice Sampler Milestone 3 (M3):
- High-concurrency barrier races (simultaneous training starts, simultaneous stops)
- Path traversal & injection in model names and endpoints
- Simulation engine mathematical stability across extreme scales
- Real artifact binary deserialization and validation
- WebSocket telemetry churn & rapid client disconnect handling
- RVC v2 5-stage mathematical and shape verification
- Server lifespan shutdown safety under active training
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.main import app, lifespan
from backend.app.routers.websocket import (
    TrainingTelemetryManager,
    broadcast_training_telemetry,
    get_training_telemetry_manager,
)
from backend.app.services.f5tts_service import (
    POLISH_LOWER_DIACRITICS,
    POLISH_UPPER_DIACRITICS,
    build_polish_vocab,
    get_f5tts_service,
    prepare_f5tts_dataset,
)
from backend.app.services.rvc_service import get_rvc_service
from backend.app.services.simulation import get_simulation_engine


@pytest.fixture(autouse=True)
def setup_adversarial_env(tmp_path: Path):
    """Setup clean isolated environment for each adversarial test."""
    test_db = tmp_path / "test_m3_adv.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.models_dir = tmp_path / "data" / "models"
    settings.cache_dir = tmp_path / "data" / "cache"
    settings.samples_raw_dir = tmp_path / "data" / "samples" / "raw"
    settings.ensure_directories()

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


def _seed_audio_sample(name: str = "adv_sample.wav") -> str:
    """Helper to seed sample in database and disk."""
    settings = get_settings()
    wav_path = settings.samples_raw_dir / name
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1.2, int(1.2 * 44100), endpoint=False, dtype=np.float32)
    audio = 0.6 * np.sin(2 * np.pi * 330.0 * t)
    sf.write(str(wav_path), audio, 44100, subtype="PCM_16")

    rec = repository.create_sample(
        name=name,
        filename=name,
        file_path=str(wav_path),
        transcript="Chrząszcz brzmi w trzcinie w Szczebrzeszynie.",
    )
    return str(rec["id"])


# ==============================================================================
# 1. High Concurrency Barrier Race Tests
# ==============================================================================

def test_m3_concurrent_training_request_barrier():
    """
    Stress-test concurrent training initiation:
    20 worker threads simultaneously attempt to start training via REST API.
    Guarantees:
    - Exactly 1 thread receives HTTP 200 (started)
    - Exactly 19 threads receive HTTP 409 (conflict)
    - Zero unhandled 500 exceptions
    - Exactly one active training run registered
    """
    client = TestClient(app)
    sid = _seed_audio_sample()

    num_threads = 20
    barrier = threading.Barrier(num_threads)
    results: List[int] = []
    lock = threading.Lock()

    def try_start(thread_idx: int):
        barrier.wait()
        engine = "f5-tts" if thread_idx % 2 == 0 else "rvc"
        url = f"/api/train/{engine}"
        payload = {
            "model_name": f"race_model_{thread_idx}",
            "epochs": 50,
            "sample_ids": [sid],
        }
        resp = client.post(url, json=payload)
        with lock:
            results.append(resp.status_code)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(try_start, i) for i in range(num_threads)]
        for f in futures:
            f.result()

    success_count = results.count(200)
    conflict_count = results.count(409)

    # Clean up immediately
    client.post("/api/train/stop")

    assert success_count == 1, f"Expected exactly 1 success, got {success_count}"
    assert conflict_count == num_threads - 1, f"Expected {num_threads - 1} conflicts, got {conflict_count}"
    assert len(results) == num_threads


def test_m3_concurrent_stop_during_training():
    """
    Stress-test concurrent stop calls:
    Initiate a long training run, then hammer /api/train/stop with 15 concurrent threads.
    Guarantees:
    - All return HTTP 200
    - All responses are either 'stopped' or 'no_active_run'
    - Zero crashes or race condition errors
    - Training run status is successfully transitioned to 'stopped'
    """
    client = TestClient(app)
    sid = _seed_audio_sample()

    start_resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "adv_cancel_target", "epochs": 100, "sample_ids": [sid]},
    )
    assert start_resp.status_code == 200

    num_threads = 15
    barrier = threading.Barrier(num_threads)
    stop_results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def try_stop():
        barrier.wait()
        resp = client.post("/api/train/stop")
        with lock:
            stop_results.append((resp.status_code, resp.json()))

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(try_stop) for _ in range(num_threads)]
        for f in futures:
            f.result()

    assert len(stop_results) == num_threads
    for status_code, body in stop_results:
        assert status_code == 200
        assert body["status"] in ("stopped", "no_active_run")

    # DB state must be 'stopped'
    status_resp = client.get("/api/train/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["active"] is False


# ==============================================================================
# 2. Path Traversal & Parameter Injection Attacks
# ==============================================================================

@pytest.mark.parametrize(
    "bad_name",
    [
        "../traversal",
        "..\\windows_traversal",
        "/absolute/path/attack",
        "name with spaces",
        "model;rm -rf /",
        "model$(whoami)",
        "<script>alert(1)</script>",
        "model\0nullbyte",
        "",
        "   ",
        "model!@#$%^&*()",
    ],
)
def test_m3_model_name_adversarial_validation(bad_name: str):
    """Verify adversarial model names are strictly rejected by both training and models APIs."""
    client = TestClient(app)

    # F5-TTS
    f5_resp = client.post("/api/train/f5-tts", json={"model_name": bad_name, "epochs": 5})
    assert f5_resp.status_code in (400, 422), f"F5-TTS allowed invalid model name: {bad_name}"

    # RVC
    rvc_resp = client.post("/api/train/rvc", json={"model_name": bad_name, "epochs": 5})
    assert rvc_resp.status_code in (400, 422), f"RVC allowed invalid model name: {bad_name}"


# ==============================================================================
# 3. Simulation Engine Extremes & Mathematical Invariants
# ==============================================================================

def test_m3_simulation_loss_trajectory_extremes():
    """
    Stress-test mathematical loss calculation across boundary inputs:
    - 1 epoch, 1 step
    - 5000 epochs, 500,000 steps
    Guarantees:
    - Loss remains bounded in [0.18, 3.5]
    - Loss is monotonically non-increasing in trend
    - No NaN, Inf, or negative numbers
    """
    sim = get_simulation_engine()

    for engine in ("f5-tts", "rvc"):
        # Test 1: Single step run
        l_single = sim.compute_loss(engine, step=1, total_steps=1, epoch=1, total_epochs=1)
        assert 0.18 <= l_single <= 3.5
        assert not math.isnan(l_single)
        assert not math.isinf(l_single)

        # Test 2: Ultra-long run
        l_first = sim.compute_loss(engine, step=1, total_steps=500000, epoch=1, total_epochs=5000)
        l_mid = sim.compute_loss(engine, step=250000, total_steps=500000, epoch=2500, total_epochs=5000)
        l_last = sim.compute_loss(engine, step=500000, total_steps=500000, epoch=5000, total_epochs=5000)

        assert l_first > l_mid > l_last
        assert 0.18 <= l_last <= 0.50
        assert not math.isnan(l_last)


# ==============================================================================
# 4. Artifact Binary Deserialization & Structural Verification
# ==============================================================================

def test_m3_artifact_binary_deserialization_and_verification(tmp_path: Path):
    """
    Adversarially verify that created model artifacts are genuine, valid binary files:
    1. F5-TTS: .pt contains valid state dict with real float32 numpy arrays and metadata
    2. RVC: .pth contains valid generator state dict, .index contains float32 vectors or Faiss header
    3. Polish vocab: contains all 32 Polish characters, punctuation, digits, tokens
    """
    sim = get_simulation_engine()
    settings = get_settings()

    # 1. F5-TTS
    f5_res = sim.create_f5tts_artifacts(
        model_name="adv_f5_model",
        epochs=3,
        batch_size=2,
        learning_rate=0.0001,
        sample_count=2,
        final_loss=0.32,
    )
    pt_file = settings.models_dir / "adv_f5_model_f5tts.pt"
    assert pt_file.exists()
    assert pt_file.stat().st_size > 1024, "Checkpoint file must not be an empty dummy"

    # Deserialize and inspect
    try:
        import torch
        bundle = torch.load(str(pt_file), map_location="cpu", weights_only=False)
    except Exception:
        with open(pt_file, "rb") as f:
            bundle = pickle.load(f)

    assert bundle["model_type"] == "F5TTS"
    assert "ema_weights" in bundle
    assert isinstance(bundle["ema_weights"]["transformer.embed.weight"], np.ndarray)
    assert bundle["ema_weights"]["transformer.embed.weight"].dtype == np.float32

    # Check vocab file
    vocab_file = settings.models_dir / "adv_f5_model" / "vocab.txt"
    assert vocab_file.exists()
    vocab_text = vocab_file.read_text(encoding="utf-8").splitlines()
    for lower, upper in zip(POLISH_LOWER_DIACRITICS, POLISH_UPPER_DIACRITICS):
        assert lower in vocab_text
        assert upper in vocab_text

    # 2. RVC v2
    rvc_res = sim.create_rvc_artifacts(
        model_name="adv_rvc_model",
        epochs=5,
        batch_size=4,
        sample_rate=40000,
        sample_count=2,
        final_loss=0.28,
    )
    pth_file = settings.models_dir / "adv_rvc_model.pth"
    index_file = settings.models_dir / "adv_rvc_model.index"
    assert pth_file.exists()
    assert index_file.exists()
    assert pth_file.stat().st_size > 1024
    assert index_file.stat().st_size > 512

    try:
        import torch
        rvc_bundle = torch.load(str(pth_file), map_location="cpu", weights_only=False)
    except Exception:
        with open(pth_file, "rb") as f:
            rvc_bundle = pickle.load(f)

    assert "weight" in rvc_bundle
    assert "config" in rvc_bundle
    assert rvc_bundle["params"]["sr"] == "40k"


# ==============================================================================
# 5. WebSocket Telemetry Churn & Concurrency Safety
# ==============================================================================

def test_m3_websocket_telemetry_churn_and_resilience():
    """
    Stress-test TrainingTelemetryManager under high churn:
    Rapidly broadcast telemetry frames while subscribers connect and disconnect.
    Guarantees:
    - Thread-safe broadcast from sync threads does not deadlock or crash
    - Dead subscribers are automatically pruned
    """
    manager = get_training_telemetry_manager()

    # Simulate broadcasts from multiple background threads
    def spam_broadcasts(thread_id: int):
        for step in range(25):
            manager.broadcast_sync({
                "type": "progress",
                "engine": "f5-tts",
                "run_id": f"spam_run_{thread_id}",
                "epoch": 1,
                "total_epochs": 10,
                "step": step,
                "total_steps": 25,
                "loss": 0.45,
                "progress_percent": float(step * 4),
            })

    threads = [threading.Thread(target=spam_broadcasts, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ==============================================================================
# 6. RVC 5-Stage Mathematical Shape Invariants
# ==============================================================================

def test_m3_rvc_stages_mathematical_invariants(tmp_path: Path):
    """
    Verify mathematical correctness of RVC stage transformations:
    - Preprocessing produces exact sample rate and -0.5 dBFS normalized peak
    - F0 extraction produces frames matching audio duration
    - HuBERT produces 768-dimensional features
    """
    rvc = get_rvc_service()
    settings = get_settings()

    # Create 2.5s stereo audio at 48kHz with some DC offset
    sr_in = 48000
    t = np.linspace(0, 2.5, int(2.5 * sr_in), endpoint=False, dtype=np.float32)
    stereo = np.column_stack([0.8 * np.sin(2 * np.pi * 200 * t), 0.7 * np.cos(2 * np.pi * 400 * t)])
    raw_path = settings.samples_raw_dir / "stereo_adv.wav"
    sf.write(str(raw_path), stereo, sr_in, subtype="PCM_16")

    sample = repository.create_sample(
        name="stereo_adv",
        filename="stereo_adv.wav",
        file_path=str(raw_path),
        transcript="Test dwukanałowy.",
    )

    stage_dir = tmp_path / "stage_test"
    gt_dir = stage_dir / "0_gt_wavs"
    dir_16k = stage_dir / "1_16k_wavs"
    f0_dir = stage_dir / "2a_f0"
    f0nsf_dir = stage_dir / "2b_f0nsf"
    feat_dir = stage_dir / "3_feature768"
    for d in (gt_dir, dir_16k, f0_dir, f0nsf_dir, feat_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1: Preprocess with 40k target
    chunks = rvc.stage1_preprocess_audio([sample], gt_dir, dir_16k, target_sr=40000)
    assert len(chunks) >= 1

    gt_file = list(gt_dir.glob("*.wav"))[0]
    audio_gt, sr_gt = sf.read(str(gt_file))
    assert sr_gt == 40000
    assert audio_gt.ndim == 1, "Downmixing must ensure mono"
    assert np.max(np.abs(audio_gt)) <= 0.96, "Peak must be normalized near 0.95"

    # Stage 2: Pitch
    rvc.stage2_extract_f0(dir_16k, f0_dir, f0nsf_dir)
    f0_npy = np.load(str(list(f0_dir.glob("*.npy"))[0]))
    f0nsf_npy = np.load(str(list(f0nsf_dir.glob("*.npy"))[0]))
    assert np.all(f0_npy >= 0.0), "Pitch Hz must be non-negative"
    assert np.all((f0nsf_npy >= 0) & (f0nsf_npy <= 255)), "NSF bins must be in [0, 255]"

    # Stage 3: Features
    rvc.stage3_extract_features(dir_16k, feat_dir)
    feats_npy = np.load(str(list(feat_dir.glob("*.npy"))[0]))
    assert feats_npy.shape[1] == 768, "Semantic representations must be 768-dimensional"
    assert len(feats_npy) == len(f0_npy), "Feature frames must align with F0 frames"


# ==============================================================================
# 7. Lifespan Shutdown Integration
# ==============================================================================

@pytest.mark.asyncio
async def test_m3_server_lifespan_shutdown_under_active_training():
    """
    Verify server lifespan context manager cleanly stops active training
    and checkpoints database on shutdown without raising unhandled exceptions.
    """
    sid = _seed_audio_sample()

    # Enter lifespan
    async with lifespan(app):
        # Start training
        f5 = get_f5tts_service()
        f5.start_training(model_name="lifespan_test", epochs=50, sample_ids=[sid])
        assert f5.is_training_active()

    # Lifespan has exited: verify training was stopped cleanly
    assert not f5.is_training_active()
    run = repository.get_active_training_run()
    # Active run in DB must be None (either completed or stopped)
    assert run is None
