"""
tests/test_m3_challenger_barrier_empirical.py
Empirical Challenger Concurrency & Cross-Engine Barrier Synchronization Stress Test.

Validates:
1. Multi-round High-concurrency cross-engine barrier race:
   - 30 threads (15 F5-TTS, 15 RVC) released simultaneously across 5 consecutive rounds.
   - In each round: EXACTLY 1 request yields HTTP 200, EXACTLY 29 yield HTTP 409.
   - Zero HTTP 500 internal server errors.
   - Zero dual execution (F5-TTS and RVC can never be simultaneously active).
2. Interleaved Start / Stop 3-Way Barrier Race:
   - 10 F5 starts, 10 RVC starts, 10 Stops released on a 30-thread barrier.
   - Mutual exclusion guaranteed: at most 1 start succeeds.
3. Jittered Cross-Engine Hammer:
   - 40 rapid interleaved requests across both engines with randomized micro-delays.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

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
def setup_stress_env(tmp_path: Path):
    """Isolate DB and storage for stress testing."""
    test_db = tmp_path / "test_stress_barrier.db"
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


def _seed_sample(name: str = "stress_sample.wav") -> str:
    settings = get_settings()
    p = settings.samples_raw_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 0.5, int(0.5 * 44100), endpoint=False, dtype=np.float32)
    sf.write(str(p), 0.3 * np.sin(2 * np.pi * 440.0 * t), 44100, subtype="PCM_16")
    rec = repository.create_sample(
        name=name,
        filename=name,
        file_path=str(p),
        transcript="Test transcript for stress barrier.",
    )
    return str(rec["id"])


def test_multi_round_cross_engine_barrier_saturation():
    """
    Stress-test: 5 consecutive rounds of 30 simultaneous threads (15 F5, 15 RVC).
    Verifies that across EVERY round:
    - Exactly 1 request gets HTTP 200
    - Exactly 29 requests get HTTP 409
    - Zero requests get HTTP 500 or any other code
    - Both engines are never active simultaneously
    """
    sid = _seed_sample()
    client = TestClient(app)

    num_threads = 30
    num_rounds = 5

    f5_service = get_f5tts_service()
    rvc_service = get_rvc_service()

    for round_idx in range(num_rounds):
        barrier = threading.Barrier(num_threads)
        results: List[Dict[str, Any]] = []
        lock = threading.Lock()
        dual_active_detected = threading.Event()

        def _worker(thread_idx: int):
            t_client = TestClient(app)
            engine = "f5-tts" if thread_idx % 2 == 0 else "rvc"
            endpoint = f"/api/train/{engine}"
            payload = (
                {"model_name": f"f5_r{round_idx}_{thread_idx}", "epochs": 50, "sample_ids": [sid]}
                if engine == "f5-tts"
                else {"model_name": f"rvc_r{round_idx}_{thread_idx}", "epochs": 50, "sample_rate": 40000, "sample_ids": [sid]}
            )

            # Wait for all threads to be ready
            barrier.wait()

            # Execute request
            resp = t_client.post(endpoint, json=payload)

            # Check if dual execution ever happens
            if f5_service.is_training_active() and rvc_service.is_training_active():
                dual_active_detected.set()

            with lock:
                results.append({
                    "round": round_idx,
                    "thread": thread_idx,
                    "engine": engine,
                    "status_code": resp.status_code,
                    "body": resp.text,
                })

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Inspect invariant during round
        assert not dual_active_detected.is_set(), (
            f"Round {round_idx}: DUAL EXECUTION DETECTED! Both F5-TTS and RVC were active simultaneously!"
        )

        codes = [r["status_code"] for r in results]
        status_200 = codes.count(200)
        status_409 = codes.count(409)
        other_codes = [c for c in codes if c not in (200, 409)]

        assert len(other_codes) == 0, f"Round {round_idx}: Unexpected status codes: {other_codes}"
        assert status_200 == 1, (
            f"Round {round_idx}: Expected exactly 1 success (200), got {status_200}. Results: {results}"
        )
        assert status_409 == num_threads - 1, (
            f"Round {round_idx}: Expected {num_threads - 1} conflicts (409), got {status_409}"
        )

        # Stop active run and verify clean idle transition
        stop_resp = client.post("/api/train/stop")
        assert stop_resp.status_code == 200
        assert stop_resp.json()["status"] == "stopped"

        # Verify services are now idle
        assert not f5_service.is_training_active()
        assert not rvc_service.is_training_active()


def test_three_way_barrier_start_and_stop_race():
    """
    Stress-test: 10 F5 starts, 10 RVC starts, and 10 Stops released on a 30-thread barrier simultaneously.
    Guarantees:
    - At most 1 start can succeed with HTTP 200.
    - All other starts MUST receive HTTP 409 Conflict.
    - Stops always return HTTP 200 (either 'stopped' or 'no_active_run').
    - Zero crashes or 500s.
    """
    sid = _seed_sample()
    client = TestClient(app)

    num_threads = 30  # 10 F5, 10 RVC, 10 Stops
    barrier = threading.Barrier(num_threads)
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _worker(thread_idx: int):
        t_client = TestClient(app)
        op_type = thread_idx % 3
        if op_type == 0:
            endpoint = "/api/train/f5-tts"
            payload = {"model_name": f"f5_3way_{thread_idx}", "epochs": 50, "sample_ids": [sid]}
        elif op_type == 1:
            endpoint = "/api/train/rvc"
            payload = {"model_name": f"rvc_3way_{thread_idx}", "epochs": 50, "sample_rate": 40000, "sample_ids": [sid]}
        else:
            endpoint = "/api/train/stop"
            payload = {}

        barrier.wait()

        resp = t_client.post(endpoint, json=payload)
        with lock:
            results.append({
                "thread": thread_idx,
                "op": "stop" if op_type == 2 else ("f5" if op_type == 0 else "rvc"),
                "status_code": resp.status_code,
                "json": resp.json() if resp.status_code == 200 else resp.text,
            })

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # Clean up
    client.post("/api/train/stop")

    start_results = [r for r in results if r["op"] in ("f5", "rvc")]
    stop_results = [r for r in results if r["op"] == "stop"]

    assert len(start_results) == 20
    assert len(stop_results) == 10

    # Stops must all be 200 with valid status
    for sr in stop_results:
        assert sr["status_code"] == 200
        assert sr["json"]["status"] in ("stopped", "no_active_run")

    # Starts: at most 1 HTTP 200 across both engines, all others 409
    start_codes = [r["status_code"] for r in start_results]
    start_200s = start_codes.count(200)
    start_409s = start_codes.count(409)

    assert start_200s in (0, 1), f"Expected at most 1 start success, got {start_200s}"
    assert start_409s == len(start_results) - start_200s


def test_jittered_cross_engine_hammer():
    """
    Stress-test: 40 threads with randomized micro-sleeps (0-5ms) before posting.
    Simulates real-world unpredictable network latency.
    Guarantees:
    - Never are both engines active at the same time.
    - Requests serialize properly without deadlock or unhandled 500 exceptions.
    """
    sid = _seed_sample()
    client = TestClient(app)

    num_threads = 40
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()
    f5_svc = get_f5tts_service()
    rvc_svc = get_rvc_service()
    dual_active_flag = threading.Event()

    def _worker(thread_idx: int):
        t_client = TestClient(app)
        engine = "f5-tts" if thread_idx % 2 == 0 else "rvc"
        endpoint = f"/api/train/{engine}"
        payload = (
            {"model_name": f"f5_jit_{thread_idx}", "epochs": 50, "sample_ids": [sid]}
            if engine == "f5-tts"
            else {"model_name": f"rvc_jit_{thread_idx}", "epochs": 50, "sample_rate": 40000, "sample_ids": [sid]}
        )

        time.sleep(random.uniform(0.0001, 0.005))

        resp = t_client.post(endpoint, json=payload)

        if f5_svc.is_training_active() and rvc_svc.is_training_active():
            dual_active_flag.set()

        with lock:
            results.append({"thread": thread_idx, "engine": engine, "status_code": resp.status_code})

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    client.post("/api/train/stop")

    assert not dual_active_flag.is_set(), "DUAL EXECUTION DETECTED under jittered hammer!"
    codes = [r["status_code"] for r in results]
    assert all(c in (200, 409) for c in codes), f"Unexpected status codes found: {codes}"
