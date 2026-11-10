"""
tests/test_m3_challenger_concurrency.py
Empirical Challenger Stress Harness for Voice Sampler Milestone 3:
1. Concurrent Training Protection (Single active run invariant, HTTP 409 Conflict validation).
2. Rapid Start-Stop Cancellation Race Conditions (Thread lifecycle, DB consistency, idempotency).
3. WebSocket Telemetry Concurrency & Churn (Multi-client broadcast, abrupt disconnection, mid-training hydration).
"""

from __future__ import annotations

import concurrent.futures
import math
from pathlib import Path
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
from backend.app.main import app
from backend.app.routers.websocket import get_training_telemetry_manager
from backend.app.services.f5tts_service import get_f5tts_service
from backend.app.services.rvc_service import get_rvc_service


@pytest.fixture(autouse=True)
def setup_challenger_environment(tmp_path: Path):
    """Isolate DB and filesystem directories per test and guarantee clean service state."""
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

    f5 = get_f5tts_service()
    f5.stop_training()
    rvc = get_rvc_service()
    rvc.cancel_training()

    ws_mgr = get_training_telemetry_manager()
    ws_mgr._latest_telemetry = None

    yield

    f5 = get_f5tts_service()
    f5.stop_training()
    rvc = get_rvc_service()
    rvc.cancel_training()
    set_db_path(None)


def _seed_test_sample(name: str = "stress_sample.wav") -> str:
    """Helper to seed a valid audio sample record on disk and in SQLite."""
    settings = get_settings()
    path = settings.samples_raw_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1.0, 44100, endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * 440.0 * t), 44100, subtype="PCM_16")

    sample = repository.create_sample(
        name=name,
        filename=name,
        file_path=str(path),
        transcript="Chrząszcz brzmi w trzcinie, a Szczebrzeszyn z tego słynie.",
    )
    return str(sample["id"])


# ==============================================================================
# DIMENSION 1: CONCURRENT TRAINING ATTEMPTS & HTTP 409 CONFLICT VALIDATION
# ==============================================================================

def test_concurrent_same_engine_f5tts_requests_barrier():
    """
    Stress-test: 10 concurrent threads hit POST /api/train/f5-tts simultaneously using Barrier.
    Verification: Exactly 1 request succeeds with HTTP 200, exactly 9 requests receive HTTP 409 Conflict.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results: List[int] = []
    results_lock = threading.Lock()

    def _worker(thread_idx: int):
        t_client = TestClient(app)
        barrier.wait()  # Synchronize release
        resp = t_client.post(
            "/api/train/f5-tts",
            json={
                "model_name": f"f5_conc_{thread_idx}",
                "epochs": 20,
                "batch_size": 2,
                "sample_ids": [sid],
            },
        )
        with results_lock:
            results.append(resp.status_code)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # Clean up active run
    client.post("/api/train/stop")

    status_counts = {200: results.count(200), 409: results.count(409)}
    assert status_counts[200] == 1, f"Expected exactly 1 HTTP 200, got: {status_counts}"
    assert status_counts[409] == num_threads - 1, f"Expected {num_threads - 1} HTTP 409, got: {status_counts}"
    assert len(results) == num_threads


def test_concurrent_same_engine_rvc_requests_barrier():
    """
    Stress-test: 10 concurrent threads hit POST /api/train/rvc simultaneously using Barrier.
    Verification: Exactly 1 request succeeds with HTTP 200, exactly 9 requests receive HTTP 409 Conflict.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results: List[int] = []
    results_lock = threading.Lock()

    def _worker(thread_idx: int):
        t_client = TestClient(app)
        barrier.wait()
        resp = t_client.post(
            "/api/train/rvc",
            json={
                "model_name": f"rvc_conc_{thread_idx}",
                "epochs": 20,
                "batch_size": 4,
                "sample_rate": 40000,
                "sample_ids": [sid],
            },
        )
        with results_lock:
            results.append(resp.status_code)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    client.post("/api/train/stop")

    status_counts = {200: results.count(200), 409: results.count(409)}
    assert status_counts[200] == 1, f"Expected exactly 1 HTTP 200, got: {status_counts}"
    assert status_counts[409] == num_threads - 1, f"Expected {num_threads - 1} HTTP 409, got: {status_counts}"


def test_concurrent_cross_engine_f5_and_rvc_requests_barrier():
    """
    Stress-test: 5 threads for F5-TTS and 5 threads for RVC hit their respective endpoints simultaneously.
    Verification: Across all 10 requests, EXACTLY 1 succeeds with HTTP 200 and EXACTLY 9 receive HTTP 409.
    Never can both F5-TTS and RVC run at the same time.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()

    def _worker(thread_idx: int):
        t_client = TestClient(app)
        engine = "f5-tts" if thread_idx % 2 == 0 else "rvc"
        endpoint = "/api/train/f5-tts" if engine == "f5-tts" else "/api/train/rvc"
        payload = (
            {"model_name": f"cross_{thread_idx}", "epochs": 20, "sample_ids": [sid]}
            if engine == "f5-tts"
            else {"model_name": f"cross_{thread_idx}", "epochs": 20, "sample_rate": 40000, "sample_ids": [sid]}
        )
        barrier.wait()
        resp = t_client.post(endpoint, json=payload)
        with results_lock:
            results.append({"thread": thread_idx, "engine": engine, "status_code": resp.status_code, "body": resp.text})

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # Cleanup
    client.post("/api/train/stop")

    codes = [r["status_code"] for r in results]
    success_200 = codes.count(200)
    conflict_409 = codes.count(409)

    # Critical invariant: At most 1 active run across all engines
    f5_active = get_f5tts_service().is_training_active()
    rvc_active = get_rvc_service().is_training_active()
    assert not (f5_active and rvc_active), "VIOLATION: Both F5-TTS and RVC services are active simultaneously!"

    assert success_200 == 1, f"Expected exactly 1 HTTP 200 across mixed engines, got {success_200} (results: {results})"
    assert conflict_409 == num_threads - 1, f"Expected {num_threads - 1} HTTP 409s, got {conflict_409} (results: {results})"


def test_sequential_interleaved_conflict_rejection():
    """
    Verify cross-engine 409 rejection in sequential order:
    1. Start F5-TTS -> RVC returns 409, F5 returns 409.
    2. Stop F5-TTS.
    3. Start RVC -> F5 returns 409, RVC returns 409.
    4. Stop RVC -> Idle.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    # Step 1: Start F5-TTS
    r1 = client.post("/api/train/f5-tts", json={"model_name": "f5_active", "epochs": 50, "sample_ids": [sid]})
    assert r1.status_code == 200

    # Step 2: Attempt RVC while F5 active -> MUST 409
    r2 = client.post("/api/train/rvc", json={"model_name": "rvc_blocked", "epochs": 50, "sample_ids": [sid]})
    assert r2.status_code == 409
    assert "already in progress" in r2.json()["detail"]

    # Step 3: Attempt F5 while F5 active -> MUST 409
    r3 = client.post("/api/train/f5-tts", json={"model_name": "f5_blocked", "epochs": 50, "sample_ids": [sid]})
    assert r3.status_code == 409
    assert "already in progress" in r3.json()["detail"]

    # Step 4: Stop F5-TTS
    r4 = client.post("/api/train/stop")
    assert r4.status_code == 200
    assert r4.json()["status"] == "stopped"

    # Step 5: Start RVC
    r5 = client.post("/api/train/rvc", json={"model_name": "rvc_active", "epochs": 50, "sample_ids": [sid]})
    assert r5.status_code == 200

    # Step 6: Attempt F5 while RVC active -> MUST 409
    r6 = client.post("/api/train/f5-tts", json={"model_name": "f5_blocked2", "epochs": 50, "sample_ids": [sid]})
    assert r6.status_code == 409
    assert "already in progress" in r6.json()["detail"]

    # Step 7: Attempt RVC while RVC active -> MUST 409
    r7 = client.post("/api/train/rvc", json={"model_name": "rvc_blocked2", "epochs": 50, "sample_ids": [sid]})
    assert r7.status_code == 409

    # Step 8: Clean stop
    r8 = client.post("/api/train/stop")
    assert r8.status_code == 200
    assert r8.json()["status"] == "stopped"


# ==============================================================================
# DIMENSION 2: RAPID START-STOP CANCELLATION RACE CONDITIONS
# ==============================================================================

def test_rapid_cancellation_immediate_stop_f5tts():
    """
    Stress-test: Start F5-TTS training and immediately call POST /api/train/stop without sleeping.
    Verification:
    - Service transitions to 'stopped'.
    - DB record status is 'stopped'.
    - Subsequent stop returns {"status": "no_active_run"}.
    - Worker thread does not hang or leak.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    start_resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "rapid_f5", "epochs": 50, "sample_ids": [sid]},
    )
    assert start_resp.status_code == 200
    run_id = start_resp.json()["run_id"]

    # Immediate stop (sub-millisecond gap)
    stop_resp = client.post("/api/train/stop")
    assert stop_resp.status_code == 200
    data = stop_resp.json()
    assert data["status"] == "stopped"
    assert data["run_id"] == run_id

    # Wait briefly for thread to settle
    time.sleep(0.05)

    # Verify DB record
    db_run = repository.get_training_run(run_id)
    assert db_run is not None
    assert db_run["status"] == "stopped"

    # Verify status endpoint reflects idle
    status_resp = client.get("/api/train/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["active"] is False

    # Second stop returns no_active_run
    stop_again = client.post("/api/train/stop")
    assert stop_again.status_code == 200
    assert stop_again.json() == {"status": "no_active_run", "run_id": None}


def test_rapid_cancellation_immediate_stop_rvc():
    """
    Stress-test: Start RVC training and immediately call POST /api/train/stop without sleeping.
    Verification:
    - Service transitions to 'stopped'.
    - DB record status is 'stopped'.
    - Subsequent stop returns {"status": "no_active_run"}.
    - Worker thread terminates cleanly.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    start_resp = client.post(
        "/api/train/rvc",
        json={"model_name": "rapid_rvc", "epochs": 50, "sample_rate": 40000, "sample_ids": [sid]},
    )
    assert start_resp.status_code == 200
    run_id = start_resp.json()["run_id"]

    # Immediate stop
    stop_resp = client.post("/api/train/stop")
    assert stop_resp.status_code == 200
    data = stop_resp.json()
    assert data["status"] == "stopped"
    assert data["run_id"] == run_id

    time.sleep(0.05)

    db_run = repository.get_training_run(run_id)
    assert db_run is not None
    assert db_run["status"] == "stopped"

    status_resp = client.get("/api/train/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["active"] is False

    stop_again = client.post("/api/train/stop")
    assert stop_again.status_code == 200
    assert stop_again.json() == {"status": "no_active_run", "run_id": None}


def test_rapid_start_stop_storm_f5tts():
    """
    Stress-test: 8 back-to-back rapid start-stop cycles for F5-TTS with tiny jitter delays.
    Validates absence of deadlocks, race corruption in SQLite, or thread accumulation.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    for cycle in range(8):
        start_resp = client.post(
            "/api/train/f5-tts",
            json={"model_name": f"storm_f5_{cycle}", "epochs": 30, "sample_ids": [sid]},
        )
        assert start_resp.status_code == 200, f"Cycle {cycle} failed to start: {start_resp.text}"
        run_id = start_resp.json()["run_id"]

        # Small micro-jitter between 0 and 15ms
        if cycle % 2 == 1:
            time.sleep(0.01)

        stop_resp = client.post("/api/train/stop")
        assert stop_resp.status_code == 200
        stop_data = stop_resp.json()
        assert stop_data["status"] in ("stopped", "no_active_run")

        # Give cooperative thread time to settle
        time.sleep(0.05)

        # Confirm DB run status
        db_run = repository.get_training_run(run_id)
        assert db_run["status"] in ("stopped", "completed")

        # Verify idle before next cycle
        status_resp = client.get("/api/train/status")
        assert status_resp.json()["active"] is False


def test_rapid_start_stop_storm_rvc():
    """
    Stress-test: 8 back-to-back rapid start-stop cycles for RVC.
    Validates pipeline stages 1-5 cooperatively cancel without orphaned lock states.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    for cycle in range(8):
        start_resp = client.post(
            "/api/train/rvc",
            json={"model_name": f"storm_rvc_{cycle}", "epochs": 30, "sample_rate": 40000, "sample_ids": [sid]},
        )
        assert start_resp.status_code == 200
        run_id = start_resp.json()["run_id"]

        if cycle % 2 == 1:
            time.sleep(0.01)

        stop_resp = client.post("/api/train/stop")
        assert stop_resp.status_code == 200
        stop_data = stop_resp.json()
        assert stop_data["status"] in ("stopped", "no_active_run")

        time.sleep(0.05)

        db_run = repository.get_training_run(run_id)
        assert db_run["status"] in ("stopped", "completed")

        status_resp = client.get("/api/train/status")
        assert status_resp.json()["active"] is False


def test_concurrent_duplicate_stop_requests():
    """
    Stress-test: Fire 5 concurrent POST /api/train/stop calls simultaneously using Barrier while a run is active.
    Verification: System processes cancellation cleanly without 500 crashes or deadlocks; exactly one reports
    the run stopped or all return valid status codes.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    start_resp = client.post(
        "/api/train/f5-tts",
        json={"model_name": "multi_stop", "epochs": 50, "sample_ids": [sid]},
    )
    assert start_resp.status_code == 200
    run_id = start_resp.json()["run_id"]

    num_stops = 5
    barrier = threading.Barrier(num_stops)
    stop_results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _stop_worker():
        t_client = TestClient(app)
        barrier.wait()
        resp = t_client.post("/api/train/stop")
        with lock:
            stop_results.append(resp.json())

    threads = [threading.Thread(target=_stop_worker) for _ in range(num_stops)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # Every response must be 200 OK with either 'stopped' or 'no_active_run'
    assert len(stop_results) == num_stops
    statuses = [r.get("status") for r in stop_results]
    assert all(s in ("stopped", "no_active_run") for s in statuses)
    assert "stopped" in statuses, f"At least one stop request must claim the run: {stop_results}"

    # Verify DB state
    time.sleep(0.05)
    db_run = repository.get_training_run(run_id)
    assert db_run["status"] == "stopped"


# ==============================================================================
# DIMENSION 3: WEBSOCKET TELEMETRY CONCURRENCY, CHURN & HYDRATION
# ==============================================================================

def test_websocket_multi_client_broadcast():
    """
    Stress-test: Connect 3 WebSocket clients simultaneously to /ws/training.
    Verify all 3 receive active training progress frames.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    with client.websocket_connect("/ws/training") as ws1, \
         client.websocket_connect("/ws/training") as ws2, \
         client.websocket_connect("/ws/training") as ws3:

        # Start short training run
        start_resp = client.post(
            "/api/train/f5-tts",
            json={"model_name": "ws_multi_client", "epochs": 2, "sample_ids": [sid]},
        )
        assert start_resp.status_code == 200

        # Receive at least 1 progress frame on each connection
        f1 = ws1.receive_json()
        f2 = ws2.receive_json()
        f3 = ws3.receive_json()

        assert f1.get("type") in ("progress", "completed", "stopped")
        assert f2.get("type") in ("progress", "completed", "stopped")
        assert f3.get("type") in ("progress", "completed", "stopped")

        client.post("/api/train/stop")


def test_websocket_abrupt_disconnect_churn():
    """
    Stress-test: Connect 3 clients, start streaming, abruptly disconnect Client 2,
    verify Client 1 and Client 3 continue receiving frames without server crash or hang.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    # Step 1: Open ws1 and ws3 as outer context, ws2 inside inner scope
    with client.websocket_connect("/ws/training") as ws1, client.websocket_connect("/ws/training") as ws3:
        # Client 2 connects and disconnects abruptly
        with client.websocket_connect("/ws/training") as ws2:
            # Start run
            client.post(
                "/api/train/f5-tts",
                json={"model_name": "ws_churn", "epochs": 10, "sample_ids": [sid]},
            )
            # Both active connections receive streaming frame
            _ = ws1.receive_json()
            _ = ws2.receive_json()
        # ws2 is now closed abruptly

        # Broadcast continues to ws1 and ws3
        f1 = ws1.receive_json()
        f3 = ws3.receive_json()
        assert f1 is not None
        assert f3 is not None
        assert "engine" in f1

        client.post("/api/train/stop")


def test_websocket_mid_training_client_hydration():
    """
    Stress-test: Start training, wait for several frames, then connect a NEW client.
    Verification: The new client immediately receives initial state hydration frame
    containing the latest telemetry snapshot.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    with client.websocket_connect("/ws/training") as ws_initial:
        start_resp = client.post(
            "/api/train/f5-tts",
            json={"model_name": "ws_hydration", "epochs": 15, "sample_ids": [sid]},
        )
        assert start_resp.status_code == 200
        run_id = start_resp.json()["run_id"]

        # Wait for a couple progress frames to be emitted
        frame1 = ws_initial.receive_json()
        frame2 = ws_initial.receive_json()
        assert frame2.get("run_id") == run_id

    # Now connect a brand new client mid-flight
    with client.websocket_connect("/ws/training") as ws_late:
        # Initial frame MUST be hydration of latest telemetry
        hydration_frame = ws_late.receive_json()
        assert hydration_frame is not None
        assert "type" in hydration_frame
        assert hydration_frame["run_id"] == run_id
        assert "timestamp" in hydration_frame
        if hydration_frame["type"] == "progress":
            assert "epoch" in hydration_frame
            assert "loss" in hydration_frame

    client.post("/api/train/stop")


def test_websocket_terminal_cancellation_frame_broadcast():
    """
    Verify connected WebSocket client receives type='stopped' frame when training is cancelled.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    with client.websocket_connect("/ws/training") as ws:
        start_resp = client.post(
            "/api/train/f5-tts",
            json={"model_name": "ws_stop_frame", "epochs": 50, "sample_ids": [sid]},
        )
        assert start_resp.status_code == 200
        run_id = start_resp.json()["run_id"]

        # Wait for initial progress frame
        _ = ws.receive_json()

        # Cancel training
        stop_resp = client.post("/api/train/stop")
        assert stop_resp.status_code == 200

        # Read frames until 'stopped' received
        stopped_found = False
        for _ in range(5):
            msg = ws.receive_json()
            if msg.get("type") == "stopped" and msg.get("run_id") == run_id:
                stopped_found = True
                assert msg.get("status") == "stopped"
                break
        assert stopped_found, "Expected WebSocket terminal 'stopped' frame not received"


def test_websocket_ping_pong_during_active_training():
    """
    Verify client can send 'ping' and receive 'pong' during active training.
    """
    client = TestClient(app)
    sid = _seed_test_sample()

    with client.websocket_connect("/ws/training") as ws:
        client.post(
            "/api/train/f5-tts",
            json={"model_name": "ws_ping_test", "epochs": 5, "sample_ids": [sid]},
        )
        ws.send_text("ping")
        resp = ws.receive_text()
        assert resp == "pong"

        client.post("/api/train/stop")
