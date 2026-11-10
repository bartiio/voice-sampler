"""
tests/test_m1_lifecycle_stress.py
Adversarial lifecycle, watchdog, and shutdown stress test for Milestone 1.
Authored by challenger_m1_2.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import psutil


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def http_get(url: str, timeout: float = 2.0) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        code = response.getcode()
        body = json.loads(response.read().decode("utf-8"))
        return code, body


def http_post(url: str, data: Optional[Dict[str, Any]] = None, timeout: float = 2.0) -> Tuple[int, Dict[str, Any]]:
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        code = response.getcode()
        body = json.loads(response.read().decode("utf-8"))
        return code, body


def wait_for_health(port: int, max_wait: float = 15.0) -> Dict[str, Any]:
    deadline = time.time() + max_wait
    url = f"http://127.0.0.1:{port}/api/health"
    last_err = None
    while time.time() < deadline:
        try:
            code, body = http_get(url, timeout=1.0)
            if code == 200 and body.get("backend_ready") is True:
                return body
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    raise TimeoutError(f"Server on port {port} failed to become healthy within {max_wait}s. Last error: {last_err}")


def test_live_spawning_and_shutdown():
    print("\n--- [TEST 1] Live Spawning, Health Polling, GPU Endpoint & Shutdown ---")
    port = 8099
    assert not is_port_in_use(port), f"Port {port} is unexpectedly in use before test"

    cmd = [sys.executable, "run_server.py", "--port", str(port), "--host", "127.0.0.1"]
    spawn_time = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"Spawned server subprocess PID: {proc.pid}")

    try:
        # Step 1: Poll health
        health = wait_for_health(port, max_wait=15.0)
        ready_duration = time.time() - spawn_time
        print(f"Health response received in {ready_duration:.2f}s: {health}")
        assert health.get("status") == "ok"
        assert health.get("version") == "0.1.0"
        assert health.get("backend_ready") is True
        assert "uptime_seconds" in health

        # Step 2: Verify GPU endpoint
        gpu_url = f"http://127.0.0.1:{port}/api/settings/gpu"
        code, gpu_body = http_get(gpu_url)
        print(f"GPU settings response (HTTP {code}): {gpu_body}")
        assert code == 200
        assert "cuda_available" in gpu_body
        assert "gpu_name" in gpu_body
        assert "vram_total_mb" in gpu_body
        assert "simulation_mode" in gpu_body
        assert "message" in gpu_body

        # Step 3: Trigger shutdown
        shutdown_url = f"http://127.0.0.1:{port}/api/shutdown"
        shutdown_start = time.time()
        s_code, s_body = http_post(shutdown_url)
        print(f"Shutdown response (HTTP {s_code}): {s_body}")
        assert s_code == 200
        assert s_body.get("status") == "shutting_down"

        # Step 4: Measure shutdown duration and verify clean exit
        proc.wait(timeout=5.0)
        shutdown_duration = time.time() - shutdown_start
        print(f"Process terminated cleanly in {shutdown_duration:.2f}s with returncode {proc.returncode}")
        assert proc.returncode == 0, f"Expected returncode 0, got {proc.returncode}"

        # Step 5: Check zero orphaned processes
        assert not psutil.pid_exists(proc.pid), f"PID {proc.pid} still exists after exit!"
        assert not is_port_in_use(port), f"Port {port} still bound after shutdown!"
        print("[TEST 1 PASSED] Server cleanly shut down with 0 orphaned processes.")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_parent_watchdog():
    print("\n--- [TEST 2] Parent Process Watchdog Termination ---")
    port = 8098
    assert not is_port_in_use(port), f"Port {port} is unexpectedly in use before test"

    # Spawn dummy parent
    dummy_parent = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    parent_pid = dummy_parent.pid
    print(f"Spawned dummy parent PID: {parent_pid}")

    # Spawn server pointing to dummy parent
    cmd = [
        sys.executable,
        "run_server.py",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--parent-pid",
        str(parent_pid),
    ]
    server_proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"Spawned server PID: {server_proc.pid} (monitoring parent {parent_pid})")

    try:
        # Wait for server to become healthy
        wait_for_health(port, max_wait=15.0)
        print("Server is healthy. Now killing dummy parent...")

        # Kill dummy parent
        kill_start = time.time()
        dummy_parent.kill()
        dummy_parent.wait()
        print(f"Dummy parent {parent_pid} terminated. Waiting for server watchdog to notice...")

        # Watchdog loop sleeps 2s, so server should detect dead parent and exit within ~4-5s
        server_proc.wait(timeout=8.0)
        detect_duration = time.time() - kill_start
        print(f"Server exited in {detect_duration:.2f}s with returncode {server_proc.returncode}")
        assert server_proc.returncode == 0, f"Expected server returncode 0, got {server_proc.returncode}"

        # Verify no orphaned processes
        assert not psutil.pid_exists(server_proc.pid), "Server PID still exists after watchdog exit!"
        assert not is_port_in_use(port), f"Port {port} still bound after watchdog exit!"
        print("[TEST 2 PASSED] Watchdog successfully detected parent death and exited cleanly.")
    finally:
        if dummy_parent.poll() is None:
            dummy_parent.kill()
            dummy_parent.wait()
        if server_proc.poll() is None:
            server_proc.kill()
            server_proc.wait()


def test_watchdog_already_dead_parent():
    print("\n--- [TEST 3] Watchdog with Non-Existent/Already Dead Parent PID ---")
    port = 8097
    assert not is_port_in_use(port), f"Port {port} is unexpectedly in use before test"

    # Find an unused PID
    all_pids = set(psutil.pids())
    fake_pid = 999999
    while fake_pid in all_pids:
        fake_pid += 1

    cmd = [
        sys.executable,
        "run_server.py",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--parent-pid",
        str(fake_pid),
    ]
    start_time = time.time()
    server_proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"Spawned server PID: {server_proc.pid} with dead parent PID {fake_pid}")

    try:
        # Watchdog should immediately see that parent does not exist and trigger shutdown
        server_proc.wait(timeout=8.0)
        exit_time = time.time() - start_time
        print(f"Server detected dead parent and exited in {exit_time:.2f}s with code {server_proc.returncode}")
        assert server_proc.returncode == 0
        assert not is_port_in_use(port)
        print("[TEST 3 PASSED] Server cleanly terminates when spawned with dead parent PID.")
    finally:
        if server_proc.poll() is None:
            server_proc.kill()
            server_proc.wait()


def test_concurrent_shutdown_requests():
    print("\n--- [TEST 4] Concurrent Shutdown Requests Stress Test ---")
    port = 8096
    assert not is_port_in_use(port), f"Port {port} is unexpectedly in use before test"

    cmd = [sys.executable, "run_server.py", "--port", str(port), "--host", "127.0.0.1"]
    server_proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"Spawned server PID: {server_proc.pid} on port {port}")

    try:
        wait_for_health(port, max_wait=15.0)
        shutdown_url = f"http://127.0.0.1:{port}/api/shutdown"

        def send_shutdown():
            try:
                code, body = http_post(shutdown_url, timeout=2.0)
                return ("ok", code, body)
            except Exception as e:
                return ("err", str(e), None)

        # Fire 5 concurrent shutdown requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_shutdown) for _ in range(5)]
            results = [f.result() for f in futures]

        print(f"Concurrent shutdown results: {results}")
        # At least one must succeed with HTTP 200
        success_count = sum(1 for status, code, _ in results if status == "ok" and code == 200)
        print(f"Successful HTTP 200 shutdown calls: {success_count}/5")
        assert success_count >= 1

        server_proc.wait(timeout=5.0)
        assert server_proc.returncode == 0
        assert not psutil.pid_exists(server_proc.pid)
        assert not is_port_in_use(port)
        print("[TEST 4 PASSED] Concurrent shutdown handled gracefully without crash.")
    finally:
        if server_proc.poll() is None:
            server_proc.kill()
            server_proc.wait()


if __name__ == "__main__":
    test_live_spawning_and_shutdown()
    test_parent_watchdog()
    test_watchdog_already_dead_parent()
    test_concurrent_shutdown_requests()
    print("\n========================================================")
    print(" ALL LIFECYCLE & WATCHDOG STRESS TESTS COMPLETED SUCCESSFULLY!")
    print("========================================================\n")
