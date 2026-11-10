"""
tests/test_m5_challenger_electron_lifecycle.py

Empirical Challenge & Verification Test Suite for Milestone 5:
- Electron Child Process Supervisor
- Dynamic Port Scanning & Conflict Resolution
- Health Check Polling & Transient Startup Delay Resilience
- Clean Shutdown Handshake & Tree-Kill Fallback
- Parent Watcher Auto-Termination

Authored by Challenger M5-2.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP port is currently open and accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def bind_dummy_socket(port: int, host: str = "127.0.0.1") -> socket.socket:
    """Bind and listen on a TCP port to simulate an occupied port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(5)
    return s


def run_node_eval(script: str, timeout: float = 15.0) -> Tuple[int, str, str]:
    """Execute a JavaScript snippet using Node.js and return (exit_code, stdout, stderr)."""
    res = subprocess.run(
        ["node", "-e", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def http_get(url: str, timeout: float = 2.0) -> Tuple[int, Dict[str, Any]]:
    import urllib.request
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = json.loads(resp.read().decode("utf-8"))
        return code, body


def http_post(url: str, data: Optional[Dict[str, Any]] = None, timeout: float = 2.0) -> Tuple[int, Dict[str, Any]]:
    import urllib.request
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = json.loads(resp.read().decode("utf-8"))
        return code, body


def wait_for_backend_health(port: int, max_wait: float = 15.0) -> Dict[str, Any]:
    """Poll GET http://127.0.0.1:{port}/api/health until backend_ready is True."""
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
    raise TimeoutError(
        f"Backend on port {port} failed to become healthy within {max_wait}s. Last error: {last_err}"
    )


# ==============================================================================
# SECTION 1: Dynamic Port Scanning & Conflict Resolution
# ==============================================================================

class TestDynamicPortScanning:
    """
    Empirically test that when default port (e.g. 8000) or candidate ports are occupied,
    the supervisor correctly scans sequentially and selects the first available port.
    """

    def test_node_find_available_port_single_conflict(self):
        """
        Verify that Node's findAvailablePort algorithm detects an occupied start port
        and cleanly increments to port + 1.
        """
        test_port = 8110
        sock = bind_dummy_socket(test_port)
        try:
            assert is_port_in_use(test_port), f"Port {test_port} must be occupied"

            # Execute the exact findAvailablePort logic implemented in electron/main.ts
            node_script = f"""
            const net = require('net');
            function findAvailablePort(startPort = {test_port}) {{
              return new Promise((resolve) => {{
                const server = net.createServer();
                server.listen(startPort, '127.0.0.1', () => {{
                  const addr = server.address();
                  const port = addr.port;
                  server.close(() => resolve(port));
                }});
                server.on('error', () => {{
                  resolve(findAvailablePort(startPort + 1));
                }});
              }});
            }}
            findAvailablePort({test_port}).then(port => console.log('RESOLVED_PORT:' + port));
            """
            code, stdout, stderr = run_node_eval(node_script)
            assert code == 0, f"Node script failed: {stderr}"
            assert f"RESOLVED_PORT:{test_port + 1}" in stdout, (
                f"Expected port {test_port + 1}, got: {stdout}"
            )
        finally:
            sock.close()

    def test_node_find_available_port_multiple_sequential_conflicts(self):
        """
        Verify that when consecutive ports (e.g. 8120, 8121, 8122) are blocked,
        the supervisor recursively scans through all conflicts and binds the first free port (8123).
        """
        base_port = 8120
        occupied_socks = [bind_dummy_socket(base_port + i) for i in range(3)]
        try:
            for i in range(3):
                assert is_port_in_use(base_port + i)

            node_script = f"""
            const net = require('net');
            function findAvailablePort(startPort = {base_port}) {{
              return new Promise((resolve) => {{
                const server = net.createServer();
                server.listen(startPort, '127.0.0.1', () => {{
                  const addr = server.address();
                  const port = addr.port;
                  server.close(() => resolve(port));
                }});
                server.on('error', () => {{
                  resolve(findAvailablePort(startPort + 1));
                }});
              }});
            }}
            findAvailablePort({base_port}).then(port => console.log('RESOLVED_PORT:' + port));
            """
            code, stdout, stderr = run_node_eval(node_script)
            assert code == 0, f"Node script failed: {stderr}"
            expected_port = base_port + 3
            assert f"RESOLVED_PORT:{expected_port}" in stdout, (
                f"Expected port {expected_port}, got: {stdout}"
            )
        finally:
            for s in occupied_socks:
                s.close()

    def test_backend_spawns_on_dynamic_port_while_8000_is_occupied(self):
        """
        Verify that when port 8000 (or candidate port) is occupied, spawning run_server.py
        with --port <dynamic_port> successfully binds, initializes FastAPI, and responds to health checks.
        """
        candidate_port = 8130
        alt_port = 8131
        sock_candidate = bind_dummy_socket(candidate_port)

        proc = None
        try:
            assert is_port_in_use(candidate_port)
            assert not is_port_in_use(alt_port)

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "run_server.py"),
                "--port",
                str(alt_port),
                "--host",
                "127.0.0.1",
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            health = wait_for_backend_health(alt_port, max_wait=15.0)
            assert health.get("status") == "ok"
            assert health.get("backend_ready") is True

            # Candidate port must still be safely occupied by our socket
            assert is_port_in_use(candidate_port)
        finally:
            sock_candidate.close()
            if proc is not None:
                try:
                    http_post(f"http://127.0.0.1:{alt_port}/api/shutdown")
                    proc.wait(timeout=4.0)
                except Exception:
                    proc.kill()
                    proc.wait()


# ==============================================================================
# SECTION 2: Health Check Polling & Transient Startup Delays
# ==============================================================================

class TestHealthCheckPolling:
    """
    Empirically test waitForBackendHealth logic:
    - Resilience to transient HTTP 503 / unready responses
    - Resilience to initial connection refused errors (cold start)
    - Graceful timeout handling without unhandled rejections
    - Validation of live backend /api/health schema and latency
    """

    def test_node_health_polling_handles_transient_503_warmup(self):
        """
        Simulate a backend warming up: returns 503 for the first 3 polls, then 200 ready.
        Verify Node.js polling loop retries and resolves true without crashing.
        """
        test_port = 8140
        node_script = f"""
        const http = require('http');

        let attempts = 0;
        const server = http.createServer((req, res) => {{
          attempts++;
          if (attempts < 4) {{
            res.writeHead(503, {{ 'Content-Type': 'application/json' }});
            res.end(JSON.stringify({{ status: 'warming_up', backend_ready: false }}));
          }} else {{
            res.writeHead(200, {{ 'Content-Type': 'application/json' }});
            res.end(JSON.stringify({{ status: 'ok', backend_ready: true, version: '0.1.0' }}));
          }}
        }});

        async function waitForBackendHealth(port, timeoutMs = 6000, intervalMs = 100) {{
          const startTime = Date.now();
          const healthUrl = `http://127.0.0.1:${{port}}/api/health`;
          while (Date.now() - startTime < timeoutMs) {{
            try {{
              const response = await fetch(healthUrl, {{ method: 'GET', signal: AbortSignal.timeout(1000) }});
              if (response.ok) {{
                const body = await response.json();
                if (body.status === 'ok' && body.backend_ready) {{
                  return true;
                }}
              }}
            }} catch {{
              // retry
            }}
            await new Promise(r => setTimeout(r, intervalMs));
          }}
          return false;
        }}

        server.listen({test_port}, '127.0.0.1', async () => {{
          const ok = await waitForBackendHealth({test_port});
          console.log(`HEALTH_RESULT:${{ok}}:ATTEMPTS:${{attempts}}`);
          server.close();
        }});
        """
        code, stdout, stderr = run_node_eval(node_script)
        assert code == 0, f"Node script error: {stderr}"
        assert "HEALTH_RESULT:true" in stdout, f"Expected true, got: {stdout}"
        assert "ATTEMPTS:4" in stdout, f"Expected 4 attempts before success, got: {stdout}"

    def test_node_health_polling_handles_connection_refused_cold_start(self):
        """
        Simulate cold start delay where port is initially closed for 1.0s before listening.
        Verify polling loop tolerates network failures (ECONNREFUSED) and detects readiness.
        """
        test_port = 8145
        node_script = f"""
        const http = require('http');

        const server = http.createServer((req, res) => {{
          res.writeHead(200, {{ 'Content-Type': 'application/json' }});
          res.end(JSON.stringify({{ status: 'ok', backend_ready: true }}));
        }});

        async function waitForBackendHealth(port, timeoutMs = 5000, intervalMs = 150) {{
          const startTime = Date.now();
          const healthUrl = `http://127.0.0.1:${{port}}/api/health`;
          while (Date.now() - startTime < timeoutMs) {{
            try {{
              const response = await fetch(healthUrl, {{ method: 'GET', signal: AbortSignal.timeout(500) }});
              if (response.ok) {{
                const body = await response.json();
                if (body.status === 'ok' && body.backend_ready) {{
                  return true;
                }}
              }}
            }} catch {{
              // Connection refused or timed out, retry
            }}
            await new Promise(r => setTimeout(r, intervalMs));
          }}
          return false;
        }}

        // Delay server start by 1000ms
        setTimeout(() => {{
          server.listen({test_port}, '127.0.0.1');
        }}, 1000);

        (async () => {{
          const ok = await waitForBackendHealth({test_port});
          console.log(`COLD_START_RESULT:${{ok}}`);
          server.close();
        }})();
        """
        code, stdout, stderr = run_node_eval(node_script)
        assert code == 0, f"Node script error: {stderr}"
        assert "COLD_START_RESULT:true" in stdout, f"Expected true, got: {stdout}"

    def test_node_health_polling_timeout_exhaustion(self):
        """
        Verify that when the backend fails to start within the timeout period,
        waitForBackendHealth cleanly returns false without crashing or hanging.
        """
        dead_port = 8149
        assert not is_port_in_use(dead_port)

        node_script = f"""
        async function waitForBackendHealth(port, timeoutMs = 1200, intervalMs = 100) {{
          const startTime = Date.now();
          const healthUrl = `http://127.0.0.1:${{port}}/api/health`;
          while (Date.now() - startTime < timeoutMs) {{
            try {{
              const response = await fetch(healthUrl, {{ method: 'GET', signal: AbortSignal.timeout(300) }});
              if (response.ok) {{
                const body = await response.json();
                if (body.status === 'ok' && body.backend_ready) return true;
              }}
            }} catch {{
              // retry
            }}
            await new Promise(r => setTimeout(r, intervalMs));
          }}
          return false;
        }}

        (async () => {{
          const t0 = Date.now();
          const ok = await waitForBackendHealth({dead_port}, 1200, 100);
          const elapsed = Date.now() - t0;
          console.log(`TIMEOUT_RESULT:${{ok}}:ELAPSED:${{elapsed}}`);
        }})();
        """
        code, stdout, stderr = run_node_eval(node_script)
        assert code == 0, f"Node script error: {stderr}"
        assert "TIMEOUT_RESULT:false" in stdout, f"Expected false, got: {stdout}"

    def test_live_backend_health_probe_latency_and_contract(self):
        """
        Measure actual live backend startup latency and verify complete GET /api/health schema.
        """
        test_port = 8150
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_server.py"),
            "--port",
            str(test_port),
            "--host",
            "127.0.0.1",
        ]
        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            health = wait_for_backend_health(test_port, max_wait=15.0)
            elapsed = time.time() - t0
            print(f"\n[Backend Healthy in {elapsed:.2f}s]")
            assert health.get("status") == "ok"
            assert health.get("version") == "0.1.0"
            assert health.get("backend_ready") is True
            assert isinstance(health.get("uptime_seconds"), (int, float))
            assert health["uptime_seconds"] >= 0.0
            assert "timestamp" in health
            # Startup latency on modern machines should easily be under 8 seconds
            assert elapsed < 8.0, f"Backend startup took too long ({elapsed:.2f}s)"
        finally:
            try:
                http_post(f"http://127.0.0.1:{test_port}/api/shutdown")
                proc.wait(timeout=4.0)
            except Exception:
                proc.kill()
                proc.wait()


# ==============================================================================
# SECTION 3: Clean Shutdown Handshake & Tree-Kill Fallback
# ==============================================================================

class TestCleanShutdownHandshake:
    """
    Empirically test the graceful shutdown protocol:
    - POST /api/shutdown initiates Uvicorn exit
    - Backend exits within the 2.0s Electron supervisor timeout window
    - Tree-kill on Windows terminates process hierarchy without orphans
    - Multiple concurrent shutdown requests succeed without crashing
    """

    def test_clean_shutdown_timing_under_2_seconds(self):
        """
        Verify that after receiving POST /api/shutdown, Uvicorn terminates the server
        and exits naturally with returncode 0 within <= 2.0 seconds.
        """
        test_port = 8160
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_server.py"),
            "--port",
            str(test_port),
            "--host",
            "127.0.0.1",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            wait_for_backend_health(test_port, max_wait=15.0)

            # Issue clean shutdown
            shutdown_url = f"http://127.0.0.1:{test_port}/api/shutdown"
            t_shutdown_start = time.time()
            code, body = http_post(shutdown_url)
            assert code == 200
            assert body.get("status") == "shutting_down"

            # Wait for process exit and record elapsed time
            proc.wait(timeout=4.0)
            shutdown_duration = time.time() - t_shutdown_start

            print(f"\n[Backend Shutdown Duration: {shutdown_duration:.2f}s, Returncode: {proc.returncode}]")
            assert proc.returncode == 0, f"Expected returncode 0, got {proc.returncode}"
            # Electron main.ts enforces a 2.0s wait before tree-kill
            assert shutdown_duration <= 2.0, (
                f"Shutdown took {shutdown_duration:.2f}s, exceeding the 2.0s Electron wait window"
            )
            assert not psutil.pid_exists(proc.pid), f"PID {proc.pid} still exists after shutdown"
            assert not is_port_in_use(test_port), f"Port {test_port} still bound after shutdown"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_tree_kill_fallback_on_windows_process_tree(self):
        """
        Verify that tree-kill cleanly eliminates a parent process and its spawned children
        on Windows, leaving zero orphaned processes.
        """
        # Node script spawns a Python parent which spawns a Python child, then invokes treeKill
        node_script = """
        const treeKill = require('tree-kill');
        const { spawn } = require('child_process');

        // Spawn python parent which spawns a background child process
        const parent = spawn('python', [
          '-c',
          'import subprocess, time, sys; ' +
          'c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); ' +
          'print("CHILD_PID:" + str(c.pid), flush=True); ' +
          'time.sleep(60)'
        ]);

        let childPid = null;
        parent.stdout.on('data', (data) => {
          const match = data.toString().match(/CHILD_PID:(\\d+)/);
          if (match) {
            childPid = parseInt(match[1], 10);
            console.log(`PARENTPID:${parent.pid}:CHILDPID:${childPid}`);

            // Invoke tree-kill on the parent PID
            setTimeout(() => {
              treeKill(parent.pid, 'SIGKILL', (err) => {
                console.log(`TREEKILL_RESULT:${err ? err.message : 'OK'}`);
              });
            }, 500);
          }
        });
        """
        code, stdout, stderr = run_node_eval(node_script, timeout=12.0)
        assert code == 0, f"Node script error: {stderr}"
        assert "TREEKILL_RESULT:OK" in stdout, f"tree-kill failed: {stdout}"

        # Parse PIDs and ensure both parent and child are dead in the OS
        import re
        m = re.search(r"PARENTPID:(\d+):CHILDPID:(\d+)", stdout)
        assert m is not None, f"Could not extract PIDs from output: {stdout}"
        parent_pid = int(m.group(1))
        child_pid = int(m.group(2))

        # Give OS a brief moment to update process table
        time.sleep(0.5)
        assert not psutil.pid_exists(parent_pid), f"Parent PID {parent_pid} was not killed!"
        assert not psutil.pid_exists(child_pid), f"Child PID {child_pid} was orphaned!"

    def test_concurrent_shutdown_flood(self):
        """
        Stress test: send 8 simultaneous POST /api/shutdown requests.
        Verify no race condition, deadlock, or unhandled 500 error occurs.
        """
        test_port = 8165
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_server.py"),
            "--port",
            str(test_port),
            "--host",
            "127.0.0.1",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            wait_for_backend_health(test_port, max_wait=15.0)
            shutdown_url = f"http://127.0.0.1:{test_port}/api/shutdown"

            def send_req():
                try:
                    c, b = http_post(shutdown_url, timeout=2.0)
                    return ("ok", c, b)
                except Exception as e:
                    return ("err", str(e), None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futures = [ex.submit(send_req) for _ in range(8)]
                results = [f.result() for f in futures]

            success_calls = sum(1 for status, code, _ in results if status == "ok" and code == 200)
            assert success_calls >= 1, "At least one shutdown call must succeed with HTTP 200"

            proc.wait(timeout=4.0)
            assert proc.returncode == 0
            assert not psutil.pid_exists(proc.pid)
            assert not is_port_in_use(test_port)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


# ==============================================================================
# SECTION 4: Parent Watcher Auto-Termination
# ==============================================================================

class TestParentWatcherTermination:
    """
    Empirically test backend's internal start_parent_watcher:
    - Simulated parent sudden death triggers backend self-shutdown within ~2-4s
    - Non-existent parent PID triggers clean immediate shutdown
    - Parent PID <= 1 boundary check safely bypasses monitoring
    """

    def test_parent_watcher_detects_abrupt_parent_crash(self):
        """
        Simulate Electron process crash: kill dummy parent process with SIGKILL.
        Verify run_server.py parent watcher detects absence and cleanly exits.
        """
        test_port = 8170
        dummy_parent = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        parent_pid = dummy_parent.pid

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_server.py"),
            "--port",
            str(test_port),
            "--host",
            "127.0.0.1",
            "--parent-pid",
            str(parent_pid),
        ]
        server_proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            wait_for_backend_health(test_port, max_wait=15.0)

            # Abruptly kill the simulated parent
            t_kill = time.time()
            dummy_parent.kill()
            dummy_parent.wait()
            assert not psutil.pid_exists(parent_pid)

            # Watcher loop sleeps 2.0s; expect termination within <= 5.0s
            server_proc.wait(timeout=6.0)
            detect_duration = time.time() - t_kill

            print(f"\n[Parent Watcher Detected Death in {detect_duration:.2f}s, Returncode: {server_proc.returncode}]")
            assert server_proc.returncode == 0, f"Expected returncode 0, got {server_proc.returncode}"
            assert detect_duration <= 5.0, f"Detection took too long ({detect_duration:.2f}s)"
            assert not psutil.pid_exists(server_proc.pid)
            assert not is_port_in_use(test_port)
        finally:
            if dummy_parent.poll() is None:
                dummy_parent.kill()
                dummy_parent.wait()
            if server_proc.poll() is None:
                server_proc.kill()
                server_proc.wait()

    def test_parent_watcher_spawned_with_nonexistent_parent_pid(self):
        """
        Spawn run_server.py with a parent PID that does not exist in the OS.
        Verify it detects parent absence on initial watch loop and terminates cleanly.
        """
        test_port = 8175
        all_pids = set(psutil.pids())
        fake_pid = 999990
        while fake_pid in all_pids:
            fake_pid += 1

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_server.py"),
            "--port",
            str(test_port),
            "--host",
            "127.0.0.1",
            "--parent-pid",
            str(fake_pid),
        ]
        server_proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Watcher should trigger shutdown on first loop cycle (~2s)
            server_proc.wait(timeout=6.0)
            assert server_proc.returncode == 0
            assert not is_port_in_use(test_port)
        finally:
            if server_proc.poll() is None:
                server_proc.kill()
                server_proc.wait()

    def test_parent_watcher_boundary_pid_zero_or_one(self):
        """
        Verify that passing --parent-pid 1 or 0 is safely ignored by start_parent_watcher
        (since parent_pid <= 1 check applies) and does not kill the server.
        """
        test_port = 8180
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_server.py"),
            "--port",
            str(test_port),
            "--host",
            "127.0.0.1",
            "--parent-pid",
            "1",
        ]
        server_proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            health = wait_for_backend_health(test_port, max_wait=15.0)
            assert health.get("status") == "ok"
            assert health.get("backend_ready") is True

            # Ensure server stays up and does not terminate immediately
            time.sleep(2.5)
            assert server_proc.poll() is None, "Server unexpectedly exited with parent-pid 1"

            # Clean shutdown
            http_post(f"http://127.0.0.1:{test_port}/api/shutdown")
            server_proc.wait(timeout=4.0)
            assert server_proc.returncode == 0
        finally:
            if server_proc.poll() is None:
                server_proc.kill()
                server_proc.wait()
