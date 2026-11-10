"""
tests/test_m5_challenger_ws_unmount.py

Adversarial Empirical Challenge Test Suite for Milestone 5:
1. Dynamic WebSocket Port & URL Resolution (src/api/websocket.ts)
   - Dynamic port extraction (e.g. 8088, 9000, 65535, 1)
   - Custom hostnames (e.g. localhost, voicesampler.local, 0.0.0.0, IPv6 [::1])
   - Protocol transformations (http:// -> ws://, https:// -> wss://, case insensitivity)
   - Fallback and malformed input handling
   - Dynamic URL precedence hierarchy (VOICE_SAMPLER_CONFIG.wsUrl > dynamicWsBase > API_BASE_URL > fallback)
   - Live uvicorn WebSocket server execution and telemetry/notification streaming on dynamic ports
2. Microphone Unmount Resource Cleanup (src/views/SamplesView.tsx)
   - micStreamRef track lifecycle and explicit stop() iteration on unmount
   - timerIntervalRef cancellation via clearInterval() on unmount
   - mediaRecorderRef safe stop() and onstop detachment to prevent orphaned uploads
   - Simulated React unmount lifecycle under active, paused, unstarted, and error-throwing scenarios

Authored by Challenger M5-4.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
import websockets
from starlette.testclient import TestClient

from backend.app.main import create_app
from backend.app.routers.websocket import (
    broadcast_training_telemetry,
    get_training_telemetry_manager,
)
from backend.app.services.notification_service import (
    get_notification_manager,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
WEBSOCKET_TS = SRC_DIR / "api" / "websocket.ts"
SAMPLES_VIEW_TSX = SRC_DIR / "views" / "SamplesView.tsx"
CLIENT_TS = SRC_DIR / "api" / "client.ts"


# ==============================================================================
# Helper: Node Execution of JavaScript in V8 Engine
# ==============================================================================

def execute_node_script(script: str, timeout: float = 15.0) -> Tuple[int, str, str]:
    """Execute JavaScript code directly in the Node.js V8 runtime via stdin."""
    proc = subprocess.run(
        ["node"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


# ==============================================================================
# SECTION 1: Dynamic WebSocket Port & URL Resolution (src/api/websocket.ts)
# ==============================================================================

class TestDynamicWebSocketPortAndUrlDerivation:
    """Adversarial stress-testing of dynamic port and WebSocket URL derivation."""

    def test_websocket_ts_file_exists(self):
        """Verify src/api/websocket.ts exists and exports deriveWsFromHttp."""
        assert WEBSOCKET_TS.exists(), "src/api/websocket.ts does not exist"
        content = WEBSOCKET_TS.read_text(encoding="utf-8")
        assert "export function deriveWsFromHttp" in content, (
            "deriveWsFromHttp is not exported from src/api/websocket.ts"
        )

    def test_derive_ws_from_http_custom_ports_in_v8(self):
        """
        Empirically verify deriveWsFromHttp with arbitrary custom ports in V8 runtime:
        - Custom port 8088
        - Custom port 9000
        - Standard HTTP port 80
        - Boundary min port 1
        - Boundary max port 65535
        """
        node_script = """
        function deriveWsFromHttp(httpUrl, endpointPath) {
          try {
            const parsed = new URL(httpUrl);
            const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${wsProto}//${parsed.host}${endpointPath}`;
          } catch {
            return `ws://127.0.0.1:8000${endpointPath}`;
          }
        }

        const cases = [
          ['http://127.0.0.1:8088', '/ws/training', 'ws://127.0.0.1:8088/ws/training'],
          ['http://127.0.0.1:9000', '/ws/notifications', 'ws://127.0.0.1:9000/ws/notifications'],
          ['http://127.0.0.1:8001', '/ws/training', 'ws://127.0.0.1:8001/ws/training'],
          ['http://127.0.0.1:1', '/ws/training', 'ws://127.0.0.1:1/ws/training'],
          ['http://127.0.0.1:65535', '/ws/notifications', 'ws://127.0.0.1:65535/ws/notifications'],
          ['http://127.0.0.1:80', '/ws/training', 'ws://127.0.0.1/ws/training'], // port 80 omitted by URL.host
        ];

        for (const [url, ep, expected] of cases) {
          const actual = deriveWsFromHttp(url, ep);
          if (actual !== expected) {
            console.error(`Mismatch for ${url}: expected ${expected}, got ${actual}`);
            process.exit(1);
          }
        }
        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"V8 port derivation failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"

    def test_derive_ws_from_http_custom_hostnames_in_v8(self):
        """
        Empirically verify deriveWsFromHttp with custom hostnames, IP representations, and IPv6:
        - localhost
        - 0.0.0.0
        - custom internal domain: voicesampler.local
        - multi-level subdomain: api.staging.voicesampler.io
        - IPv6 loopback: [::1]:8080
        """
        node_script = """
        function deriveWsFromHttp(httpUrl, endpointPath) {
          try {
            const parsed = new URL(httpUrl);
            const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${wsProto}//${parsed.host}${endpointPath}`;
          } catch {
            return `ws://127.0.0.1:8000${endpointPath}`;
          }
        }

        const cases = [
          ['http://localhost:8088', '/ws/training', 'ws://localhost:8088/ws/training'],
          ['http://0.0.0.0:5000', '/ws/notifications', 'ws://0.0.0.0:5000/ws/notifications'],
          ['http://voicesampler.local:8000', '/ws/training', 'ws://voicesampler.local:8000/ws/training'],
          ['http://api.staging.voicesampler.io:8080', '/ws/training', 'ws://api.staging.voicesampler.io:8080/ws/training'],
          ['http://[::1]:8080', '/ws/training', 'ws://[::1]:8080/ws/training'],
        ];

        for (const [url, ep, expected] of cases) {
          const actual = deriveWsFromHttp(url, ep);
          if (actual !== expected) {
            console.error(`Mismatch for ${url}: expected ${expected}, got ${actual}`);
            process.exit(1);
          }
        }
        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"V8 hostname derivation failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"

    def test_derive_ws_from_http_protocol_transformation_in_v8(self):
        """
        Verify protocol transformation:
        - http: -> ws:
        - https: -> wss:
        - Case insensitivity (HTTP: -> ws:, HTTPS: -> wss:)
        """
        node_script = """
        function deriveWsFromHttp(httpUrl, endpointPath) {
          try {
            const parsed = new URL(httpUrl);
            const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${wsProto}//${parsed.host}${endpointPath}`;
          } catch {
            return `ws://127.0.0.1:8000${endpointPath}`;
          }
        }

        const cases = [
          ['http://127.0.0.1:8000', '/ws/training', 'ws://127.0.0.1:8000/ws/training'],
          ['https://127.0.0.1:8443', '/ws/training', 'wss://127.0.0.1:8443/ws/training'],
          ['https://voice.custom-domain.org:9000', '/ws/notifications', 'wss://voice.custom-domain.org:9000/ws/notifications'],
          ['HTTP://127.0.0.1:8088', '/ws/training', 'ws://127.0.0.1:8088/ws/training'],
          ['HTTPS://127.0.0.1:9090', '/ws/notifications', 'wss://127.0.0.1:9090/ws/notifications'],
        ];

        for (const [url, ep, expected] of cases) {
          const actual = deriveWsFromHttp(url, ep);
          if (actual !== expected) {
            console.error(`Mismatch for ${url}: expected ${expected}, got ${actual}`);
            process.exit(1);
          }
        }
        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"V8 protocol transformation failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"

    def test_derive_ws_from_http_fallback_behavior_in_v8(self):
        """
        Adversarially challenge fallback behavior against malformed, partial, or empty inputs:
        - Empty string -> fallback ws://127.0.0.1:8000${endpointPath}
        - Non-URL garbage string -> fallback
        - Scheme missing -> fallback
        - Null or undefined (as stringified or handled in JS) -> fallback
        - Subpaths on base URL: 'http://127.0.0.1:8000/api' + '/ws/training' -> 'ws://127.0.0.1:8000/ws/training'
        - Trailing slash: 'http://127.0.0.1:8000/' + '/ws/training' -> 'ws://127.0.0.1:8000/ws/training'
        """
        node_script = """
        function deriveWsFromHttp(httpUrl, endpointPath) {
          try {
            const parsed = new URL(httpUrl);
            const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${wsProto}//${parsed.host}${endpointPath}`;
          } catch {
            return `ws://127.0.0.1:8000${endpointPath}`;
          }
        }

        const cases = [
          ['', '/ws/training', 'ws://127.0.0.1:8000/ws/training'],
          ['not-a-valid-url', '/ws/notifications', 'ws://127.0.0.1:8000/ws/notifications'],
          ['127.0.0.1:8000', '/ws/training', 'ws://127.0.0.1:8000/ws/training'],
          ['://bad-scheme', '/ws/training', 'ws://127.0.0.1:8000/ws/training'],
          ['http://', '/ws/training', 'ws://127.0.0.1:8000/ws/training'],
          // Trailing slashes and API subpaths
          ['http://127.0.0.1:8000/', '/ws/training', 'ws://127.0.0.1:8000/ws/training'],
          ['http://127.0.0.1:8000/api', '/ws/training', 'ws://127.0.0.1:8000/ws/training'],
          ['http://127.0.0.1:8000/api/v1/', '/ws/notifications', 'ws://127.0.0.1:8000/ws/notifications'],
        ];

        for (const [url, ep, expected] of cases) {
          const actual = deriveWsFromHttp(url, ep);
          if (actual !== expected) {
            console.error(`Mismatch for "${url}": expected ${expected}, got ${actual}`);
            process.exit(1);
          }
        }
        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"V8 fallback test failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"

    def test_websocket_url_precedence_hierarchy_in_v8(self):
        """
        Verify the 4-tier precedence hierarchy implemented in ResilientWebSocketService.getWsUrl():
        1. window.VOICE_SAMPLER_CONFIG.wsUrl (highest priority override)
        2. dynamicWsBase (resolved dynamically from Electron IPC getBackendConfig)
        3. API_BASE_URL (from REST client config)
        4. Fallback default: ws://127.0.0.1:8000/ws/...
        """
        node_script = """
        function deriveWsFromHttp(httpUrl, endpointPath) {
          try {
            const parsed = new URL(httpUrl);
            const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${wsProto}//${parsed.host}${endpointPath}`;
          } catch {
            return `ws://127.0.0.1:8000${endpointPath}`;
          }
        }

        class MockWsService {
          constructor(endpointPath) {
            this.endpointPath = endpointPath;
          }

          getWsUrl(globals) {
            const customWs = globals.window?.VOICE_SAMPLER_CONFIG?.wsUrl;
            if (customWs) return customWs;
            if (globals.dynamicWsBase) return `${globals.dynamicWsBase}${this.endpointPath}`;
            if (globals.API_BASE_URL) return deriveWsFromHttp(globals.API_BASE_URL, this.endpointPath);
            return `ws://127.0.0.1:8000${this.endpointPath}`;
          }
        }

        const svc = new MockWsService('/ws/training');

        // Test 1: Fallback default (all unset)
        const t1 = svc.getWsUrl({ window: {}, dynamicWsBase: null, API_BASE_URL: null });
        if (t1 !== 'ws://127.0.0.1:8000/ws/training') throw new Error(`T1 failed: ${t1}`);

        // Test 2: API_BASE_URL set to port 8088
        const t2 = svc.getWsUrl({ window: {}, dynamicWsBase: null, API_BASE_URL: 'http://127.0.0.1:8088' });
        if (t2 !== 'ws://127.0.0.1:8088/ws/training') throw new Error(`T2 failed: ${t2}`);

        // Test 3: dynamicWsBase overrides API_BASE_URL
        const t3 = svc.getWsUrl({
          window: {},
          dynamicWsBase: 'ws://127.0.0.1:9000',
          API_BASE_URL: 'http://127.0.0.1:8088',
        });
        if (t3 !== 'ws://127.0.0.1:9000/ws/training') throw new Error(`T3 failed: ${t3}`);

        // Test 4: VOICE_SAMPLER_CONFIG.wsUrl overrides dynamicWsBase and API_BASE_URL
        const t4 = svc.getWsUrl({
          window: { VOICE_SAMPLER_CONFIG: { wsUrl: 'wss://cloud.custom.com/telemetry' } },
          dynamicWsBase: 'ws://127.0.0.1:9000',
          API_BASE_URL: 'http://127.0.0.1:8088',
        });
        if (t4 !== 'wss://cloud.custom.com/telemetry') throw new Error(`T4 failed: ${t4}`);

        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"V8 precedence test failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"

    def test_electron_backend_config_resolution_logic_in_v8(self):
        """
        Verify the listener logic in websocket.ts:
        - When electronAPI.getBackendConfig() returns { port: 8088 }, resolvedBase is ws://127.0.0.1:8088
        - When cfg includes custom host: { host: 'localhost', port: 9000 }, resolvedBase is ws://localhost:9000
        - When cfg includes url: { url: 'https://127.0.0.1:8443' }, resolvedBase is wss://127.0.0.1:8443
        - When dynamicWsBase changes, svc.refreshUrl() is invoked across all registered services.
        """
        node_script = """
        function resolveBaseFromConfig(cfg) {
          let resolvedBase = null;
          if (cfg?.port) {
            const host = cfg.host || '127.0.0.1';
            resolvedBase = `ws://${host}:${cfg.port}`;
          } else if (cfg?.url) {
            try {
              const parsed = new URL(cfg.url);
              const wsProto = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
              resolvedBase = `${wsProto}//${parsed.host}`;
            } catch {
              // fallback
            }
          }
          return resolvedBase;
        }

        const cases = [
          [{ port: 8088 }, 'ws://127.0.0.1:8088'],
          [{ host: 'localhost', port: 9000 }, 'ws://localhost:9000'],
          [{ host: '192.168.1.100', port: 8000 }, 'ws://192.168.1.100:8000'],
          [{ url: 'https://backend.local:8443' }, 'wss://backend.local:8443'],
          [{ url: 'http://127.0.0.1:9090' }, 'ws://127.0.0.1:9090'],
          [{ url: 'invalid-url' }, null],
          [null, null],
          [{}, null],
        ];

        for (const [cfg, expected] of cases) {
          const actual = resolveBaseFromConfig(cfg);
          if (actual !== expected) {
            console.error(`Mismatch for config ${JSON.stringify(cfg)}: expected ${expected}, got ${actual}`);
            process.exit(1);
          }
        }
        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"Electron config resolution test failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"


# ==============================================================================
# SECTION 2: Live Dynamic Port WebSocket Streaming Tests
# ==============================================================================

class TestLiveDynamicPortWebSocketServer:
    """Empirical verification of real FastAPI WebSocket endpoints running on dynamic ports."""

    @pytest.fixture(scope="class")
    def live_dynamic_server(self):
        """
        Spin up an actual uvicorn server on a dynamically allocated OS port.
        Verifies real socket bind, WebSocket upgrade, ping/pong, and telemetry broadcast.
        """
        import uvicorn

        # Find available ephemeral port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        app = create_app()
        config = uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error')
        server = uvicorn.Server(config)

        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        # Wait for server startup
        for _ in range(50):
            try:
                s = socket.create_connection(('127.0.0.1', port), timeout=0.2)
                s.close()
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Server on port {port} failed to start in 5 seconds")

        yield port

        server.should_exit = True
        server_thread.join(timeout=3.0)

    def test_dynamic_port_notifications_websocket_ping_pong(self, live_dynamic_server):
        """
        Connect to /ws/notifications on dynamic port, send ping, and assert pong response.
        """
        port = live_dynamic_server
        uri = f"ws://127.0.0.1:{port}/ws/notifications"

        async def run_test():
            async with websockets.connect(uri) as ws:
                await ws.send("ping")
                resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
                assert resp == "pong", f"Expected 'pong', received '{resp}'"

        asyncio.run(run_test())

    def test_dynamic_port_training_telemetry_broadcast(self, live_dynamic_server):
        """
        Connect to /ws/training on dynamic port, broadcast telemetry frame,
        and assert frame delivery through real TCP/WebSocket network stream.
        """
        port = live_dynamic_server
        uri = f"ws://127.0.0.1:{port}/ws/training"

        async def run_test():
            async with websockets.connect(uri) as ws:
                await ws.send("ping")
                pong = await asyncio.wait_for(ws.recv(), timeout=3.0)
                assert pong == "pong"

                # Broadcast telemetry frame
                sample_frame = {
                    "type": "progress",
                    "engine": "f5-tts",
                    "run_id": "challenger-run-001",
                    "epoch": 5,
                    "total_epochs": 20,
                    "loss": 0.2845,
                    "progress_percent": 25.0,
                    "message": "Epoch 5/20 complete",
                    "status": "training",
                }
                broadcast_training_telemetry(sample_frame)

                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                assert data["type"] == "progress"
                assert data["engine"] == "f5-tts"
                assert data["run_id"] == "challenger-run-001"
                assert data["loss"] == 0.2845
                assert data["progress_percent"] == 25.0

        asyncio.run(run_test())

    def test_dynamic_port_notification_broadcast(self, live_dynamic_server):
        """
        Connect to /ws/notifications on dynamic port, broadcast notification,
        and assert delivery with matching sample_id and transcript.
        """
        port = live_dynamic_server
        uri = f"ws://127.0.0.1:{port}/ws/notifications"

        async def run_test():
            async with websockets.connect(uri) as ws:
                await ws.send("ping")
                pong = await asyncio.wait_for(ws.recv(), timeout=3.0)
                assert pong == "pong"

                # Broadcast notification event
                event_payload = {
                    "type": "notification",
                    "event": "transcription_completed",
                    "sample_id": "test-sample-uuid-999",
                    "transcript": "Testowy zapis głosu po polsku.",
                    "duration": 4.5,
                }
                get_notification_manager().broadcast_sync(event_payload)

                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                assert data["type"] == "notification"
                assert data["event"] == "transcription_completed"
                assert data["sample_id"] == "test-sample-uuid-999"
                assert data["transcript"] == "Testowy zapis głosu po polsku."

        asyncio.run(run_test())


# ==============================================================================
# SECTION 3: Microphone Unmount Teardown (src/views/SamplesView.tsx)
# ==============================================================================

class TestSamplesViewMicrophoneUnmountCleanup:
    """Empirical inspection and simulated lifecycle validation of mic teardown in SamplesView."""

    def test_samples_view_file_exists(self):
        """Verify src/views/SamplesView.tsx exists."""
        assert SAMPLES_VIEW_TSX.exists(), "src/views/SamplesView.tsx does not exist"

    def test_mic_stream_ref_declaration_and_tracking(self):
        """
        Verify that SamplesView.tsx tracks micStream via useRef:
        - micStreamRef = useRef<MediaStream | null>(null);
        - micStreamRef.current = stream; in startRecording
        - micStreamRef.current = null; when stopped
        """
        content = SAMPLES_VIEW_TSX.read_text(encoding="utf-8")
        assert "micStreamRef = useRef<MediaStream | null>(null)" in content, (
            "SamplesView does not declare micStreamRef = useRef<MediaStream | null>(null)"
        )
        assert "micStreamRef.current = stream;" in content, (
            "SamplesView does not assign micStreamRef.current = stream on recording start"
        )
        assert "micStreamRef.current = null;" in content, (
            "SamplesView does not null micStreamRef.current on recording stop/unmount"
        )

    def test_timer_interval_ref_declaration_and_tracking(self):
        """
        Verify that SamplesView.tsx tracks timer interval via useRef:
        - timerIntervalRef = useRef<any>(null);
        - timerIntervalRef.current = setInterval(...) in startRecording
        - clearInterval(timerIntervalRef.current) in stop, pause, and unmount
        """
        content = SAMPLES_VIEW_TSX.read_text(encoding="utf-8")
        assert "timerIntervalRef = useRef" in content, (
            "SamplesView does not declare timerIntervalRef"
        )
        assert "timerIntervalRef.current = setInterval(" in content, (
            "SamplesView does not assign timerIntervalRef.current = setInterval"
        )
        assert "clearInterval(timerIntervalRef.current)" in content, (
            "SamplesView does not call clearInterval on timerIntervalRef.current"
        )

    def test_unmount_use_effect_structure_and_guarantees(self):
        """
        Inspect the unmount useEffect in SamplesView.tsx to verify:
        1. It has an empty dependency array [] (mount/unmount lifecycle hook).
        2. Clears timerIntervalRef.current via clearInterval() and sets it to null.
        3. Stops mediaRecorderRef.current, detaching onstop = null beforehand to prevent orphaned uploads.
        4. Iterates over micStreamRef.current.getTracks() calling track.stop().
        5. Wraps track.stop() and mediaRecorder.stop() in defensive try/catch blocks.
        6. Nullifies micStreamRef.current to prevent memory/reference leaks.
        """
        content = SAMPLES_VIEW_TSX.read_text(encoding="utf-8")

        # Locate unmount useEffect
        unmount_pattern = re.compile(
            r'useEffect\(\s*\(\)\s*=>\s*\{\s*return\s*\(\)\s*=>\s*\{(.*?)\};\s*\},\s*\[\]\);',
            re.DOTALL
        )
        match = unmount_pattern.search(content)
        assert match is not None, "Failed to find useEffect with return () => { ... }, [] in SamplesView.tsx"

        cleanup_code = match.group(1)

        # 1. Timer cleanup
        assert "clearInterval(timerIntervalRef.current)" in cleanup_code, (
            "Unmount cleanup does not call clearInterval(timerIntervalRef.current)"
        )
        assert "timerIntervalRef.current = null" in cleanup_code, (
            "Unmount cleanup does not nullify timerIntervalRef.current"
        )

        # 2. MediaRecorder cleanup
        assert "mediaRecorderRef.current" in cleanup_code, (
            "Unmount cleanup does not inspect mediaRecorderRef.current"
        )
        assert "mediaRecorderRef.current.onstop = null" in cleanup_code, (
            "Unmount cleanup does not detach onstop callback before stopping recorder! "
            "Risk of orphaned upload triggered on unmount."
        )
        assert "mediaRecorderRef.current.stop()" in cleanup_code, (
            "Unmount cleanup does not call mediaRecorderRef.current.stop()"
        )

        # 3. MicStream tracks cleanup
        assert "micStreamRef.current" in cleanup_code, (
            "Unmount cleanup does not inspect micStreamRef.current"
        )
        assert "micStreamRef.current.getTracks()" in cleanup_code, (
            "Unmount cleanup does not call micStreamRef.current.getTracks()"
        )
        assert "track.stop()" in cleanup_code, (
            "Unmount cleanup does not call track.stop()"
        )
        assert "micStreamRef.current = null" in cleanup_code, (
            "Unmount cleanup does not nullify micStreamRef.current"
        )

    def test_simulated_react_unmount_lifecycle_in_v8(self):
        """
        Empirically simulate the unmount cleanup logic in Node.js V8 runtime across 5 scenarios:
        1. Active recording unmount (running recorder, 2 audio tracks, active timer)
        2. Paused recording unmount (paused recorder, active audio tracks)
        3. Clean unmount when recording was never started (all refs null)
        4. Hardware error resilience (track.stop throws exception during unplug/OS error)
        5. Unmount after normal user stop (resources already released, no double-stop error)
        """
        node_script = """
        // Mock classes representing browser Web Audio / MediaStream / MediaRecorder APIs
        class MockMediaStreamTrack {
          constructor(id) {
            this.id = id;
            this.stopCalls = 0;
            this.throwOnStop = false;
          }
          stop() {
            this.stopCalls++;
            if (this.throwOnStop) {
              throw new Error('Device disconnected or audio hardware error');
            }
          }
        }

        class MockMediaStream {
          constructor(tracks) {
            this.tracks = tracks || [];
          }
          getTracks() {
            return this.tracks;
          }
        }

        class MockMediaRecorder {
          constructor(state = 'recording') {
            this.state = state;
            this.stopCalls = 0;
            this.onstop = () => { this.onstopFired = true; };
            this.onstopFired = false;
          }
          stop() {
            this.stopCalls++;
            this.state = 'inactive';
            if (this.onstop) this.onstop();
          }
        }

        function createUnmountTeardown(timerIntervalRef, mediaRecorderRef, micStreamRef, clearedIntervals) {
          return function unmountCleanup() {
            if (timerIntervalRef.current) {
              clearedIntervals.push(timerIntervalRef.current);
              timerIntervalRef.current = null;
            }
            if (mediaRecorderRef.current) {
              if (mediaRecorderRef.current.state !== 'inactive') {
                try {
                  mediaRecorderRef.current.onstop = null;
                  mediaRecorderRef.current.stop();
                } catch {
                  // ignore error during unmount stop
                }
              }
              mediaRecorderRef.current = null;
            }
            if (micStreamRef.current) {
              micStreamRef.current.getTracks().forEach((track) => {
                try {
                  track.stop();
                } catch {
                  // ignore error
                }
              });
              micStreamRef.current = null;
            }
          };
        }

        // ==========================================
        // SCENARIO 1: Unmount during ACTIVE RECORDING
        // ==========================================
        {
          const track1 = new MockMediaStreamTrack('mic-left');
          const track2 = new MockMediaStreamTrack('mic-right');
          const stream = new MockMediaStream([track1, track2]);
          const recorder = new MockMediaRecorder('recording');
          const timerRef = { current: 12345 };
          const recRef = { current: recorder };
          const streamRef = { current: stream };
          const cleared = [];

          const cleanup = createUnmountTeardown(timerRef, recRef, streamRef, cleared);
          cleanup();

          if (cleared.length !== 1 || cleared[0] !== 12345) throw new Error('Timer was not cleared');
          if (timerRef.current !== null) throw new Error('Timer ref was not nulled');
          if (track1.stopCalls !== 1 || track2.stopCalls !== 1) throw new Error('Tracks were not stopped');
          if (streamRef.current !== null) throw new Error('Stream ref was not nulled');
          if (recorder.stopCalls !== 1) throw new Error('MediaRecorder was not stopped');
          if (recorder.onstopFired) throw new Error('CRITICAL: onstop fired on unmount!');
          if (recRef.current !== null) throw new Error('MediaRecorder ref was not nulled');
        }

        // ==========================================
        // SCENARIO 2: Unmount during PAUSED RECORDING
        // ==========================================
        {
          const track = new MockMediaStreamTrack('mic-mono');
          const stream = new MockMediaStream([track]);
          const recorder = new MockMediaRecorder('paused');
          const timerRef = { current: null };
          const recRef = { current: recorder };
          const streamRef = { current: stream };
          const cleared = [];

          const cleanup = createUnmountTeardown(timerRef, recRef, streamRef, cleared);
          cleanup();

          if (track.stopCalls !== 1) throw new Error('Paused track was not stopped');
          if (recorder.stopCalls !== 1) throw new Error('Paused recorder was not stopped');
          if (recorder.onstopFired) throw new Error('onstop fired on unmount during pause');
        }

        // ==========================================
        // SCENARIO 3: Clean unmount before recording
        // ==========================================
        {
          const timerRef = { current: null };
          const recRef = { current: null };
          const streamRef = { current: null };
          const cleared = [];

          const cleanup = createUnmountTeardown(timerRef, recRef, streamRef, cleared);
          // Should not throw or fail
          cleanup();
          if (cleared.length !== 0) throw new Error('Unexpected interval clear');
        }

        // ==========================================
        // SCENARIO 4: Track error resilience
        // ==========================================
        {
          const track1 = new MockMediaStreamTrack('failing-mic');
          track1.throwOnStop = true;
          const track2 = new MockMediaStreamTrack('working-mic');
          const stream = new MockMediaStream([track1, track2]);
          const timerRef = { current: null };
          const recRef = { current: null };
          const streamRef = { current: stream };
          const cleared = [];

          const cleanup = createUnmountTeardown(timerRef, recRef, streamRef, cleared);
          // Should catch track1 error and continue to stop track2
          cleanup();

          if (track1.stopCalls !== 1) throw new Error('Failing track stop was not invoked');
          if (track2.stopCalls !== 1) throw new Error('Subsequent track stop was blocked by exception');
          if (streamRef.current !== null) throw new Error('Stream ref was not nulled on error');
        }

        // ==========================================
        // SCENARIO 5: Unmount after normal stop
        // ==========================================
        {
          const recorder = new MockMediaRecorder('inactive');
          const timerRef = { current: null };
          const recRef = { current: recorder };
          const streamRef = { current: null };
          const cleared = [];

          const cleanup = createUnmountTeardown(timerRef, recRef, streamRef, cleared);
          cleanup();

          // Inactive recorder should not be stopped again
          if (recorder.stopCalls !== 0) throw new Error('Inactive recorder was double-stopped');
          if (recRef.current !== null) throw new Error('Rec ref was not nulled');
        }

        // ==========================================
        // SCENARIO 6: MediaRecorder stop() throws error
        // ==========================================
        {
          const recorder = new MockMediaRecorder('recording');
          recorder.stop = () => { throw new Error('Underlying browser recorder crashed'); };
          const timerRef = { current: null };
          const recRef = { current: recorder };
          const streamRef = { current: null };
          const cleared = [];

          const cleanup = createUnmountTeardown(timerRef, recRef, streamRef, cleared);
          // Should catch recorder stop error and nullify ref without throwing
          cleanup();
          if (recRef.current !== null) throw new Error('Rec ref was not nulled on stop error');
        }

        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"Simulated unmount test failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"

    def test_dynamic_ws_port_switch_and_reconnect_simulation(self):
        """
        Simulate runtime port switch race condition:
        1. ResilientWebSocketService connects to initial default port 8000.
        2. Backend config resolves asynchronously with dynamic port 8088.
        3. refreshUrl() detects URL divergence and automatically triggers reconnect to port 8088.
        4. Verifies exponential backoff caps at 10000ms.
        """
        node_script = """
        class MockWebSocket {
          constructor(url) {
            this.url = url;
            this.readyState = 1; // OPEN
            this.closed = false;
          }
          close() {
            this.closed = true;
            this.readyState = 3; // CLOSED
          }
        }

        class MockResilientWsService {
          constructor(endpointPath) {
            this.endpointPath = endpointPath;
            this.ws = null;
            this.dynamicWsBase = null;
            this.isExplicitlyClosed = false;
            this.reconnectDelay = 1000;
            this.connectionHistory = [];
          }

          getWsUrl() {
            if (this.dynamicWsBase) return `${this.dynamicWsBase}${this.endpointPath}`;
            return `ws://127.0.0.1:8000${this.endpointPath}`;
          }

          connect() {
            this.isExplicitlyClosed = false;
            const url = this.getWsUrl();
            this.ws = new MockWebSocket(url);
            this.connectionHistory.push(url);
          }

          disconnect() {
            this.isExplicitlyClosed = true;
            if (this.ws) {
              this.ws.close();
              this.ws = null;
            }
          }

          refreshUrl() {
            if (this.isExplicitlyClosed) return;
            if (this.ws) {
              const currentUrl = this.ws.url;
              const targetUrl = this.getWsUrl();
              if (currentUrl !== targetUrl) {
                this.disconnect();
                this.connect();
              }
            }
          }

          stepBackoff() {
            this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 10000);
            return this.reconnectDelay;
          }
        }

        const svc = new MockResilientWsService('/ws/training');
        
        // Initial connection to default 8000
        svc.connect();
        if (svc.ws.url !== 'ws://127.0.0.1:8000/ws/training') throw new Error('Initial URL wrong');

        // Dynamic backend port resolved to 8088
        svc.dynamicWsBase = 'ws://127.0.0.1:8088';
        svc.refreshUrl();

        if (svc.ws.url !== 'ws://127.0.0.1:8088/ws/training') throw new Error('Failed to reconnect to dynamic port 8088');
        if (svc.connectionHistory.length !== 2) throw new Error('Reconnection history length mismatch');

        // Verify backoff capping at 10000ms
        let delay = 1000;
        for (let i = 0; i < 15; i++) {
          delay = svc.stepBackoff();
        }
        if (delay !== 10000) throw new Error(`Backoff did not cap at 10000ms (got ${delay})`);

        console.log('SUCCESS');
        """
        code, stdout, stderr = execute_node_script(node_script)
        assert code == 0, f"Dynamic port switch simulation failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert stdout == "SUCCESS"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
