"""
tests/test_m5_challenger_frontend.py

Empirical Challenge & Verification Test Suite for Milestone 5 Frontend:
- Static TypeScript Compilation (Renderer & Electron)
- Build Output & Asset Reference Integrity (Relative Paths for file://)
- Electron Bundle Compatibility (CommonJS syntax, no broken ESM)
- Security & Leakage Audit (No hardcoded credentials, source map leakage detection)
- React 19 Component Contracts (JSX key props, null-safety, cleanup, diacritics)
- API Client Contract Alignment with FastAPI Backend

Authored by Challenger M5-1.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
DIST_ELECTRON_DIR = PROJECT_ROOT / "dist-electron"
SRC_DIR = PROJECT_ROOT / "src"
ELECTRON_SRC_DIR = PROJECT_ROOT / "electron"


# ==============================================================================
# Helper Functions
# ==============================================================================

def run_cmd(cmd: List[str], cwd: Path = PROJECT_ROOT, timeout: float = 60.0) -> Tuple[int, str, str]:
    """Execute shell command and return (returncode, stdout, stderr)."""
    shell_mode = sys.platform == "win32"
    res = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        shell=shell_mode,
        timeout=timeout,
    )
    return res.returncode, res.stdout.strip(), res.stderr.strip()


# ==============================================================================
# SECTION 1: Static TypeScript Checks & Build Pipeline
# ==============================================================================

class TestStaticTypesAndCompilation:
    """Empirical static compilation checks for renderer and Electron processes."""

    def test_tsc_renderer_typecheck_zero_errors(self):
        """Verify `npx tsc --noEmit` runs with 0 errors across all src/ files."""
        code, stdout, stderr = run_cmd(["npx", "tsc", "--noEmit"])
        assert code == 0, f"TypeScript typecheck failed (exit {code}):\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert "error TS" not in stdout, f"TypeScript emitted errors:\n{stdout}"

    def test_tsc_electron_typecheck_zero_errors(self):
        """Verify `npx tsc -p tsconfig.node.json --noEmit` runs with 0 errors for electron/."""
        code, stdout, stderr = run_cmd(["npx", "tsc", "-p", "tsconfig.node.json", "--noEmit"])
        assert code == 0, f"Electron TypeScript typecheck failed (exit {code}):\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        assert "error TS" not in stdout, f"Electron TS emitted errors:\n{stdout}"

    def test_production_build_succeeds(self):
        """Verify `npm run build` generates production bundle cleanly."""
        code, stdout, stderr = run_cmd(["npm", "run", "build"], timeout=90.0)
        assert code == 0, f"npm run build failed (exit {code}):\nSTDOUT: {stdout}\nSTDERR: {stderr}"


# ==============================================================================
# SECTION 2: Build Output & Asset Integrity
# ==============================================================================

class TestBuildArtifactsAndBundleIntegrity:
    """Empirical inspection of dist/ and dist-electron/ bundle contents."""

    def test_dist_index_html_exists(self):
        """Verify that dist/index.html is created and non-empty."""
        index_html = DIST_DIR / "index.html"
        assert index_html.exists(), "dist/index.html does not exist"
        assert index_html.stat().st_size > 100, "dist/index.html is suspiciously small or empty"

    def test_dist_index_html_loads_assets_via_relative_paths(self):
        """
        Verify that dist/index.html loads bundled scripts and styles via relative paths (./assets/...)
        and NOT absolute paths (/assets/...). Absolute paths fail in packaged Electron under file://.
        """
        index_html = DIST_DIR / "index.html"
        assert index_html.exists(), "dist/index.html missing"
        content = index_html.read_text(encoding="utf-8")

        # Find script tags with src
        script_srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', content)
        assert len(script_srcs) > 0, "No <script src=...> found in dist/index.html"
        for src in script_srcs:
            assert src.startswith("./assets/") or not src.startswith("/"), (
                f"Script src '{src}' is not relative! Must start with './' to work in Electron file:// protocol."
            )

        # Find link tags with href for stylesheets
        css_hrefs = re.findall(r'<link[^>]+rel=["\']stylesheet["\'][^>]+href=["\']([^"\']+)["\']', content)
        if not css_hrefs:
            css_hrefs = re.findall(r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']stylesheet["\']', content)
        assert len(css_hrefs) > 0, "No stylesheet <link href=...> found in dist/index.html"
        for href in css_hrefs:
            assert href.startswith("./assets/") or not href.startswith("/"), (
                f"CSS href '{href}' is not relative! Must start with './' to work in Electron file:// protocol."
            )

    def test_dist_electron_scripts_exist_and_commonjs(self):
        """Verify dist-electron/main.js and preload.js exist and are valid CommonJS."""
        main_js = DIST_ELECTRON_DIR / "main.js"
        preload_js = DIST_ELECTRON_DIR / "preload.js"

        assert main_js.exists(), "dist-electron/main.js does not exist"
        assert preload_js.exists(), "dist-electron/preload.js does not exist"

        # Syntax check via node -c
        code_main, _, err_main = run_cmd(["node", "-c", str(main_js)])
        assert code_main == 0, f"dist-electron/main.js has syntax errors: {err_main}"

        code_pre, _, err_pre = run_cmd(["node", "-c", str(preload_js)])
        assert code_pre == 0, f"dist-electron/preload.js has syntax errors: {err_pre}"

        # Check for CommonJS signatures and absence of top-level ESM exports
        main_content = main_js.read_text(encoding="utf-8")
        assert "require(" in main_content, "main.js does not appear to be CommonJS (missing require)"
        assert not re.search(r"^\s*export\s+(default|const|let|var|function|class)\b", main_content, re.M), (
            "main.js contains raw ESM export statements!"
        )

        preload_content = preload_js.read_text(encoding="utf-8")
        assert "require(" in preload_content, "preload.js missing require()"
        assert not re.search(r"^\s*export\s+(default|const|let|var|function|class)\b", preload_content, re.M), (
            "preload.js contains raw ESM export statements!"
        )

    def test_no_credentials_or_secrets_leaked_in_bundles(self):
        """Verify no passwords, API tokens, bearer keys, or secret credentials are in production bundles."""
        js_files = list((DIST_DIR / "assets").glob("*.js"))
        assert len(js_files) > 0, "No JS files found in dist/assets"

        sensitive_patterns = [
            re.compile(r'(?:api[_-]?key|secret_key|private_key|auth_token)\s*[:=]\s*["\'][A-Za-z0-9_\-\.]{12,}["\']', re.I),
            re.compile(r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*', re.I),
            re.compile(r'-----BEGIN (?:RSA )?PRIVATE KEY-----'),
        ]

        for js_file in js_files:
            content = js_file.read_text(encoding="utf-8")
            for pattern in sensitive_patterns:
                matches = pattern.findall(content)
                assert len(matches) == 0, (
                    f"Possible secret credential leaked in bundle {js_file.name}: {matches[:3]}"
                )

    def test_source_map_leakage_audit(self):
        """
        Audit for raw source maps in production output:
        Empirically checks if dist/ contains .map files or sourceMappingURL references.
        When build.sourcemap is enabled in vite.config.ts, .map files are emitted into dist/assets.
        """
        map_files = list(DIST_DIR.glob("**/*.map"))
        # In production bundles, shipping raw source maps exposes the entire source tree
        # We record whether raw .map files are present
        if map_files:
            file_names = [f.name for f in map_files]
            total_size_kb = sum(f.stat().st_size for f in map_files) / 1024.0
            print(f"\n[CHALLENGE OBSERVATION]: Found {len(map_files)} raw source map files ({total_size_kb:.1f} KB): {file_names}")
            # This directly identifies the finding requested in mission item 1:
            # "Verify that no raw source maps or debug credentials leak into production bundles"
            assert len(map_files) == 0, (
                f"Raw source map files leaked into production dist/ bundle ({len(map_files)} files, "
                f"{total_size_kb:.1f} KB). Files: {file_names}. In vite.config.ts, 'build.sourcemap' is true."
            )


# ==============================================================================
# SECTION 3: Frontend Component Contracts & Static Code Analysis
# ==============================================================================

class TestFrontendComponentContracts:
    """Stress-test React component contracts, JSX key props, null safety, diacritics, and cleanup."""

    def test_all_jsx_list_iterations_have_key_props(self):
        """
        Inspect all .tsx files in src/ for array .map() rendering JSX.
        Ensure that every single mapped element has an explicit key prop.
        """
        tsx_files = list(SRC_DIR.glob("**/*.tsx"))
        assert len(tsx_files) > 0, "No TSX files found in src/"

        map_jsx_pattern = re.compile(r'\.map\s*\(\s*(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>\s*(?:\{[^{}]*return\s+|\s*)(<[a-zA-Z0-9_]+)')
        
        missing_keys = []
        for file in tsx_files:
            content = file.read_text(encoding="utf-8")
            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if ".map(" in line and not line.strip().startswith("//"):
                    # Check the next 4 lines for JSX tag and key= prop
                    chunk = "\n".join(lines[idx:idx + 5])
                    # If it returns a JSX element like <tr, <div, <button, <option, <label
                    tag_match = re.search(r'=>\s*(?:\(?\s*)<([a-zA-Z0-9_]+)', chunk)
                    if tag_match:
                        tag_name = tag_match.group(1)
                        if tag_name not in ("React.Fragment",):
                            # Must contain key=
                            if "key=" not in chunk:
                                missing_keys.append(f"{file.name}:{idx + 1} -> <{tag_name}> missing key prop")

        assert len(missing_keys) == 0, f"Found JSX list iterations without key prop:\n" + "\n".join(missing_keys)

    def test_waveform_visualizer_safety_and_resource_teardown(self):
        """
        Verify that WaveformVisualizer:
        1. Never connects audio source to audioContext.destination (feedback prevention).
        2. Closes AudioContext in cleanup.
        3. Cancels animation frames on unmount/stop.
        4. Handles webkitAudioContext fallback safely.
        """
        file_path = SRC_DIR / "components" / "WaveformVisualizer.tsx"
        assert file_path.exists(), "WaveformVisualizer.tsx not found"
        content = file_path.read_text(encoding="utf-8")

        # Feedback prevention: source.connect must connect to analyser, NEVER destination
        assert "source.connect(analyser)" in content, "Missing source.connect(analyser)"
        assert "source.connect(audioCtx.destination)" not in content, (
            "CRITICAL: WaveformVisualizer connects source to audioCtx.destination, causing microphone feedback loop!"
        )

        # Cleanup verification
        assert "audioContextRef.current.close()" in content or "audioCtx.close()" in content, (
            "WaveformVisualizer does not close AudioContext on teardown!"
        )
        assert "cancelAnimationFrame" in content, "Missing cancelAnimationFrame cleanup"
        assert "sourceRef.current.disconnect()" in content or "source.disconnect()" in content, (
            "Missing source node disconnection"
        )

    def test_polish_diacritics_support_in_views(self):
        """
        Verify that Polish diacritics are provided across Views:
        - ą, ć, ę, ł, ń, ó, ś, ź, ż (lowercase)
        - Ą, Ć, Ę, Ł, Ń, Ó, Ś, Ź, Ż (uppercase)
        - Both SamplesView and GenerateView have diacritic toolbars
        """
        samples_view = (SRC_DIR / "views" / "SamplesView.tsx").read_text(encoding="utf-8")
        generate_view = (SRC_DIR / "views" / "GenerateView.tsx").read_text(encoding="utf-8")

        diacritics_lower = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż']
        diacritics_upper = ['Ą', 'Ć', 'Ę', 'Ł', 'Ń', 'Ó', 'Ś', 'Ź', 'Ż']

        for char in diacritics_lower + diacritics_upper:
            assert char in samples_view, f"Polish diacritic '{char}' missing from SamplesView.tsx"
            assert char in generate_view, f"Polish diacritic '{char}' missing from GenerateView.tsx"

    def test_unicode_nfc_normalization_in_api_client(self):
        """
        Verify that transcripts are normalized to Unicode NFC before dispatching to backend.
        This prevents mismatched combining diacritics from breaking Whisper and F5-TTS vocabularies.
        """
        client_code = (SRC_DIR / "api" / "client.ts").read_text(encoding="utf-8")
        assert "normalize('NFC')" in client_code, (
            "client.ts does not normalize Polish text to NFC before API dispatch"
        )

    def test_settings_view_null_safety(self):
        """
        Verify that SettingsView handles missing GPU info, null VRAM, and disconnected backend
        without throwing unhandled exceptions.
        """
        settings_code = (SRC_DIR / "views" / "SettingsView.tsx").read_text(encoding="utf-8")
        # Null-safe access checks
        assert "gpuInfo?.cuda_available" in settings_code or "gpuInfo &&" in settings_code
        assert "formatVram" in settings_code
        assert "formatUptime" in settings_code
        assert "DISCONNECTED" in settings_code
        assert "SIMULATION MODE" in settings_code

    def test_audio_player_safe_division_and_error_handling(self):
        """
        Verify AudioPlayer guards against duration = 0 division by zero and catches play() rejections.
        """
        player_code = (SRC_DIR / "components" / "AudioPlayer.tsx").read_text(encoding="utf-8")
        assert "audio.play().catch(" in player_code, "AudioPlayer does not catch audio.play() promise rejection"
        assert "duration > 0 ? duration : 1" in player_code or "duration || 1" in player_code, (
            "AudioPlayer does not guard scrubber max against duration <= 0"
        )


# ==============================================================================
# SECTION 4: API Client Contract Alignment with FastAPI Backend
# ==============================================================================

class TestApiClientContractAlignment:
    """Verify that all REST API endpoints defined in frontend match backend routers."""

    def test_frontend_endpoints_exist_in_backend_routers(self):
        """
        Extract API endpoint paths from src/api/client.ts and verify they correspond
        to registered routes in the FastAPI application.
        """
        client_code = (SRC_DIR / "api" / "client.ts").read_text(encoding="utf-8")

        # Extract all string literals matching /api/...
        called_endpoints = set(re.findall(r'[\'"`](/api/[a-zA-Z0-9_\-\/]+)', client_code))
        assert len(called_endpoints) >= 12, f"Expected at least 12 API endpoints, found: {called_endpoints}"

        from backend.app.main import create_app
        app = create_app()

        registered_paths = set()
        for route in app.routes:
            if hasattr(route, 'path'):
                registered_paths.add(route.path.rstrip('/'))
            elif hasattr(route, 'original_router') and hasattr(route, 'include_context'):
                prefix = route.include_context.prefix
                for sub in route.original_router.routes:
                    sub_path = getattr(sub, 'path', '')
                    registered_paths.add((prefix + sub_path).rstrip('/'))

        assert len(registered_paths) > 0, "No registered paths found in FastAPI app!"

        # Normalize frontend template literals like /api/samples/${sampleId} -> /api/samples/{...}
        for endpoint in called_endpoints:
            # Strip trailing slash
            ep = endpoint.rstrip("/")
            # Check direct match or parameterized match
            matched = False
            for reg in registered_paths:
                # Direct match
                if ep == reg:
                    matched = True
                    break
                # Parameterized match e.g. /api/samples/{sample_id} vs /api/samples
                if reg.startswith(ep + "/{") or ep.startswith(reg.split("/{")[0]):
                    matched = True
                    break
            assert matched, f"Frontend calls endpoint '{endpoint}', but no matching route found in FastAPI app routes: {sorted(registered_paths)}"


# ==============================================================================
# SECTION 5: Electron Security & Supervisor Settings
# ==============================================================================

class TestElectronSecurityConfiguration:
    """Verify BrowserWindow security parameters and IPC boundaries in electron/main.ts."""

    def test_browser_window_security_webpreferences(self):
        """
        Verify BrowserWindow security flags:
        - contextIsolation: true
        - nodeIntegration: false
        - sandbox: true
        - webSecurity: true
        """
        main_ts = (ELECTRON_SRC_DIR / "main.ts").read_text(encoding="utf-8")
        assert "contextIsolation: true" in main_ts, "Security violation: contextIsolation is not true"
        assert "nodeIntegration: false" in main_ts, "Security violation: nodeIntegration is not false"
        assert "sandbox: true" in main_ts, "Security violation: sandbox is not true"
        assert "webSecurity: true" in main_ts, "Security violation: webSecurity is not true"

    def test_open_external_protocol_validation(self):
        """
        Verify openExternal handler only allows http:// and https:// URLs.
        Arbitrary protocols like file:// or custom URIs could allow command execution.
        """
        main_ts = (ELECTRON_SRC_DIR / "main.ts").read_text(encoding="utf-8")
        assert "url.startsWith('http://') || url.startsWith('https://')" in main_ts, (
            "openExternal does not validate protocols to only http/https!"
        )

    def test_media_permission_request_handler(self):
        """
        Verify session permission request handler specifically grants permission for 'media'
        to allow getUserMedia microphone capture.
        """
        main_ts = (ELECTRON_SRC_DIR / "main.ts").read_text(encoding="utf-8")
        assert "setPermissionRequestHandler" in main_ts
        assert "permission === 'media'" in main_ts


if __name__ == "__main__":
    pytest.main(["-v", __file__])
