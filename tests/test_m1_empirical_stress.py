"""
tests/test_m1_empirical_stress.py
Empirical Stress and Adversarial Edge Case Verification Suite for Milestone 1.
Authored by challenger_m1_1.

Covers:
1. Concurrent SQLite reads & writes from multiple threads simultaneously (WAL mode resilience).
2. Malformed / unexpected queries to /api/settings/gpu and /api/health.
3. Unicode edge cases in repository operations (full Polish alphabet with diacritics: Zażółć gęślą jaźń).
4. Testing dev/simulation flag overrides (VOICE_SAMPLER_DEV_MODE=0 vs 1, SIMULATE_TRAINING, VOICE_SAMPLER_SIMULATION_MODE).
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List
import unicodedata
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.db.connection import (
    checkpoint_db,
    get_db_connection,
    init_storage_directories,
    set_db_path,
)
from backend.app.db.repository import (
    complete_training_run,
    count_generations,
    count_models,
    count_samples,
    create_generation,
    create_model,
    create_sample,
    create_training_run,
    delete_generation,
    delete_model,
    delete_sample,
    get_active_training_run,
    get_generation,
    get_model,
    get_model_by_name,
    get_sample,
    get_training_run,
    init_db,
    list_generations,
    list_models,
    list_samples,
    list_training_runs,
    update_sample,
    update_sample_transcript,
    update_training_run_progress,
)
from backend.app.main import create_app
from backend.app.services.gpu_service import GPUService, get_gpu_service


# ==============================================================================
# SECTION 1: CONCURRENT SQLITE READS & WRITES FROM MULTIPLE THREADS
# ==============================================================================

class TestConcurrentSQLiteStress:
    """Stress tests concurrent SQLite operations in WAL mode with high thread count."""

    def test_concurrent_threads_heavy_read_write(self, tmp_path: Path):
        """
        Spawns 20 concurrent worker threads (10 writers, 10 readers) executing
        rapid simultaneous transactions against a real file-based SQLite database.
        Synchronized with a threading Barrier to trigger maximum write contention.
        """
        db_file = tmp_path / "stress_concurrency.db"
        init_db(db_path=db_file)

        num_threads = 20
        iterations_per_thread = 40
        barrier = threading.Barrier(num_threads)
        errors: List[Exception] = []
        errors_lock = threading.Lock()

        def writer_worker(thread_idx: int):
            try:
                barrier.wait(timeout=5.0)
                for i in range(iterations_per_thread):
                    # Write operation 1: create sample
                    sid = f"sample_t{thread_idx}_{i}"
                    create_sample(
                        name=f"Wątek {thread_idx} Próbka {i}",
                        filename=f"audio_t{thread_idx}_{i}.wav",
                        file_path=f"/data/samples/audio_t{thread_idx}_{i}.wav",
                        duration_seconds=3.14 + (i * 0.1),
                        transcript=f"Transkrypcja współbieżna krok {i}",
                        sample_id=sid,
                        db_path=db_file,
                    )

                    # Write operation 2: update transcript
                    update_sample_transcript(
                        sample_id=sid,
                        transcript=f"Zaktualizowana transkrypcja krok {i} dla wątku {thread_idx}",
                        status="verified",
                        db_path=db_file,
                    )

                    # Write operation 3: create training run
                    rid = f"run_t{thread_idx}_{i}"
                    create_training_run(
                        engine="f5-tts" if i % 2 == 0 else "rvc",
                        model_name=f"model_t{thread_idx}_{i}",
                        total_epochs=10,
                        total_steps=100,
                        run_id=rid,
                        db_path=db_file,
                    )

                    # Write operation 4: update training run progress
                    update_training_run_progress(
                        run_id=rid,
                        current_epoch=1,
                        current_step=10,
                        current_loss=0.45,
                        progress_percent=10.0,
                        status_message="Trening w toku",
                        db_path=db_file,
                    )

                    # Intermittent small yield
                    time.sleep(0.001)

            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def reader_worker(thread_idx: int):
            try:
                barrier.wait(timeout=5.0)
                for _ in range(iterations_per_thread):
                    # Read operation 1: count samples
                    _ = count_samples(db_path=db_file)

                    # Read operation 2: list samples
                    samples = list_samples(db_path=db_file)
                    if samples:
                        # Read operation 3: get sample by id
                        _ = get_sample(samples[0]["id"], db_path=db_file)

                    # Read operation 4: list training runs
                    _ = list_training_runs(limit=10, db_path=db_file)

                    # Read operation 5: get active training run
                    _ = get_active_training_run(db_path=db_file)

                    time.sleep(0.001)

            except Exception as e:
                with errors_lock:
                    errors.append(e)

        threads: List[threading.Thread] = []
        # 10 writers
        for t in range(10):
            th = threading.Thread(target=writer_worker, args=(t,))
            threads.append(th)
        # 10 readers
        for t in range(10, 20):
            th = threading.Thread(target=reader_worker, args=(t,))
            threads.append(th)

        # Start all threads
        for th in threads:
            th.start()

        # Await completion
        for th in threads:
            th.join(timeout=30.0)

        # Assert no lockouts, deadlock, or operational errors occurred
        assert len(errors) == 0, f"Concurrent stress test encountered {len(errors)} errors: {errors}"

        # Verify database consistency after stress
        with get_db_connection(db_file) as conn:
            integrity = conn.execute("PRAGMA integrity_check;").fetchone()
            assert integrity[0] == "ok", f"Integrity check failed: {integrity[0]}"

            fk_check = conn.execute("PRAGMA foreign_key_check;").fetchall()
            assert len(fk_check) == 0, f"Foreign key check failed: {fk_check}"

            # Verify total records written (10 writers * 40 iterations = 400 samples)
            sample_cnt = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            assert sample_cnt == 400, f"Expected 400 samples, found {sample_cnt}"

            run_cnt = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            assert run_cnt == 400, f"Expected 400 training runs, found {run_cnt}"

        # Verify WAL checkpoint under post-stress condition
        checkpoint_db(db_path=db_file)


# ==============================================================================
# SECTION 2: MALFORMED / UNEXPECTED QUERIES TO /api/settings/gpu & /api/health
# ==============================================================================

class TestMalformedQueriesAndInputs:
    """Stress tests API endpoints against malformed, invalid, unexpected inputs."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health_malformed_and_unexpected_inputs(self):
        """Probes /api/health with unexpected parameters, methods, bodies, and headers."""
        # 1. Unexpected query parameters (should safely ignore extra query params and return 200)
        res = self.client.get("/api/health?random=true&foo=bar&backend_ready=false&inject=<script>")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["backend_ready"] is True

        # 2. Sending unexpected JSON body to GET request
        res = self.client.request(
            "GET",
            "/api/health",
            json={"malicious_key": "malicious_payload", "admin": True},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # 3. Disallowed HTTP methods
        for method in ["POST", "PUT", "DELETE", "PATCH"]:
            res = self.client.request(method, "/api/health")
            assert res.status_code == 405, f"Method {method} returned {res.status_code}, expected 405"

        # 4. Unusual / malformed Accept headers
        res = self.client.get("/api/health", headers={"Accept": "application/xml, text/csv; q=0.8"})
        assert res.status_code == 200

        # 5. Excessively long header
        res = self.client.get("/api/health", headers={"X-Stress-Header": "A" * 8192})
        assert res.status_code == 200

    def test_gpu_settings_malformed_and_unexpected_inputs(self):
        """Probes /api/settings/gpu with unexpected data types, methods, and queries."""
        # 1. Valid boolean query parameter coercions
        for val in ["true", "True", "1", "TRUE", "yes"]:
            res = self.client.get(f"/api/settings/gpu?refresh={val}")
            assert res.status_code == 200
            assert "cuda_available" in res.json()

        for val in ["false", "False", "0", "FALSE", "no"]:
            res = self.client.get(f"/api/settings/gpu?refresh={val}")
            assert res.status_code == 200
            assert "cuda_available" in res.json()

        # 2. Invalid boolean query parameter (MUST return 422 Unprocessable Entity, NEVER 500)
        invalid_refresh_vals = [
            "notabool",
            "12345",
            "null",
            "undefined",
            "true_or_false",
            "{}",
            "[]",
        ]
        for val in invalid_refresh_vals:
            res = self.client.get(f"/api/settings/gpu?refresh={val}")
            assert res.status_code == 422, f"Expected 422 for refresh='{val}', got {res.status_code}"
            assert res.json().get("error_code") == "VALIDATION_ERROR"

        # 3. Extremely long query string
        huge_val = "x" * 10000
        res = self.client.get(f"/api/settings/gpu?refresh={huge_val}")
        assert res.status_code == 422
        assert res.json().get("error_code") == "VALIDATION_ERROR"

        # 4. Unknown query parameters should be silently ignored
        res = self.client.get("/api/settings/gpu?unknown_flag=test&gpu_override=RTX5090&vram=128000")
        assert res.status_code == 200
        assert "cuda_available" in res.json()

        # 5. Method not allowed for /api/settings/gpu
        for method in ["POST", "PUT", "DELETE", "PATCH"]:
            res = self.client.request(method, "/api/settings/gpu")
            assert res.status_code == 405, f"Method {method} returned {res.status_code}, expected 405"

    def test_general_settings_disallowed_methods(self):
        """Probes /api/settings base endpoint."""
        res = self.client.get("/api/settings")
        assert res.status_code == 200
        data = res.json()
        assert "app_name" in data
        assert "gpu" in data

        for method in ["POST", "PUT", "DELETE", "PATCH"]:
            res = self.client.request(method, "/api/settings")
            assert res.status_code == 405


# ==============================================================================
# SECTION 3: UNICODE EDGE CASES & POLISH DIACRITICS ("Zażółć gęślą jaźń")
# ==============================================================================

class TestUnicodePolishDiacriticsStress:
    """Stress tests full Polish alphabet with diacritics and edge-case Unicode in DB."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path: Path):
        self.db_path = tmp_path / "unicode_stress.db"
        init_db(db_path=self.db_path)

    def test_full_polish_alphabet_pangrams_and_diacritics(self):
        """
        Verify correct storage, retrieval, filtering, and indexing of all 9 Polish diacritics
        in lowercase and uppercase across all database models and tables.
        """
        # All 9 Polish diacritics in lowercase and uppercase
        polish_chars_lower = "ąćęłńóśźż"
        polish_chars_upper = "ĄĆĘŁŃÓŚŹŻ"
        pangram = "Zażółć gęślą jaźń"
        pangram_upper = "ZAŻÓŁĆ GĘŚLĄ JAŹŃ"
        tongue_twister = "W Szczebrzeszynie chrząszcz brzmi w trzcinie i Szczebrzeszyn z tego słynie."
        poetic_passage = (
            "Litwo! Ojczyzno moja! ty jesteś jak zdrowie. "
            "Ile cię trzeba cenić, ten tylko się dowie, kto cię stracił. "
            "Dziś piękność twą w całej ozdobie widzę i opisuję, bo tęsknię po tobie."
        )

        # 1. Samples Table
        sample = create_sample(
            name=f"Nagranie: {pangram}",
            filename=f"{pangram.replace(' ', '_')}.wav",
            file_path=f"/data/samples/{pangram.replace(' ', '_')}.wav",
            preprocessed_path=f"/data/samples/prep_{pangram.replace(' ', '_')}.wav",
            duration_seconds=12.5,
            transcript=f"{pangram} | {pangram_upper} | {tongue_twister} | {poetic_passage}",
            sample_id="sample_polish_diacritics",
            db_path=self.db_path,
        )

        assert sample is not None
        assert sample["name"] == f"Nagranie: {pangram}"
        assert pangram in sample["transcript"]
        for ch in polish_chars_lower:
            assert ch in sample["transcript"].lower()
        for ch in polish_chars_upper:
            assert ch in sample["transcript"]

        # Update sample transcript with uppercase pangram
        updated_sample = update_sample_transcript(
            sample_id="sample_polish_diacritics",
            transcript=f"{pangram_upper} (Zaktualizowano pomyślnie)",
            status="manual_edycja_żółć",
            db_path=self.db_path,
        )
        assert updated_sample["transcript"] == f"{pangram_upper} (Zaktualizowano pomyślnie)"
        assert updated_sample["transcript_status"] == "manual_edycja_żółć"

        # Retrieve and verify list_samples with custom status containing diacritics
        filtered_samples = list_samples(status="manual_edycja_żółć", db_path=self.db_path)
        assert len(filtered_samples) == 1
        assert filtered_samples[0]["id"] == "sample_polish_diacritics"

        # 2. Models Table
        model_name = f"Model_{pangram.replace(' ', '_')}_v1"
        model = create_model(
            name=model_name,
            engine="f5-tts",
            checkpoint_path=f"/data/models/f5_tts/{model_name}.pt",
            parameters={
                "język": "polski",
                "częstotliwość_próbkowania": 40000,
                "znaki_specjalne": polish_chars_lower + polish_chars_upper,
                "opis": f"Wytrenowano z użyciem: {tongue_twister}",
            },
            model_id="model_polish_diacritics",
            db_path=self.db_path,
        )
        assert model["name"] == model_name
        assert model["parameters"]["język"] == "polski"
        assert model["parameters"]["częstotliwość_próbkowania"] == 40000

        # Retrieve model by exact Polish name
        fetched_model = get_model_by_name(model_name, db_path=self.db_path)
        assert fetched_model is not None
        assert fetched_model["id"] == "model_polish_diacritics"

        # 3. Training Runs Table
        run = create_training_run(
            engine="f5-tts",
            model_name=model_name,
            config={"język": "pl", "opis": pangram},
            run_id="run_polish_diacritics",
            db_path=self.db_path,
        )
        assert run["config"]["opis"] == pangram

        # Update training run with Polish status message
        updated_run = update_training_run_progress(
            run_id="run_polish_diacritics",
            current_epoch=5,
            current_step=50,
            current_loss=0.123,
            progress_percent=50.0,
            status_message=f"Trwa ekstrakcja cech F0 (RMVPE) dla próbki: {pangram}",
            db_path=self.db_path,
        )
        assert pangram in updated_run["status_message"]

        # Complete run with Polish error message
        completed_run = complete_training_run(
            run_id="run_polish_diacritics",
            status="completed",
            error_message="Brak błędów. Zażółć gęślą jaźń pomyślnie przetworzona.",
            db_path=self.db_path,
        )
        assert "Brak błędów" in completed_run["error_message"]

        # 4. Generations Table
        gen = create_generation(
            type="tts",
            model_id=model["id"],
            output_audio_path=f"/data/generated/tts/{pangram.replace(' ', '_')}.wav",
            output_filename=f"{pangram.replace(' ', '_')}.wav",
            input_text=poetic_passage,
            reference_sample_id=sample["id"],
            parameters={"współczynnik_wyszukiwania": 0.75, "tonacja": "+2 półtony"},
            gen_id="gen_polish_diacritics",
            db_path=self.db_path,
        )
        assert gen is not None
        assert gen["input_text"] == poetic_passage
        assert gen["parameters"]["tonacja"] == "+2 półtony"

    def test_unicode_normalization_nfc_and_nfd(self):
        """
        Verify NFC (composed) and NFD (decomposed) forms of Polish characters
        behave reliably without data corruption or crashes in SQLite.
        """
        text_nfc = unicodedata.normalize("NFC", "Zażółć gęślą jaźń")
        text_nfd = unicodedata.normalize("NFD", "Zażółć gęślą jaźń")

        # They are character-sequence distinct in Python byte representation
        assert text_nfc != text_nfd
        assert len(text_nfd) > len(text_nfc)

        # Store NFC
        s1 = create_sample(
            name="Sample NFC",
            filename="nfc.wav",
            file_path="/data/samples/nfc.wav",
            transcript=text_nfc,
            sample_id="sample_nfc",
            db_path=self.db_path,
        )
        # Store NFD
        s2 = create_sample(
            name="Sample NFD",
            filename="nfd.wav",
            file_path="/data/samples/nfd.wav",
            transcript=text_nfd,
            sample_id="sample_nfd",
            db_path=self.db_path,
        )

        res1 = get_sample("sample_nfc", db_path=self.db_path)
        res2 = get_sample("sample_nfd", db_path=self.db_path)

        assert res1["transcript"] == text_nfc
        assert res2["transcript"] == text_nfd

    def test_sql_injection_and_quotes_in_polish_text(self):
        """
        Adversarial SQL injection attempts embedded inside Polish text.
        Parameterized queries must store injection payloads as literal strings.
        """
        malicious_transcripts = [
            "Zażółć'; DROP TABLE samples; --",
            "Gęślą' OR '1'='1",
            "Jaźń\"; DELETE FROM models; --",
            "'; UPDATE samples SET transcript='HACKED' WHERE '1'='1",
        ]

        for idx, payload in enumerate(malicious_transcripts):
            sid = f"sample_inj_{idx}"
            create_sample(
                name=f"Injection Test {idx}",
                filename=f"inj_{idx}.wav",
                file_path=f"/data/inj_{idx}.wav",
                transcript=payload,
                sample_id=sid,
                db_path=self.db_path,
            )

            # Verify payload is preserved exactly as text
            s = get_sample(sid, db_path=self.db_path)
            assert s["transcript"] == payload

        # Verify table samples still exists and has not been dropped or tampered
        assert count_samples(db_path=self.db_path) == len(malicious_transcripts)

    def test_huge_polish_text_transcript(self):
        """Stress tests extremely large Polish text payloads (10,000+ words)."""
        huge_polish_text = ("Zażółć gęślą jaźń. W Szczebrzeszynie chrząszcz brzmi w trzcinie. " * 500)
        assert len(huge_polish_text) > 30000

        sid = "sample_huge_polish"
        s = create_sample(
            name="Huge Polish Transcript",
            filename="huge.wav",
            file_path="/data/huge.wav",
            transcript=huge_polish_text,
            sample_id=sid,
            db_path=self.db_path,
        )
        assert len(s["transcript"]) == len(huge_polish_text)
        assert s["transcript"] == huge_polish_text


# ==============================================================================
# SECTION 4: TESTING DEV / SIMULATION FLAG OVERRIDES
# ==============================================================================

class TestDevSimulationFlagOverrides:
    """Stress tests DEV and SIMULATION mode environment variable overrides."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Preserve and restore environment variables after each test."""
        keys = [
            "VOICE_SAMPLER_DEV_MODE",
            "VOICE_SAMPLER_SIMULATION_MODE",
            "SIMULATE_TRAINING",
            "VOICE_SAMPLER_LOG_LEVEL",
        ]
        saved = {k: os.environ.get(k) for k in keys}
        yield
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_gpu_service_dev_mode_flag_parsing(self):
        """Verify is_dev_mode_enabled correctly parses all truthy and falsy string variants."""
        service = GPUService()

        # Truthy variants for VOICE_SAMPLER_DEV_MODE
        for truthy in ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]:
            os.environ["VOICE_SAMPLER_DEV_MODE"] = truthy
            assert service.is_dev_mode_enabled() is True, f"Failed for truthy value: {truthy}"

        # Falsy variants for VOICE_SAMPLER_DEV_MODE
        for falsy in ["0", "false", "False", "FALSE", "no", "NO", "off", "OFF"]:
            os.environ["VOICE_SAMPLER_DEV_MODE"] = falsy
            assert service.is_dev_mode_enabled() is False, f"Failed for falsy value: {falsy}"

        # Unset / empty
        os.environ.pop("VOICE_SAMPLER_DEV_MODE", None)
        assert service.is_dev_mode_enabled() is None

        # Fallback to SIMULATE_TRAINING
        os.environ["SIMULATE_TRAINING"] = "1"
        assert service.is_dev_mode_enabled() is True
        os.environ["SIMULATE_TRAINING"] = "0"
        assert service.is_dev_mode_enabled() is False
        os.environ.pop("SIMULATE_TRAINING", None)

        # Fallback to VOICE_SAMPLER_SIMULATION_MODE
        os.environ["VOICE_SAMPLER_SIMULATION_MODE"] = "yes"
        assert service.is_dev_mode_enabled() is True
        os.environ["VOICE_SAMPLER_SIMULATION_MODE"] = "no"
        assert service.is_dev_mode_enabled() is False

    def test_simulation_forced_when_cuda_present(self):
        """
        When a CUDA GPU is present, setting VOICE_SAMPLER_DEV_MODE=1 MUST force
        simulation_mode=True and provide an informative message stating simulation is forced.
        """
        service = GPUService()

        # Mock PyTorch CUDA as available
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
        mock_torch.cuda.mem_get_info.return_value = (20 * 1024**3, 24 * 1024**3)
        mock_torch.cuda.memory_allocated.return_value = 512 * 1024**2
        mock_torch.version.cuda = "12.4"
        mock_torch.__version__ = "2.5.0"

        with patch("backend.app.services.gpu_service.torch", mock_torch), \
             patch("backend.app.services.gpu_service.TORCH_AVAILABLE", True):

            # Case A: Dev mode forced ON (VOICE_SAMPLER_DEV_MODE=1)
            os.environ["VOICE_SAMPLER_DEV_MODE"] = "1"
            info_forced = service.get_gpu_info(force_refresh=True)

            assert info_forced["cuda_available"] is True
            assert info_forced["simulation_mode"] is True
            assert "simulation mode is forced" in info_forced["message"]

            # Case B: Dev mode explicitly OFF (VOICE_SAMPLER_DEV_MODE=0)
            os.environ["VOICE_SAMPLER_DEV_MODE"] = "0"
            info_real = service.get_gpu_info(force_refresh=True)

            assert info_real["cuda_available"] is True
            assert info_real["simulation_mode"] is False
            assert info_real["message"] is None

            # Case C: Unset (should naturally default to simulation_mode=False because CUDA is available)
            os.environ.pop("VOICE_SAMPLER_DEV_MODE", None)
            info_default = service.get_gpu_info(force_refresh=True)

            assert info_default["cuda_available"] is True
            assert info_default["simulation_mode"] is False
            assert info_default["message"] is None

    def test_simulation_behavior_when_cuda_absent(self):
        """
        When CUDA is absent, the system must report cuda_available=False,
        and simulation_mode=True (unless overridden), with clear message.
        """
        service = GPUService()

        # Mock CUDA absent
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("backend.app.services.gpu_service.torch", mock_torch), \
             patch("backend.app.services.gpu_service.TORCH_AVAILABLE", True):

            # Default without env overrides
            os.environ.pop("VOICE_SAMPLER_DEV_MODE", None)
            info = service.get_gpu_info(force_refresh=True)

            assert info["cuda_available"] is False
            assert info["simulation_mode"] is True
            assert "No CUDA GPU detected" in info["message"]

    def test_fastapi_settings_endpoint_reflects_dev_mode(self):
        """
        Verify that GET /api/settings/gpu reflects dynamic dev mode environment changes
        when queried through FastAPI TestClient.
        """
        # TestClient with DEV_MODE=1
        os.environ["VOICE_SAMPLER_DEV_MODE"] = "1"
        client = TestClient(create_app())

        res1 = self.client_gpu_call(client)
        assert res1["simulation_mode"] is True

        # TestClient with DEV_MODE=0
        os.environ["VOICE_SAMPLER_DEV_MODE"] = "0"
        client2 = TestClient(create_app())
        res2 = self.client_gpu_call(client2)
        # On non-CUDA machines, setting DEV_MODE=0 turns simulation_mode to False,
        # but warning message still alerts that CUDA is not detected.
        assert res2["simulation_mode"] is False
        assert "No CUDA GPU detected" in res2["message"]

    def client_gpu_call(self, client: TestClient) -> Dict[str, Any]:
        res = client.get("/api/settings/gpu?refresh=true")
        assert res.status_code == 200
        return res.json()
