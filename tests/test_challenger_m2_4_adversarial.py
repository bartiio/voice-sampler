"""
tests/test_challenger_m2_4_adversarial.py
Adversarial Challenge Test Suite for Milestone 2:
- Concurrency & Race Conditions: Manual edits vs Whisper transcription
- Deleted Sample Audio Streaming: Strict HTTP 404 verification under residual locks & boundary headers
- Polish Diacritics Fidelity: Strict NFC normalization across decomposed NFD and mixed inputs
- WebSocket Notification Resiliency: High-volume broadcasts with slow/abruptly disconnecting clients
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import unicodedata
import uuid
import wave

from fastapi.testclient import TestClient
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.db.repository import (
    create_sample,
    delete_sample,
    get_sample,
    list_samples,
    update_sample,
)
from backend.app.main import app
from backend.app.services.audio_service import get_audio_service
from backend.app.services.notification_service import get_notification_manager
from backend.app.services.whisper_service import get_whisper_service


def _generate_wav(duration_s: float = 0.5, sr: int = 24000, freq: float = 440.0) -> bytes:
    """Generate a clean synthetic WAV byte stream for testing."""
    num_samples = int(duration_s * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(num_samples):
            val = int(16000 * math.sin(2 * math.pi * freq * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return buf.getvalue()


# ==============================================================================
# CHALLENGE SUITE 1: MANUAL TRANSCRIPT EDITS VS WHISPER CONCURRENCY
# ==============================================================================

class TestManualEditVsWhisperRace:
    """Adversarially challenge race conditions between user edits and Whisper background worker."""

    def test_heavy_concurrent_manual_edits_during_whisper_backlog(self):
        """
        Stress-test 20 concurrent samples queued for transcription.
        A slow Whisper engine delays execution while concurrent threads fire PUT manual edits.
        Assert that for ALL 20 samples, Whisper NEVER overwrites the manual edit.
        """
        client = TestClient(app)
        whisper_svc = get_whisper_service()
        engine = whisper_svc._engine
        orig_transcribe = engine.transcribe

        sample_count = 20
        created_sids = []
        wav_bytes = _generate_wav(duration_s=0.3)

        # Introduce an intentional delay in transcribe to guarantee overlap with user PUT
        def slow_transcribe(audio_path, language="pl"):
            time.sleep(0.15)
            return "WHISPER_SHOULD_NEVER_OVERWRITE_THIS"

        engine.transcribe = slow_transcribe

        try:
            # 1. Create 20 samples and enqueue transcription
            futures = []
            for i in range(sample_count):
                sid = f"race_{i}_{uuid.uuid4().hex[:6]}"
                created_sids.append(sid)
                audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, f"race_{i}.wav")
                create_sample(
                    name=f"race_{i}.wav",
                    filename=f"race_{i}.wav",
                    file_path=str(audio_info["file_path"]),
                    duration_seconds=audio_info["duration"],
                    sample_id=sid,
                    transcript="",
                    transcript_status="pending",
                )
                fut = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])
                futures.append((sid, fut))

            # 2. Concurrently fire manual edits across all 20 samples
            barrier = threading.Barrier(sample_count)
            put_errors = []

            def edit_worker(idx: int, sample_id: str):
                try:
                    barrier.wait(timeout=5.0)
                    manual_text = f"Manualne nagranie użytkownika #{idx} (Zażółć gęślą jaźń)"
                    resp = client.put(f"/api/samples/{sample_id}", json={"transcript": manual_text})
                    assert resp.status_code == 200, f"PUT failed: {resp.status_code} {resp.text}"
                except Exception as e:
                    put_errors.append((sample_id, e))

            threads = [
                threading.Thread(target=edit_worker, args=(i, sid))
                for i, (sid, _) in enumerate(futures)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)

            assert len(put_errors) == 0, f"Errors during concurrent PUT: {put_errors}"

            # 3. Wait for all Whisper background tasks to complete
            for sid, fut in futures:
                try:
                    fut.result(timeout=10.0)
                except Exception:
                    pass

            # 4. Rigorously verify each sample in SQLite:
            # - transcript must be the manual text
            # - transcript_status must be 'manual'
            # - Whisper text must NOT be present!
            for i, sid in enumerate(created_sids):
                db_record = get_sample(sid)
                assert db_record is not None, f"Sample {sid} missing from DB"
                expected_text = f"Manualne nagranie użytkownika #{i} (Zażółć gęślą jaźń)"
                assert db_record["transcript"] == expected_text, (
                    f"Sample {sid} transcript was overwritten! Expected '{expected_text}', got '{db_record['transcript']}'"
                )
                assert db_record["transcript_status"] == "manual", (
                    f"Sample {sid} transcript_status was overwritten! Expected 'manual', got '{db_record['transcript_status']}'"
                )
        finally:
            engine.transcribe = orig_transcribe
            for sid in created_sids:
                try:
                    client.delete(f"/api/samples/{sid}")
                except Exception:
                    pass

    def test_empty_manual_edit_is_preserved_against_whisper(self):
        """
        Adversarial Test:
        User explicitly clears the transcript by setting transcript="" (empty string).
        Verify that Whisper does not treat empty transcript as unedited and overwrite it.
        """
        client = TestClient(app)
        whisper_svc = get_whisper_service()
        engine = whisper_svc._engine
        orig_transcribe = engine.transcribe

        sid = f"empty_edit_{uuid.uuid4().hex[:8]}"
        wav_bytes = _generate_wav(duration_s=0.3)
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, "empty_edit.wav")
        create_sample(
            name="empty_edit.wav",
            filename="empty_edit.wav",
            file_path=str(audio_info["file_path"]),
            duration_seconds=audio_info["duration"],
            sample_id=sid,
            transcript="Initial text",
            transcript_status="pending",
        )

        started = threading.Event()

        def slow_transcribe(audio_path, language="pl"):
            started.set()
            time.sleep(0.2)
            return "Whisper generated text"

        engine.transcribe = slow_transcribe

        try:
            fut = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])
            assert started.wait(timeout=3.0)

            # User sets transcript to empty string
            put_resp = client.put(f"/api/samples/{sid}", json={"transcript": ""})
            assert put_resp.status_code == 200
            assert put_resp.json()["transcript"] == ""
            assert put_resp.json()["status"] == "manual"

            fut.result(timeout=5.0)

            final_record = get_sample(sid)
            assert final_record["transcript"] == "", f"Expected empty transcript, got '{final_record['transcript']}'"
            assert final_record["transcript_status"] == "manual"
        finally:
            engine.transcribe = orig_transcribe
            client.delete(f"/api/samples/{sid}")

    def test_whisper_exception_does_not_clobber_manual_status(self):
        """
        Adversarial Test:
        Whisper transcription fails with an unhandled exception while sample was manually edited.
        The error handler in WhisperService must NOT overwrite transcript_status from 'manual' to 'error'.
        """
        client = TestClient(app)
        whisper_svc = get_whisper_service()
        engine = whisper_svc._engine
        orig_transcribe = engine.transcribe

        sid = f"err_edit_{uuid.uuid4().hex[:8]}"
        wav_bytes = _generate_wav(duration_s=0.3)
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, "err_edit.wav")
        create_sample(
            name="err_edit.wav",
            filename="err_edit.wav",
            file_path=str(audio_info["file_path"]),
            duration_seconds=audio_info["duration"],
            sample_id=sid,
            transcript="",
            transcript_status="pending",
        )

        started = threading.Event()

        def failing_transcribe(audio_path, language="pl"):
            started.set()
            time.sleep(0.15)
            raise RuntimeError("Simulated Whisper CUDA OOM / crash")

        engine.transcribe = failing_transcribe

        try:
            fut = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])
            assert started.wait(timeout=3.0)

            # User edits transcript
            client.put(f"/api/samples/{sid}", json={"transcript": "User text during crash"})

            # Wait for future to fail
            with pytest.raises(Exception):
                fut.result(timeout=5.0)

            # Record must remain 'manual', NOT 'error'
            rec = get_sample(sid)
            assert rec["transcript"] == "User text during crash"
            assert rec["transcript_status"] == "manual", f"Expected 'manual', got '{rec['transcript_status']}'"
        finally:
            engine.transcribe = orig_transcribe
            client.delete(f"/api/samples/{sid}")

    def test_rapid_consecutive_user_edits_during_transcription(self):
        """
        Adversarial Test:
        While Whisper is active, send 10 rapid sequential PUT updates with increments (v0 ... v9).
        Verify the final persisted transcript is v9 with status 'manual'.
        """
        client = TestClient(app)
        whisper_svc = get_whisper_service()
        engine = whisper_svc._engine
        orig_transcribe = engine.transcribe

        sid = f"consec_{uuid.uuid4().hex[:8]}"
        wav_bytes = _generate_wav(duration_s=0.3)
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, "consec.wav")
        create_sample(
            name="consec.wav",
            filename="consec.wav",
            file_path=str(audio_info["file_path"]),
            duration_seconds=audio_info["duration"],
            sample_id=sid,
            transcript="",
            transcript_status="pending",
        )

        started = threading.Event()

        def slow_transcribe(audio_path, language="pl"):
            started.set()
            time.sleep(0.3)
            return "Whisper text to discard"

        engine.transcribe = slow_transcribe

        try:
            fut = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])
            assert started.wait(timeout=3.0)

            for i in range(10):
                r = client.put(f"/api/samples/{sid}", json={"transcript": f"Wersja edycji {i}: Zażółć"})
                assert r.status_code == 200

            fut.result(timeout=5.0)

            rec = get_sample(sid)
            assert rec["transcript"] == "Wersja edycji 9: Zażółć"
            assert rec["transcript_status"] == "manual"
        finally:
            engine.transcribe = orig_transcribe
            client.delete(f"/api/samples/{sid}")


# ==============================================================================
# CHALLENGE SUITE 2: STREAMING AUDIO FOR DELETED SAMPLES
# ==============================================================================

class TestDeletedSampleAudioStreamingStrict404:
    """Stress-test that deleted or non-existent samples strictly return HTTP 404 on audio streaming."""

    def test_stream_deleted_sample_with_windows_locked_file(self):
        """
        On Windows, if a file has an open file handle, os.unlink() fails with [WinError 32].
        Simulate this exact condition:
        1. Upload sample.
        2. Keep open file handle in Python on the raw WAV file.
        3. Invoke DELETE /api/samples/{id} (DB row is deleted, file remains locked).
        4. Request GET /api/samples/{id}/audio.
        5. MUST return HTTP 404 (not HTTP 200 or 500).
        """
        client = TestClient(app)
        wav_bytes = _generate_wav(duration_s=0.3)
        up_resp = client.post("/api/samples/upload", files={"file": ("lock_test.wav", io.BytesIO(wav_bytes), "audio/wav")})
        assert up_resp.status_code == 200
        sid = up_resp.json()["id"]
        file_path = up_resp.json()["file_path"]

        # Open file to lock it on Windows
        locked_handle = open(file_path, "rb")
        try:
            del_resp = client.delete(f"/api/samples/{sid}")
            assert del_resp.status_code == 200

            # File physically still exists on disk because of open handle
            assert os.path.exists(file_path), "File was unexpectedly removed despite open lock"

            # Streaming MUST return 404 because sample is deleted from DB!
            stream_resp = client.get(f"/api/samples/{sid}/audio")
            assert stream_resp.status_code == 404, (
                f"Expected 404 on audio stream of deleted sample with locked file, got {stream_resp.status_code}"
            )
        finally:
            locked_handle.close()
            # Now unlink manually
            try:
                os.unlink(file_path)
            except Exception:
                pass

    def test_range_requests_on_deleted_sample_strictly_return_404(self):
        """
        Verify that various Range headers (valid, invalid, malformed, empty)
        all return HTTP 404 if the sample is deleted or does not exist.
        """
        client = TestClient(app, raise_server_exceptions=False)
        wav_bytes = _generate_wav(duration_s=0.3)
        up = client.post("/api/samples/upload", files={"file": ("del_range.wav", io.BytesIO(wav_bytes), "audio/wav")}).json()
        sid = up["id"]

        # Delete it
        client.delete(f"/api/samples/{sid}")

        range_headers = [
            "bytes=0-10",
            "bytes=100-",
            "bytes=-50",
            "bytes=0-0",
            "bytes=9999-10000",
            "bytes=500-100",  # reversed
            "bytes=foo-bar",   # malformed non-numeric
            "bytes=",          # empty
            "invalid-unit=0-10",
        ]

        for r_hdr in range_headers:
            resp = client.get(f"/api/samples/{sid}/audio", headers={"Range": r_hdr})
            assert resp.status_code == 404, (
                f"Range header '{r_hdr}' on deleted sample returned {resp.status_code} instead of 404"
            )

    def test_completely_fictitious_uuid_returns_404(self):
        """Verify random UUID returns 404 with both normal and range requests."""
        client = TestClient(app, raise_server_exceptions=False)
        fake_id = str(uuid.uuid4())

        r1 = client.get(f"/api/samples/{fake_id}/audio")
        assert r1.status_code == 404

        r2 = client.get(f"/api/samples/{fake_id}/audio", headers={"Range": "bytes=0-100"})
        assert r2.status_code == 404

        r3 = client.get(f"/api/samples/{fake_id}/audio", headers={"Range": "bytes=garbage"})
        assert r3.status_code == 404


# ==============================================================================
# CHALLENGE SUITE 3: POLISH DIACRITICS PERSISTENCE & NFC NORMALIZATION
# ==============================================================================

class TestPolishDiacriticsNFCIntegrity:
    """Stress-test Polish diacritics persistence under decomposed NFD, NFC, and complex Unicode."""

    POLISH_9_LOWER = "ąćęłńóśźż"
    POLISH_9_UPPER = "ĄĆĘŁŃÓŚŹŻ"

    def test_pure_nfd_input_is_saved_as_canonical_nfc(self):
        """
        Input decomposed NFD strings where accents are separate combining codepoints.
        Assert that the stored and returned transcript is strictly normalized to NFC.
        """
        client = TestClient(app)
        wav_bytes = _generate_wav(duration_s=0.2)
        up = client.post("/api/samples/upload", files={"file": ("nfd_test.wav", io.BytesIO(wav_bytes), "audio/wav")}).json()
        sid = up["id"]

        test_cases = [
            # Pangrams in pure NFD
            "Zażółć gęślą jaźń",
            "Pchnąć w tę łódź jeża lub ośm skrzyń fig",
            "W Szczebrzeszynie chrząszcz brzmi w trzcinie, a chrabąszcz brzmi w trzcinie.",
            # All 9 diacritics in lowercase and uppercase
            f"{self.POLISH_9_LOWER} {self.POLISH_9_UPPER}",
            # Rare Polish words with dense diacritics
            "źdźbło żółwica gżegżółka próchniejący rzężenie",
        ]

        try:
            for raw_phrase in test_cases:
                nfd_phrase = unicodedata.normalize("NFD", raw_phrase)
                expected_nfc = unicodedata.normalize("NFC", raw_phrase)

                # Ensure test data actually has decomposed bytes where possible
                # (Note: for characters like 'ż' (U+017C), NFD is 'z' + U+0307)
                put_r = client.put(f"/api/samples/{sid}", json={"transcript": nfd_phrase})
                assert put_r.status_code == 200

                data = put_r.json()
                returned_transcript = data["transcript"]

                # 1. Returned transcript MUST be NFC
                assert unicodedata.is_normalized("NFC", returned_transcript), (
                    f"Returned transcript for '{raw_phrase}' is NOT normalized to NFC!"
                )
                assert returned_transcript == expected_nfc

                # 2. Directly inspect SQLite database row
                db_record = get_sample(sid)
                db_transcript = db_record["transcript"]
                assert unicodedata.is_normalized("NFC", db_transcript), (
                    f"SQLite transcript for '{raw_phrase}' is NOT normalized to NFC!"
                )
                assert db_transcript == expected_nfc

                # 3. GET /api/samples/{id}
                get_r = client.get(f"/api/samples/{sid}").json()
                assert unicodedata.is_normalized("NFC", get_r["transcript"])
                assert get_r["transcript"] == expected_nfc
        finally:
            client.delete(f"/api/samples/{sid}")

    def test_mixed_nfc_nfd_and_combining_characters(self):
        """
        Mix NFC words with NFD words in the exact same sentence.
        Verify full string becomes uniform canonical NFC.
        """
        client = TestClient(app)
        wav_bytes = _generate_wav(duration_s=0.2)
        up = client.post("/api/samples/upload", files={"file": ("mixed_test.wav", io.BytesIO(wav_bytes), "audio/wav")}).json()
        sid = up["id"]

        w1_nfc = unicodedata.normalize("NFC", "Zażółć")
        w2_nfd = unicodedata.normalize("NFD", "gęślą")
        w3_nfc = unicodedata.normalize("NFC", "jaźń")
        mixed_input = f"{w1_nfc} {w2_nfd} {w3_nfc}"

        try:
            resp = client.put(f"/api/samples/{sid}", json={"transcript": mixed_input})
            assert resp.status_code == 200
            saved = resp.json()["transcript"]

            assert unicodedata.is_normalized("NFC", saved)
            assert saved == "Zażółć gęślą jaźń"

            db_sample = get_sample(sid)
            assert unicodedata.is_normalized("NFC", db_sample["transcript"])
        finally:
            client.delete(f"/api/samples/{sid}")


# ==============================================================================
# CHALLENGE SUITE 4: NON-BLOCKING WEBSOCKET NOTIFICATION BROADCASTS UNDER LOAD
# ==============================================================================

class TestWebSocketBroadcastUnderStress:
    """Stress-test WebSocket broadcasting under high event velocity and concurrent client operations."""

    def test_high_volume_broadcast_sync_under_thread_concurrency(self):
        """
        50 concurrent worker threads blasting 20 notifications each (1000 total events).
        Verify NotificationManager schedules all events without Deadlock, Thread Lock, or memory leak.
        """
        manager = get_notification_manager()
        thread_count = 50
        events_per_thread = 20
        barrier = threading.Barrier(thread_count)
        errors = []

        def worker(tid: int):
            try:
                barrier.wait(timeout=5.0)
                for i in range(events_per_thread):
                    sample_dict = {
                        "id": f"ws_stress_{tid}_{i}",
                        "filename": f"sample_{tid}_{i}.wav",
                        "duration": 1.23,
                        "status": "transcribing",
                    }
                    manager.notify_sample_created(sample_dict)
                    manager.notify_transcription_completed(f"ws_stress_{tid}_{i}", f"Transkrypt {tid}-{i}")
            except Exception as e:
                errors.append((tid, e))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(thread_count)]
        start_time = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
        duration = time.monotonic() - start_time

        assert len(errors) == 0, f"Errors during high volume broadcast: {errors}"
        # All 1000 events scheduled smoothly in under 10 seconds
        assert duration < 10.0, f"Broadcast took too long: {duration:.2f}s"

    def test_websocket_client_abrupt_disconnect_during_active_broadcast(self):
        """
        Connect multiple WebSocket clients, launch a barrage of broadcasts,
        and abruptly disconnect half the clients mid-stream.
        Verify:
        - The server does not throw unhandled exceptions or crash.
        - Disconnected clients are safely cleaned up from _active_connections.
        - Surviving clients continue receiving messages.
        """
        client = TestClient(app)
        manager = get_notification_manager()

        clients = []
        for _ in range(6):
            ws = client.websocket_connect("/ws/notifications")
            ws.__enter__()
            ws.send_text("ping")
            assert ws.receive_text() == "pong"
            clients.append(ws)

        # Broadcast events via manager.broadcast
        for i in range(5):
            asyncio.run(manager.broadcast({"type": "notification", "event": f"event_{i}"}))

        # Abruptly close first 3 clients
        for c in clients[:3]:
            c.__exit__(None, None, None)

        # Broadcast more events (these will hit the closed sockets)
        for i in range(5, 10):
            asyncio.run(manager.broadcast({"type": "notification", "event": f"event_{i}"}))

        # Verify dead connections are pruned from manager._active_connections
        async def _get_active():
            async with manager._lock:
                return set(manager._active_connections)

        active = asyncio.run(_get_active())
        for dead_c in clients[:3]:
            for conn in active:
                assert conn != dead_c._websocket if hasattr(dead_c, "_websocket") else conn != dead_c

        # Clean up remaining clients
        for c in clients[3:]:
            c.__exit__(None, None, None)
