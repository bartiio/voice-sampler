"""
Unit tests for Samples API and WebSocket endpoints (/api/samples, /ws/notifications).
"""

import io
import math
import struct
import wave
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app


def _generate_test_wav(duration_s=1.0, sr=24000):
    num_samples = int(duration_s * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(num_samples):
            val = int(16000 * math.sin(2 * math.pi * 440.0 * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(app)


def test_samples_empty_listing(client):
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)


def test_upload_sample_and_crud_lifecycle(client):
    wav_bytes = _generate_test_wav(duration_s=1.0)
    files = {"file": ("test_upload.wav", io.BytesIO(wav_bytes), "audio/wav")}

    # 1. Upload sample
    up_resp = client.post("/api/samples/upload", files=files)
    assert up_resp.status_code == 200
    sample = up_resp.json()
    sample_id = sample["id"]
    assert sample["filename"] == "test_upload.wav"
    assert "duration" in sample
    assert "status" in sample
    assert "transcript" in sample
    assert len(sample["transcript"]) > 0

    # 2. Get sample by ID
    get_resp = client.get(f"/api/samples/{sample_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == sample_id

    # 3. Update transcript with Polish diacritics
    new_text = "Nowa polska transkrypcja: Zażółć gęślą jaźń!"
    put_resp = client.put(f"/api/samples/{sample_id}", json={"transcript": new_text})
    assert put_resp.status_code == 200
    assert put_resp.json()["transcript"] == new_text

    # 4. Stream audio with full content (HTTP 200)
    stream_resp = client.get(f"/api/samples/{sample_id}/audio")
    assert stream_resp.status_code == 200
    assert stream_resp.headers.get("accept-ranges") == "bytes"
    assert stream_resp.headers.get("content-type") == "audio/wav"
    assert len(stream_resp.content) > 0

    # 5. Stream audio with byte range (HTTP 206)
    range_resp = client.get(f"/api/samples/{sample_id}/audio", headers={"Range": "bytes=0-100"})
    assert range_resp.status_code == 206
    assert len(range_resp.content) == 101
    assert "Content-Range" in range_resp.headers

    # 6. Stream audio with reversed range (HTTP 416)
    bad_range_resp = client.get(f"/api/samples/{sample_id}/audio", headers={"Range": "bytes=500-200"})
    assert bad_range_resp.status_code == 416

    # 7. Delete sample
    del_resp = client.delete(f"/api/samples/{sample_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 8. Verify 404 after deletion
    assert client.get(f"/api/samples/{sample_id}").status_code == 404
    assert client.get(f"/api/samples/{sample_id}/audio").status_code == 404


def test_upload_empty_file_returns_400(client):
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/api/samples/upload", files=files)
    assert response.status_code == 400


def test_record_sample_endpoint(client):
    wav_bytes = _generate_test_wav(duration_s=0.5)
    files = {"file": ("mic_take.wav", io.BytesIO(wav_bytes), "audio/wav")}
    rec_resp = client.post("/api/samples/record", files=files)
    assert rec_resp.status_code == 200
    data = rec_resp.json()
    assert "polsk" in data["transcript"].lower()
    assert data["status"] == "transcribed"


def test_websocket_notifications_endpoint(client):
    with client.websocket_connect("/ws/notifications") as ws:
        ws.send_text("ping")
        resp = ws.receive_text()
        assert resp == "pong"


def test_malformed_range_returns_416_with_content_range_header(client):
    wav_bytes = _generate_test_wav(duration_s=0.5)
    files = {"file": ("range_err.wav", io.BytesIO(wav_bytes), "audio/wav")}
    up = client.post("/api/samples/upload", files=files).json()
    sid = up["id"]

    try:
        # Non-numeric range: MUST return 416, NOT 500
        r = client.get(f"/api/samples/{sid}/audio", headers={"Range": "bytes=abc-def"})
        assert r.status_code == 416
        assert "content-range" in r.headers
        assert r.headers["content-range"].startswith("bytes */")

        # Non-numeric start: MUST return 416
        r2 = client.get(f"/api/samples/{sid}/audio", headers={"Range": "bytes=foo-100"})
        assert r2.status_code == 416
        assert "content-range" in r2.headers

        # Non-numeric end: MUST return 416
        r3 = client.get(f"/api/samples/{sid}/audio", headers={"Range": "bytes=10-bar"})
        assert r3.status_code == 416
        assert "content-range" in r3.headers

        # Suffix of 0: MUST return 416
        r4 = client.get(f"/api/samples/{sid}/audio", headers={"Range": "bytes=-0"})
        assert r4.status_code == 416
        assert "content-range" in r4.headers
    finally:
        client.delete(f"/api/samples/{sid}")


def test_put_sample_transcript_normalizes_nfd_to_nfc(client):
    """Verify that decomposed NFD Polish characters are normalized to NFC upon PUT."""
    import unicodedata

    wav_bytes = _generate_test_wav(duration_s=0.5)
    files = {"file": ("nfd_norm.wav", io.BytesIO(wav_bytes), "audio/wav")}
    up = client.post("/api/samples/upload", files=files).json()
    sid = up["id"]

    try:
        raw_text = "Zażółć gęślą jaźń — chrząszcz brzmi w trzcinie."
        nfd_text = unicodedata.normalize("NFD", raw_text)
        assert not unicodedata.is_normalized("NFC", nfd_text)

        put_resp = client.put(f"/api/samples/{sid}", json={"transcript": nfd_text})
        assert put_resp.status_code == 200
        saved = put_resp.json()["transcript"]

        # Assert saved transcript is strictly NFC
        assert unicodedata.is_normalized("NFC", saved)
        assert saved == unicodedata.normalize("NFC", raw_text)
        assert put_resp.json()["status"] == "manual"

        # Assert GET returns NFC
        get_resp = client.get(f"/api/samples/{sid}").json()
        assert unicodedata.is_normalized("NFC", get_resp["transcript"])
        assert get_resp["status"] == "manual"
    finally:
        client.delete(f"/api/samples/{sid}")


def test_upload_sample_notification_event_ordering(client):
    """Verify that sample_created is broadcast before transcription_started/completed."""
    from backend.app.services.notification_service import get_notification_manager

    events = []
    mgr = get_notification_manager()
    orig_broadcast_sync = mgr.broadcast_sync

    def intercept_broadcast(event):
        events.append(event.get("event"))
        orig_broadcast_sync(event)

    mgr.broadcast_sync = intercept_broadcast

    try:
        wav_bytes = _generate_test_wav(duration_s=0.5)
        files = {"file": ("order.wav", io.BytesIO(wav_bytes), "audio/wav")}
        up = client.post("/api/samples/upload", files=files)
        assert up.status_code == 200
        sid = up.json()["id"]

        # Ensure sample_created occurred before transcription events
        assert "sample_created" in events
        created_idx = events.index("sample_created")
        if "transcription_started" in events:
            assert created_idx < events.index("transcription_started")
        if "transcription_completed" in events:
            assert created_idx < events.index("transcription_completed")
    finally:
        mgr.broadcast_sync = orig_broadcast_sync
        client.delete(f"/api/samples/{sid}")


def test_deleted_sample_with_residual_file_returns_404(client):
    """Verify GET /api/samples/{id}/audio returns 404 even if audio file is still on disk."""
    import os
    wav_bytes = _generate_test_wav(duration_s=0.3)
    up = client.post("/api/samples/upload", files={"file": ("resid.wav", io.BytesIO(wav_bytes), "audio/wav")}).json()
    sid = up["id"]
    file_path = up["file_path"]

    # Keep file handle open to simulate Windows file lock
    with open(file_path, "rb") as f:
        del_resp = client.delete(f"/api/samples/{sid}")
        assert del_resp.status_code == 200

        # Stream endpoint must check DB and return 404!
        stream_resp = client.get(f"/api/samples/{sid}/audio")
        assert stream_resp.status_code == 404

    # Now unlink residual file
    if os.path.exists(file_path):
        os.unlink(file_path)
