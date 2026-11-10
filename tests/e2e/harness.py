"""
Voice Sampler E2E Test Harness
Provides:
- Synthetic audio generator for deterministic audio testing (WAV, MP3, FLAC, corruptions)
- High-level test client supporting REST and WebSocket
- Self-contained EmbeddedMockServer adhering to PROJECT.md specifications
- Test client factory supporting both live external server and embedded ASGI server
"""

import io
import math
import os
import struct
import tempfile
import time
import urllib.parse
import uuid
import wave
from typing import Any, Dict, List, Optional, Tuple, Union

import warnings

warnings.filterwarnings("ignore")

import requests
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.testclient import TestClient


# ==============================================================================
# 1. Synthetic Audio Generator
# ==============================================================================

class SyntheticAudioGenerator:
    """
    Generates deterministic in-memory audio buffers for test fixtures.
    Produces valid PCM 16-bit WAVs, mock formats, silence, and corrupted streams.
    """

    @staticmethod
    def generate_sine_wav(
        duration_s: float = 2.0,
        sample_rate: int = 24000,
        frequency: float = 440.0,
        amplitude: float = 0.5,
        channels: int = 1,
    ) -> bytes:
        """Generates a valid 16-bit PCM RIFF WAV sine wave."""
        num_samples = int(duration_s * sample_rate)
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Generate frames
            frames = bytearray()
            max_val = 32767 * amplitude
            for i in range(num_samples):
                val = int(max_val * math.sin(2.0 * math.pi * frequency * (i / sample_rate)))
                # Clamp to 16-bit signed int
                val = max(-32768, min(32767, val))
                sample_bytes = struct.pack("<h", val)
                for _ in range(channels):
                    frames.extend(sample_bytes)

            wav_file.writeframes(frames)

        return buffer.getvalue()

    @staticmethod
    def generate_speech_dummy_wav(
        duration_s: float = 3.0,
        sample_rate: int = 24000,
        base_freq: float = 140.0,
        channels: int = 1,
    ) -> bytes:
        """
        Generates a multi-harmonic audio stream simulating human vocal formants.
        Useful for Whisper transcription and F5/RVC pitch extractor tests.
        """
        num_samples = int(duration_s * sample_rate)
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            frames = bytearray()
            for i in range(num_samples):
                t = i / sample_rate
                # Envelope: attack, sustain, decay
                env = min(1.0, t * 5.0) * min(1.0, (duration_s - t) * 5.0)
                # Formant harmonics: fundamental (140Hz), F1 (560Hz), F2 (1200Hz), F3 (2400Hz)
                val = (
                    0.4 * math.sin(2 * math.pi * base_freq * t)
                    + 0.3 * math.sin(2 * math.pi * (base_freq * 4.0) * t)
                    + 0.2 * math.sin(2 * math.pi * (base_freq * 8.5) * t)
                    + 0.1 * math.sin(2 * math.pi * (base_freq * 17.0) * t)
                ) * env
                int_val = int(val * 24000)
                int_val = max(-32768, min(32767, int_val))
                sample_bytes = struct.pack("<h", int_val)
                for _ in range(channels):
                    frames.extend(sample_bytes)

            wav_file.writeframes(frames)

        return buffer.getvalue()

    @staticmethod
    def generate_silence_wav(duration_s: float = 1.0, sample_rate: int = 24000) -> bytes:
        """Generates a valid WAV with absolute silence (zero amplitude)."""
        num_samples = int(duration_s * sample_rate)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * num_samples)
        return buffer.getvalue()

    @staticmethod
    def generate_corrupted_wav() -> bytes:
        """Generates a corrupt audio file with invalid RIFF headers."""
        return b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00INVALID_DATA_CORRUPT"

    @staticmethod
    def generate_mock_mp3(size_bytes: int = 1024) -> bytes:
        """Generates a dummy MP3 container with valid ID3v2 header, mock tag, and sync frames."""
        id3_header = b"ID3\x03\x00\x00\x00\x00\x00\x20"  # 10 bytes ID3v2.3
        mock_tag = b"MOCK_MP3_PAYLOAD\x00"  # 17 bytes
        padding = b"\x00" * max(0, 32 - len(mock_tag))
        # MP3 sync frame: 0xFFFB (MPEG 1 Layer 3, 128kbps, 44.1kHz)
        sync_frames = (b"\xff\xfb\x90\x64" + b"\x55" * 100) * (size_bytes // 104)
        return id3_header + mock_tag + padding + sync_frames

    @staticmethod
    def generate_mock_flac(size_bytes: int = 1024) -> bytes:
        """Generates a dummy FLAC container with valid fLaC magic marker and mock tag."""
        flac_header = b"fLaC\x00\x00\x00\x22"  # STREAMINFO block
        mock_tag = b"MOCK_FLAC_PAYLOAD\x00"  # 18 bytes
        stream_info = b"\x10\x00\x10\x00\x00\x00\x00\x00\x00\x00\x0a\xc4\x42\xf0\x00\x00\x00\x00" + (b"\x00" * 16)
        remaining = max(0, size_bytes - len(flac_header) - len(mock_tag) - len(stream_info))
        payload = b"\x00" * remaining
        return flac_header + mock_tag + stream_info + payload

    @staticmethod
    def generate_mock_audio_marker(duration_s: float = 1.0) -> bytes:
        """Generates an explicit mock audio test payload with MOCK_AUDIO_PAYLOAD prefix."""
        return f"MOCK_AUDIO_PAYLOAD:{duration_s}".encode("utf-8")


# ==============================================================================
# 2. Embedded Mock Server (Adheres 100% to PROJECT.md Contract)
# ==============================================================================

class TrainF5TTSRequest(BaseModel):
    model_name: str
    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 0.0001
    sample_ids: Optional[List[str]] = None


class TrainRVCRequest(BaseModel):
    model_name: str
    epochs: int = 20
    batch_size: int = 4
    sample_rate: int = 40000
    sample_ids: Optional[List[str]] = None


class GenerateTTSRequest(BaseModel):
    text: str
    model_name: str
    reference_sample_id: str
    speed: float = 1.0


class GenerateSTSRequest(BaseModel):
    input_sample_id: str
    model_name: str
    pitch_shift: int = 0
    index_rate: float = 0.75


class UpdateTranscriptRequest(BaseModel):
    transcript: str


class EmbeddedMockServer:
    """
    In-memory FastAPI application strictly implementing PROJECT.md endpoints and schemas.
    Allows complete offline E2E verification of Tiers 1-4 without external dependencies.
    """

    def __init__(self):
        self.app = FastAPI(title="Voice Sampler Backend (Mock/Sim)")
        self.samples: Dict[str, Dict[str, Any]] = {}
        self.sample_audio: Dict[str, bytes] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.generated_audio: Dict[str, bytes] = {}
        self.active_training: Optional[Dict[str, Any]] = None
        self.ws_subscribers: List[WebSocket] = []
        self._setup_cors()
        self._setup_routes()

    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        app = self.app

        @app.get("/api/health")
        def get_health():
            return {
                "status": "ok",
                "version": "0.1.0",
                "backend_ready": True,
            }

        @app.post("/api/shutdown")
        def post_shutdown():
            return {"status": "shutting_down"}

        @app.get("/api/settings/gpu")
        def get_gpu_settings():
            # Matches PROJECT.md AMD Radeon / Non-CUDA simulation response
            return {
                "cuda_available": False,
                "gpu_name": "AMD Radeon 740M Graphics",
                "vram_total_mb": 512,
                "vram_allocated_mb": 0,
                "cuda_version": None,
                "simulation_mode": True,
                "message": "No CUDA GPU detected. GPU is required for real training. Running in simulation mode.",
            }

        @app.get("/api/samples")
        def get_samples():
            return {"samples": list(self.samples.values())}

        @app.post("/api/samples/upload")
        async def upload_sample(file: UploadFile):
            content = await file.read()
            if not content or len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file provided")

            sample_id = str(uuid.uuid4())
            filename = file.filename or "sample.wav"
            # Calculate mock duration
            duration = round(max(1.0, len(content) / 48000.0), 2)

            sample_record = {
                "id": sample_id,
                "filename": filename,
                "duration": duration,
                "transcript": "To jest przykładowy tekst nagrania w języku polskim.",
                "status": "transcribed",
                "created_at": "2026-11-06T10:00:00Z",
            }
            self.samples[sample_id] = sample_record
            self.sample_audio[sample_id] = content
            return sample_record

        @app.post("/api/samples/record")
        async def record_sample(file: UploadFile):
            content = await file.read()
            if not content or len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty recording buffer")

            sample_id = str(uuid.uuid4())
            filename = file.filename or f"recording_{sample_id[:8]}.wav"
            duration = round(max(0.5, len(content) / 48000.0), 2)

            sample_record = {
                "id": sample_id,
                "filename": filename,
                "duration": duration,
                "transcript": "Nagranie z mikrofonu z polskimi znakami: zażółć gęślą jaźń.",
                "status": "transcribed",
                "created_at": "2026-11-06T10:00:00Z",
            }
            self.samples[sample_id] = sample_record
            self.sample_audio[sample_id] = content
            return sample_record

        @app.put("/api/samples/{sample_id}")
        def update_sample(sample_id: str, req: UpdateTranscriptRequest):
            if sample_id not in self.samples:
                raise HTTPException(status_code=404, detail="Sample not found")

            self.samples[sample_id]["transcript"] = req.transcript
            return self.samples[sample_id]

        @app.delete("/api/samples/{sample_id}")
        def delete_sample(sample_id: str):
            if sample_id not in self.samples:
                raise HTTPException(status_code=404, detail="Sample not found")

            del self.samples[sample_id]
            if sample_id in self.sample_audio:
                del self.sample_audio[sample_id]
            return {"status": "deleted", "id": sample_id}

        @app.get("/api/samples/{sample_id}/audio")
        def get_sample_audio(sample_id: str, request: Request):
            if sample_id not in self.sample_audio:
                raise HTTPException(status_code=404, detail="Audio file not found")

            audio_data = self.sample_audio[sample_id]
            total_len = len(audio_data)
            range_header = request.headers.get("Range")

            if range_header and range_header.startswith("bytes="):
                # Byte-range partial content support (HTTP 206)
                parts = range_header.replace("bytes=", "").split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else total_len - 1

                if start >= total_len or end >= total_len or start > end:
                    raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

                chunk = audio_data[start : end + 1]
                headers = {
                    "Content-Range": f"bytes {start}-{end}/{total_len}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(chunk)),
                    "Content-Type": "audio/wav",
                }
                return Response(content=chunk, status_code=206, headers=headers)

            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={"Accept-Ranges": "bytes", "Content-Length": str(total_len)},
            )

        @app.post("/api/train/f5-tts")
        def start_f5_tts(req: TrainF5TTSRequest):
            if not req.model_name or not req.model_name.strip():
                raise HTTPException(status_code=400, detail="Model name cannot be empty")
            if req.epochs < 1:
                raise HTTPException(status_code=400, detail="Epochs must be >= 1")
            if req.learning_rate <= 0:
                raise HTTPException(status_code=400, detail="Learning rate must be > 0")
            if req.sample_ids is not None:
                for sid in req.sample_ids:
                    if sid not in self.samples:
                        raise HTTPException(status_code=400, detail=f"Sample id {sid} not found")

            run_id = str(uuid.uuid4())
            self.active_training = {
                "run_id": run_id,
                "engine": "f5-tts",
                "model_name": req.model_name,
                "status": "training",
                "epochs": req.epochs,
            }

            # Register model checkpoint in registry
            self.models[req.model_name] = {
                "name": req.model_name,
                "engine": "f5-tts",
                "checkpoint_path": f"data/models/{req.model_name}_f5tts.pt",
                "vocab_path": f"data/models/{req.model_name}_vocab.txt",
                "created_at": "2026-11-06T10:05:00Z",
            }

            return {
                "run_id": run_id,
                "status": "started",
                "engine": "f5-tts",
                "model_name": req.model_name,
            }

        @app.post("/api/train/rvc")
        def start_rvc(req: TrainRVCRequest):
            if not req.model_name or not req.model_name.strip():
                raise HTTPException(status_code=400, detail="Model name cannot be empty")
            if req.epochs < 1:
                raise HTTPException(status_code=400, detail="Epochs must be >= 1")
            if req.sample_rate not in (32000, 40000, 48000):
                raise HTTPException(status_code=400, detail="Sample rate must be 32k, 40k, or 48k")
            if req.sample_ids is not None:
                for sid in req.sample_ids:
                    if sid not in self.samples:
                        raise HTTPException(status_code=400, detail=f"Sample id {sid} not found")

            run_id = str(uuid.uuid4())
            self.active_training = {
                "run_id": run_id,
                "engine": "rvc",
                "model_name": req.model_name,
                "status": "training",
                "epochs": req.epochs,
            }

            # Register model checkpoint in registry
            self.models[req.model_name] = {
                "name": req.model_name,
                "engine": "rvc",
                "checkpoint_path": f"data/models/{req.model_name}.pth",
                "index_path": f"data/models/{req.model_name}.index",
                "created_at": "2026-11-06T10:05:00Z",
            }

            return {
                "run_id": run_id,
                "status": "started",
                "engine": "rvc",
                "model_name": req.model_name,
            }

        @app.post("/api/train/stop")
        def stop_training():
            if not self.active_training or self.active_training.get("status") != "training":
                return {"status": "no_active_run"}

            run_id = self.active_training["run_id"]
            self.active_training["status"] = "stopped"
            return {"status": "stopped", "run_id": run_id}

        @app.get("/api/models")
        def get_models():
            return {"models": list(self.models.values())}

        @app.post("/api/generate/tts")
        def generate_tts(req: GenerateTTSRequest):
            if not req.text or not req.text.strip():
                raise HTTPException(status_code=400, detail="Text cannot be empty")
            if req.model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found")
            if req.reference_sample_id not in self.samples:
                raise HTTPException(status_code=404, detail=f"Reference sample '{req.reference_sample_id}' not found")

            gen_id = str(uuid.uuid4())
            filename = f"tts_{gen_id[:8]}.wav"
            # Produce 24kHz synthesized audio dummy
            audio_bytes = SyntheticAudioGenerator.generate_speech_dummy_wav(
                duration_s=2.5, sample_rate=24000, base_freq=160.0
            )
            self.generated_audio[filename] = audio_bytes

            return {
                "generation_id": gen_id,
                "audio_url": f"/api/audio/generated/{filename}",
                "filename": filename,
            }

        @app.post("/api/generate/sts")
        def generate_sts(req: GenerateSTSRequest):
            if req.input_sample_id not in self.samples:
                raise HTTPException(status_code=404, detail=f"Input sample '{req.input_sample_id}' not found")
            if req.model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found")
            if req.pitch_shift < -24 or req.pitch_shift > 24:
                raise HTTPException(status_code=400, detail="Pitch shift must be between -24 and +24 semitones")
            if req.index_rate < 0.0 or req.index_rate > 1.0:
                raise HTTPException(status_code=400, detail="Index rate must be between 0.0 and 1.0")

            gen_id = str(uuid.uuid4())
            filename = f"sts_{gen_id[:8]}.wav"
            # Produce 40kHz converted audio dummy
            audio_bytes = SyntheticAudioGenerator.generate_sine_wav(
                duration_s=3.0, sample_rate=40000, frequency=220.0 * (2 ** (req.pitch_shift / 12.0))
            )
            self.generated_audio[filename] = audio_bytes

            return {
                "generation_id": gen_id,
                "audio_url": f"/api/audio/generated/{filename}",
                "filename": filename,
            }

        @app.get("/api/audio/generated/{filename}")
        def get_generated_audio(filename: str, request: Request):
            if filename not in self.generated_audio:
                raise HTTPException(status_code=404, detail="Generated audio not found")

            audio_data = self.generated_audio[filename]
            total_len = len(audio_data)
            range_header = request.headers.get("Range")

            if range_header and range_header.startswith("bytes="):
                parts = range_header.replace("bytes=", "").split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else total_len - 1

                if start >= total_len or end >= total_len or start > end:
                    raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

                chunk = audio_data[start : end + 1]
                headers = {
                    "Content-Range": f"bytes {start}-{end}/{total_len}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(chunk)),
                    "Content-Type": "audio/wav",
                }
                return Response(content=chunk, status_code=206, headers=headers)

            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={"Accept-Ranges": "bytes", "Content-Length": str(total_len)},
            )

        @app.get("/api/audio/download/{filename}")
        def download_generated_audio(filename: str):
            if filename not in self.generated_audio:
                raise HTTPException(status_code=404, detail="Audio file not found")

            content = self.generated_audio[filename]
            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "audio/wav",
                "Content-Length": str(len(content)),
            }
            return Response(content=content, headers=headers)

        @app.websocket("/ws/training")
        async def ws_training(websocket: WebSocket):
            await websocket.accept()
            self.ws_subscribers.append(websocket)
            try:
                # Stream initial simulated telemetry progress
                for epoch in range(1, 4):
                    msg = {
                        "type": "progress",
                        "engine": "f5-tts",
                        "run_id": str(uuid.uuid4()),
                        "epoch": epoch,
                        "total_epochs": 3,
                        "step": epoch * 10,
                        "total_steps": 30,
                        "loss": round(0.5 / epoch, 4),
                        "progress_percent": round((epoch / 3.0) * 100, 1),
                        "message": f"Epoch {epoch}/3 completed",
                        "status": "training",
                        "timestamp": "2026-11-06T10:00:00Z",
                    }
                    await websocket.send_json(msg)

                # Send completion frame
                completed_msg = {
                    "type": "completed",
                    "status": "completed",
                    "model_name": "test_voice",
                    "checkpoint_file": "data/models/test_voice_f5tts.pt",
                }
                await websocket.send_json(completed_msg)

                # Keep connection alive for client pings
                while True:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.ws_subscribers:
                    self.ws_subscribers.remove(websocket)

    def get_app(self) -> FastAPI:
        return self.app


# ==============================================================================
# 3. High-Level Test Client Wrapper
# ==============================================================================

class VoiceSamplerTestClient:
    """
    Unified client supporting both live remote servers and embedded ASGI apps.
    Provides standard REST request methods, file upload helpers, and WebSocket support.
    """

    def __init__(self, base_url: Optional[str] = None, app: Optional[FastAPI] = None):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.app = app
        if not self.base_url and not self.app:
            # Default: create embedded mock app
            self.mock_server = EmbeddedMockServer()
            self.app = self.mock_server.get_app()
        else:
            self.mock_server = None

        if self.app:
            self.test_client = TestClient(self.app)
            self.is_live = False
        else:
            self.session = requests.Session()
            self.is_live = True

    # --- HTTP Helpers ---

    def get(self, path: str, headers: Optional[Dict[str, str]] = None) -> Any:
        if self.is_live:
            return self.session.get(f"{self.base_url}{path}", headers=headers)
        return self.test_client.get(path, headers=headers)

    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if self.is_live:
            return self.session.post(
                f"{self.base_url}{path}", json=json, data=data, files=files, headers=headers
            )
        return self.test_client.post(path, json=json, data=data, files=files, headers=headers)

    def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if self.is_live:
            return self.session.put(f"{self.base_url}{path}", json=json, headers=headers)
        return self.test_client.put(path, json=json, headers=headers)

    def delete(self, path: str, headers: Optional[Dict[str, str]] = None) -> Any:
        if self.is_live:
            return self.session.delete(f"{self.base_url}{path}", headers=headers)
        return self.test_client.delete(path, headers=headers)

    # --- Domain Endpoint Convenience Methods ---

    def get_health(self) -> Any:
        return self.get("/api/health")

    def post_shutdown(self) -> Any:
        return self.post("/api/shutdown")

    def get_gpu_settings(self) -> Any:
        return self.get("/api/settings/gpu")

    def get_samples(self) -> Any:
        return self.get("/api/samples")

    def upload_sample(
        self,
        audio_bytes: bytes,
        filename: str = "sample.wav",
        content_type: str = "audio/wav",
    ) -> Any:
        files = {"file": (filename, io.BytesIO(audio_bytes), content_type)}
        return self.post("/api/samples/upload", files=files)

    def record_sample(
        self,
        audio_bytes: bytes,
        filename: str = "mic_sample.wav",
        content_type: str = "audio/wav",
    ) -> Any:
        files = {"file": (filename, io.BytesIO(audio_bytes), content_type)}
        return self.post("/api/samples/record", files=files)

    def update_sample_transcript(self, sample_id: str, transcript: str) -> Any:
        return self.put(f"/api/samples/{sample_id}", json={"transcript": transcript})

    def delete_sample(self, sample_id: str) -> Any:
        return self.delete(f"/api/samples/{sample_id}")

    def get_sample_audio(self, sample_id: str, byte_range: Optional[str] = None) -> Any:
        headers = {"Range": byte_range} if byte_range else None
        return self.get(f"/api/samples/{sample_id}/audio", headers=headers)

    def train_f5tts(
        self,
        model_name: str,
        epochs: int = 10,
        batch_size: int = 2,
        learning_rate: float = 0.0001,
        sample_ids: Optional[List[str]] = None,
    ) -> Any:
        payload = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sample_ids": sample_ids,
        }
        return self.post("/api/train/f5-tts", json=payload)

    def train_rvc(
        self,
        model_name: str,
        epochs: int = 20,
        batch_size: int = 4,
        sample_rate: int = 40000,
        sample_ids: Optional[List[str]] = None,
    ) -> Any:
        payload = {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "sample_rate": sample_rate,
            "sample_ids": sample_ids,
        }
        return self.post("/api/train/rvc", json=payload)

    def stop_training(self) -> Any:
        return self.post("/api/train/stop")

    def get_models(self) -> Any:
        return self.get("/api/models")

    def generate_tts(
        self,
        text: str,
        model_name: str,
        reference_sample_id: str,
        speed: float = 1.0,
    ) -> Any:
        payload = {
            "text": text,
            "model_name": model_name,
            "reference_sample_id": reference_sample_id,
            "speed": speed,
        }
        return self.post("/api/generate/tts", json=payload)

    def generate_sts(
        self,
        input_sample_id: str,
        model_name: str,
        pitch_shift: int = 0,
        index_rate: float = 0.75,
    ) -> Any:
        payload = {
            "input_sample_id": input_sample_id,
            "model_name": model_name,
            "pitch_shift": pitch_shift,
            "index_rate": index_rate,
        }
        return self.post("/api/generate/sts", json=payload)

    def get_generated_audio(self, filename: str, byte_range: Optional[str] = None) -> Any:
        headers = {"Range": byte_range} if byte_range else None
        return self.get(f"/api/audio/generated/{filename}", headers=headers)

    def download_generated_audio(self, filename: str) -> Any:
        return self.get(f"/api/audio/download/{filename}")

    # --- WebSocket Support ---

    def websocket_training(self):
        """Context manager for WebSocket training telemetry."""
        if not self.is_live:
            return self.test_client.websocket_connect("/ws/training")
        raise NotImplementedError("Live external WebSocket requires websockets client; use Starlette in embedded mode")


# ==============================================================================
# 4. Global Client Factory
# ==============================================================================

def get_test_client(target_url: Optional[str] = None, force_mock: bool = False) -> VoiceSamplerTestClient:
    """
    Factory returning an appropriate VoiceSamplerTestClient.
    If target_url is provided and live server is responding, connects to live server.
    Otherwise returns an embedded test client.
    """
    if force_mock:
        return VoiceSamplerTestClient()

    env_url = target_url or os.environ.get("VOICE_SAMPLER_TEST_URL", "http://127.0.0.1:8000")
    try:
        r = requests.get(f"{env_url}/api/health", timeout=0.8)
        if r.status_code == 200:
            return VoiceSamplerTestClient(base_url=env_url)
    except Exception:
        pass

    # Fallback to embedded mock client
    return VoiceSamplerTestClient()
