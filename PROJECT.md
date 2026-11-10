# Project: Voice Sampler

## Architecture

Voice Sampler is a cross-platform desktop application for recording voice samples, automatically transcribing them with Whisper (with Polish language priority), training custom voice cloning models (F5-TTS for Text-to-Speech and RVC v2 for Speech-to-Speech / Voice Conversion), and generating speech inference in the cloned voice.

```
+----------------------------------------------------------------------------------+
|                              Electron Desktop Shell                              |
|  +----------------------------------------------------------------------------+  |
|  |                     React 19 + TypeScript + Vite Frontend                  |  |
|  |                                                                            |  |
|  |   [Samples View]          [Training View]       [Generate View]  [Settings]|  |
|  |   - Waveform Recorder     - F5-TTS Engine       - TTS (F5-TTS)   - GPU Info|  |
|  |   - Drag & Drop Upload    - RVC v2 Engine       - STS (RVC v2)   - Backend |  |
|  |   - Audio Player & List   - Hyperparams         - Player & DL      Health  |  |
|  |   - Inline Transcript Ed  - Live WS Telemetry                              |  |
|  +----------------------------------------------------------------------------+  |
|        │ REST (Axios/Fetch)               ▲ WebSocket (Telemetry & Events)       |
|        ▼                                  │                                      |
+--------┼──────────────────────────────────┼────────────────────────────────------+
         │ Child Process Lifecycle          │
         ▼ (Managed by Electron Main)       │
+-------------------------------------------┴--------------------------------------+
|                           FastAPI Python Backend                                 |
|                                                                                  |
|   /api/health  /api/settings/gpu  /api/samples  /api/train  /api/generate        |
|   /ws/training (telemetry)       /ws/notifications (transcription events)        |
|                                                                                  |
|   +-----------------------+ +--------------------+ +-------------------------+   |
|   |   Whisper Engine      | |  Training Pipeline | |   Inference Engine      |   |
|   |  (Polish priority,    | |  - F5-TTS Prep     | |  - F5-TTS Inference    |   |
|   |   faster-whisper/ct2, | |  - RVC v2 Pipeline | |  - RVC v2 Inference    |   |
|   |   async background)   | |  - Dev/Simulation  | |  - Audio Exporter      |   |
|   +-----------------------+ +--------------------+ +-------------------------+   |
|                                       │                                          |
|   +-----------------------------------┴--------------------------------------+   |
|   | Storage Layer: SQLite (data/db/voice_sampler.db) + Filesystem (data/)    |   |
|   +--------------------------------------------------------------------------+   |
+----------------------------------------------------------------------------------+
```

## Feature Inventory
Every feature from the Survey phase appears here with its assigned milestone:
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Backend Lifecycle & Scaffolding | FastAPI initialization, lifespan management, CORS, graceful shutdown | M1 | Survey 2 |
| 2 | SQLite Storage & DB Schemas | SQLite DB (`data/db/voice_sampler.db`), WAL mode, tables for samples, models, runs | M1 | Survey 2 |
| 3 | GPU & System Detection | Detect NVIDIA GPU, VRAM, CUDA version via PyTorch, AMD/WMI fallback | M1 | Survey 1, 2 |
| 4 | Backend Health & Settings API | `GET /api/health`, `GET /api/settings/gpu` with status & warnings | M1 | Survey 1, 2 |
| 5 | Audio Storage Layout | Dedicated directories for raw/preprocessed samples, models, and generated audio | M1 | Survey 2 |
| 6 | Microphone Audio Recording Ingestion | Accept audio blobs from browser recorder, normalize to 16-bit PCM WAV | M2 | Survey 1, 2 |
| 7 | Multi-Format File Upload | Drag-and-drop & file upload support for WAV, MP3, FLAC with PyAV/soundfile | M2 | Survey 1, 2 |
| 8 | Sample CRUD & Transcript Editing | `GET /api/samples`, `PUT /api/samples/{id}` for in-place transcript persistence | M2 | Survey 1, 2 |
| 9 | Audio File Streaming | Byte-range audio streaming (`/api/samples/{id}/audio`) for waveform scrubbers | M2 | Survey 1, 2 |
| 10 | Whisper Polish Transcription | Auto-transcribe samples using Whisper with Polish language priority in background | M2 | Survey 2 |
| 11 | F5-TTS Dataset Preparation | Resample audio to 24kHz mono WAV, generate pipe-delimited metadata.csv & vocab.txt | M3 | Survey 3 |
| 12 | F5-TTS Model Training | Fine-tune F5-TTS base model with configurable epochs, batch size, learning rate | M3 | Survey 3 |
| 13 | RVC v2 Dataset Preparation | Preprocess WAVs, RMVPE F0 pitch extraction, HuBERT 768-dim feature extraction | M3 | Survey 3 |
| 14 | RVC v2 Model Training | NSF-VITS model fine-tuning and Faiss index creation with configurable params | M3 | Survey 3 |
| 15 | WebSocket Telemetry Streaming | Real-time telemetry streaming (`ws://127.0.0.1:{port}/ws/training`) | M3 | Survey 2, 3 |
| 16 | Model Registry & Checkpoints | Persist model checkpoints to `data/models/` and query via `/api/models` | M3 | Survey 2, 3 |
| 17 | Dev/Simulation Fallback Engine | Realistic simulation of training & artifacts on non-CUDA machines | M3 | Survey 1, 2, 3 |
| 18 | F5-TTS Speech Generation (TTS) | Synthesize speech from text + reference audio + trained model (`/api/generate/tts`) | M4 | Survey 3 |
| 19 | RVC Voice Conversion (STS) | Convert voice with input audio + model + pitch shift + index rate (`/api/generate/sts`)| M4 | Survey 3 |
| 20 | Generated Audio Playback & Download | Stream and download generated audio (`/api/audio/generated/{id}`, `/api/audio/download/{id}`) | M4 | Survey 2, 3 |
| 21 | Electron Main Process & Lifecycle | Auto-spawn FastAPI child process, health polling, graceful kill with tree-kill | M5 | Survey 1 |
| 22 | React 19 Frontend Shell & Navigation | Sidebar navigation with 4 sections: Samples, Training, Generate, Settings | M5 | Survey 1 |
| 23 | Samples UI (Recorder & Waveform) | 60fps canvas waveform visualizer, MediaRecorder mic recording, upload dropzone | M5 | Survey 1 |
| 24 | Training UI & Live Metrics | Engine selector, hyperparameter inputs, start/stop buttons, real-time telemetry | M5 | Survey 1, 3 |
| 25 | Generate UI (TTS & STS Tabs) | Text input, file upload/recorder, sliders for pitch/index rate, audio player & save | M5 | Survey 1, 3 |
| 26 | Settings UI & GPU Status Banner | GPU details, CUDA status, warning banner when CUDA is missing, backend health | M5 | Survey 1, 2 |
| 27 | Opaque-Box E2E Test Suite | Automated test harness covering Tiers 1-4 across all features and acceptance criteria | M6 / E2E Track | Survey 1, 2, 3 |
| 28 | Adversarial Hardening & Verification | White-box adversarial testing (Tier 5), edge cases, documentation (README.md) | M6 | Survey 1, 2, 3 |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Backend Foundation & Hardware API | FastAPI scaffolding, SQLite persistence, GPU detection, Health & Settings endpoints, directory structure | none | DONE |
| M2 | Sample Management & Whisper Polish | Sample upload/record endpoints, audio streaming, CRUD, async Polish Whisper worker, transcript persistence | M1 | DONE |
| M3 | ML Training Pipelines & Telemetry | F5-TTS & RVC v2 dataset preparation & training pipelines, WebSocket telemetry, model registry, dev/sim engine | M1, M2 | DONE |
| M4 | Voice Generation (TTS & STS) | F5-TTS inference, RVC v2 conversion, audio player streaming, and download endpoints | M3 | DONE |
| M5 | Desktop UI Shell & App Integration | Electron main process, child process supervisor, React 19 + Vite frontend with 4 views (Samples, Training, Generate, Settings), recording waveform, build scripts | M1, M2, M4 | DONE |
| M6 | Final Verification & Adversarial Hardening | Phase 1: 100% pass of E2E test suite (Tiers 1-4); Phase 2: Tier 5 adversarial testing, complete README.md, quality gates | M5, TEST_READY.md | DONE |

*Note: The **E2E Testing Track** runs in parallel with M1–M5, creating the test infrastructure and test suites for Tiers 1–4, and concludes by publishing `TEST_READY.md` for M6.*

## Interface Contracts

### 1. Electron ↔ FastAPI Process Supervisor
- Electron spawns backend: `python run_server.py --port 8000` (or dynamic free port).
- Startup readiness: Poll `GET http://127.0.0.1:8000/api/health` until HTTP 200 returned (timeout 15s).
- Shutdown handshake:
  1. Electron invokes `POST http://127.0.0.1:8000/api/shutdown`.
  2. Electron waits up to 2 seconds, then issues `tree-kill(pid, 'SIGKILL')` to terminate any orphaned child processes.

### 2. REST API Endpoints
- `GET /api/health` -> `{"status": "ok", "version": "0.1.0", "backend_ready": true}`
- `GET /api/settings/gpu` ->
  ```json
  {
    "cuda_available": false,
    "gpu_name": "AMD Radeon 740M Graphics",
    "vram_total_mb": 512,
    "vram_allocated_mb": 0,
    "cuda_version": null,
    "simulation_mode": true,
    "message": "No CUDA GPU detected. GPU is required for real training. Running in simulation mode."
  }
  ```
- `GET /api/samples` -> `{"samples": [{"id": "uuid", "filename": "sample.wav", "duration": 5.4, "transcript": "...", "status": "transcribed", "created_at": "..."}]}`
- `POST /api/samples/upload` -> Accepts `multipart/form-data` with `file: UploadFile`. Returns sample metadata.
- `POST /api/samples/record` -> Accepts `multipart/form-data` with recorded WAV/WebM blob. Saves normalized WAV.
- `PUT /api/samples/{id}` -> `{"transcript": "Updated text"}` -> Updates transcript in DB.
- `DELETE /api/samples/{id}` -> Deletes sample audio and DB record.
- `GET /api/samples/{id}/audio` -> Streams audio file with `Accept-Ranges: bytes` (HTTP 206 support).
- `POST /api/train/f5-tts` ->
  ```json
  {"model_name": "voice_1", "epochs": 10, "batch_size": 2, "learning_rate": 0.0001, "sample_ids": ["..."]}
  ```
- `POST /api/train/rvc` ->
  ```json
  {"model_name": "voice_1", "epochs": 20, "batch_size": 4, "sample_rate": 40000, "sample_ids": ["..."]}
  ```
- `POST /api/train/stop` -> Cancels active training run.
- `GET /api/models` -> `{"models": [{"name": "voice_1", "engine": "f5-tts", "checkpoint_path": "...", "created_at": "..."}]}`
- `POST /api/generate/tts` ->
  ```json
  {"text": "Cześć, to jest mój sklonowany głos.", "model_name": "voice_1", "reference_sample_id": "sample_1"}
  ```
  Returns `{"generation_id": "uuid", "audio_url": "/api/audio/generated/xyz.wav", "filename": "xyz.wav"}`
- `POST /api/generate/sts` ->
  ```json
  {"input_sample_id": "sample_2", "model_name": "voice_1", "pitch_shift": 0, "index_rate": 0.75}
  ```
  Returns `{"generation_id": "uuid", "audio_url": "/api/audio/generated/abc.wav", "filename": "abc.wav"}`
- `GET /api/audio/generated/{filename}` -> Streams audio for playback.
- `GET /api/audio/download/{filename}` -> Returns `Content-Disposition: attachment; filename=...`.

### 3. WebSocket Telemetry Protocol (`/ws/training`)
- Client connects to `ws://127.0.0.1:8000/ws/training`.
- Server streams JSON frames:
  ```json
  {
    "type": "progress",
    "engine": "f5-tts",
    "run_id": "uuid",
    "epoch": 3,
    "total_epochs": 10,
    "step": 45,
    "total_steps": 150,
    "loss": 0.3421,
    "progress_percent": 30.0,
    "message": "Epoch 3/10 completed (loss: 0.3421)",
    "status": "training",
    "timestamp": "2026-11-06T09:00:00Z"
  }
  ```
- Terminal events:
  - Completion: `{"type": "completed", "status": "completed", "model_name": "...", "checkpoint_file": "..."}`
  - Error: `{"type": "error", "status": "failed", "error_message": "..."}`

## Code Layout

```
c:\Users\bartoszk\Desktop\voice sampler\
├── package.json                   # Root package config (scripts: dev, build, lint, test)
├── tsconfig.json                  # Root TypeScript configuration
├── tsconfig.node.json             # Electron main/preload TypeScript config
├── vite.config.ts                 # Vite + Electron build plugin config
├── tailwind.config.js             # Tailwind CSS styling config
├── postcss.config.js              # PostCSS config
├── index.html                     # Frontend entry point
├── run_server.py                  # CLI Python runner for FastAPI backend
├── requirements.txt               # Python dependencies
├── README.md                      # Comprehensive user & developer setup guide
│
├── electron/                      # Electron application shell
│   ├── main.ts                    # Main process, window creation, child process manager
│   └── preload.ts                 # Preload context bridge (IPC APIs)
│
├── src/                           # React 19 Frontend
│   ├── main.tsx                   # React root entry
│   ├── App.tsx                    # Main app shell with sidebar layout & routing
│   ├── index.css                  # Global styles & Tailwind directives
│   ├── api/                       # API clients (REST Axios client & WebSocket service)
│   │   ├── client.ts              # REST client wrapper
│   │   ├── websocket.ts           # Resilient WebSocket client with auto-reconnect
│   │   └── types.ts               # TypeScript data transfer models
│   ├── components/                # Shared UI components
│   │   ├── Sidebar.tsx            # Accessible navigation sidebar
│   │   ├── AudioPlayer.tsx        # Audio playback bar with scrubber & time display
│   │   ├── WaveformCanvas.tsx     # Real-time microphone audio visualizer
│   │   └── StatusBadge.tsx        # Status indicators (health, training, GPU)
│   └── views/                     # Main section views
│       ├── SamplesView.tsx        # Mic recorder, drag-drop upload, sample table, transcript editor
│       ├── TrainingView.tsx       # F5-TTS & RVC selector, hyperparams, live telemetry
│       ├── GenerateView.tsx       # TTS tab, STS tab, inference controls, player & download
│       └── SettingsView.tsx       # GPU diagnostic card, backend health indicator
│
├── backend/                       # Python FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app creation, lifespan, route inclusion
│   │   ├── config.py              # Application settings, paths, simulation flag
│   │   ├── db/                    # SQLite database & models
│   │   │   ├── __init__.py
│   │   │   ├── connection.py      # SQLite connection manager (WAL mode)
│   │   │   └── repository.py      # CRUD for samples, models, runs
│   │   ├── routers/               # API endpoint routers
│   │   │   ├── __init__.py
│   │   │   ├── health.py          # /api/health and /api/shutdown
│   │   │   ├── settings.py        # /api/settings/gpu
│   │   │   ├── samples.py         # /api/samples (upload, record, stream, CRUD)
│   │   │   ├── train.py           # /api/train (start, stop, status)
│   │   │   ├── generate.py        # /api/generate (tts, sts, audio streaming)
│   │   │   ├── models.py          # /api/models (registry query)
│   │   │   └── websocket.py       # /ws/training and /ws/notifications
│   │   ├── services/              # Core business & ML logic
│   │   │   ├── __init__.py
│   │   │   ├── gpu_service.py     # GPU & CUDA detection (PyTorch + WMI)
│   │   │   ├── audio_service.py   # Audio normalization, resampling, WAV encoding
│   │   │   ├── whisper_service.py # Polish Whisper transcription (faster-whisper/ct2)
│   │   │   ├── f5tts_service.py   # F5-TTS dataset prep, training, and inference
│   │   │   ├── rvc_service.py     # RVC v2 dataset prep, training, and inference
│   │   │   └── simulation.py      # Realism simulation engine for non-CUDA hosts
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logger.py          # Standardized logging
│
├── tests/                         # E2E & Integration Test Suites
│   ├── e2e/
│   │   ├── harness.py             # Test harness & HTTP/WS client
│   │   ├── test_tier1_features.py # Tier 1: Feature coverage (all endpoints & flows)
│   │   ├── test_tier2_boundary.py # Tier 2: Boundary & corner cases
│   │   ├── test_tier3_combos.py   # Tier 3: Cross-feature combinations
│   │   └── test_tier4_scenarios.py# Tier 4: Real-world Polish voice cloning scenarios
│   └── run_e2e_tests.py           # E2E test runner executing all tiers
│
└── data/                          # Runtime data (persisted, gitignored)
    ├── db/                        # SQLite database
    ├── samples/                   # Uploaded and recorded samples
    ├── models/                    # Model checkpoints (F5-TTS, RVC .pth/.index)
    └── generated/                 # Output generated audio files
```
