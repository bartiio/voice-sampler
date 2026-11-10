"""
backend/app/main.py
FastAPI application instance for Voice Sampler desktop application.
Configures CORS, lifespan events (storage, database, and hardware checks),
centralized exception handlers, and modular API routers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.db.connection import checkpoint_db
from backend.app.db.repository import init_db
from backend.app.routers import health, settings
from backend.app.services.gpu_service import detect_system_gpu
from backend.app.utils.logger import setup_logger

# Standard logging configuration
logger = setup_logger("voice_sampler.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI Lifespan Context Manager.
    Handles startup initialization (directories, DB, hardware checks) and graceful teardown.
    """
    # === STARTUP SEQUENCE ===
    app_settings = get_settings()
    app.state.start_time = time.time()
    logger.info(
        f"Starting {app_settings.app_name} v{app_settings.app_version} "
        f"({'DEV' if app_settings.dev_mode else 'PROD'} mode)..."
    )

    # 1. Initialize filesystem storage directories
    created_dirs = app_settings.ensure_directories()
    logger.info(f"Verified {len(created_dirs)} runtime data directories under {app_settings.data_dir}")

    # 2. Initialize SQLite database schema
    try:
        init_db()
        app.state.db_initialized = True
        logger.info(f"SQLite database schema initialized at {app_settings.db_path}")
    except Exception as e:
        logger.error(f"Database initialization error: {e}", exc_info=True)
        app.state.db_initialized = False

    # 3. Hardware & GPU diagnostics logging
    try:
        gpu_info = detect_system_gpu()
        cuda_ok = gpu_info.get("cuda_available", False)
        gpu_name = gpu_info.get("gpu_name") or "Unknown"
        vram = gpu_info.get("vram_total_mb", 0)
        logger.info(f"Hardware Diagnostics: GPU='{gpu_name}' ({vram}MB VRAM), CUDA={cuda_ok}")
        if not cuda_ok:
            logger.info("Running in simulation mode (CUDA GPU not detected).")
    except Exception as e:
        logger.warning(f"Failed to query GPU diagnostics on startup: {e}")

    # 4. Register active asyncio event loop in NotificationManager and TrainingTelemetryManager
    try:
        from backend.app.services.notification_service import get_notification_manager
        get_notification_manager().register_loop(asyncio.get_running_loop())
        logger.info("Registered active event loop in NotificationManager")
    except Exception as e:
        logger.warning(f"Failed to register event loop in NotificationManager: {e}")

    try:
        from backend.app.routers.websocket import get_training_telemetry_manager
        get_training_telemetry_manager().register_loop(asyncio.get_running_loop())
        logger.info("Registered active event loop in TrainingTelemetryManager")
    except Exception as e:
        logger.warning(f"Failed to register event loop in TrainingTelemetryManager: {e}")

    logger.info("FastAPI backend startup complete. Ready for requests.")

    yield

    # === SHUTDOWN SEQUENCE ===
    logger.info("FastAPI backend executing shutdown teardown...")
    try:
        from backend.app.services.f5tts_service import get_f5tts_service
        f5 = get_f5tts_service()
        if f5.is_training_active():
            f5.stop_training()
    except Exception as e:
        logger.warning(f"Error stopping F5-TTS on shutdown: {e}")

    try:
        from backend.app.services.rvc_service import get_rvc_service
        rvc = get_rvc_service()
        if rvc.is_training_active():
            rvc.cancel_training()
    except Exception as e:
        logger.warning(f"Error stopping RVC on shutdown: {e}")

    try:
        from backend.app.services.whisper_service import get_whisper_service
        get_whisper_service().shutdown(wait=False)
    except Exception as e:
        logger.warning(f"Error during shutdown Whisper service: {e}")
    try:
        checkpoint_db()
    except Exception as e:
        logger.warning(f"Error during shutdown database checkpoint: {e}")
    logger.info("FastAPI backend shutdown finished.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app_settings = get_settings()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=app_settings.app_description,
        lifespan=lifespan,
        docs_url="/docs" if app_settings.dev_mode else None,
        redoc_url="/redoc" if app_settings.dev_mode else None,
    )

    # 1. Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Configure Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "status_code": exc.status_code,
                "error_code": getattr(exc, "error_code", "HTTP_EXCEPTION"),
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": exc.errors(),
                "status_code": 422,
                "error_code": "VALIDATION_ERROR",
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error occurred.",
                "status_code": 500,
                "error_code": "INTERNAL_SERVER_ERROR",
            },
        )

    # 3. Mount API Routers
    # Milestone 1: Health & Shutdown
    app.include_router(health.router, prefix="/api", tags=["health"])

    # Milestone 1: Settings & GPU
    app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

    # Future Milestones: Samples (M2), Training (M3), Models (M3), Generate (M4), WebSockets (M3)
    try:
        from backend.app.routers import samples as samples_router
        app.include_router(samples_router.router, prefix="/api/samples", tags=["samples"])
    except ImportError:
        pass

    try:
        from backend.app.routers import train as train_router
        app.include_router(train_router.router, prefix="/api/train", tags=["training"])
    except ImportError:
        pass

    try:
        from backend.app.routers import models as models_router
        app.include_router(models_router.router, prefix="/api/models", tags=["models"])
    except ImportError:
        pass

    try:
        from backend.app.routers import generate as generate_router
        app.include_router(generate_router.router, prefix="/api/generate", tags=["generation"])
    except ImportError:
        pass

    try:
        from backend.app.routers import audio as audio_router
        app.include_router(audio_router.router, prefix="/api/audio", tags=["audio"])
    except ImportError:
        pass

    try:
        from backend.app.routers import websocket as ws_router
        app.include_router(ws_router.router, prefix="/ws", tags=["websockets"])
    except ImportError:
        pass

    return app


app = create_app()
