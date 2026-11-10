"""
backend/app/routers/health.py
Health check and process shutdown endpoints for Voice Sampler.
Used by Electron to monitor backend readiness and gracefully terminate the child process.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
import signal
import sys
import time
from typing import Optional

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

from backend.app.config import get_settings

logger = logging.getLogger("voice_sampler.health")
router = APIRouter()


class HealthResponse(BaseModel):
    """Schema returned by GET /api/health."""
    status: str = Field(default="ok", description="Service health status")
    version: str = Field(..., description="API software version")
    backend_ready: bool = Field(default=True, description="Readiness flag for Electron polling")
    uptime_seconds: float = Field(..., description="Elapsed server uptime in seconds")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the response")


class ShutdownResponse(BaseModel):
    """Schema returned by POST /api/shutdown."""
    status: str = Field(default="shutting_down", description="Shutdown acknowledgement")
    message: str = Field(default="Server shutdown initiated cleanly", description="Status message")


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Backend Health & Liveness Probe",
    description="Returns backend readiness status, version, and uptime for Electron startup polling.",
)
async def get_health(request: Request) -> HealthResponse:
    """Return health check response for Electron startup poll."""
    settings = get_settings()
    start_time: Optional[float] = getattr(request.app.state, "start_time", None)
    uptime = round(time.time() - start_time, 2) if start_time else 0.0

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        backend_ready=True,
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/shutdown",
    response_model=ShutdownResponse,
    status_code=status.HTTP_200_OK,
    summary="Graceful Server Shutdown",
    description="Initiates graceful termination of the FastAPI/Uvicorn backend process upon Electron app exit.",
)
async def shutdown_server(request: Request) -> ShutdownResponse:
    """
    Initiate graceful termination of the Uvicorn server loop.
    Returns HTTP 200 before signaling Uvicorn to exit.
    """
    logger.info("Received POST /api/shutdown from client. Triggering shutdown...")
    server = getattr(request.app.state, "server", None)

    def _do_shutdown() -> None:
        logger.info("Executing server exit signal...")
        if server is not None:
            # Setting should_exit triggers Uvicorn's graceful teardown
            server.should_exit = True
        else:
            # Fallback if server reference is missing
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                sys.exit(0)

    loop = asyncio.get_running_loop()
    # Schedule exit 100ms later so HTTP 200 payload is flushed to client
    loop.call_later(0.1, _do_shutdown)

    return ShutdownResponse(
        status="shutting_down",
        message="Server shutdown initiated cleanly",
    )
