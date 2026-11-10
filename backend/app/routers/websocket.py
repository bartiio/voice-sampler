"""
backend/app/routers/websocket.py
WebSocket endpoints for real-time notifications and training telemetry.
Part of Voice Sampler Milestone 2 (M2) and Milestone 3 (M3).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.services.notification_service import get_notification_manager

logger = logging.getLogger("voice_sampler.routers.websocket")
router = APIRouter()


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class TrainingTelemetryManager:
    """
    Manages active WebSocket connections to /ws/training and delivers real-time
    training progress frames, epoch metrics, and terminal status events.
    Thread-safe bridge between synchronous training worker threads and the asyncio event loop.
    """

    _instance: Optional[TrainingTelemetryManager] = None

    def __init__(self) -> None:
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._latest_telemetry: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls) -> TrainingTelemetryManager:
        if cls._instance is None:
            cls._instance = TrainingTelemetryManager()
        return cls._instance

    def register_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Register or update reference to running asyncio event loop."""
        if loop is not None and not loop.is_closed():
            self._loop = loop
        elif self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket client for training telemetry."""
        await websocket.accept()
        self.register_loop()
        async with self._lock:
            self._active_connections.add(websocket)
        logger.info(f"Training WS client connected. Total subscribers: {len(self._active_connections)}")

        # Send latest telemetry snapshot immediately on connection for instant UI hydration
        if self._latest_telemetry is not None:
            try:
                await websocket.send_json(self._latest_telemetry)
            except Exception as e:
                logger.debug(f"Failed to send initial telemetry snapshot: {e}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected WebSocket client."""
        async with self._lock:
            self._active_connections.discard(websocket)
        logger.info(f"Training WS client disconnected. Total subscribers: {len(self._active_connections)}")

    async def broadcast(self, frame: Dict[str, Any]) -> None:
        """Asynchronously broadcast JSON telemetry payload to all active clients."""
        if "timestamp" not in frame:
            frame["timestamp"] = _utc_now_iso()

        self._latest_telemetry = frame

        async with self._lock:
            connections = list(self._active_connections)

        dead_connections: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(frame)
            except (WebSocketDisconnect, RuntimeError, Exception) as exc:
                logger.debug(f"Failed to deliver telemetry frame ({exc}); scheduling removal.")
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                for dead in dead_connections:
                    self._active_connections.discard(dead)

    def broadcast_sync(self, frame: Dict[str, Any]) -> None:
        """
        Thread-safe synchronous bridge allowing background training threads
        to broadcast telemetry frames through the asyncio event loop.
        """
        self.register_loop()
        if self._loop and self._loop.is_running():
            try:
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None

                if current_loop is self._loop:
                    self._loop.create_task(self.broadcast(frame))
                else:
                    asyncio.run_coroutine_threadsafe(self.broadcast(frame), self._loop)
            except Exception as e:
                logger.debug(f"Error dispatching sync training telemetry: {e}")
        else:
            # Event loop not currently running (e.g. CLI runner or isolated unit test)
            logger.debug(f"No running event loop available for training telemetry: {frame.get('type')}")


def get_training_telemetry_manager() -> TrainingTelemetryManager:
    """Return TrainingTelemetryManager singleton instance."""
    return TrainingTelemetryManager.get_instance()


def broadcast_training_telemetry(frame: Dict[str, Any]) -> None:
    """Global convenience helper to broadcast training telemetry from any context."""
    get_training_telemetry_manager().broadcast_sync(frame)


@router.websocket("/notifications")
async def websocket_notifications(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sample and transcription events.
    Broadcasting is managed via NotificationManager.
    """
    manager = get_notification_manager()
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as exc:
        logger.debug(f"Notification WebSocket connection terminated: {exc}")
        await manager.disconnect(websocket)


@router.websocket("/training")
async def websocket_training(websocket: WebSocket):
    """
    WebSocket endpoint for training telemetry streaming.
    Provides real-time loss curves, epoch metrics, and terminal status events.
    """
    manager = get_training_telemetry_manager()
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as exc:
        logger.debug(f"Training WebSocket connection terminated: {exc}")
        await manager.disconnect(websocket)
