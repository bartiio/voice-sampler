"""
backend/app/services/notification_service.py
WebSocket notification manager providing thread-safe real-time event broadcasting.
Part of Voice Sampler Milestone 2 (M2).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("voice_sampler.notifications")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NotificationManager:
    """
    Manages active WebSocket connections and delivers real-time notifications.
    Thread-safe bridge between synchronous worker threads and asyncio event loop.
    """

    _instance: Optional[NotificationManager] = None

    def __init__(self) -> None:
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_instance(cls) -> NotificationManager:
        if cls._instance is None:
            cls._instance = NotificationManager()
        return cls._instance

    def register_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Register the running asyncio event loop."""
        if loop is not None:
            self._loop = loop
        elif self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.register_loop()
        async with self._lock:
            self._active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self._active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected WebSocket."""
        async with self._lock:
            self._active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self._active_connections)}")

    async def broadcast(self, event: Dict[str, Any]) -> None:
        """Broadcast JSON payload to all active WebSocket connections asynchronously."""
        if "timestamp" not in event:
            event["timestamp"] = _utc_now_iso()

        async with self._lock:
            connections = list(self._active_connections)

        dead_connections: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(event)
            except (WebSocketDisconnect, RuntimeError, Exception) as exc:
                logger.debug(f"Failed to send to client ({exc}). Marking for removal.")
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                for dead in dead_connections:
                    self._active_connections.discard(dead)

    def broadcast_sync(self, event: Dict[str, Any]) -> None:
        """
        Thread-safe synchronous bridge for background worker threads and async handlers.
        Schedules broadcast on the registered asyncio event loop without deadlocks or thread blocking.
        """
        self.register_loop()
        if self._loop and self._loop.is_running():
            try:
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None

                if current_loop is self._loop:
                    self._loop.create_task(self.broadcast(event))
                else:
                    asyncio.run_coroutine_threadsafe(self.broadcast(event), self._loop)
            except Exception as e:
                logger.debug(f"Error scheduling sync broadcast: {e}")
        else:
            logger.debug(f"No running event loop available for sync broadcast: {event.get('event')}")

    # Convenience Notification Helpers
    def notify_sample_created(self, sample: Dict[str, Any]) -> None:
        event = {
            "type": "notification",
            "event": "sample_created",
            "sample_id": sample.get("id"),
            "filename": sample.get("filename"),
            "duration": sample.get("duration", sample.get("duration_seconds", 0.0)),
            "status": sample.get("status", sample.get("transcript_status", "pending")),
        }
        self.broadcast_sync(event)

    def notify_transcription_started(self, sample_id: str, filename: str = "") -> None:
        event = {
            "type": "notification",
            "event": "transcription_started",
            "sample_id": sample_id,
            "filename": filename,
            "status": "transcribing",
        }
        self.broadcast_sync(event)

    def notify_transcription_completed(self, sample_id: str, transcript: str, duration: float = 0.0) -> None:
        event = {
            "type": "notification",
            "event": "transcription_completed",
            "sample_id": sample_id,
            "status": "transcribed",
            "transcript": transcript,
            "duration": duration,
        }
        self.broadcast_sync(event)

    def notify_transcription_failed(self, sample_id: str, error_message: str) -> None:
        event = {
            "type": "notification",
            "event": "transcription_failed",
            "sample_id": sample_id,
            "status": "error",
            "error_message": error_message,
        }
        self.broadcast_sync(event)

    def notify_sample_updated(self, sample_id: str, transcript: str, status: str) -> None:
        event = {
            "type": "notification",
            "event": "sample_updated",
            "sample_id": sample_id,
            "transcript": transcript,
            "status": status,
        }
        self.broadcast_sync(event)

    def notify_sample_deleted(self, sample_id: str) -> None:
        event = {
            "type": "notification",
            "event": "sample_deleted",
            "sample_id": sample_id,
            "status": "deleted",
        }
        self.broadcast_sync(event)


def get_notification_manager() -> NotificationManager:
    return NotificationManager.get_instance()
