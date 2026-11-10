#!/usr/bin/env python3
"""
run_server.py
CLI entry point to launch the Voice Sampler FastAPI backend.
Spawned by Electron main process on application launch:
    python run_server.py --port 8000
Supports graceful shutdown via /api/shutdown or SIGINT/SIGTERM,
and monitors parent PID to automatically terminate if Electron quits.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import threading
import time

import uvicorn

from backend.app.utils.logger import setup_logger

logger = setup_logger("voice_sampler.runner")


def start_parent_watcher(parent_pid: int, server: uvicorn.Server) -> None:
    """
    Monitor parent process (e.g. Electron) and cleanly stop backend if parent dies.
    Prevents orphaned Python background processes if Electron crashes or is killed.
    """
    if parent_pid <= 1:
        return

    try:
        import psutil
    except ImportError:
        logger.warning("psutil not available; parent process monitoring disabled.")
        return

    def _watch() -> None:
        logger.info(f"Parent process watcher active for PID {parent_pid}")
        while not server.should_exit:
            time.sleep(2.0)
            try:
                if not psutil.pid_exists(parent_pid):
                    logger.warning(
                        f"Parent process {parent_pid} no longer exists. Initiating auto-shutdown..."
                    )
                    server.should_exit = True
                    break
            except Exception as e:
                logger.debug(f"Parent check exception: {e}")
                break

    watcher = threading.Thread(target=_watch, daemon=True, name="ParentWatcher")
    watcher.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Sampler Backend Server")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.getenv("VOICE_SAMPLER_PORT", "8000")),
        help="Port to bind the server to (default: 8000 or VOICE_SAMPLER_PORT)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("VOICE_SAMPLER_HOST", "127.0.0.1"),
        help="Host address to bind to (default: 127.0.0.1 or VOICE_SAMPLER_HOST)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=os.getenv("VOICE_SAMPLER_DEV_MODE", "0").lower() in ("1", "true", "yes"),
        help="Enable development mode with API docs and verbose logging",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code change (development only)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("VOICE_SAMPLER_LOG_LEVEL", "info").lower(),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Server log level",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="Explicit parent PID to monitor for automatic termination",
    )

    args = parser.parse_args()

    # Propagate CLI arguments to environment variables so config.py picks them up
    os.environ["VOICE_SAMPLER_PORT"] = str(args.port)
    os.environ["VOICE_SAMPLER_HOST"] = args.host
    if args.dev:
        os.environ["VOICE_SAMPLER_DEV_MODE"] = "1"
    os.environ["VOICE_SAMPLER_LOG_LEVEL"] = args.log_level

    # Ensure project root is on Python sys.path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    logger.info(
        f"Starting Voice Sampler Backend on {args.host}:{args.port} "
        f"(dev_mode={args.dev})..."
    )

    if args.reload:
        # In reload mode, uvicorn spawns a supervisor subprocess
        uvicorn.run(
            "backend.app.main:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        # Direct Server instance allows clean shutdown via server.should_exit
        from backend.app.main import app

        config = uvicorn.Config(
            app=app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            access_log=True,
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        app.state.server = server

        # Start parent PID monitoring if running as a child process of Electron
        monitored_pid = args.parent_pid or os.getppid()
        if monitored_pid and monitored_pid > 1:
            start_parent_watcher(monitored_pid, server)

        # Run the server event loop
        server.run()
        logger.info("Voice Sampler Backend process exited cleanly.")


if __name__ == "__main__":
    main()
