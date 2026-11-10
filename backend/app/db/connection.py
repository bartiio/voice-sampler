"""
backend/app/db/connection.py
Thread-safe SQLite connection management, PRAGMA configuration,
and runtime storage directory initialization.
Part of Voice Sampler Milestone 1 (M1).
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
from typing import Dict, Generator, Optional, Union

from backend.app.config import get_settings

logger = logging.getLogger("voice_sampler.db.connection")

# Thread-safe database path override (e.g. for testing with ":memory:" or custom temp files)
_db_path_override: Optional[Union[Path, str]] = None
_shared_test_conn: Optional[sqlite3.Connection] = None


def init_storage_directories(base_data_dir: Optional[Union[Path, str]] = None) -> Dict[str, Path]:
    """
    Ensure all required runtime data and model storage directories exist.
    Returns a dictionary of confirmed Path objects.
    """
    if base_data_dir is not None:
        root = Path(base_data_dir).resolve()
    else:
        settings = get_settings()
        root = settings.data_dir or (settings.project_root / "data")

    dirs: Dict[str, Path] = {
        "data_root": root,
        "db": root / "db",
        "samples": root / "samples",
        "samples_raw": root / "samples" / "raw",
        "samples_preprocessed": root / "samples" / "preprocessed",
        "models": root / "models",
        "models_f5_tts": root / "models" / "f5_tts",
        "models_rvc": root / "models" / "rvc",
        "generated": root / "generated",
        "generated_tts": root / "generated" / "tts",
        "generated_sts": root / "generated" / "sts",
        "cache": root / "cache",
    }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def set_db_path(path: Optional[Union[Path, str]]) -> None:
    """Override the active database path (useful for automated testing)."""
    global _db_path_override, _shared_test_conn
    _db_path_override = path
    if path == ":memory:":
        # Maintain a persistent connection for in-memory DB during tests
        _shared_test_conn = _create_raw_connection(":memory:")
    else:
        if _shared_test_conn is not None:
            try:
                _shared_test_conn.close()
            except Exception:
                pass
            _shared_test_conn = None


def get_active_db_path() -> Union[Path, str]:
    """Return the currently configured database path."""
    if _db_path_override is not None:
        return _db_path_override
    env_path = os.getenv("VOICE_SAMPLER_DB_PATH")
    if env_path:
        return Path(env_path).resolve()
    settings = get_settings()
    return settings.db_path or (settings.data_dir / "db" / "voice_sampler.db")


def _create_raw_connection(target_path: Union[Path, str]) -> sqlite3.Connection:
    """Instantiate and configure a raw sqlite3 connection with proper PRAGMAs."""
    if isinstance(target_path, Path):
        # Guarantee parent directory exists before connecting
        target_path.parent.mkdir(parents=True, exist_ok=True)
        db_str = str(target_path)
    else:
        db_str = target_path

    conn = sqlite3.connect(
        database=db_str,
        timeout=30.0,
        check_same_thread=False,
        isolation_level=None,  # Autocommit mode; transactions managed explicitly
    )
    conn.row_factory = sqlite3.Row

    # Execute critical PRAGMAs
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    if db_str != ":memory:":
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA busy_timeout = 30000;")
    cur.close()

    return conn


@contextmanager
def get_db_connection(
    db_path: Optional[Union[Path, str]] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager providing a configured SQLite connection.
    Commits on successful completion, rolls back on exception, and closes cleanly.
    """
    global _shared_test_conn
    target = db_path if db_path is not None else get_active_db_path()

    # Reuse shared in-memory connection if testing against :memory:
    if target == ":memory:" and _shared_test_conn is not None:
        try:
            _shared_test_conn.execute("BEGIN IMMEDIATE;")
            yield _shared_test_conn
            if _shared_test_conn.in_transaction:
                _shared_test_conn.execute("COMMIT;")
        except Exception:
            try:
                if _shared_test_conn.in_transaction:
                    _shared_test_conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        return

    conn = _create_raw_connection(target)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        yield conn
        if conn.in_transaction:
            conn.execute("COMMIT;")
    except Exception:
        try:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency provider for route handlers."""
    with get_db_connection() as conn:
        yield conn


def checkpoint_db(db_path: Optional[Union[Path, str]] = None) -> None:
    """
    Execute WAL checkpoint with TRUNCATE to flush all WAL frames into the main .db file.
    Should be called during application shutdown.
    """
    target = db_path if db_path is not None else get_active_db_path()
    if target == ":memory:":
        return
    try:
        conn = _create_raw_connection(target)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()
        logger.info(f"Database WAL checkpointed successfully for {target}")
    except Exception as e:
        logger.warning(f"Database checkpoint warning: {e}")
