"""
backend/app/db/repository.py
Database schema initialization and CRUD repository methods.
Part of Voice Sampler Milestone 1 (M1).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

from backend.app.db.connection import get_db_connection


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema DDL Definition
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    preprocessed_path TEXT,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    sample_rate INTEGER NOT NULL DEFAULT 44100,
    channels INTEGER NOT NULL DEFAULT 1,
    transcript TEXT NOT NULL DEFAULT '',
    transcript_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_created_at ON samples(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(transcript_status);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    engine TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    index_path TEXT,
    vocab_path TEXT,
    config_path TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    sample_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_models_engine ON models(engine);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_models_name ON models(name);

CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    engine TEXT NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    config_json TEXT NOT NULL DEFAULT '{}',
    current_epoch INTEGER NOT NULL DEFAULT 0,
    total_epochs INTEGER NOT NULL DEFAULT 0,
    current_step INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    current_loss REAL,
    progress_percent REAL NOT NULL DEFAULT 0.0,
    status_message TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_created_at ON training_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    model_id TEXT NOT NULL,
    input_text TEXT,
    input_audio_path TEXT,
    reference_sample_id TEXT,
    output_audio_path TEXT NOT NULL,
    output_filename TEXT NOT NULL,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    sample_rate INTEGER NOT NULL DEFAULT 24000,
    channels INTEGER NOT NULL DEFAULT 1,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (reference_sample_id) REFERENCES samples(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_generations_type ON generations(type);
CREATE INDEX IF NOT EXISTS idx_generations_model_id ON generations(model_id);
CREATE INDEX IF NOT EXISTS idx_generations_created_at ON generations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_generations_filename ON generations(output_filename);
"""


def init_db(db_path: Optional[Union[Path, str]] = None) -> None:
    """Initialize database tables and indexes."""
    with get_db_connection(db_path) as conn:
        conn.executescript(SCHEMA_DDL)


# ---------------------------------------------------------------------------
# Samples Repository CRUD
# ---------------------------------------------------------------------------

def create_sample(
    name: str,
    filename: str,
    file_path: str,
    preprocessed_path: Optional[str] = None,
    duration_seconds: float = 0.0,
    sample_rate: int = 44100,
    channels: int = 1,
    transcript: str = "",
    transcript_status: str = "pending",
    sample_id: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Insert a new voice sample record and return it."""
    sid = sample_id or str(uuid.uuid4())
    now = _utc_now_iso()

    sql = """
    INSERT INTO samples (
        id, name, filename, file_path, preprocessed_path,
        duration_seconds, sample_rate, channels, transcript,
        transcript_status, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        sid, name, filename, file_path, preprocessed_path,
        duration_seconds, sample_rate, channels, transcript,
        transcript_status, now, now
    )

    with get_db_connection(db_path) as conn:
        conn.execute(sql, params)

    return get_sample(sid, db_path=db_path)  # type: ignore


def get_sample(
    sample_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve sample by unique ID."""
    sql = "SELECT * FROM samples WHERE id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (sample_id,)).fetchone()
        return dict(row) if row else None


def list_samples(
    status: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """List all samples ordered by creation date descending, optionally filtered by status."""
    if status:
        sql = "SELECT * FROM samples WHERE transcript_status = ? ORDER BY created_at DESC"
        params: tuple = (status,)
    else:
        sql = "SELECT * FROM samples ORDER BY created_at DESC"
        params = ()

    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def update_sample(
    sample_id: str,
    db_path: Optional[Union[Path, str]] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Update arbitrary fields of a sample record and set updated_at."""
    if not kwargs:
        return get_sample(sample_id, db_path=db_path)

    allowed = {
        "name", "filename", "file_path", "preprocessed_path",
        "duration_seconds", "sample_rate", "channels",
        "transcript", "transcript_status"
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return get_sample(sample_id, db_path=db_path)

    updates["updated_at"] = _utc_now_iso()
    set_clause = ", ".join(f"{col} = ?" for col in updates.keys())
    values = list(updates.values())
    values.append(sample_id)

    sql = f"UPDATE samples SET {set_clause} WHERE id = ?"
    with get_db_connection(db_path) as conn:
        conn.execute(sql, tuple(values))

    return get_sample(sample_id, db_path=db_path)


def update_sample_transcript(
    sample_id: str,
    transcript: str,
    status: str = "manual",
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Convenience helper to update transcript text and status."""
    return update_sample(
        sample_id,
        transcript=transcript,
        transcript_status=status,
        db_path=db_path,
    )


def update_sample_conditional(
    sample_id: str,
    expected_status: Union[str, Sequence[str]],
    db_path: Optional[Union[Path, str]] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Atomically update sample fields only if current transcript_status equals expected_status.

    If expected_status is a string, evaluates `transcript_status = ?`.
    If expected_status is a list/tuple/set, evaluates `transcript_status IN (?, ...)`.
    Executes atomically in SQLite.
    If cur.rowcount == 0 (record missing or status mismatch), returns None (transition rejected).
    If cur.rowcount > 0, returns the updated sample record.
    """
    if not kwargs:
        return None

    allowed = {
        "name", "filename", "file_path", "preprocessed_path",
        "duration_seconds", "sample_rate", "channels",
        "transcript", "transcript_status"
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return None

    updates["updated_at"] = _utc_now_iso()
    set_clause = ", ".join(f"{col} = ?" for col in updates.keys())
    values = list(updates.values())
    values.append(sample_id)

    if isinstance(expected_status, (list, tuple, set)):
        status_list = list(expected_status)
        placeholders = ", ".join("?" for _ in status_list)
        where_clause = f"WHERE id = ? AND transcript_status IN ({placeholders})"
        values.extend(status_list)
    else:
        where_clause = "WHERE id = ? AND transcript_status = ?"
        values.append(expected_status)

    sql = f"UPDATE samples SET {set_clause} {where_clause}"
    with get_db_connection(db_path) as conn:
        cur = conn.execute(sql, tuple(values))
        if cur.rowcount == 0:
            return None

    return get_sample(sample_id, db_path=db_path)


def update_sample_status_conditional(
    sample_id: str,
    new_status: str,
    expected_status: Union[str, Sequence[str]],
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Atomically update status only if current status matches expected_status."""
    res = update_sample_conditional(
        sample_id=sample_id,
        expected_status=expected_status,
        db_path=db_path,
        transcript_status=new_status,
    )
    return res is not None


def delete_sample(
    sample_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Delete a sample by ID. Returns True if a record was removed."""
    sql = "DELETE FROM samples WHERE id = ?"
    with get_db_connection(db_path) as conn:
        cur = conn.execute(sql, (sample_id,))
        return cur.rowcount > 0


def delete_all_samples(
    only_missing_files: bool = False,
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """
    Delete all samples, or only samples whose audio files no longer exist on disk.
    Returns the count of deleted sample records.
    """
    with get_db_connection(db_path) as conn:
        if not only_missing_files:
            cur = conn.execute("DELETE FROM samples")
            return cur.rowcount
        else:
            rows = conn.execute("SELECT id, file_path, preprocessed_path FROM samples").fetchall()
            to_delete = []
            for r in rows:
                fp = r["file_path"]
                pp = r["preprocessed_path"]
                fp_exists = bool(fp and os.path.exists(fp))
                pp_exists = bool(pp and os.path.exists(pp))
                if not fp_exists and not pp_exists:
                    to_delete.append((r["id"],))
            if to_delete:
                conn.executemany("DELETE FROM samples WHERE id = ?", to_delete)
            return len(to_delete)


def count_samples(db_path: Optional[Union[Path, str]] = None) -> int:
    """Return total count of samples."""
    with get_db_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Models Repository CRUD
# ---------------------------------------------------------------------------

def create_model(
    name: str,
    engine: str,
    checkpoint_path: str,
    index_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    config_path: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    sample_count: int = 0,
    model_id: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Register a trained model checkpoint."""
    mid = model_id or str(uuid.uuid4())
    now = _utc_now_iso()
    param_json = json.dumps(parameters or {})

    sql = """
    INSERT INTO models (
        id, name, engine, checkpoint_path, index_path, vocab_path,
        config_path, parameters_json, sample_count, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        mid, name, engine, checkpoint_path, index_path, vocab_path,
        config_path, param_json, sample_count, now, now
    )

    with get_db_connection(db_path) as conn:
        conn.execute(sql, params)

    return get_model(mid, db_path=db_path)  # type: ignore


def get_model(
    model_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve model by unique ID."""
    sql = "SELECT * FROM models WHERE id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (model_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["parameters"] = json.loads(res.get("parameters_json") or "{}")
        except Exception:
            res["parameters"] = {}
        return res


def get_model_by_name(
    name: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve model by its unique name."""
    sql = "SELECT * FROM models WHERE name = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (name,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["parameters"] = json.loads(res.get("parameters_json") or "{}")
        except Exception:
            res["parameters"] = {}
        return res


def list_models(
    engine: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """List all models ordered by creation date descending, optionally filtered by engine."""
    if engine:
        sql = "SELECT * FROM models WHERE engine = ? ORDER BY created_at DESC"
        params: tuple = (engine,)
    else:
        sql = "SELECT * FROM models ORDER BY created_at DESC"
        params = ()

    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["parameters"] = json.loads(item.get("parameters_json") or "{}")
            except Exception:
                item["parameters"] = {}
            result.append(item)
        return result


def delete_model(
    model_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Delete a model registry record."""
    sql = "DELETE FROM models WHERE id = ?"
    with get_db_connection(db_path) as conn:
        cur = conn.execute(sql, (model_id,))
        return cur.rowcount > 0


def count_models(db_path: Optional[Union[Path, str]] = None) -> int:
    """Return total count of registered models."""
    with get_db_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM models").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Training Runs Repository CRUD
# ---------------------------------------------------------------------------

def create_training_run(
    engine: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    total_epochs: int = 0,
    total_steps: int = 0,
    run_id: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Create a new training run record."""
    rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
    now = _utc_now_iso()
    cfg_json = json.dumps(config or {})

    sql = """
    INSERT INTO training_runs (
        id, engine, model_name, status, config_json,
        current_epoch, total_epochs, current_step, total_steps,
        current_loss, progress_percent, status_message, error_message,
        created_at, updated_at, completed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        rid, engine, model_name, "queued", cfg_json,
        0, total_epochs, 0, total_steps,
        None, 0.0, "Queued for execution", None,
        now, now, None
    )

    with get_db_connection(db_path) as conn:
        conn.execute(sql, params)

    return get_training_run(rid, db_path=db_path)  # type: ignore


def get_training_run(
    run_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve training run by ID."""
    sql = "SELECT * FROM training_runs WHERE id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (run_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["config"] = json.loads(res.get("config_json") or "{}")
        except Exception:
            res["config"] = {}
        return res


def get_active_training_run(
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Return currently running or queued training run, if any."""
    sql = "SELECT * FROM training_runs WHERE status IN ('queued', 'running') ORDER BY created_at DESC LIMIT 1"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["config"] = json.loads(res.get("config_json") or "{}")
        except Exception:
            res["config"] = {}
        return res


def update_training_run_progress(
    run_id: str,
    current_epoch: int,
    current_step: int,
    current_loss: Optional[float] = None,
    progress_percent: float = 0.0,
    status_message: Optional[str] = None,
    status: str = "running",
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Update training telemetry state."""
    now = _utc_now_iso()
    sql = """
    UPDATE training_runs SET
        current_epoch = ?,
        current_step = ?,
        current_loss = ?,
        progress_percent = ?,
        status_message = ?,
        status = ?,
        updated_at = ?
    WHERE id = ?
    """
    params = (
        current_epoch, current_step, current_loss,
        progress_percent, status_message, status, now, run_id
    )
    with get_db_connection(db_path) as conn:
        conn.execute(sql, params)

    return get_training_run(run_id, db_path=db_path)


def complete_training_run(
    run_id: str,
    status: str = "completed",
    error_message: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Conclude training run as completed, failed, or cancelled."""
    now = _utc_now_iso()
    sql = """
    UPDATE training_runs SET
        status = ?,
        error_message = ?,
        updated_at = ?,
        completed_at = ?
    WHERE id = ?
    """
    params = (status, error_message, now, now, run_id)
    with get_db_connection(db_path) as conn:
        conn.execute(sql, params)

    return get_training_run(run_id, db_path=db_path)


def list_training_runs(
    limit: int = 20,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """List recent training runs."""
    sql = "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?"
    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
        res = []
        for r in rows:
            item = dict(r)
            try:
                item["config"] = json.loads(item.get("config_json") or "{}")
            except Exception:
                item["config"] = {}
            res.append(item)
        return res


# ---------------------------------------------------------------------------
# Generations Repository CRUD
# ---------------------------------------------------------------------------

def create_generation(
    type: str,
    model_id: str,
    output_audio_path: str,
    output_filename: str,
    input_text: Optional[str] = None,
    input_audio_path: Optional[str] = None,
    reference_sample_id: Optional[str] = None,
    duration_seconds: float = 0.0,
    sample_rate: int = 24000,
    channels: int = 1,
    parameters: Optional[Dict[str, Any]] = None,
    gen_id: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Record an audio generation (TTS or STS)."""
    gid = gen_id or str(uuid.uuid4())
    now = _utc_now_iso()
    param_json = json.dumps(parameters or {})

    sql = """
    INSERT INTO generations (
        id, type, model_id, input_text, input_audio_path,
        reference_sample_id, output_audio_path, output_filename,
        duration_seconds, sample_rate, channels, parameters_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        gid, type, model_id, input_text, input_audio_path,
        reference_sample_id, output_audio_path, output_filename,
        duration_seconds, sample_rate, channels, param_json, now
    )

    with get_db_connection(db_path) as conn:
        conn.execute(sql, params)

    return get_generation(gid, db_path=db_path)  # type: ignore


def get_generation(
    gen_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve generation by ID."""
    sql = "SELECT * FROM generations WHERE id = ?"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (gen_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["parameters"] = json.loads(res.get("parameters_json") or "{}")
        except Exception:
            res["parameters"] = {}
        return res


def get_generation_by_filename(
    filename: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve generation record by output filename."""
    sql = "SELECT * FROM generations WHERE output_filename = ? LIMIT 1"
    with get_db_connection(db_path) as conn:
        row = conn.execute(sql, (filename,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["parameters"] = json.loads(res.get("parameters_json") or "{}")
        except Exception:
            res["parameters"] = {}
        return res


def list_generations(
    gen_type: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """List generation records."""
    if gen_type:
        sql = "SELECT * FROM generations WHERE type = ? ORDER BY created_at DESC LIMIT ?"
        params: tuple = (gen_type, limit)
    else:
        sql = "SELECT * FROM generations ORDER BY created_at DESC LIMIT ?"
        params = (limit,)

    with get_db_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["parameters"] = json.loads(item.get("parameters_json") or "{}")
            except Exception:
                item["parameters"] = {}
            result.append(item)
        return result


def delete_generation(
    gen_id: str,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Delete a generation record."""
    sql = "DELETE FROM generations WHERE id = ?"
    with get_db_connection(db_path) as conn:
        cur = conn.execute(sql, (gen_id,))
        return cur.rowcount > 0


def count_generations(db_path: Optional[Union[Path, str]] = None) -> int:
    """Return total count of generations."""
    with get_db_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM generations").fetchone()
        return row[0] if row else 0
