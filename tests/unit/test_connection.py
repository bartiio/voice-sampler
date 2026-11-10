"""
Unit tests for SQLite connection manager, PRAGMA setup, and transaction handling.
"""

from pathlib import Path
import sqlite3
import pytest

from backend.app.db.connection import (
    checkpoint_db,
    get_active_db_path,
    get_db_connection,
    init_storage_directories,
    set_db_path,
)


def test_init_storage_directories(tmp_path):
    dirs = init_storage_directories(base_data_dir=tmp_path)
    assert dirs["data_root"] == tmp_path
    assert dirs["db"].exists()
    assert dirs["samples_raw"].exists()
    assert dirs["samples_preprocessed"].exists()
    assert dirs["models_f5_tts"].exists()
    assert dirs["models_rvc"].exists()
    assert dirs["generated_tts"].exists()
    assert dirs["generated_sts"].exists()
    assert dirs["cache"].exists()


def test_get_db_connection_pragmas(tmp_path):
    test_db = tmp_path / "test.db"
    with get_db_connection(db_path=test_db) as conn:
        # Check foreign keys
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1

        # Check WAL journal mode
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal.lower() == "wal"

        # Check busy timeout
        timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert timeout == 30000

        # Check row factory
        row = conn.execute("SELECT 1 AS col_a, 'hello' AS col_b").fetchone()
        assert row["col_a"] == 1
        assert row["col_b"] == "hello"


def test_get_db_connection_rollback(tmp_path):
    test_db = tmp_path / "test_rollback.db"
    with get_db_connection(db_path=test_db) as conn:
        conn.execute("CREATE TABLE items (id TEXT PRIMARY KEY, val TEXT);")

    # Raise an error inside context to trigger rollback
    with pytest.raises(RuntimeError):
        with get_db_connection(db_path=test_db) as conn:
            conn.execute("INSERT INTO items (id, val) VALUES ('1', 'saved');")
            raise RuntimeError("Forced abort")

    # Verify that the record was rolled back
    with get_db_connection(db_path=test_db) as conn:
        row = conn.execute("SELECT COUNT(*) FROM items WHERE id = '1'").fetchone()
        assert row[0] == 0


def test_checkpoint_db(tmp_path):
    test_db = tmp_path / "test_checkpoint.db"
    with get_db_connection(db_path=test_db) as conn:
        conn.execute("CREATE TABLE checkpoint_test (id INT);")
        conn.execute("INSERT INTO checkpoint_test VALUES (1);")

    # Checkpoint should succeed without error
    checkpoint_db(db_path=test_db)
    assert test_db.exists()
