"""
backend/app/routers/models.py
REST API endpoints for Model Registry queries, checkpoint inspection, and model deletion.
Part of Voice Sampler Milestone 3 (M3).
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.config import get_settings
from backend.app.db import repository

logger = logging.getLogger("voice_sampler.routers.models")
router = APIRouter()


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class ModelResponse(BaseModel):
    """Model checkpoint metadata."""
    id: str
    name: str
    engine: str
    checkpoint_path: str
    index_path: Optional[str] = None
    vocab_path: Optional[str] = None
    config_path: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    sample_count: int = 0
    created_at: str
    updated_at: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of registered models."""
    models: List[Dict[str, Any]] = Field(default_factory=list)


class DeleteModelResponse(BaseModel):
    """Response confirming model deletion."""
    status: str = "deleted"
    name: str
    id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_checkpoint_path(path_str: Optional[str]) -> Optional[str]:
    """Ensures checkpoint path uses normalized relative format 'data/models/...'."""
    if not path_str:
        return path_str
    # Replace backslashes with forward slashes for cross-platform API contract consistency
    norm = path_str.replace("\\", "/")
    if "data/models/" in norm:
        idx = norm.find("data/models/")
        return norm[idx:]
    return norm


def _format_model_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Formats a database model row for API response."""
    formatted = dict(record)
    if "checkpoint_path" in formatted:
        formatted["checkpoint_path"] = _normalize_checkpoint_path(formatted["checkpoint_path"])
    if "index_path" in formatted and formatted["index_path"]:
        formatted["index_path"] = _normalize_checkpoint_path(formatted["index_path"])
    if "vocab_path" in formatted and formatted["vocab_path"]:
        formatted["vocab_path"] = _normalize_checkpoint_path(formatted["vocab_path"])
    if "config_path" in formatted and formatted["config_path"]:
        formatted["config_path"] = _normalize_checkpoint_path(formatted["config_path"])
    return formatted


# ---------------------------------------------------------------------------
# Router Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all registered models",
)
def list_models(engine: Optional[str] = Query(None, description="Filter by engine ('f5-tts' or 'rvc')")) -> Dict[str, Any]:
    """
    Returns an array of all registered models in the SQLite database,
    optionally filtered by engine ('f5-tts' or 'rvc').
    Returns {"models": []} when the registry is empty.
    """
    records = repository.list_models(engine=engine)
    formatted_models = [_format_model_record(r) for r in records]
    return {"models": formatted_models}


@router.get(
    "/{name}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get model details by name",
)
def get_model(name: str) -> Dict[str, Any]:
    """
    Retrieves detailed metadata for a registered model by name.
    Returns 404 if the model does not exist.
    """
    record = repository.get_model_by_name(name)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found",
        )
    return _format_model_record(record)


@router.delete(
    "/{name}",
    response_model=DeleteModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete model and its checkpoint artifacts",
)
def delete_model(name: str) -> Dict[str, Any]:
    """
    Deletes the model record from the SQLite database and safely removes
    all associated on-disk checkpoint files, indexes, and directories.
    Returns 404 if the model does not exist.
    """
    record = repository.get_model_by_name(name)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{name}' not found",
        )

    model_id = str(record["id"])
    settings = get_settings()

    # 1. Unlink specific artifact files recorded in the database
    for path_key in ("checkpoint_path", "index_path", "vocab_path", "config_path"):
        fpath_str = record.get(path_key)
        if fpath_str:
            p = Path(fpath_str)
            if not p.is_absolute():
                p = (settings.project_root / p).resolve()
            if p.is_file():
                try:
                    p.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Could not unlink model artifact file {p}: {e}")

    # 2. Remove dedicated model directory (data/models/{name})
    dedicated_dir = (settings.models_dir / name).resolve()
    if dedicated_dir.is_dir():
        try:
            shutil.rmtree(str(dedicated_dir), ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not remove model directory {dedicated_dir}: {e}")

    # 3. Clean up only flat convenience files belonging strictly to this model
    target_patterns = [
        f"{name}.*",
        f"{name}_f5tts.*",
        f"{name}_config.json",
        f"{name}_vocab.txt",
        f"{name}_index.index",
    ]
    try:
        for pat in target_patterns:
            for residual in settings.models_dir.glob(pat):
                if residual.is_file():
                    residual.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Error cleaning residual files for model {name}: {e}")

    # 4. Remove database record
    repository.delete_model(model_id)
    logger.info(f"Model '{name}' (id: {model_id}) deleted successfully")

    return {
        "status": "deleted",
        "name": name,
        "id": model_id,
    }
