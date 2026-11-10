"""
Unit tests for FastAPI routers (/api/health, /api/settings, /api/shutdown) and middlewares.
"""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["backend_ready"] is True
    assert "uptime_seconds" in data
    assert "timestamp" in data


def test_settings_gpu_endpoint(client):
    response = client.get("/api/settings/gpu")
    assert response.status_code == 200
    data = response.json()
    assert "cuda_available" in data
    assert "gpu_name" in data
    assert "vram_total_mb" in data
    assert "simulation_mode" in data


def test_general_settings_endpoint(client):
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Voice Sampler API"
    assert data["version"] == "0.1.0"
    assert "gpu" in data
    assert "data_dir" in data
    assert "db_path" in data


def test_cors_preflight(client):
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_shutdown_endpoint(client):
    mock_server = MagicMock()
    app.state.server = mock_server

    response = client.post("/api/shutdown")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "shutting_down"
