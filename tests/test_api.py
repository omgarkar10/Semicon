from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

import backend.api as api_module
from backend.inference import InferenceService


class FakeRestorer(torch.nn.Module):
    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_module.app)


@pytest.fixture
def client_with_model(client: TestClient):
    service = InferenceService(model=FakeRestorer(scale=2), require_checkpoint=False)
    old_service = api_module._service
    old_error = api_module._service_error
    api_module._service = service
    api_module._service_error = None
    yield client, service
    api_module._service = old_service
    api_module._service_error = old_error


def test_health_endpoint(client: TestClient):
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["server"] == "online"
    assert "model_loaded" in payload
    assert "status" in payload


def test_model_endpoint_without_checkpoint(client: TestClient, monkeypatch):
    def _mark_unavailable() -> None:
        api_module._service = None
        api_module._service_error = "Trained checkpoint not found."

    monkeypatch.setattr(api_module, "_try_load_service", _mark_unavailable)
    api_module._service = None
    api_module._service_error = "Trained checkpoint not found."

    response = client.get("/api/model")

    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].lower()


def test_model_endpoint_with_fake_service(client_with_model):
    client, _service = client_with_model

    response = client.get("/api/model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "rrdb_restorer"
    assert payload["scale"] == 2
    assert payload["tta_count"] == 8
    assert payload["checkpoint_loaded"] is False


def test_restore_rejects_invalid_extension(client_with_model):
    client, _service = client_with_model

    response = client.post(
        "/api/restore",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .npy files are supported."


def test_restore_rejects_invalid_npy(client_with_model):
    client, _service = client_with_model

    response = client.post(
        "/api/restore",
        files={"file": ("sample.npy", b"not-a-numpy-file", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Invalid NumPy file" in response.json()["detail"]


def test_restore_rejects_invalid_shape(client_with_model):
    client, _service = client_with_model
    arr = np.zeros((3, 16, 16), dtype=np.float32)

    response = client.post(
        "/api/restore",
        files={"file": ("sample.npy", _npy_bytes(arr), "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported shape" in response.json()["detail"]


def test_restore_valid_npy_request(client_with_model):
    client, _service = client_with_model
    arr = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)

    response = client.post(
        "/api/restore",
        files={"file": ("sample.npy", _npy_bytes(arr), "application/octet-stream")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["restored_npy_filename"] == "sample.npy"
    assert payload["input_shape"] == [16, 16]
    assert payload["output_shape"] == [32, 32]
    assert payload["tta_count"] == 8
    assert payload["inference_time"] >= 0.0
    assert isinstance(payload["device"], str)
    assert payload["image_png"]

    restored_bytes = base64.b64decode(payload["restored_npy"])
    restored = np.load(io.BytesIO(restored_bytes), allow_pickle=False)
    assert restored.shape == (32, 32)
    assert restored.dtype == np.float32
    assert np.isfinite(restored).all()
    assert restored.min() >= 0.0
    assert restored.max() <= 1.0


def test_restore_valid_hw1_request(client_with_model):
    client, _service = client_with_model
    arr = np.ones((12, 12, 1), dtype=np.float32)

    response = client.post(
        "/api/restore",
        files={"file": ("channel.npy", _npy_bytes(arr), "application/octet-stream")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["input_shape"] == [12, 12]
    assert payload["output_shape"] == [24, 24]
