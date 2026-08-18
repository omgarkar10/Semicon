"""FastAPI application for SEMICON AI image restoration."""

from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import imageio.v3 as iio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from backend.inference import (
    CheckpointNotFoundError,
    InferenceService,
    InvalidInputArrayError,
    get_inference_service,
    validate_input_array,
)
from semicon.config import ROOT

logger = logging.getLogger(__name__)

FRONTEND_DIR = ROOT / "frontend"

_service: InferenceService | None = None
_service_error: str | None = None


def _try_load_service() -> None:
    """Load the inference service once; record errors without crashing the API."""
    global _service, _service_error
    if _service is not None:
        return
    try:
        _service = get_inference_service()
        _service_error = None
        logger.info("Inference service loaded on %s", _service.device)
    except CheckpointNotFoundError as exc:
        _service = None
        _service_error = str(exc)
        logger.error("Inference service unavailable: %s", exc)


def _require_service() -> InferenceService:
    _try_load_service()
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail=_service_error or "Inference service is unavailable.",
        )
    return _service


def _safe_filename(name: str | None) -> str:
    raw = Path(name or "restored.npy").name
    if not raw.lower().endswith(".npy"):
        raw = f"{Path(raw).stem}.npy"
    return raw


def _load_uploaded_npy(content: bytes) -> np.ndarray:
    try:
        arr = np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid NumPy file: {exc}") from exc

    if not isinstance(arr, np.ndarray):
        raise HTTPException(status_code=400, detail="Uploaded file did not contain a NumPy array.")

    try:
        validate_input_array(arr)
    except InvalidInputArrayError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return arr


def _array_to_png_base64(arr: np.ndarray) -> str:
    clipped = np.clip(arr, 0.0, 1.0)
    uint8 = (clipped * 255.0).round().astype(np.uint8)
    png_bytes = iio.imwrite("<bytes>", uint8, extension=".png")
    return base64.b64encode(png_bytes).decode("ascii")


def _array_to_npy_base64(arr: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, arr.astype(np.float32, copy=False))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_service()
    yield


app = FastAPI(
    title="SEMICON AI",
    description="Semiconductor image restoration API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    _try_load_service()
    model_loaded = _service is not None
    return {
        "status": "online" if model_loaded else "degraded",
        "server": "online",
        "model_loaded": model_loaded,
        "model_error": _service_error,
        "device": str(_service.device) if _service is not None else None,
    }


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    service = _require_service()
    return service.model_info


@app.post("/api/restore")
async def restore(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    if not filename.lower().endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    arr = _load_uploaded_npy(content)
    service = _require_service()

    try:
        restored_arr, metadata = service.restore_image(arr)
    except InvalidInputArrayError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    restored_name = _safe_filename(filename)
    return {
        "image_png": _array_to_png_base64(restored_arr),
        "restored_npy": _array_to_npy_base64(restored_arr),
        "restored_npy_filename": restored_name,
        "input_shape": list(metadata["input_shape"]),
        "output_shape": list(metadata["output_shape"]),
        "inference_time": metadata["inference_time"],
        "device": metadata["device"],
        "tta_count": metadata["tta_count"],
    }


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)
