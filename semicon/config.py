from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else ROOT / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_data_path(cfg: dict[str, Any], key: str) -> Path:
    rel = Path(cfg["data"][key])
    if rel.is_absolute():
        return rel
    return Path(cfg["_root"]) / rel
