"""Lightweight config loader for shared settings (e.g., class labels).

Supports simple YAML or JSON files. Primary use: provide a single source
of truth for label names that is shared across annotation, packing, and
training/inference utilities.

Example dataset.yaml
  labels:
    - neutral
    - smile
    - frown

Usage
  from expression_recognition.config.schemas import load_config, get_labels
  cfg = load_config("configs/dataset.yaml")
  labels = get_labels(cfg)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML optional; JSON still supported


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    # Try YAML if available and extension suggests YAML/YML
    if yaml is not None and p.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    # Fallback to JSON
    try:
        return json.loads(text)
    except Exception as e:
        # If YAML is installed, attempt YAML regardless of extension
        if yaml is not None:
            return yaml.safe_load(text) or {}
        raise e


def get_labels(cfg: Dict[str, Any]) -> List[str]:
    """Extract label list from a config dictionary.

    Recognized locations:
    - cfg["labels"]
    - cfg["data"]["labels"]
    - cfg["dataset"]["labels"]
    Returns [] if none found.
    """
    # direct
    if isinstance(cfg.get("labels"), list):
        return [str(x) for x in cfg["labels"]]
    # nested
    for k in ("data", "dataset"):
        sub = cfg.get(k)
        if isinstance(sub, dict) and isinstance(sub.get("labels"), list):
            return [str(x) for x in sub["labels"]]
    return []

