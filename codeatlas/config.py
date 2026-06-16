"""
codeatlas/config.py
-------------------
Configuration loader for the CodeAtlas extension layer.

Design rules:
- Reads conf/codeatlas.yaml relative to the RAGFlow repo root.
- Falls back to sensible defaults if the file is absent (safe for testing).
- Exposes a single module-level ``_cfg`` dict so every submodule can import
  from here without re-parsing YAML on every call.
- Zero modifications to RAGFlow code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locate conf/codeatlas.yaml
# ---------------------------------------------------------------------------
# We resolve upward from this file: codeatlas/config.py -> codeatlas/ -> repo root
_CODEATLAS_PKG = Path(__file__).parent          # …/codeatlas/
_REPO_ROOT = _CODEATLAS_PKG.parent              # …/<repo root>/

# Allow override via environment variable for testing or Docker deployments.
_CONFIG_PATH = Path(
    os.environ.get("CODEATLAS_CONFIG", str(_REPO_ROOT / "conf" / "codeatlas.yaml"))
)

# ---------------------------------------------------------------------------
# Internal default values — used when the file is missing or a key is absent
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "log_level": "INFO",
    "features": {},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict where *override* values win, recursing into nested dicts."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    """Parse a YAML file, returning an empty dict on any error."""
    try:
        import yaml  # PyYAML is a RAGFlow dependency — available in venv
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("codeatlas", {})
    except FileNotFoundError:
        logging.getLogger("codeatlas.config").warning(
            "Config file not found at %s — using defaults.", path
        )
        return {}
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("codeatlas.config").error(
            "Failed to parse %s: %s — using defaults.", path, exc
        )
        return {}


# ---------------------------------------------------------------------------
# Module-level config dict (lazy-loaded on first access)
# ---------------------------------------------------------------------------
_cfg: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Return the parsed CodeAtlas configuration dict.

    The first call parses ``conf/codeatlas.yaml`` and caches the result.
    Subsequent calls return the cached value.
    """
    global _cfg  # noqa: PLW0603
    if _cfg is None:
        raw = _load_yaml(_CONFIG_PATH)
        _cfg = _deep_merge(_DEFAULTS, raw)
    return _cfg


def get(key: str, default: Any = None) -> Any:
    """Convenience accessor: ``config.get("log_level")``."""
    return get_config().get(key, default)
