"""
codeatlas/flags.py
------------------
Feature flag registry for the CodeAtlas extension layer.

Design rules:
- ``is_enabled(name)`` is the only public API needed by all subsequent phases.
- Default return value is **always** ``False`` for any unknown flag.
- Flags are sourced from ``conf/codeatlas.yaml`` → ``codeatlas.features``.
- Zero RAGFlow modifications.
"""

from __future__ import annotations

from typing import Any

from codeatlas.config import get_config


def is_enabled(name: str) -> bool:
    """Return ``True`` if the named feature flag is explicitly enabled.

    Parameters
    ----------
    name:
        Flag name as defined under ``codeatlas.features`` in
        ``conf/codeatlas.yaml``.  Examples: ``"arch_explorer"``,
        ``"code_search"``.

    Returns
    -------
    bool
        ``True`` only when the flag is present **and** set to ``true`` in
        the config file.  Missing or ``null`` flags return ``False``.
    """
    features: dict[str, Any] = get_config().get("features", {})
    return bool(features.get(name, False))


def all_flags() -> dict[str, bool]:
    """Return a snapshot of every flag and its current state.

    Useful for health-check endpoints or startup logging.
    """
    features: dict[str, Any] = get_config().get("features", {})
    return {k: bool(v) for k, v in features.items()}
