"""
codeatlas/logger.py
-------------------
Logging factory for the CodeAtlas extension layer.

Design rules:
- All CodeAtlas loggers live under the ``codeatlas`` namespace so they can
  be filtered independently from RAGFlow's own loggers.
- Log level is read from ``conf/codeatlas.yaml`` → ``codeatlas.log_level``.
- RAGFlow's existing logging configuration is never modified.
- Zero RAGFlow modifications.
"""

from __future__ import annotations

import logging
from typing import Final

from codeatlas.config import get

# Mapping from YAML string to logging level constants.
_LEVEL_MAP: Final[dict[str, int]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_DEFAULT_LEVEL: Final[int] = logging.INFO


def _resolve_level() -> int:
    """Read log_level from config and convert to a logging constant."""
    raw: str = str(get("log_level", "INFO")).upper()
    return _LEVEL_MAP.get(raw, _DEFAULT_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Return a ``logging.Logger`` scoped under ``codeatlas.<name>``.

    Parameters
    ----------
    name:
        Sub-namespace for the logger, e.g. ``"flags"`` → ``codeatlas.flags``.
        Pass ``__name__`` from the calling module for automatic namespacing.

    Returns
    -------
    logging.Logger
        A logger that respects the level set in ``conf/codeatlas.yaml``.
        If no handler is configured by RAGFlow's logging setup, a
        ``NullHandler`` is attached to avoid "No handlers found" warnings.
    """
    # Ensure the root codeatlas logger has at least a NullHandler so library
    # users who haven't configured logging don't see spurious warnings.
    root_logger = logging.getLogger("codeatlas")
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())

    # Sub-logger inherits the root codeatlas level unless overridden.
    root_logger.setLevel(_resolve_level())

    # Derive the full dotted name: callers pass __name__ which is already
    # "codeatlas.something", or a plain name like "config".
    full_name = name if name.startswith("codeatlas.") else f"codeatlas.{name}"
    return logging.getLogger(full_name)
