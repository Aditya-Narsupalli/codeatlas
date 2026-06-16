"""
codeatlas
=========
CodeAtlas — extension layer over RAGFlow for code-aware knowledge retrieval.

This package is the single import point for all CodeAtlas functionality.
Every feature is off by default and guarded by a feature flag.

Phases implemented so far:
  Phase 1 — Module scaffold + feature flags (this package)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "CodeAtlas"

# Re-export the most-used symbols so callers can do:
#   from codeatlas import flags, config, get_logger
from codeatlas import config, flags
from codeatlas.logger import get_logger

__all__ = [
    "__version__",
    "config",
    "flags",
    "get_logger",
]
