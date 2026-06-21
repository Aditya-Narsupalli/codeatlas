# agent/component/codeatlas/__init__.py
#
# CodeAtlas — agent component namespace
# ---------------------------------------------------------------------------
# This sub-package holds all CodeAtlas agent components.
#
# Registration
# ------------
# RAGFlow's agent/component/__init__.py auto-discovers components by
# scanning *.py files DIRECTLY inside agent/component/ (it does not walk
# subdirectories — see _import_submodules() in that file). A nested package
# like agent/component/codeatlas/ is therefore NOT auto-discovered.
#
# To make CodeAtlas components visible to the existing component_class()
# lookup and the agent canvas component registry, re-export them from this
# file so a future explicit import in agent/component/__init__.py picks
# them up with a single line, e.g.:
#
#   from agent.component.codeatlas import ReadingOrderComponent
#
# That activation line is intentionally NOT added in Phase 11 (the roadmap
# defers FACTORY/registration wiring to a later activation step, matching
# the pattern used in Phases 2, 3, and 9).
#
# Phase 11 added: ReadingOrderComponent
# Phase 12 adds: ArchReportComponent
# ---------------------------------------------------------------------------

from agent.component.codeatlas.reading_order import (
    ReadingOrderComponent,
    ReadingOrderParam,
    CandidateDocument,
    ReadingOrderItem,
)
from agent.component.codeatlas.report import (
    ArchReportComponent,
    ArchReportParam,
    GraphData,
    GraphNodeData,
    GraphEdgeData,
    RetrievedChunk,
)

__all__ = [
    "ReadingOrderComponent",
    "ReadingOrderParam",
    "CandidateDocument",
    "ReadingOrderItem",
    "ArchReportComponent",
    "ArchReportParam",
    "GraphData",
    "GraphNodeData",
    "GraphEdgeData",
    "RetrievedChunk",
]
