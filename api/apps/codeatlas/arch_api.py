# api/apps/codeatlas/arch_api.py
#
# CodeAtlas — Phase 9: Architecture Explorer API
# ---------------------------------------------------------------------------
# Quart route handlers for the two graph query endpoints.
#
# Endpoints
# ---------
#   GET /api/codeatlas/graph
#       Returns the full architecture graph for the kb_id supplied in the
#       query string.
#       Response: { "nodes": [...], "edges": [...] }
#       Auth: login_required (returns 401 if unauthenticated)
#
#   GET /api/codeatlas/graph/node/<node_id>
#       Returns a single node and its directly connected edges.
#       Response: { "node": {...}, "edges": [...] }
#       Returns 404 (never 500) if node_id is unknown.
#       Auth: login_required
#
# Auth
# ----
# Uses the existing @login_required decorator from api/apps/__init__.py,
# which covers JWT, API key, and session-cookie auth and raises
# QuartAuthUnauthorized on failure (handled globally as 401).
#
# DB access
# ---------
# All DB reads are lazy (inside the route handler).  The module is
# importable without a live database — ORM models are imported at
# call time, not at module level.
#
# Scope (Phase 9 only)
# --------------------
# - Read-only GET endpoints
# - No graph mutation
# - No UI / frontend
# - No pagination (full graph returned; pagination is Phase N+)
# - No call-graph traversal (Phase 16+)
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging

from quart import request

from api.apps import login_required, current_user
from api.utils.api_utils import get_json_result
from common.constants import RetCode

from api.apps.codeatlas import blueprint

_log = logging.getLogger("codeatlas.arch_api")


# ---------------------------------------------------------------------------
# Internal helpers — DB access (lazy imports, no top-level ORM dependency)
# ---------------------------------------------------------------------------

def _node_to_dict(node) -> dict:
    """Serialize an ArchGraphNode ORM instance to a plain dict."""
    return {
        "id":         node.id,
        "kb_id":      node.kb_id,
        "symbol":     node.symbol,
        "kind":       node.kind,
        "file":       node.file,
        "start_line": node.start_line,
        "end_line":   node.end_line,
        "language":   node.language,
    }


def _edge_to_dict(edge) -> dict:
    """Serialize an ArchGraphEdge ORM instance to a plain dict."""
    return {
        "id":        edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "edge_type": edge.edge_type,
        "kb_id":     edge.kb_id,
    }


def _fetch_graph(kb_id: str | None) -> tuple[list[dict], list[dict]]:
    """
    Fetch all nodes and edges for *kb_id* from the Phase 7 tables.

    If *kb_id* is None or empty, returns all nodes and edges across all KBs
    (useful for admin-level graph views; can be scoped later).

    Returns
    -------
    (nodes, edges) : tuple[list[dict], list[dict]]
    """
    from api.db.db_models import ArchGraphNode, ArchGraphEdge  # noqa: PLC0415

    if kb_id:
        node_query = ArchGraphNode.select().where(ArchGraphNode.kb_id == kb_id)
        edge_query = ArchGraphEdge.select().where(ArchGraphEdge.kb_id == kb_id)
    else:
        node_query = ArchGraphNode.select()
        edge_query = ArchGraphEdge.select()

    nodes = [_node_to_dict(n) for n in node_query]
    edges = [_edge_to_dict(e) for e in edge_query]
    return nodes, edges


def _fetch_node(node_id: str) -> dict | None:
    """
    Fetch a single node by *node_id*.  Returns None if not found.
    Never raises — all exceptions are caught and logged.
    """
    from api.db.db_models import ArchGraphNode  # noqa: PLC0415
    try:
        node = ArchGraphNode.get_by_id(node_id)
        return _node_to_dict(node)
    except ArchGraphNode.DoesNotExist:
        return None
    except Exception as exc:  # noqa: BLE001
        _log.error("_fetch_node(%r): unexpected error — %s", node_id, exc)
        return None


def _fetch_edges_for_node(node_id: str) -> list[dict]:
    """
    Fetch all edges where *node_id* is the source or target.
    Returns [] on any error.
    """
    from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
    try:
        edges = ArchGraphEdge.select().where(
            (ArchGraphEdge.source_id == node_id) |
            (ArchGraphEdge.target_id == node_id)
        )
        return [_edge_to_dict(e) for e in edges]
    except Exception as exc:  # noqa: BLE001
        _log.error("_fetch_edges_for_node(%r): unexpected error — %s", node_id, exc)
        return []


# ---------------------------------------------------------------------------
# Route: GET /api/codeatlas/graph
# ---------------------------------------------------------------------------

@blueprint.route("/graph", methods=["GET"])
@login_required
async def get_graph():
    """
    Return the full architecture graph for a knowledge base.

    Query parameters
    ----------------
    kb_id : str, optional
        Knowledge-base ID to filter by.  If omitted, all nodes and edges
        across all knowledge bases are returned.

    Response
    --------
    200 OK::

        {
            "code": 0,
            "message": "success",
            "data": {
                "nodes": [
                    {
                        "id": "...",
                        "kb_id": "...",
                        "symbol": "my_func",
                        "kind": "function",
                        "file": "rag/nlp/query.py",
                        "start_line": 42,
                        "end_line": 55,
                        "language": "python"
                    },
                    ...
                ],
                "edges": [
                    {
                        "id": "...",
                        "source_id": "...",
                        "target_id": "...",
                        "edge_type": "import",
                        "kb_id": "..."
                    },
                    ...
                ]
            }
        }

    An empty graph (no nodes ingested yet) returns ``{"nodes": [], "edges": []}``
    with HTTP 200 — never a 404 or 500.

    Auth
    ----
    Requires a valid JWT, API key, or session cookie.
    Returns 401 if unauthenticated.
    """
    kb_id: str = (request.args.get("kb_id") or "").strip()

    try:
        nodes, edges = _fetch_graph(kb_id or None)
    except Exception as exc:  # noqa: BLE001
        _log.exception("get_graph: DB error — %s", exc)
        return get_json_result(
            code=RetCode.DATA_ERROR,
            message=f"Failed to query graph: {exc}",
        ), 500

    _log.info(
        "get_graph: kb_id=%r  nodes=%d  edges=%d  user=%s",
        kb_id or "(all)", len(nodes), len(edges),
        getattr(current_user, "id", "?"),
    )

    return get_json_result(
        code=RetCode.SUCCESS,
        message="success",
        data={"nodes": nodes, "edges": edges},
    )


# ---------------------------------------------------------------------------
# Route: GET /api/codeatlas/graph/node/<node_id>
# ---------------------------------------------------------------------------

@blueprint.route("/graph/node/<node_id>", methods=["GET"])
@login_required
async def get_node(node_id: str):
    """
    Return a single node and its directly connected edges.

    Path parameters
    ---------------
    node_id : str
        The 32-char hex node ID (as stored in arch_graph_nodes.id).

    Response — found (200)::

        {
            "code": 0,
            "message": "success",
            "data": {
                "node": {
                    "id": "...",
                    "kb_id": "...",
                    "symbol": "MyClass",
                    "kind": "class",
                    "file": "rag/nlp/query.py",
                    "start_line": 1,
                    "end_line": 40,
                    "language": "python"
                },
                "edges": [...]
            }
        }

    Response — not found (404)::

        {
            "code": 404,
            "message": "Node not found: <node_id>"
        }

    This endpoint NEVER returns 500 for an unknown node ID.
    All DB errors are caught and returned as structured error responses.

    Auth
    ----
    Requires a valid JWT, API key, or session cookie.
    Returns 401 if unauthenticated.
    """
    if not node_id or not node_id.strip():
        return get_json_result(
            code=RetCode.ARGUMENT_ERROR,
            message="node_id path parameter is required",
        ), 400

    try:
        node_dict = _fetch_node(node_id)
    except Exception as exc:  # noqa: BLE001
        # This branch is a safety net; _fetch_node already catches internally.
        _log.exception("get_node(%r): unexpected error — %s", node_id, exc)
        return get_json_result(
            code=RetCode.DATA_ERROR,
            message="Internal error querying node",
        ), 500

    if node_dict is None:
        _log.debug("get_node: node_id=%r not found", node_id)
        return get_json_result(
            code=RetCode.NOT_FOUND,
            message=f"Node not found: {node_id}",
        ), 404

    try:
        edges = _fetch_edges_for_node(node_id)
    except Exception as exc:  # noqa: BLE001
        _log.error("get_node: edge fetch failed for %r — %s", node_id, exc)
        edges = []   # Degrade gracefully: return node without edges

    _log.info(
        "get_node: node_id=%r  symbol=%r  edges=%d  user=%s",
        node_id, node_dict.get("symbol"), len(edges),
        getattr(current_user, "id", "?"),
    )

    return get_json_result(
        code=RetCode.SUCCESS,
        message="success",
        data={"node": node_dict, "edges": edges},
    )
