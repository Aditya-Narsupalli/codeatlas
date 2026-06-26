# api/apps/codeatlas/flow_api.py
#
# CodeAtlas — Phase 16: Call Flow API
# ---------------------------------------------------------------------------
# Quart route handler for the call-chain flow endpoint.
#
# Endpoint
# --------
#   GET /api/codeatlas/flow/<symbol>
#       Traces the call chain starting from <symbol> in the kb_id's graph.
#
#       Response (200)::
#
#           {
#             "code": 0,
#             "data": {
#               "symbol": "parse_commit_log",
#               "chain": [
#                 {"symbol": "parse_commit_log", "file": "rag/nlp/git.py", "kind": "function"},
#                 {"symbol": "helper_util",       "file": "rag/nlp/util.py", "kind": "function"}
#               ]
#             }
#           }
#
#       Response (404) for unknown symbol::
#
#           {"code": 404, "message": "Symbol not found: <symbol>"}
#
#       Never returns 500 for a missing or unknown symbol.
#
# Auth
# ----
# Uses @login_required from api/apps/__init__.py — same as arch_api.py.
# Returns 401 if unauthenticated.
#
# Pattern
# -------
# Follows Phase 9 arch_api.py exactly:
#   - imports blueprint from api.apps.codeatlas
#   - lazy ORM / DB imports inside route handlers
#   - get_json_result() for all responses
#   - RetCode constants for numeric codes
#   - _log = logging.getLogger(...)
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging

from quart import request

from api.apps import login_required, current_user
from api.utils.api_utils import get_json_result
from common.constants import RetCode

from api.apps.codeatlas import blueprint

_log = logging.getLogger("codeatlas.flow_api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_builder(kb_id: str):
    """
    Return a CallGraphBuilder bound to the live ORM.
    Imported lazily so the module is importable without a live DB.
    """
    from codeatlas.analysis.call_graph import CallGraphBuilder  # noqa: PLC0415
    return CallGraphBuilder()


# ---------------------------------------------------------------------------
# Route: GET /api/codeatlas/flow/<symbol>
# ---------------------------------------------------------------------------

@blueprint.route("/flow/<symbol>", methods=["GET"])
@login_required
async def get_flow(symbol: str):
    """
    Return the ordered call chain starting from *symbol*.

    Path parameters
    ---------------
    symbol : str
        Name of the entry-point function or method, e.g. ``parse_commit_log``.

    Query parameters
    ----------------
    kb_id : str, optional
        Knowledge-base ID to scope the query.  If omitted, the builder
        queries all call edges across all KBs for this symbol.
    max_depth : int, optional
        Maximum BFS depth (default 10, capped at 20).

    Response — found (200)::

        {
            "code": 0,
            "message": "success",
            "data": {
                "symbol": "<symbol>",
                "chain": [
                    {"symbol": "...", "file": "...", "kind": "function|class"},
                    ...
                ]
            }
        }

    Response — unknown symbol (404)::

        {"code": 404, "message": "Symbol not found: <symbol>"}

    This endpoint NEVER returns 500 for an unknown or missing symbol.

    Auth
    ----
    Requires a valid JWT, API key, or session cookie.
    Returns 401 if unauthenticated.
    """
    if not symbol or not symbol.strip():
        return get_json_result(
            code=RetCode.ARGUMENT_ERROR,
            message="symbol path parameter is required",
        ), 400

    kb_id: str = (request.args.get("kb_id") or "").strip()

    try:
        max_depth_raw = request.args.get("max_depth", "10")
        max_depth = max(1, min(20, int(max_depth_raw)))
    except (ValueError, TypeError):
        max_depth = 10

    _log.info(
        "get_flow: symbol=%r kb_id=%r max_depth=%d user=%s",
        symbol, kb_id or "(all)", max_depth,
        getattr(current_user, "id", "?"),
    )

    # Resolve the entry symbol to a node and trace the call chain
    try:
        builder = _get_builder(kb_id)
        chain = builder.get_call_chain(
            kb_id=kb_id,
            entry_symbol=symbol,
            max_depth=max_depth,
        )
    except Exception as exc:  # noqa: BLE001
        _log.exception("get_flow: unexpected error for symbol=%r: %s", symbol, exc)
        return get_json_result(
            code=RetCode.DATA_ERROR,
            message=f"Failed to query call chain: {exc}",
        ), 500

    # Empty chain means symbol not found (node doesn't exist in the graph)
    if not chain:
        _log.debug("get_flow: symbol %r not found in kb_id=%r", symbol, kb_id)
        return get_json_result(
            code=RetCode.NOT_FOUND,
            message=f"Symbol not found: {symbol}",
        ), 404

    return get_json_result(
        code=RetCode.SUCCESS,
        message="success",
        data={
            "symbol": symbol,
            "chain":  chain,
        },
    )
