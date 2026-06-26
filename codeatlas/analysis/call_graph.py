# codeatlas/analysis/call_graph.py
#
# CodeAtlas — Phase 16: Call Graph Builder
# ---------------------------------------------------------------------------
# Extends arch_graph_edges with "call" edges by:
#   1. Loading existing ArchGraphNode rows for a kb_id (symbol lookup)
#   2. Reading source files via SymbolExtractor paths
#   3. Walking function bodies with tree-sitter to find call sites
#   4. Resolving callee names against the known symbol map
#   5. Inserting only the missing call edges (idempotent)
#
# Design rules
# ------------
# - NO new DB tables.  Writes only to the existing arch_graph_edges table
#   with edge_type="call".  Import edges (edge_type="import") are never
#   touched or re-inserted.
# - Call graph logic is ISOLATED from ArchGraphBuilder (Phase 8).  The two
#   builders can run independently in any order.
# - All ORM imports are lazy (inside methods) — module importable without DB.
# - _CallOrmAdapter protocol enables in-memory testing without a live server.
# - Deterministic edge IDs reuse Phase 8's _stable_edge_id formula so the
#   same call pair always hashes to the same 32-char hex.
# - "import edges remain untouched" is enforced by filtering on edge_type:
#   the builder only deletes/inserts edges where edge_type="call".
#
# Language support
# ----------------
# Phase 16: Python only.  The _CallExtractor class is designed so that
# additional languages can be added by registering new grammar handlers
# inside _build_extractor() without touching any other method.
#
# MissingGrammarError is raised BEFORE any DB writes so the builder always
# fails cleanly rather than inserting a partial call graph.
# ---------------------------------------------------------------------------

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeatlas.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

class CallGraphError(RuntimeError):
    """Raised when the call graph builder encounters an unrecoverable error."""


class MissingGrammarError(ImportError):
    """
    Raised when tree-sitter or a required language grammar is not installed.
    Consistent with the same exception in Phase 8 (arch_graph.py).
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CallGraphResult:
    """Returned by CallGraphBuilder.build()."""
    kb_id: str
    call_edges_inserted: int
    symbols_resolved: int
    symbols_unresolved: int

    def __repr__(self) -> str:
        return (
            f"CallGraphResult(kb_id={self.kb_id!r}, "
            f"call_edges={self.call_edges_inserted}, "
            f"resolved={self.symbols_resolved}, "
            f"unresolved={self.symbols_unresolved})"
        )


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------

@dataclass
class _NodeRecord:
    """Lightweight view of one ArchGraphNode row."""
    node_id: str
    symbol: str
    kind: str
    file: str
    language: str
    start_line: int
    end_line: int


@dataclass
class _RawCall:
    """One caller → callee relationship found during AST walking."""
    caller_symbol: str   # name of the function/method doing the calling
    callee_name: str     # raw name as it appears in the call expression
    source_file: str


# ---------------------------------------------------------------------------
# ID helpers (mirror Phase 8 convention exactly)
# ---------------------------------------------------------------------------

def _stable_edge_id(kb_id: str, source_id: str, target_id: str, edge_type: str) -> str:
    """Deterministic 32-char hex edge ID, identical formula to Phase 8."""
    key = f"{kb_id}:{source_id}:{target_id}:{edge_type}"
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Language-specific AST call extractor
# ---------------------------------------------------------------------------

class _CallExtractor:
    """
    Extracts raw (caller, callee) pairs from source text using tree-sitter.

    Currently supports Python only.  Add further languages by extending
    ``_extract_python`` with a language dispatch table.
    """

    def __init__(self) -> None:
        self._py_parser = None  # lazy: built on first use

    def extract(self, source_bytes: bytes, language: str, filename: str) -> list[_RawCall]:
        """
        Return all (caller_symbol, callee_name, source_file) tuples found
        in *source_bytes*.

        Silently returns [] for unsupported languages — no exception raised
        so the outer loop can continue with other files.
        """
        if language == "python":
            return self._extract_python(source_bytes, filename)
        _log.debug("_CallExtractor: unsupported language %r — skipping %s", language, filename)
        return []

    def _get_py_parser(self):
        """Build (and cache) a tree-sitter Parser for Python."""
        if self._py_parser is not None:
            return self._py_parser
        import tree_sitter          # noqa: PLC0415
        import tree_sitter_python   # noqa: PLC0415
        lang = tree_sitter.Language(tree_sitter_python.language())
        self._py_parser = tree_sitter.Parser(lang)
        return self._py_parser

    def _extract_python(self, source_bytes: bytes, filename: str) -> list[_RawCall]:
        """
        Walk the tree-sitter CST to find all function call sites.

        Caller tracking:
          The extractor tracks the innermost function_definition or
          decorated_definition node to identify which function/method
          each call site belongs to.

        Callee extraction:
          - ``identifier`` callee  → bare function call, e.g. ``helper(x)``
          - ``attribute`` callee   → method call, e.g. ``obj.method(x)``;
            we record only the attribute name (``method``), not the receiver.
            This matches the roadmap's "object.method()" requirement and
            makes resolution language-agnostic.
        """
        try:
            parser = self._get_py_parser()
            tree = parser.parse(source_bytes)
        except Exception as exc:  # noqa: BLE001
            _log.warning("_extract_python: parse failed for %s — %s", filename, exc)
            return []

        results: list[_RawCall] = []
        self._walk_py(tree.root_node, caller_stack=[], results=results, filename=filename)
        return results

    def _walk_py(
        self,
        node: Any,
        caller_stack: list[str],
        results: list[_RawCall],
        filename: str,
    ) -> None:
        """
        Recursive CST walk.  Pushes/pops the caller name onto a stack so
        nested definitions are handled correctly.

        Circular calls (a function calling itself) are allowed — they
        produce a self-loop edge which is filtered out later in the builder
        (matching Phase 8's "no self-loops" policy).
        """
        pushed = False

        # Track entry into a function/method definition
        if node.type in ("function_definition", "decorated_definition"):
            target = node
            if node.type == "decorated_definition":
                # Unwrap to the inner function/class
                for child in node.children:
                    if child.type == "function_definition":
                        target = child
                        break
            name_node = target.child_by_field_name("name")
            if name_node:
                caller_stack.append(name_node.text.decode("utf-8", errors="replace"))
                pushed = True

        # Detect call expression
        if node.type == "call" and caller_stack:
            caller = caller_stack[-1]
            func_node = node.children[0] if node.children else None
            if func_node is not None:
                if func_node.type == "identifier":
                    callee = func_node.text.decode("utf-8", errors="replace")
                    results.append(_RawCall(caller, callee, filename))
                elif func_node.type == "attribute":
                    attr = func_node.child_by_field_name("attribute")
                    if attr:
                        callee = attr.text.decode("utf-8", errors="replace")
                        results.append(_RawCall(caller, callee, filename))

        # Recurse
        for child in node.children:
            self._walk_py(child, caller_stack, results, filename)

        if pushed:
            caller_stack.pop()


# ---------------------------------------------------------------------------
# ORM adapter  (swapped for in-memory substitute in tests)
# ---------------------------------------------------------------------------

class _CallOrmAdapter:
    """
    Reads nodes and writes call edges via lazy ORM imports.
    Tests inject a _TestCallOrmAdapter that uses in-memory Peewee models.
    """

    def load_nodes(self, kb_id: str) -> list[_NodeRecord]:
        """Return all ArchGraphNode rows for kb_id."""
        from api.db.db_models import ArchGraphNode  # noqa: PLC0415
        rows = ArchGraphNode.select().where(ArchGraphNode.kb_id == kb_id)
        return [
            _NodeRecord(
                node_id=r.id,
                symbol=r.symbol,
                kind=r.kind,
                file=r.file,
                language=r.language,
                start_line=r.start_line,
                end_line=r.end_line,
            )
            for r in rows
        ]

    def edge_exists(self, edge_id: str) -> bool:
        """Return True if an edge with this id already exists."""
        from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
        return ArchGraphEdge.select().where(ArchGraphEdge.id == edge_id).exists()

    def insert_edges(self, rows: list[dict]) -> int:
        """Bulk-insert edge rows; return count inserted."""
        from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
        if not rows:
            return 0
        ArchGraphEdge.insert_many(rows).execute()
        return len(rows)

    def count_call_edges(self, kb_id: str) -> int:
        """Count existing call edges for kb_id."""
        from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
        return (
            ArchGraphEdge.select()
            .where(ArchGraphEdge.kb_id == kb_id, ArchGraphEdge.edge_type == "call")
            .count()
        )

    def count_import_edges(self, kb_id: str) -> int:
        """Count import edges — used to verify they were not touched."""
        from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
        return (
            ArchGraphEdge.select()
            .where(ArchGraphEdge.kb_id == kb_id, ArchGraphEdge.edge_type == "import")
            .count()
        )

    def load_nodes_with_files(self, kb_id: str) -> list[_NodeRecord]:
        """Alias kept for semantic clarity at call sites."""
        return self.load_nodes(kb_id)


# ---------------------------------------------------------------------------
# Public CallGraphBuilder
# ---------------------------------------------------------------------------

class CallGraphBuilder:
    """
    Extends the existing arch_graph_edges table with call-type edges.

    Lifecycle
    ---------
    1. Load all ArchGraphNode rows for *kb_id* — these are the "known symbols".
    2. For each unique source file referenced by those nodes, read the file
       from disk and extract all call pairs using ``_CallExtractor``.
    3. Resolve callee names against the known-symbol map.
    4. Insert only new edges (idempotent: duplicate IDs are skipped via
       ``edge_exists()`` pre-check).
    5. Import edges are never touched.

    Usage
    -----
    ::

        builder = CallGraphBuilder()
        result = builder.build(kb_id="kb_abc123")
        print(result)

    For testing, inject a ``_TestCallOrmAdapter``::

        builder = CallGraphBuilder(orm=test_adapter)

    Parameters
    ----------
    orm : _CallOrmAdapter | None
        Persistence adapter.  ``None`` → live ORM (needs running DB).
    """

    def __init__(self, orm: _CallOrmAdapter | None = None) -> None:
        self._orm = orm or _CallOrmAdapter()
        self._extractor = _CallExtractor()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, kb_id: str) -> CallGraphResult:
        """
        Build (or update) the call graph for *kb_id*.

        Parameters
        ----------
        kb_id : str
            Knowledge-base identifier.

        Returns
        -------
        CallGraphResult

        Raises
        ------
        MissingGrammarError
            If tree-sitter or tree-sitter-python is not installed.
        ValueError
            If kb_id is empty.
        """
        if not kb_id or not kb_id.strip():
            raise ValueError("kb_id must be a non-empty string")

        # Verify grammar before any DB access
        self._assert_grammar_available()

        # ── Step 1: Load existing nodes ───────────────────────────────────
        nodes = self._orm.load_nodes(kb_id)
        if not nodes:
            _log.info("CallGraphBuilder.build(%r): no nodes found — empty call graph", kb_id)
            return CallGraphResult(kb_id=kb_id, call_edges_inserted=0,
                                   symbols_resolved=0, symbols_unresolved=0)

        # ── Step 2: Build symbol lookup maps ──────────────────────────────
        # symbol_name → node_id  (last write wins for duplicates)
        symbol_to_id: dict[str, str] = {n.symbol: n.node_id for n in nodes}
        # node_id → _NodeRecord  (for caller resolution)
        node_by_id: dict[str, _NodeRecord] = {n.node_id: n for n in nodes}
        # source_file → language (from the first node that mentions it)
        file_to_language: dict[str, str] = {}
        for n in nodes:
            if n.file and n.file not in file_to_language:
                file_to_language[n.file] = n.language
        # symbol_name → source_file  (for caller file resolution)
        symbol_to_file: dict[str, str] = {n.symbol: n.file for n in nodes}

        _log.info(
            "CallGraphBuilder.build(%r): %d nodes, %d unique files",
            kb_id, len(nodes), len(file_to_language),
        )

        # ── Step 3: Extract raw calls from source files ───────────────────
        raw_calls: list[_RawCall] = []
        for file_path, language in file_to_language.items():
            file_calls = self._extract_calls_from_file(file_path, language)
            raw_calls.extend(file_calls)

        _log.debug("CallGraphBuilder: %d raw call pairs extracted", len(raw_calls))

        # ── Step 4: Resolve and build edge rows ───────────────────────────
        resolved = 0
        unresolved = 0
        new_edge_rows: list[dict] = []
        seen_edge_ids: set[str] = set()

        for rc in raw_calls:
            caller_id = symbol_to_id.get(rc.caller_symbol)
            callee_id = symbol_to_id.get(rc.callee_name)

            if caller_id is None or callee_id is None:
                unresolved += 1
                continue

            if caller_id == callee_id:
                # No self-loop edges (consistent with Phase 8)
                continue

            eid = _stable_edge_id(kb_id, caller_id, callee_id, "call")
            if eid in seen_edge_ids:
                continue
            seen_edge_ids.add(eid)

            # Skip if already persisted (idempotent)
            if self._orm.edge_exists(eid):
                continue

            new_edge_rows.append({
                "id":        eid,
                "source_id": caller_id,
                "target_id": callee_id,
                "edge_type": "call",
                "kb_id":     kb_id,
            })
            resolved += 1

        # ── Step 5: Persist ───────────────────────────────────────────────
        inserted = self._orm.insert_edges(new_edge_rows)

        _log.info(
            "CallGraphBuilder.build(%r) done: %d call edges inserted, "
            "%d unresolved, %d skipped (duplicates)",
            kb_id, inserted, unresolved,
            len(seen_edge_ids) - inserted,
        )

        return CallGraphResult(
            kb_id=kb_id,
            call_edges_inserted=inserted,
            symbols_resolved=resolved,
            symbols_unresolved=unresolved,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_calls_from_file(
        self, file_path: str, language: str
    ) -> list[_RawCall]:
        """Read *file_path* from disk and return raw call pairs."""
        p = Path(file_path)
        if not p.exists():
            _log.debug("_extract_calls_from_file: %s not on disk — skipping", file_path)
            return []
        try:
            source_bytes = p.read_bytes()
            return self._extractor.extract(source_bytes, language, file_path)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "_extract_calls_from_file: error reading %s — %s", file_path, exc
            )
            return []

    @staticmethod
    def _assert_grammar_available() -> None:
        """Raise MissingGrammarError before any DB writes if grammar missing."""
        missing: list[str] = []
        try:
            import tree_sitter        # noqa: F401, PLC0415
        except ImportError:
            missing.append("tree-sitter")
        try:
            import tree_sitter_python  # noqa: F401, PLC0415
        except ImportError:
            missing.append("tree-sitter-python")
        if missing:
            raise MissingGrammarError(
                f"Required package(s) not installed: {' '.join(missing)}. "
                f"Run: pip install {' '.join(missing)}"
            )

    # ------------------------------------------------------------------
    # Graph query helper  (used by flow_api.py)
    # ------------------------------------------------------------------

    def get_call_chain(
        self,
        kb_id: str,
        entry_symbol: str,
        max_depth: int = 10,
    ) -> list[dict]:
        """
        Return the ordered call chain starting from *entry_symbol*.

        Performs a BFS over existing call edges in the database.  Stops at
        *max_depth* to prevent runaway traversal on cyclic graphs.

        Parameters
        ----------
        kb_id : str
            Knowledge base to query.
        entry_symbol : str
            Name of the entry-point function/method.
        max_depth : int
            Maximum BFS depth.

        Returns
        -------
        list[dict]
            Ordered list of ``{"symbol": ..., "file": ..., "kind": ...}``
            dicts representing the call chain.  Returns ``[]`` if the symbol
            is unknown or has no outgoing call edges.
        """
        nodes = self._orm.load_nodes(kb_id)
        symbol_to_node: dict[str, _NodeRecord] = {n.symbol: n for n in nodes}
        id_to_node: dict[str, _NodeRecord]    = {n.node_id: n for n in nodes}

        entry = symbol_to_node.get(entry_symbol)
        if entry is None:
            return []

        # BFS
        visited: set[str] = set()
        queue: list[str] = [entry.node_id]
        chain: list[dict] = []

        depth = 0
        while queue and depth < max_depth:
            next_queue: list[str] = []
            for node_id in queue:
                if node_id in visited:
                    continue
                visited.add(node_id)
                node = id_to_node.get(node_id)
                if node is None:
                    continue
                chain.append({
                    "symbol": node.symbol,
                    "file":   node.file,
                    "kind":   node.kind,
                })
                # Load outgoing call edges for this node
                callees = self._load_callees(kb_id, node_id)
                next_queue.extend(c for c in callees if c not in visited)
            queue = next_queue
            depth += 1

        return chain

    def _load_callees(self, kb_id: str, source_id: str) -> list[str]:
        """Return target_ids of all call edges from source_id."""
        try:
            from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
            rows = ArchGraphEdge.select().where(
                ArchGraphEdge.kb_id     == kb_id,
                ArchGraphEdge.source_id == source_id,
                ArchGraphEdge.edge_type == "call",
            )
            return [r.target_id for r in rows]
        except Exception as exc:  # noqa: BLE001
            _log.warning("_load_callees(%r): %s", source_id, exc)
            return []
