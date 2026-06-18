# codeatlas/analysis/arch_graph.py
#
# CodeAtlas — Phase 8: Graph Builder Worker
# ---------------------------------------------------------------------------
# Worker callable that reads code symbols (via Phase 6 SymbolExtractor),
# resolves import-based edges between them, and writes the results into the
# Phase 7 arch_graph_nodes / arch_graph_edges tables.
#
# Public API
# ----------
#   ArchGraphBuilder.build(kb_id, sources)
#       Build (or rebuild) the full architecture graph for *kb_id*.
#       *sources* is a list of items that can be:
#         - str / Path   → file or directory on disk  (extract_from_directory)
#         - list[dict]   → Phase 3 chunk dicts        (extract from chunks)
#       Idempotent: calling build() twice on the same input produces the
#       same row counts.  Existing rows for *kb_id* are deleted and
#       rewritten on each call.
#
# Scope (Phase 8 only)
# --------------------
# - Write nodes to arch_graph_nodes
# - Write import-resolved edges to arch_graph_edges
# - Raise MissingGrammarError if tree-sitter grammar unavailable
# - No API endpoints (Phase 9)
# - No call graph (Phase 16)
# - No worker scheduling / task_executor wiring
#
# Integration note
# ----------------
# This module imports the Phase 7 ORM models at call time via a lazy import
# (inside build()) rather than at module load time.  This keeps the module
# importable in standalone tests without a live database connection.
#
# The lazy-import pattern is:
#
#   from api.db.db_models import ArchGraphNode, ArchGraphEdge
#
# Callers that have a live RAGFlow database simply call build(); tests
# pass an ORM adapter (see _OrmAdapter below) that uses an in-memory
# SQLite database with equivalent Peewee models.
# ---------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from codeatlas.analysis.symbol_extractor import Symbol, SymbolExtractor
from codeatlas.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed exceptions  (roadmap requires "typed error, not silent failure")
# ---------------------------------------------------------------------------

class MissingGrammarError(ImportError):
    """
    Raised when tree-sitter or a required language grammar is not installed.

    Example::

        raise MissingGrammarError(
            "tree-sitter-python is required but not installed. "
            "Run: pip install tree-sitter tree-sitter-python"
        )
    """


class GraphBuildError(RuntimeError):
    """Raised when the graph builder encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stable_node_id(kb_id: str, symbol_name: str, file_path: str) -> str:
    """
    Return a stable 32-char hex ID for a node, deterministic on its content.

    Using a content-hash rather than a random UUID means that re-running
    build() on identical input produces identical IDs — which is what makes
    the DELETE + re-insert idempotency strategy give identical row counts.
    """
    key = f"{kb_id}:{file_path}:{symbol_name}"
    return hashlib.md5(key.encode()).hexdigest()   # 32 hex chars


def _stable_edge_id(kb_id: str, source_id: str, target_id: str, edge_type: str) -> str:
    """Return a stable 32-char hex ID for an edge."""
    key = f"{kb_id}:{source_id}:{target_id}:{edge_type}"
    return hashlib.md5(key.encode()).hexdigest()


def _module_name_from_path(file_path: str) -> str:
    """
    Convert a file path to a Python dotted module name.

    Examples
    --------
    ``"rag/nlp/query.py"``  →  ``"rag.nlp.query"``
    ``"codeatlas/flags.py"`` → ``"codeatlas.flags"``
    """
    p = Path(file_path)
    # Strip .py / .pyi suffix
    if p.suffix in (".py", ".pyi"):
        p = p.with_suffix("")
    # Replace OS separators with dots
    parts = list(p.parts)
    return ".".join(parts)


def _resolve_edges(
    symbols: list[Symbol],
    node_id_map: dict[tuple[str, str], str],
) -> list[tuple[str, str, str]]:
    """
    Resolve import-based edges between symbols.

    Strategy
    --------
    For each symbol *S*, look at *S.imports*.  For each import string *imp*,
    check whether any other symbol's file maps to a module name that *imp*
    starts with (or equals).  If so, add an ``"import"`` edge from S → each
    symbol in that target file.

    Parameters
    ----------
    symbols :
        Full list of Symbol objects extracted for this KB.
    node_id_map :
        Maps ``(file_path, symbol_name)`` → node_id (32-char hex).

    Returns
    -------
    list[tuple[str, str, str]]
        Each tuple is ``(source_node_id, target_node_id, edge_type)``.
        Duplicates are removed.
    """
    # Build a lookup: module_name → list of (file, symbol_name, node_id)
    # for every file that contains at least one symbol
    file_to_module: dict[str, str] = {}
    module_to_symbols: dict[str, list[tuple[str, str, str]]] = {}

    for sym in symbols:
        mod = _module_name_from_path(sym.file)
        file_to_module[sym.file] = mod
        if mod not in module_to_symbols:
            module_to_symbols[mod] = []
        nid = node_id_map.get((sym.file, sym.name))
        if nid:
            module_to_symbols[mod].append((sym.file, sym.name, nid))

    edges: set[tuple[str, str, str]] = set()

    for src_sym in symbols:
        src_nid = node_id_map.get((src_sym.file, src_sym.name))
        if src_nid is None:
            continue

        for imp in src_sym.imports:
            # Direct module match (e.g. import "rag.nlp.query")
            # or prefix match (e.g. "from rag.nlp import query" → imp="rag.nlp.query")
            for mod, targets in module_to_symbols.items():
                # imp == mod  OR  imp starts with mod (package import)
                if imp == mod or imp.startswith(mod + "."):
                    for _file, _sym_name, tgt_nid in targets:
                        if tgt_nid != src_nid:  # no self-loops
                            edges.add((src_nid, tgt_nid, "import"))

    return list(edges)


# ---------------------------------------------------------------------------
# ORM adapter — thin wrapper so build() can work with test or live DB
# ---------------------------------------------------------------------------

class _OrmAdapter:
    """
    Abstract interface over the Phase 7 ORM models.

    The default implementation imports the live RAGFlow models.
    Tests substitute a ``_TestOrmAdapter`` that uses in-memory SQLite models.
    """

    def delete_nodes_for_kb(self, kb_id: str) -> int:
        """Delete all nodes for kb_id, return deleted count."""
        from api.db.db_models import ArchGraphNode  # noqa: PLC0415
        return ArchGraphNode.delete().where(ArchGraphNode.kb_id == kb_id).execute()

    def delete_edges_for_kb(self, kb_id: str) -> int:
        """Delete all edges for kb_id, return deleted count."""
        from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
        return ArchGraphEdge.delete().where(ArchGraphEdge.kb_id == kb_id).execute()

    def insert_nodes(self, rows: list[dict]) -> int:
        """Bulk-insert node rows, return inserted count."""
        from api.db.db_models import ArchGraphNode  # noqa: PLC0415
        if not rows:
            return 0
        ArchGraphNode.insert_many(rows).execute()
        return len(rows)

    def insert_edges(self, rows: list[dict]) -> int:
        """Bulk-insert edge rows, return inserted count."""
        from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
        if not rows:
            return 0
        ArchGraphEdge.insert_many(rows).execute()
        return len(rows)

    def count_nodes(self, kb_id: str) -> int:
        from api.db.db_models import ArchGraphNode  # noqa: PLC0415
        return ArchGraphNode.select().where(ArchGraphNode.kb_id == kb_id).count()

    def count_edges(self, kb_id: str) -> int:
        from api.db.db_models import ArchGraphEdge  # noqa: PLC0415
        return ArchGraphEdge.select().where(ArchGraphEdge.kb_id == kb_id).count()


# ---------------------------------------------------------------------------
# Public builder class
# ---------------------------------------------------------------------------

class ArchGraphBuilder:
    """
    Extracts code symbols and writes an architecture graph to the database.

    Usage
    -----
    ::

        from codeatlas.analysis.arch_graph import ArchGraphBuilder

        builder = ArchGraphBuilder()
        result = builder.build(
            kb_id="kb_abc123",
            sources=["path/to/repo/src/"],
        )
        print(result)
        # GraphBuildResult(nodes=42, edges=17, kb_id='kb_abc123')

    Idempotency
    -----------
    ``build()`` deletes all existing rows for *kb_id* before inserting new
    ones.  Because node and edge IDs are content-hash-derived (not random
    UUIDs), two builds on identical input produce identical rows — row counts
    are stable and no duplicate rows accumulate.

    Sources
    -------
    Each element of *sources* may be:

    - A ``str`` or ``Path`` pointing to a directory → all ``.py`` files
      under that directory are extracted recursively.
    - A ``str`` or ``Path`` pointing to a single ``.py`` file → extracted
      directly.
    - A ``list[dict]`` of Phase 3 chunk dicts → extracted via
      ``SymbolExtractor.extract(chunks)``.
    """

    def __init__(self, orm: _OrmAdapter | None = None) -> None:
        self._orm = orm or _OrmAdapter()
        self._extractor = SymbolExtractor()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(
        self,
        kb_id: str,
        sources: list[Any] | None = None,
    ) -> "GraphBuildResult":
        """
        Build (or rebuild) the architecture graph for *kb_id*.

        Parameters
        ----------
        kb_id : str
            Knowledge-base identifier.  All written rows carry this value.
        sources : list | None
            List of source specifications.  Each element is a directory
            path (str/Path), a single ``.py`` file path, or a list of
            Phase 3 chunk dicts.  Pass ``None`` or ``[]`` for an empty build.

        Returns
        -------
        GraphBuildResult
            Named-tuple-like dataclass with ``.nodes`` and ``.edges`` counts.

        Raises
        ------
        MissingGrammarError
            If tree-sitter or tree-sitter-python is not installed and is
            needed to parse the provided sources.
        GraphBuildError
            On any other unrecoverable error during the build.
        """
        if not kb_id or not kb_id.strip():
            raise ValueError("kb_id must be a non-empty string")

        # ── Verify tree-sitter availability before touching the DB ────────
        self._assert_grammar_available()

        sources = sources or []

        _log.info("ArchGraphBuilder.build: kb_id=%r, %d source(s)", kb_id, len(sources))

        # ── Extract all symbols from all sources ──────────────────────────
        symbols = self._collect_symbols(sources)
        _log.info("ArchGraphBuilder: extracted %d symbols", len(symbols))

        # ── Build node rows ───────────────────────────────────────────────
        node_id_map: dict[tuple[str, str], str] = {}
        node_rows: list[dict] = []

        for sym in symbols:
            nid = _stable_node_id(kb_id, sym.name, sym.file)
            key = (sym.file, sym.name)
            if key in node_id_map:
                # Duplicate (same symbol/file) — skip; keeps idempotent counts
                continue
            node_id_map[key] = nid
            node_rows.append({
                "id":         nid,
                "kb_id":      kb_id,
                "symbol":     sym.name,
                "kind":       sym.kind,
                "file":       sym.file,
                "start_line": sym.start_line,
                "end_line":   sym.end_line,
                "language":   sym.language,
            })

        # ── Resolve import edges ───────────────────────────────────────────
        raw_edges = _resolve_edges(symbols, node_id_map)
        edge_rows: list[dict] = []
        seen_edge_ids: set[str] = set()

        for src_nid, tgt_nid, etype in raw_edges:
            eid = _stable_edge_id(kb_id, src_nid, tgt_nid, etype)
            if eid in seen_edge_ids:
                continue
            seen_edge_ids.add(eid)
            edge_rows.append({
                "id":        eid,
                "source_id": src_nid,
                "target_id": tgt_nid,
                "edge_type": etype,
                "kb_id":     kb_id,
            })

        # ── Atomic delete + re-insert ─────────────────────────────────────
        # Delete first so that a re-run always converges to the same state.
        # No transaction wrapper here — we follow RAGFlow's pattern of
        # relying on connection-level atomicity; Phase 9 can add transactions.
        deleted_nodes = self._orm.delete_nodes_for_kb(kb_id)
        deleted_edges = self._orm.delete_edges_for_kb(kb_id)
        _log.debug(
            "ArchGraphBuilder: deleted %d nodes, %d edges for kb_id=%r",
            deleted_nodes, deleted_edges, kb_id,
        )

        inserted_nodes = self._orm.insert_nodes(node_rows)
        inserted_edges = self._orm.insert_edges(edge_rows)

        _log.info(
            "ArchGraphBuilder.build done: kb_id=%r  nodes=%d  edges=%d",
            kb_id, inserted_nodes, inserted_edges,
        )
        return GraphBuildResult(
            kb_id=kb_id,
            nodes=inserted_nodes,
            edges=inserted_edges,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_grammar_available() -> None:
        """
        Raise ``MissingGrammarError`` if tree-sitter or tree-sitter-python
        is not installed.

        This check runs before any DB writes so the builder fails cleanly
        rather than inserting a partial graph.
        """
        missing: list[str] = []
        try:
            import tree_sitter  # noqa: F401, PLC0415
        except ImportError:
            missing.append("tree-sitter")
        try:
            import tree_sitter_python  # noqa: F401, PLC0415
        except ImportError:
            missing.append("tree-sitter-python")

        if missing:
            pkg_list = " ".join(missing)
            raise MissingGrammarError(
                f"Required tree-sitter grammar package(s) not installed: "
                f"{pkg_list}. "
                f"Install with: pip install {pkg_list}"
            )

    def _collect_symbols(self, sources: list[Any]) -> list[Symbol]:
        """
        Iterate over *sources* and collect all Symbol objects.

        Each element in *sources* is handled as:
        - ``list``  → Phase 3 chunk dicts  → ``extract(chunks)``
        - ``Path``/``str`` pointing to a directory → ``extract_from_directory``
        - ``Path``/``str`` pointing to a ``.py`` file → ``extract_from_files``
        """
        all_symbols: list[Symbol] = []

        for src in sources:
            if isinstance(src, list):
                # Phase 3 chunk dicts
                syms = self._extractor.extract(src)
                _log.debug(
                    "_collect_symbols: %d symbols from chunk list (%d chunks)",
                    len(syms), len(src),
                )
                all_symbols.extend(syms)

            elif isinstance(src, (str, Path)):
                p = Path(src)
                if p.is_dir():
                    syms = self._extractor.extract_from_directory(p)
                    _log.debug(
                        "_collect_symbols: %d symbols from directory %s",
                        len(syms), p,
                    )
                elif p.is_file():
                    syms = self._extractor.extract_from_files([p])
                    _log.debug(
                        "_collect_symbols: %d symbols from file %s",
                        len(syms), p,
                    )
                else:
                    _log.warning(
                        "_collect_symbols: source %r not found, skipping", str(p)
                    )
                    syms = []
                all_symbols.extend(syms)

            else:
                _log.warning(
                    "_collect_symbols: unknown source type %s, skipping",
                    type(src).__name__,
                )

        return all_symbols


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class GraphBuildResult:
    """
    Returned by ``ArchGraphBuilder.build()``.

    Attributes
    ----------
    kb_id : str
        Knowledge base identifier the graph was built for.
    nodes : int
        Number of node rows written to ``arch_graph_nodes``.
    edges : int
        Number of edge rows written to ``arch_graph_edges``.
    """

    __slots__ = ("kb_id", "nodes", "edges")

    def __init__(self, kb_id: str, nodes: int, edges: int) -> None:
        self.kb_id = kb_id
        self.nodes = nodes
        self.edges = edges

    def __repr__(self) -> str:
        return (
            f"GraphBuildResult(kb_id={self.kb_id!r}, "
            f"nodes={self.nodes}, edges={self.edges})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphBuildResult):
            return NotImplemented
        return (self.kb_id, self.nodes, self.edges) == (
            other.kb_id, other.nodes, other.edges,
        )
