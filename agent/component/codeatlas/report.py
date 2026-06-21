# agent/component/codeatlas/report.py
#
# CodeAtlas — Phase 12: Architecture Report Agent
# ---------------------------------------------------------------------------
# Second agent component in the codeatlas namespace.  Reads the architecture
# graph (Phase 9 API output) and retrieved document/code chunks (Phase 5
# search), and synthesises a structured Markdown report describing the
# system's components and their dependencies.
#
# Architecture
# ------------
# This module mirrors the split established in Phase 11's reading_order.py:
#
#   1. ArchReportCore  — pure orchestration logic.  No dependency on
#      agent.canvas.Graph, no LLMBundle construction, no DB/HTTP connection
#      at import time.  Fully unit-testable by injecting a callable LLM
#      function plus an already-fetched graph dict and chunk list directly.
#
#   2. ArchReportComponent / ArchReportParam — the thin RAGFlow
#      ComponentBase adapter that the agent canvas runner actually
#      instantiates.  It wires ArchReportCore to the live LLMBundle, the
#      Phase 9 graph API, and Phase 5 search results.
#
# This split exists for the same reason as Phase 11: ComponentBase.__init__
# requires a live agent.canvas.Graph instance and LLM.__init__ makes live
# calls to get_model_type_by_name()/LLMBundle() — neither of which should
# run during import or in a unit test.
#
# Scope (Phase 12 only)
# ----------------------
# - ArchReportComponent callable from RAGFlow's agent runner
# - Markdown report: ≥1 component table, ≥1 dependency section, ≥1 heading
#   per major component
# - Graceful, no-crash handling of an empty graph ("No graph data available")
# - No UI, no new API endpoints, no graph schema changes, no FACTORY wiring
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from codeatlas.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GraphNodeData:
    """
    One node from the Phase 9 graph API response.

    Field names match the JSON keys returned by
    GET /api/codeatlas/graph → data.nodes[i] exactly, so a caller can build
    this directly from the parsed API response with **kwargs-style mapping.
    """
    id: str
    symbol: str
    kind: str            # "function" | "class"
    file: str
    language: str = "python"
    start_line: int = 0
    end_line: int = 0
    kb_id: str = ""


@dataclass
class GraphEdgeData:
    """
    One edge from the Phase 9 graph API response.

    Field names match GET /api/codeatlas/graph → data.edges[i] exactly.
    """
    id: str
    source_id: str
    target_id: str
    edge_type: str = "import"
    kb_id: str = ""


@dataclass
class GraphData:
    """Container mirroring the Phase 9 API's {"nodes": [...], "edges": [...]} shape."""
    nodes: list[GraphNodeData] = field(default_factory=list)
    edges: list[GraphEdgeData] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, data: dict) -> "GraphData":
        """
        Build a GraphData from the raw dict returned by Phase 9's
        GET /api/codeatlas/graph (the value of response["data"]).

        Tolerant of missing/empty keys — never raises on malformed input;
        returns an empty GraphData instead so callers can rely on
        ``is_empty`` rather than catching exceptions.
        """
        if not isinstance(data, dict):
            return cls()

        raw_nodes = data.get("nodes") or []
        raw_edges = data.get("edges") or []

        nodes: list[GraphNodeData] = []
        for n in raw_nodes:
            if not isinstance(n, dict) or not n.get("id"):
                continue
            nodes.append(GraphNodeData(
                id=str(n.get("id", "")),
                symbol=str(n.get("symbol", "") or "(unnamed)"),
                kind=str(n.get("kind", "function")),
                file=str(n.get("file", "") or "(unknown file)"),
                language=str(n.get("language", "python")),
                start_line=int(n.get("start_line", 0) or 0),
                end_line=int(n.get("end_line", 0) or 0),
                kb_id=str(n.get("kb_id", "")),
            ))

        edges: list[GraphEdgeData] = []
        for e in raw_edges:
            if not isinstance(e, dict) or not e.get("source_id") or not e.get("target_id"):
                continue
            edges.append(GraphEdgeData(
                id=str(e.get("id", "")),
                source_id=str(e["source_id"]),
                target_id=str(e["target_id"]),
                edge_type=str(e.get("edge_type", "import")),
                kb_id=str(e.get("kb_id", "")),
            ))

        return cls(nodes=nodes, edges=edges)

    @property
    def is_empty(self) -> bool:
        return not self.nodes


@dataclass
class RetrievedChunk:
    """
    One retrieved document/code chunk, as produced by Phase 5 search
    (post-``CodeSearchReranker.rerank()``) or any RAGFlow retrieval result.

    Only the fields the report actually uses are modeled; other chunk dict
    keys (vector, positions, etc.) are simply ignored by the caller when
    constructing these.
    """
    content: str
    symbol: str = ""
    source_type: str = ""   # "code" | "git" | "" (doc)
    docnm: str = ""


# ---------------------------------------------------------------------------
# Core orchestration logic — no canvas, no live LLM/API construction
# ---------------------------------------------------------------------------

class ArchReportCore:
    """
    Pure orchestration core for the Architecture Report agent.

    Has no dependency on agent.canvas.Graph, ComponentBase, LLMBundle, or
    HTTP clients.  The LLM call is injected as a plain callable and the
    graph/chunk data are passed in directly, so this class is fully
    unit-testable with a mock LLM function and in-memory fixtures.

    Usage
    -----
    ::

        core = ArchReportCore(llm_call=my_chat_function)
        markdown = core.generate_report(graph_data, chunks)

    Parameters
    ----------
    llm_call : Callable[[str, str], str] | None
        A function ``(sys_prompt: str, user_prompt: str) -> str`` that
        invokes the chat model and returns prose explanation text for the
        report's narrative sections.  If ``None``, a deterministic
        template-only report is produced (no LLM call at all) — used for
        the empty-graph case and as a safe fallback when no model is
        configured.
    max_nodes_in_table : int
        Safety cap on how many component rows are rendered in the overview
        table.  Defaults to 200; large repos are summarised rather than
        producing an unbounded table.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str, str], str]] = None,
        max_nodes_in_table: int = 200,
    ) -> None:
        self._llm_call = llm_call
        self._max_nodes_in_table = max_nodes_in_table

    def generate_report(
        self,
        graph: GraphData,
        chunks: Optional[list[RetrievedChunk]] = None,
    ) -> str:
        """
        Generate a Markdown architecture report from *graph* and *chunks*.

        Parameters
        ----------
        graph : GraphData
            Architecture graph data (Phase 9 API output, already parsed via
            ``GraphData.from_api_response``).
        chunks : list[RetrievedChunk] | None
            Retrieved document/code chunks (Phase 5 search results) used to
            enrich the narrative.  May be empty or ``None``.

        Returns
        -------
        str
            Valid Markdown text.  Always non-empty.  If *graph* is empty,
            returns a short report explicitly stating "No graph data
            available" rather than raising or producing a blank document.
        """
        chunks = chunks or []

        if graph.is_empty:
            _log.info("ArchReportCore: empty graph — producing graceful placeholder report")
            return self._empty_graph_report()

        sections: list[str] = []
        sections.append("# Architecture Report\n")
        sections.append(self._build_summary_line(graph, chunks))
        sections.append(self._build_components_section(graph))
        sections.append(self._build_dependencies_section(graph))
        sections.append(self._build_narrative_section(graph, chunks))

        report = "\n".join(s for s in sections if s)
        return report.strip() + "\n"

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_graph_report() -> str:
        """
        Graceful placeholder report for an empty/missing graph.

        Still valid Markdown with a heading, so downstream Markdown
        renderers never see a blank or malformed document.
        """
        return (
            "# Architecture Report\n\n"
            "No graph data available.\n\n"
            "The architecture graph for this knowledge base is empty — "
            "either no code repository has been ingested yet, or the "
            "graph builder (Phase 8) has not run for this knowledge base. "
            "Ingest a code repository and rebuild the graph to generate a "
            "full report.\n"
        )

    @staticmethod
    def _build_summary_line(graph: GraphData, chunks: list[RetrievedChunk]) -> str:
        n_classes = sum(1 for n in graph.nodes if n.kind == "class")
        n_functions = sum(1 for n in graph.nodes if n.kind == "function")
        return (
            f"_{len(graph.nodes)} components "
            f"({n_classes} classes, {n_functions} functions), "
            f"{len(graph.edges)} dependencies, "
            f"{len(chunks)} supporting document chunk(s)._\n"
        )

    def _build_components_section(self, graph: GraphData) -> str:
        """
        Build the '## Components' section: one heading per major component
        (file grouping) plus the required overview table.
        """
        lines: list[str] = ["## Components\n"]

        # Overview table — required by roadmap: "component overview table"
        lines.append("| Component | Type | File | Description |")
        lines.append("|---|---|---|---|")

        nodes_for_table = graph.nodes[: self._max_nodes_in_table]
        for node in nodes_for_table:
            description = self._describe_node(node)
            lines.append(
                f"| {self._escape_md(node.symbol)} "
                f"| {node.kind} "
                f"| `{self._escape_md(node.file)}` "
                f"| {self._escape_md(description)} |"
            )

        if len(graph.nodes) > self._max_nodes_in_table:
            lines.append(
                f"\n_... and {len(graph.nodes) - self._max_nodes_in_table} "
                f"more component(s), truncated for readability._"
            )

        lines.append("")  # blank line before headings

        # Required: "at least one heading per major component"
        # We group by file and emit one heading per file (a "major
        # component" in this codebase = a source file's public symbols),
        # which scales sanely even for large graphs while satisfying the
        # acceptance criterion for any nonempty graph.
        by_file: dict[str, list[GraphNodeData]] = {}
        for node in nodes_for_table:
            by_file.setdefault(node.file, []).append(node)

        for file_path, file_nodes in by_file.items():
            lines.append(f"### {file_path}\n")
            kinds = ", ".join(sorted({n.kind for n in file_nodes}))
            symbols = ", ".join(self._escape_md(n.symbol) for n in file_nodes[:10])
            more = len(file_nodes) - 10
            suffix = f", and {more} more" if more > 0 else ""
            lines.append(
                f"Contains {len(file_nodes)} {kinds} symbol(s): {symbols}{suffix}.\n"
            )

        return "\n".join(lines)

    @staticmethod
    def _describe_node(node: GraphNodeData) -> str:
        """Generate a short human-readable description for a table row."""
        kind_label = "Class" if node.kind == "class" else "Function"
        loc = (
            f"lines {node.start_line}-{node.end_line}"
            if node.start_line and node.end_line
            else "location unknown"
        )
        return f"{kind_label} defined at {loc} in {node.language}."

    def _build_dependencies_section(self, graph: GraphData) -> str:
        """
        Build the '## Dependencies' section as a bullet list of
        component → component edges, matching the roadmap's example:
            - component A → component B
        """
        lines: list[str] = ["## Dependencies\n"]

        if not graph.edges:
            lines.append("_No dependency edges recorded for this graph._\n")
            return "\n".join(lines)

        node_by_id = {n.id: n for n in graph.nodes}

        seen: set[tuple[str, str, str]] = set()
        for edge in graph.edges:
            src = node_by_id.get(edge.source_id)
            tgt = node_by_id.get(edge.target_id)
            src_label = src.symbol if src else edge.source_id[:12]
            tgt_label = tgt.symbol if tgt else edge.target_id[:12]
            key = (src_label, tgt_label, edge.edge_type)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {self._escape_md(src_label)} "
                f"→ {self._escape_md(tgt_label)} "
                f"_(`{edge.edge_type}`)_"
            )

        lines.append("")
        return "\n".join(lines)

    def _build_narrative_section(
        self,
        graph: GraphData,
        chunks: list[RetrievedChunk],
    ) -> str:
        """
        Build a 'Human-readable explanations' narrative section.

        Calls the injected LLM if available; otherwise produces a
        deterministic template summary so a report is always generated
        even with no chat model configured.
        """
        if self._llm_call is None:
            return self._template_narrative(graph, chunks)

        sys_prompt = (
            "You are a senior software architect writing the narrative "
            "summary of an architecture report. Given a list of components "
            "and their dependencies, plus supporting document excerpts, "
            "write 2-4 short paragraphs in Markdown explaining the system's "
            "structure in plain English. Do not repeat the raw table data "
            "verbatim — synthesize an explanation. Do not use a top-level "
            "heading (the caller adds its own)."
        )
        user_prompt = self._build_narrative_prompt(graph, chunks)

        try:
            narrative = self._llm_call(sys_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "ArchReportCore: LLM narrative call failed (%s) — falling "
                "back to template narrative",
                exc,
            )
            return self._template_narrative(graph, chunks)

        narrative = (narrative or "").strip()
        if not narrative:
            return self._template_narrative(graph, chunks)

        return "## Summary\n\n" + narrative + "\n"

    @staticmethod
    def _build_narrative_prompt(
        graph: GraphData,
        chunks: list[RetrievedChunk],
    ) -> str:
        lines = ["Components:"]
        for n in graph.nodes[:50]:
            lines.append(f"- {n.symbol} ({n.kind}) in {n.file}")

        lines.append("\nDependencies:")
        node_by_id = {n.id: n for n in graph.nodes}
        for e in graph.edges[:50]:
            src = node_by_id.get(e.source_id)
            tgt = node_by_id.get(e.target_id)
            lines.append(
                f"- {src.symbol if src else e.source_id} "
                f"-> {tgt.symbol if tgt else e.target_id} ({e.edge_type})"
            )

        if chunks:
            lines.append("\nSupporting context:")
            for c in chunks[:10]:
                snippet = (c.content or "").strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "…"
                lines.append(f"- {snippet}")

        return "\n".join(lines)

    @staticmethod
    def _template_narrative(
        graph: GraphData,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Deterministic fallback narrative when no LLM is available."""
        files = sorted({n.file for n in graph.nodes})
        files_preview = ", ".join(f"`{f}`" for f in files[:5])
        more = len(files) - 5
        suffix = f", and {more} more file(s)" if more > 0 else ""

        lines = [
            "## Summary\n",
            (
                f"This architecture graph spans {len(files)} file(s), "
                f"including {files_preview}{suffix}. "
                f"It contains {len(graph.nodes)} symbol(s) connected by "
                f"{len(graph.edges)} dependency edge(s)."
            ),
        ]
        if chunks:
            lines.append(
                f"\n{len(chunks)} supporting document chunk(s) were "
                f"retrieved alongside this graph but a narrative model was "
                f"not available to synthesise them into prose."
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _escape_md(text: str) -> str:
        """Escape pipe characters so table cells don't break Markdown tables."""
        return (text or "").replace("|", "\\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# RAGFlow ComponentBase adapter
# ---------------------------------------------------------------------------
# Imported lazily inside the class body's methods (not at module level) so
# that importing this file never requires a live RAGFlow environment.
# This mirrors the lazy-import pattern used in
# agent/component/codeatlas/reading_order.py (Phase 11) and
# codeatlas/analysis/arch_graph.py (Phase 8).
# ---------------------------------------------------------------------------

def _lazy_base_classes():
    """Import agent.component.base lazily; raise a clear error if unavailable."""
    from agent.component.base import ComponentBase, ComponentParamBase  # noqa: PLC0415
    return ComponentBase, ComponentParamBase


try:
    _ComponentBase, _ComponentParamBase = _lazy_base_classes()
    _BASE_AVAILABLE = True
except ImportError:
    # RAGFlow agent runtime not installed — define minimal stand-ins so this
    # module still imports cleanly (e.g. in isolated unit tests).
    _BASE_AVAILABLE = False

    class _ComponentParamBase:  # type: ignore[no-redef]
        def __init__(self):
            self.outputs = {}
            self.inputs = {}

        def check(self):
            pass

    class _ComponentBase:  # type: ignore[no-redef]
        def __init__(self, canvas, id, param):
            self._canvas = canvas
            self._id = id
            self._param = param


class ArchReportParam(_ComponentParamBase):
    """
    Parameters for ArchReportComponent.

    Attributes
    ----------
    llm_id : str
        Chat model identifier, resolved the same way as agent/component/llm.py.
        May be left empty to force the deterministic template-only narrative
        (no LLM call at all).
    kb_id : str
        Knowledge base ID whose architecture graph should be reported on.
    query : str
        Optional search query used to retrieve supporting chunks via
        Phase 5's CodeSearchReranker.  If empty, no chunk retrieval is
        attempted and the report is built from graph data alone.
    top_n_chunks : int
        Maximum number of supporting chunks to retrieve. Defaults to 10.
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm_id = ""
        self.kb_id = ""
        self.query = ""
        self.top_n_chunks = 10

    def check(self):
        if self.top_n_chunks < 0:
            raise ValueError("[ArchReport] top_n_chunks cannot be negative")


class ArchReportComponent(_ComponentBase):
    """
    RAGFlow agent component: Architecture Report.

    Reads the Phase 9 architecture graph API for ``param.kb_id``, optionally
    retrieves supporting chunks via Phase 5 search for ``param.query``, and
    sets the output variable ``report_markdown`` to a Markdown string
    matching the roadmap's required structure:

        # Architecture Report
        ## Components
        | Component | Type | File | Description |
        ## Dependencies
        - component A -> component B

    This class is intentionally a thin adapter: all report-building logic
    lives in ``ArchReportCore``, which is independently unit-testable.
    """

    component_name = "ArchReport"

    def __init__(self, canvas, component_id, param: ArchReportParam) -> None:
        if not _BASE_AVAILABLE:
            raise ImportError(
                "ArchReportComponent requires the RAGFlow agent runtime "
                "(agent.component.base). It cannot be instantiated outside "
                "a live RAGFlow environment. Use ArchReportCore directly "
                "for unit testing."
            )
        super().__init__(canvas, component_id, param)
        self._core: Optional[ArchReportCore] = None  # built lazily in _invoke

    # ------------------------------------------------------------------
    # RAGFlow lifecycle
    # ------------------------------------------------------------------

    def _invoke(self, **kwargs) -> None:
        """
        Synchronous invocation entry point used by ComponentBase.invoke().

        Fetches the architecture graph (Phase 9) and supporting chunks
        (Phase 5), builds an ArchReportCore bound to the live chat model,
        and writes the resulting Markdown to the ``report_markdown`` output.
        """
        graph = self._fetch_graph()
        chunks = self._fetch_chunks()

        core = self._get_core()
        report_markdown = core.generate_report(graph, chunks)
        self.set_output("report_markdown", report_markdown)

    # ------------------------------------------------------------------
    # Internal helpers — live RAGFlow integration points
    # ------------------------------------------------------------------

    def _get_core(self) -> ArchReportCore:
        """Build (and cache) an ArchReportCore bound to the live chat model."""
        if self._core is not None:
            return self._core

        llm_call = None
        if self._param.llm_id:
            try:
                from common.constants import LLMType  # noqa: PLC0415
                from api.db.services.llm_service import LLMBundle  # noqa: PLC0415
                from api.db.joint_services.tenant_model_service import (  # noqa: PLC0415
                    get_model_config_from_provider_instance,
                )

                tenant_id = self._canvas.get_tenant_id()
                chat_model_config = get_model_config_from_provider_instance(
                    tenant_id, LLMType.CHAT, self._param.llm_id
                )
                chat_mdl = LLMBundle(tenant_id, chat_model_config)

                def llm_call(sys_prompt: str, user_prompt: str) -> str:  # noqa: PLW0640
                    return chat_mdl.chat(
                        sys_prompt, [{"role": "user", "content": user_prompt}], {}
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "ArchReportComponent: could not build LLM bundle (%s) — "
                    "falling back to template-only narrative",
                    exc,
                )
                llm_call = None

        self._core = ArchReportCore(llm_call=llm_call)
        return self._core

    def _fetch_graph(self) -> GraphData:
        """
        Fetch the architecture graph for ``param.kb_id`` directly from the
        Phase 7 ORM models (same data source the Phase 9 API route reads
        from), avoiding an internal HTTP round-trip.

        Returns an empty GraphData (never raises) if the KB has no graph
        data yet or the query fails.
        """
        if not self._param.kb_id:
            _log.info("ArchReportComponent: no kb_id configured — empty graph")
            return GraphData()

        try:
            from api.db.db_models import ArchGraphNode, ArchGraphEdge  # noqa: PLC0415

            node_rows = list(
                ArchGraphNode.select().where(ArchGraphNode.kb_id == self._param.kb_id)
            )
            edge_rows = list(
                ArchGraphEdge.select().where(ArchGraphEdge.kb_id == self._param.kb_id)
            )

            api_shape = {
                "nodes": [
                    {
                        "id": n.id, "kb_id": n.kb_id, "symbol": n.symbol,
                        "kind": n.kind, "file": n.file,
                        "start_line": n.start_line, "end_line": n.end_line,
                        "language": n.language,
                    }
                    for n in node_rows
                ],
                "edges": [
                    {
                        "id": e.id, "source_id": e.source_id,
                        "target_id": e.target_id, "edge_type": e.edge_type,
                        "kb_id": e.kb_id,
                    }
                    for e in edge_rows
                ],
            }
            return GraphData.from_api_response(api_shape)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "ArchReportComponent: graph fetch failed (%s) — "
                "returning empty graph rather than crashing",
                exc,
            )
            return GraphData()

    def _fetch_chunks(self) -> list[RetrievedChunk]:
        """
        Retrieve supporting document/code chunks for ``param.query`` using
        Phase 5's CodeSearchReranker on top of the canvas's retriever, if
        both a query and a retriever are available.

        Returns [] (never raises) if no query is configured or retrieval
        fails for any reason.
        """
        if not self._param.query or not self._param.kb_id:
            return []

        try:
            from rag.retrieval.code_search import CodeSearchReranker  # noqa: PLC0415

            retriever = getattr(self._canvas, "get_retriever", None)
            if not callable(retriever):
                _log.info(
                    "ArchReportComponent: canvas has no retriever — "
                    "skipping chunk retrieval"
                )
                return []

            raw_chunks = retriever()  # best-effort: canvas-specific hook
            if not raw_chunks:
                return []

            reranker = CodeSearchReranker()
            ranked = reranker.rerank(raw_chunks, query=self._param.query)

            chunks: list[RetrievedChunk] = []
            for c in ranked[: self._param.top_n_chunks]:
                chunks.append(RetrievedChunk(
                    content=str(c.get("content_with_weight", "")),
                    symbol=str(c.get("symbol_kwd", "")),
                    source_type=str(c.get("source_type_kwd", "")),
                    docnm=str(c.get("docnm_kwd", "")),
                ))
            return chunks
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "ArchReportComponent: chunk retrieval failed (%s) — "
                "continuing with graph-only report",
                exc,
            )
            return []

    def thoughts(self) -> str:
        return "Synthesizing the architecture report from the graph..."
