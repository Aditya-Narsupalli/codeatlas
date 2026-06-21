# agent/component/codeatlas/reading_order.py
#
# CodeAtlas — Phase 11: Suggested Reading Order Agent
# ---------------------------------------------------------------------------
# First agent component in the codeatlas namespace.  Retrieves candidate
# documents/chunks from a knowledge base, prompts an LLM to sort them by
# conceptual dependency, and returns an ordered list with a rationale for
# each item.
#
# Architecture
# ------------
# This module is split into two layers:
#
#   1. ReadingOrderCore   — pure orchestration logic.  No dependency on
#      agent.canvas.Graph, no LLMBundle construction, no DB connection at
#      import time.  Fully unit-testable by injecting a callable LLM
#      function and a list of candidate documents directly.
#
#   2. ReadingOrderComponent / ReadingOrderParam — the thin RAGFlow
#      ComponentBase adapter that the agent canvas runner actually
#      instantiates.  It wires ReadingOrderCore to the live LLMBundle and
#      canvas-provided KB/document list, following the same pattern as
#      agent/component/categorize.py (which extends agent/component/llm.py).
#
# This split exists because ComponentBase.__init__ requires a live
# agent.canvas.Graph instance and LLM.__init__ makes live calls to
# get_model_type_by_name() / LLMBundle() — neither of which should run
# during import or in a unit test.  ReadingOrderCore has zero such
# dependencies, matching the roadmap's framing: "Pure orchestration — no
# new infrastructure at risk."
#
# Scope (Phase 11 only)
# ----------------------
# - ReadingOrderComponent callable from RAGFlow's agent runner
# - Ordered list output with per-item rationale
# - Graceful, no-crash handling of an empty KB (returns [], never fabricates
#   documents)
# - No UI, no API endpoints, no DB tables, no FACTORY/task_executor wiring
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from codeatlas.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandidateDocument:
    """
    One candidate document/chunk to be ordered.

    Attributes
    ----------
    doc_id : str
        Stable identifier (chunk_id, doc_id, or symbol node id).
    title : str
        Display title — filename, symbol name, or document name.
    summary : str
        Short text snippet used to build the LLM prompt.  Truncated
        upstream by the caller if very long; this component does not
        re-truncate.
    """
    doc_id: str
    title: str
    summary: str = ""


@dataclass
class ReadingOrderItem:
    """
    One entry in the ordered reading list.

    Matches the roadmap's example output structure:
        {"order": 1, "document": "...", "reason": "..."}
    """
    order: int
    document: str
    reason: str
    doc_id: str = ""

    def to_dict(self) -> dict:
        """Serialize to the exact roadmap output shape."""
        return {
            "order": self.order,
            "document": self.document,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Core orchestration logic — no canvas, no live LLM construction
# ---------------------------------------------------------------------------

class ReadingOrderCore:
    """
    Pure orchestration core for the Suggested Reading Order agent.

    Has no dependency on agent.canvas.Graph, ComponentBase, or LLMBundle.
    The LLM call is injected as a plain callable, so this class is fully
    unit-testable with a mock LLM function and an in-memory document list.

    Usage
    -----
    ::

        core = ReadingOrderCore(llm_call=my_chat_function)
        items = core.order_documents(candidates)
        for item in items:
            print(item.order, item.document, item.reason)

    Parameters
    ----------
    llm_call : Callable[[str, str], str]
        A function ``(sys_prompt: str, user_prompt: str) -> str`` that
        invokes the chat model and returns its raw text response.
        The component adapter wires this to ``LLMBundle.chat()``.
    max_candidates : int
        Safety cap on how many documents are sent to the LLM in one prompt.
        Defaults to 30 — well above the 5-document acceptance scenario,
        while bounding prompt size for very large KBs.
    """

    DEFAULT_SYS_PROMPT = (
        "You are an expert technical writer who helps engineers decide the "
        "best order to read a set of documents so that each document only "
        "depends on concepts already introduced in earlier ones.\n\n"
        "You will be given a numbered list of candidate documents, each "
        "with a short summary. Decide the optimal reading order based on "
        "conceptual dependency: foundational or prerequisite material first, "
        "more specific or derived material later.\n\n"
        "Respond with ONLY a JSON array, no other text. Each element must "
        "be an object with exactly these keys:\n"
        '  "doc_id": the original document id, copied exactly as given\n'
        '  "reason": one sentence explaining why this document belongs at '
        "this point in the sequence\n\n"
        "The array order IS the reading order — element 0 should be read "
        "first. Include every candidate document exactly once."
    )

    def __init__(
        self,
        llm_call: Callable[[str, str], str],
        max_candidates: int = 30,
    ) -> None:
        self._llm_call = llm_call
        self._max_candidates = max_candidates

    def order_documents(
        self,
        candidates: list[CandidateDocument],
    ) -> list[ReadingOrderItem]:
        """
        Return *candidates* sorted by conceptual dependency, with rationale.

        Parameters
        ----------
        candidates : list[CandidateDocument]
            Candidate documents retrieved from the KB.  An empty list
            returns an empty result with no LLM call and no error.

        Returns
        -------
        list[ReadingOrderItem]
            One item per candidate, ``order`` starting at 1, in the
            LLM-suggested reading sequence.  If the LLM response cannot be
            parsed, falls back to the original candidate order with a
            generic rationale rather than raising.
        """
        if not candidates:
            _log.info("ReadingOrderCore: no candidate documents — returning []")
            return []

        capped = candidates[: self._max_candidates]
        if len(capped) < len(candidates):
            _log.warning(
                "ReadingOrderCore: %d candidates exceeds max_candidates=%d, "
                "truncating",
                len(candidates), self._max_candidates,
            )

        user_prompt = self._build_user_prompt(capped)

        try:
            raw_response = self._llm_call(self.DEFAULT_SYS_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "ReadingOrderCore: LLM call failed (%s) — falling back to "
                "input order",
                exc,
            )
            return self._fallback_order(capped)

        parsed = self._parse_llm_response(raw_response, capped)
        if parsed is None:
            _log.warning(
                "ReadingOrderCore: could not parse LLM response — falling "
                "back to input order"
            )
            return self._fallback_order(capped)

        return parsed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(candidates: list[CandidateDocument]) -> str:
        """Build the user-turn prompt listing all candidate documents."""
        lines = ["Candidate documents:\n"]
        for i, doc in enumerate(candidates, start=1):
            summary = doc.summary.strip() or "(no summary available)"
            # Keep each summary reasonably short in the prompt
            if len(summary) > 400:
                summary = summary[:400] + "…"
            lines.append(
                f'{i}. doc_id="{doc.doc_id}"  title="{doc.title}"\n'
                f"   summary: {summary}\n"
            )
        lines.append(
            "\nReturn the JSON array now, ordering ALL of the above "
            "documents by conceptual dependency."
        )
        return "\n".join(lines)

    def _parse_llm_response(
        self,
        raw_response: str,
        candidates: list[CandidateDocument],
    ) -> Optional[list[ReadingOrderItem]]:
        """
        Parse the LLM's JSON array response into ReadingOrderItem objects.

        Returns None if parsing fails or the response is structurally
        invalid (caller falls back to input order in that case).
        """
        json_text = self._extract_json_array(raw_response)
        if json_text is None:
            return None

        try:
            entries = json.loads(json_text)
        except (json.JSONDecodeError, TypeError):
            _log.debug("ReadingOrderCore: JSON decode failed for: %r", json_text[:200])
            return None

        if not isinstance(entries, list) or not entries:
            return None

        by_id = {doc.doc_id: doc for doc in candidates}
        items: list[ReadingOrderItem] = []
        seen_ids: set[str] = set()

        for idx, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            doc_id = str(entry.get("doc_id", "")).strip()
            reason = str(entry.get("reason", "")).strip() or "No rationale provided."

            doc = by_id.get(doc_id)
            if doc is None:
                # LLM hallucinated or mangled an id — skip rather than
                # fabricate a fake document
                _log.debug(
                    "ReadingOrderCore: LLM returned unknown doc_id %r, skipping",
                    doc_id,
                )
                continue

            seen_ids.add(doc_id)
            items.append(
                ReadingOrderItem(
                    order=len(items) + 1,
                    document=doc.title,
                    reason=reason,
                    doc_id=doc_id,
                )
            )

        # Append any candidates the LLM omitted, preserving their original
        # relative order, so no real document is silently dropped.
        for doc in candidates:
            if doc.doc_id not in seen_ids:
                items.append(
                    ReadingOrderItem(
                        order=len(items) + 1,
                        document=doc.title,
                        reason="Included for completeness; not ranked by the model.",
                        doc_id=doc.doc_id,
                    )
                )

        return items if items else None

    @staticmethod
    def _extract_json_array(text: str) -> Optional[str]:
        """
        Extract a JSON array substring from *text*, tolerating markdown
        code fences or leading/trailing prose the LLM might add despite
        instructions.
        """
        if not text:
            return None
        # Strip ```json ... ``` or ``` ... ``` fences if present
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)
        # Otherwise find the first '[' to the matching last ']'
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return None

    @staticmethod
    def _fallback_order(candidates: list[CandidateDocument]) -> list[ReadingOrderItem]:
        """
        Deterministic fallback: preserve input order, generic rationale.
        Used when the LLM call fails or its response can't be parsed.
        Never fabricates documents — only re-labels the real candidates.
        """
        return [
            ReadingOrderItem(
                order=i,
                document=doc.title,
                reason=(
                    "Reading order could not be determined by the model; "
                    "shown in retrieval order."
                ),
                doc_id=doc.doc_id,
            )
            for i, doc in enumerate(candidates, start=1)
        ]


# ---------------------------------------------------------------------------
# RAGFlow ComponentBase adapter
# ---------------------------------------------------------------------------
# Imported lazily inside the class body's methods (not at module level) so
# that importing this file never requires a live RAGFlow environment.
# This mirrors the lazy-import pattern used in codeatlas/analysis/arch_graph.py.
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


class ReadingOrderParam(_ComponentParamBase):
    """
    Parameters for ReadingOrderComponent.

    Attributes
    ----------
    llm_id : str
        Chat model identifier, resolved the same way as agent/component/llm.py.
    top_n : int
        Maximum number of candidate documents to retrieve from the KB.
        Defaults to 10.
    kb_ids : list[str]
        Knowledge base IDs to draw candidates from. Empty list means
        "use the canvas's configured KB context" (resolved at invoke time).
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm_id = ""
        self.top_n = 10
        self.kb_ids: list[str] = []

    def check(self):
        if self.top_n <= 0:
            raise ValueError("[ReadingOrder] top_n must be positive")


class ReadingOrderComponent(_ComponentBase):
    """
    RAGFlow agent component: Suggested Reading Order.

    Retrieves up to ``param.top_n`` candidate documents from the configured
    knowledge base(s), asks the LLM to order them by conceptual dependency,
    and sets the output variable ``reading_order`` to a JSON-serializable
    list matching the roadmap's example structure::

        [{"order": 1, "document": "...", "reason": "..."}, ...]

    This class is intentionally a thin adapter: all ordering logic lives in
    ``ReadingOrderCore``, which is independently unit-testable.
    """

    component_name = "ReadingOrder"

    def __init__(self, canvas, component_id, param: ReadingOrderParam) -> None:
        if not _BASE_AVAILABLE:
            raise ImportError(
                "ReadingOrderComponent requires the RAGFlow agent runtime "
                "(agent.component.base). It cannot be instantiated outside "
                "a live RAGFlow environment. Use ReadingOrderCore directly "
                "for unit testing."
            )
        super().__init__(canvas, component_id, param)
        self._core: Optional[ReadingOrderCore] = None  # built lazily in _invoke

    # ------------------------------------------------------------------
    # RAGFlow lifecycle
    # ------------------------------------------------------------------

    def _invoke(self, **kwargs) -> None:
        """
        Synchronous invocation entry point used by ComponentBase.invoke().

        Retrieves candidate documents from the canvas's KB context, builds
        a ReadingOrderCore bound to the live chat model, and writes the
        ordered list to the ``reading_order`` output.
        """
        candidates = self._fetch_candidates()

        if not candidates:
            _log.info(
                "ReadingOrderComponent: empty KB / no candidates — "
                "returning empty reading order"
            )
            self.set_output("reading_order", [])
            return

        core = self._get_core()
        items = core.order_documents(candidates)
        self.set_output("reading_order", [item.to_dict() for item in items])

    # ------------------------------------------------------------------
    # Internal helpers — live RAGFlow integration points
    # ------------------------------------------------------------------

    def _get_core(self) -> ReadingOrderCore:
        """Build (and cache) a ReadingOrderCore bound to the live chat model."""
        if self._core is not None:
            return self._core

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

        def llm_call(sys_prompt: str, user_prompt: str) -> str:
            return chat_mdl.chat(
                sys_prompt, [{"role": "user", "content": user_prompt}], {}
            )

        self._core = ReadingOrderCore(llm_call=llm_call)
        return self._core

    def _fetch_candidates(self) -> list[CandidateDocument]:
        """
        Retrieve candidate documents from the configured knowledge base(s).

        Uses the canvas's retrieval helpers if available. Returns an empty
        list (never fabricated documents) if the KB has no chunks or the
        retrieval call fails.
        """
        kb_ids = self._param.kb_ids or self._get_canvas_kb_ids()
        if not kb_ids:
            _log.info("ReadingOrderComponent: no kb_ids configured — empty candidates")
            return []

        try:
            from api.db.services.knowledgebase_service import KnowledgebaseService  # noqa: PLC0415
            from api.db.services.document_service import DocumentService  # noqa: PLC0415

            candidates: list[CandidateDocument] = []
            for kb_id in kb_ids:
                docs, _total = DocumentService.get_by_kb_id(
                    kb_id, 1, self._param.top_n, "create_time", True, "", [], []
                )
                for d in docs:
                    candidates.append(
                        CandidateDocument(
                            doc_id=str(d.get("id", "")),
                            title=str(d.get("name", "untitled")),
                            summary=str(d.get("description", "") or ""),
                        )
                    )
            return candidates[: self._param.top_n]
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "ReadingOrderComponent: candidate retrieval failed (%s) — "
                "returning empty list rather than crashing",
                exc,
            )
            return []

    def _get_canvas_kb_ids(self) -> list[str]:
        """Best-effort lookup of KB ids from the canvas, if exposed."""
        try:
            getter = getattr(self._canvas, "get_kb_ids", None)
            if callable(getter):
                return list(getter() or [])
        except Exception:  # noqa: BLE001
            pass
        return []

    def thoughts(self) -> str:
        return "Working out the best order to read these documents..."
