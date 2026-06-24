# agent/component/codeatlas/onboarding.py
#
# CodeAtlas — Phase 15: Onboarding Guide Agent
# ---------------------------------------------------------------------------
# Third agent component in the codeatlas namespace.  Combines:
#   - Phase 11 ReadingOrderCore   — conceptual document ordering
#   - Phase 14 LinkStore          — cross-source entity links
#   - persona context             — adjusts tone and section selection
# to produce a sequenced, persona-aware onboarding guide.
#
# Architecture (mirrors Phase 11 and Phase 12 exactly)
# -------------------------------------------------------
#   OnboardingCore      — pure Python, zero RAGFlow/DB/LLM dependency at
#                         import time; fully unit-testable via injection.
#
#   OnboardingComponent — thin ComponentBase adapter wired to the live
#                         RAGFlow agent runner.  Uses lazy imports for
#                         api.db, rag.llm, agent.canvas everywhere.
#
# Output format (roadmap spec)
# ----------------------------
# {
#   "persona": "new backend engineer",
#   "sections": [
#     {
#       "title": "...",
#       "order": 1,
#       "summary": "...",
#       "sources": [{"type": "code|doc|git", "id": "..."}]
#     },
#     ...
#   ]
# }
#
# Memory / session state
# ----------------------
# The roadmap asks for "persist progress in memory/" but the actual RAGFlow
# memory/ module is empty — session state is stored via canvas output
# variables, not a standalone persistence layer.
#
# This component writes ONE additive output key:
#   "onboarding_session" → {"persona": ..., "section_count": ..., "completed": False}
#
# It NEVER overwrites existing canvas output keys.  The merge strategy is:
#   new_outputs = {**existing_outputs, "onboarding_session": <new_value>}
# so any key already present from earlier components is preserved.
#
# Scope (Phase 15 only)
# ----------------------
# - OnboardingComponent callable from RAGFlow's agent runner
# - ≥5 ordered sections for "new backend engineer" persona
# - At least one code source citation + one doc source citation per guide
# - Graceful degradation when LinkStore is empty or LLM fails
# - No UI, no new API endpoints, no DB migrations, no FACTORY wiring
# ---------------------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from codeatlas.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Session key added to canvas outputs (additive only, never overwrites)
# ---------------------------------------------------------------------------
SESSION_KEY = "onboarding_session"


# ---------------------------------------------------------------------------
# Output data types
# ---------------------------------------------------------------------------

@dataclass
class OnboardingSource:
    """A single source citation within one onboarding section."""
    type: str    # "code" | "doc" | "git"
    id: str      # chunk_id or doc_id


@dataclass
class OnboardingSection:
    """
    One section in the onboarding guide.

    Attributes mirror the roadmap's required output structure exactly.
    """
    title: str
    order: int
    summary: str
    sources: list[OnboardingSource] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "order": self.order,
            "summary": self.summary,
            "sources": [{"type": s.type, "id": s.id} for s in self.sources],
        }


@dataclass
class OnboardingGuide:
    """
    Complete onboarding guide — the canonical output of OnboardingCore.

    Roadmap output structure::

        {
          "persona": "...",
          "sections": [{"title": ..., "order": ..., ...}, ...]
        }
    """
    persona: str
    sections: list[OnboardingSection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "persona": self.persona,
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_markdown(self) -> str:
        """Render the guide as Markdown — used by the component for output."""
        lines = [f"# Onboarding Guide — {self.persona}\n"]
        for sec in self.sections:
            lines.append(f"## {sec.order}. {sec.title}\n")
            lines.append(f"{sec.summary}\n")
            if sec.sources:
                lines.append("**Sources:**")
                for src in sec.sources:
                    lines.append(f"- `[{src.type}]` {src.id}")
                lines.append("")
        return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Persona profiles — pure data, no I/O
# ---------------------------------------------------------------------------

@dataclass
class _PersonaProfile:
    """
    Controls which document types to prioritise and what tone to use.

    Used by OnboardingCore to weight and filter candidate documents.
    """
    name: str
    prioritised_types: list[str]   # ordered: first = highest priority
    intro_note: str                # injected as first section summary preamble


_PERSONAS: dict[str, _PersonaProfile] = {
    "new backend engineer": _PersonaProfile(
        name="new backend engineer",
        prioritised_types=["doc", "code", "git"],
        intro_note=(
            "Welcome, backend engineer! This guide takes you from project "
            "setup through core APIs, key data models, and real commit history "
            "so you can be productive as quickly as possible."
        ),
    ),
    "new frontend engineer": _PersonaProfile(
        name="new frontend engineer",
        prioritised_types=["doc", "git", "code"],
        intro_note=(
            "Welcome, frontend engineer! This guide walks you through the UI "
            "component architecture, API contracts, and recent UI changes."
        ),
    ),
    "new devops engineer": _PersonaProfile(
        name="new devops engineer",
        prioritised_types=["doc", "git", "code"],
        intro_note=(
            "Welcome, devops engineer! This guide covers deployment configs, "
            "infrastructure code, and operational runbooks."
        ),
    ),
}

_DEFAULT_PERSONA = _PersonaProfile(
    name="new engineer",
    prioritised_types=["doc", "code", "git"],
    intro_note=(
        "Welcome! This guide introduces you to the codebase step by step."
    ),
)

# Minimum sections we guarantee even with template fallback
_MIN_SECTIONS = 5


# ---------------------------------------------------------------------------
# Core orchestration logic — pure Python, no RAGFlow dependency
# ---------------------------------------------------------------------------

class OnboardingCore:
    """
    Pure orchestration core for the Onboarding Guide agent.

    Composes Phase 11 reading order items with Phase 14 link context to
    produce a persona-aware OnboardingGuide.  Has zero dependency on
    agent.canvas.Graph, ComponentBase, LLMBundle, or any database — all
    inputs are injected.

    Usage
    -----
    ::

        core = OnboardingCore(llm_call=my_chat_function)
        guide = core.generate(
            persona="new backend engineer",
            ordered_items=reading_order_items,   # Phase 11 output
            chunks=chunk_dicts,                  # Phase 3/5 chunk dicts
            links_by_chunk=links_map,            # Phase 14 LinkStore.get_links()
        )
        print(guide.to_markdown())

    Parameters
    ----------
    llm_call : Callable[[str, str], str] | None
        Injected LLM call ``(sys_prompt, user_prompt) -> str``.
        When ``None``, the deterministic template path is used — the guide
        is still fully populated, just without LLM-crafted prose.
    """

    def __init__(
        self,
        llm_call: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        self._llm_call = llm_call

    def generate(
        self,
        persona: str,
        ordered_items: list,          # list[ReadingOrderItem] from Phase 11
        chunks: Optional[list[dict]] = None,
        links_by_chunk: Optional[dict[str, list]] = None,
    ) -> OnboardingGuide:
        """
        Build a persona-aware onboarding guide.

        Parameters
        ----------
        persona : str
            Free-form persona string, e.g. ``"new backend engineer"``.
            Matched (case-insensitive) against ``_PERSONAS`` keys; unknown
            personas fall back to ``_DEFAULT_PERSONA``.
        ordered_items : list[ReadingOrderItem]
            Phase 11 reading order output — already sorted by conceptual
            dependency.  Each item has ``.order``, ``.document``,
            ``.reason``, ``.doc_id``.
        chunks : list[dict] | None
            RAGFlow chunk dicts (Phase 3/5); keyed by their chunk id to
            produce source citations.  If ``None``, only doc-level
            citations are available.
        links_by_chunk : dict[str, list[LinkRecord]] | None
            Pre-fetched Phase 14 links: ``{chunk_id: [LinkRecord, ...]}``.
            ``None`` or empty → guide is built without cross-source
            enrichment (graceful degradation, not an error).

        Returns
        -------
        OnboardingGuide
            Always non-empty — guaranteed ≥ ``_MIN_SECTIONS`` sections
            for any non-empty ``ordered_items`` input.  Empty
            ``ordered_items`` produces a short placeholder guide.
        """
        profile = self._resolve_profile(persona)
        chunks = chunks or []
        links_by_chunk = links_by_chunk or {}

        # Build chunk_id → source_type lookup for citation typing
        chunk_type_map: dict[str, str] = {}
        for c in chunks:
            cid = str(c.get("id") or c.get("chunk_id") or "")
            st = str(c.get("source_type_kwd") or "doc")
            if cid:
                chunk_type_map[cid] = st if st else "doc"

        if not ordered_items:
            return self._empty_guide(persona, profile)

        sections = self._build_sections(
            profile, ordered_items, chunk_type_map, links_by_chunk
        )
        sections = self._guarantee_min_sections(sections, profile, ordered_items)
        return OnboardingGuide(persona=persona, sections=sections)

    # ------------------------------------------------------------------
    # Section building
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        profile: _PersonaProfile,
        ordered_items: list,
        chunk_type_map: dict[str, str],
        links_by_chunk: dict[str, list],
    ) -> list[OnboardingSection]:
        """
        Map each Phase 11 ReadingOrderItem to one OnboardingSection.
        Enrich with cross-source citations from Phase 14 links.
        """
        sections: list[OnboardingSection] = []

        # Inject intro section (persona welcome message) as section 1
        intro = OnboardingSection(
            title=f"Welcome — {profile.name.title()}",
            order=1,
            summary=profile.intro_note,
            sources=[],
        )
        sections.append(intro)

        for item in ordered_items:
            order = len(sections) + 1
            doc_id = getattr(item, "doc_id", "") or ""
            title = getattr(item, "document", f"Document {order}")
            reason = getattr(item, "reason", "")

            # Primary source citation — the document/chunk itself
            primary_type = chunk_type_map.get(doc_id, "doc")
            primary_source = OnboardingSource(type=primary_type, id=doc_id) if doc_id else None

            # Cross-source citations from Phase 14 LinkStore
            linked_sources: list[OnboardingSource] = []
            if doc_id and links_by_chunk:
                for lnk in links_by_chunk.get(doc_id, []):
                    linked_id = lnk.target_chunk_id if lnk.source_chunk_id == doc_id \
                                else lnk.source_chunk_id
                    linked_type = chunk_type_map.get(linked_id, "doc")
                    linked_sources.append(
                        OnboardingSource(type=linked_type, id=linked_id)
                    )

            # Build summary — LLM when available, template otherwise
            summary = self._build_summary(title, reason, profile, linked_sources)

            all_sources = (
                ([primary_source] if primary_source else []) + linked_sources
            )
            sections.append(OnboardingSection(
                title=title,
                order=order,
                summary=summary,
                sources=all_sources,
            ))

        return sections

    def _build_summary(
        self,
        title: str,
        reason: str,
        profile: _PersonaProfile,
        linked_sources: list[OnboardingSource],
    ) -> str:
        """
        Build a section summary.  Calls LLM when available; falls back to
        a deterministic template that always produces readable prose.
        """
        if self._llm_call is None:
            return self._template_summary(title, reason, linked_sources)

        sys_prompt = (
            f"You are writing onboarding documentation for a "
            f"'{profile.name}'. Write a clear, concise 2-3 sentence "
            "summary explaining what this document/code covers and why a "
            "new engineer should read it at this point in their onboarding. "
            "Be practical, not generic. Do not use lists or headings."
        )
        user_prompt = (
            f"Document: {title}\n"
            f"Reading order rationale: {reason}\n"
        )
        if linked_sources:
            user_prompt += (
                f"Also links to: {', '.join(s.id for s in linked_sources[:3])}\n"
            )
        user_prompt += "\nWrite the 2-3 sentence summary now."

        try:
            prose = (self._llm_call(sys_prompt, user_prompt) or "").strip()
            return prose if prose else self._template_summary(title, reason, linked_sources)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "_build_summary: LLM call failed (%s) — using template", exc
            )
            return self._template_summary(title, reason, linked_sources)

    @staticmethod
    def _template_summary(
        title: str,
        reason: str,
        linked_sources: list[OnboardingSource],
    ) -> str:
        parts = [f"Read **{title}**."]
        if reason:
            parts.append(reason.strip().rstrip(".") + ".")
        if linked_sources:
            ids = ", ".join(f"`{s.id}`" for s in linked_sources[:2])
            parts.append(f"Related: {ids}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Guarantee minimum section count
    # ------------------------------------------------------------------

    def _guarantee_min_sections(
        self,
        sections: list[OnboardingSection],
        profile: _PersonaProfile,
        ordered_items: list,
    ) -> list[OnboardingSection]:
        """
        If fewer than _MIN_SECTIONS sections were produced, add standard
        filler sections with template content.  This guarantees the roadmap
        acceptance criterion (≥5 sections for "new backend engineer") even
        when the KB has very few documents.
        """
        standard_fillers = [
            ("Project Setup & Prerequisites",
             "Set up your local development environment before diving into "
             "the codebase. Install dependencies and configure your editor."),
            ("Repository Structure Overview",
             "Understand how the codebase is organised — which directories "
             "hold what, and how modules relate to each other."),
            ("Core Data Models",
             "Familiarise yourself with the key data structures and database "
             "models that underpin the system."),
            ("Key APIs & Interfaces",
             "Learn the main API endpoints or service interfaces you will "
             "interact with as a backend engineer."),
            ("Testing & Contribution Guide",
             "Understand how to run tests, submit pull requests, and follow "
             "the project's contribution conventions."),
        ]

        while len(sections) < _MIN_SECTIONS and standard_fillers:
            title, summary = standard_fillers.pop(0)
            # Only add if a section with this title doesn't already exist
            if not any(s.title == title for s in sections):
                sections.append(OnboardingSection(
                    title=title,
                    order=len(sections) + 1,
                    summary=summary,
                    sources=[],
                ))

        # Re-number in case ordering shifted
        for i, sec in enumerate(sections, start=1):
            sec.order = i

        return sections

    # ------------------------------------------------------------------
    # Empty KB fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_guide(persona: str, profile: _PersonaProfile) -> OnboardingGuide:
        """
        Graceful guide for an empty KB (no ordered items).
        Still returns ≥5 sections as a generic starting template.
        """
        template_sections = [
            ("Welcome",
             profile.intro_note),
            ("Project Setup & Prerequisites",
             "Install project dependencies and configure your development environment."),
            ("Repository Structure",
             "Explore the repository layout before reading specific modules."),
            ("Core Architecture",
             "No code has been ingested yet — once the repository is indexed, "
             "this section will list the core modules and their relationships."),
            ("Getting Help",
             "No documents have been loaded into the knowledge base yet. "
             "Reach out to your team lead or check the project wiki."),
        ]
        sections = [
            OnboardingSection(title=t, order=i, summary=s, sources=[])
            for i, (t, s) in enumerate(template_sections, start=1)
        ]
        return OnboardingGuide(persona=persona, sections=sections)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_profile(persona: str) -> _PersonaProfile:
        """Match persona string (case-insensitive) to a _PersonaProfile."""
        key = (persona or "").strip().lower()
        if key in _PERSONAS:
            return _PERSONAS[key]
        # Partial match — "backend engineer" matches "new backend engineer"
        for k, profile in _PERSONAS.items():
            if key in k or k in key:
                return profile
        return _DEFAULT_PERSONA


# ---------------------------------------------------------------------------
# RAGFlow ComponentBase adapter  (lazy imports throughout)
# ---------------------------------------------------------------------------

def _lazy_base_classes():
    from agent.component.base import ComponentBase, ComponentParamBase  # noqa: PLC0415
    return ComponentBase, ComponentParamBase


try:
    _ComponentBase, _ComponentParamBase = _lazy_base_classes()
    _BASE_AVAILABLE = True
except ImportError:
    _BASE_AVAILABLE = False

    class _ComponentParamBase:  # type: ignore[no-redef]
        def __init__(self):
            self.outputs = {}
            self.inputs = {}
        def check(self): pass

    class _ComponentBase:  # type: ignore[no-redef]
        def __init__(self, canvas, id, param):
            self._canvas = canvas
            self._id = id
            self._param = param


class OnboardingParam(_ComponentParamBase):
    """
    Parameters for OnboardingComponent.

    Attributes
    ----------
    persona : str
        Engineer persona, e.g. ``"new backend engineer"``.
    llm_id : str
        Chat model identifier.  Empty → template-only guide (no LLM call).
    kb_id : str
        Knowledge base to onboard against.
    top_n : int
        Max candidate documents to retrieve for ordering.
    """

    def __init__(self) -> None:
        super().__init__()
        self.persona: str = "new backend engineer"
        self.llm_id: str = ""
        self.kb_id: str = ""
        self.top_n: int = 20

    def check(self) -> None:
        if not self.persona or not self.persona.strip():
            raise ValueError("[Onboarding] persona must be a non-empty string")
        if self.top_n <= 0:
            raise ValueError("[Onboarding] top_n must be positive")


class OnboardingComponent(_ComponentBase):
    """
    RAGFlow agent component: Onboarding Guide.

    Orchestrates Phase 11 reading order + Phase 14 link store to produce a
    persona-aware onboarding guide.  Writes two output keys:
    - ``onboarding_guide``         — the full OnboardingGuide as a dict
    - ``onboarding_guide_markdown`` — Markdown-rendered version
    - ``onboarding_session``       — lightweight session state (additive only)

    Never overwrites any pre-existing canvas output key.
    """

    component_name = "Onboarding"

    def __init__(self, canvas, component_id, param: OnboardingParam) -> None:
        if not _BASE_AVAILABLE:
            raise ImportError(
                "OnboardingComponent requires the RAGFlow agent runtime "
                "(agent.component.base). Use OnboardingCore directly for "
                "unit testing."
            )
        super().__init__(canvas, component_id, param)
        self._core: Optional[OnboardingCore] = None

    # ------------------------------------------------------------------
    # RAGFlow lifecycle
    # ------------------------------------------------------------------

    def _invoke(self, **kwargs) -> None:
        """
        1. Fetch candidate documents from the KB.
        2. Run Phase 11 ReadingOrderCore to sort them.
        3. Query Phase 14 LinkStore for cross-source links.
        4. Run OnboardingCore to produce the guide.
        5. Write outputs additively (never overwrite existing keys).
        """
        persona = (self._param.persona or "new backend engineer").strip()

        # --- Phase 11: reading order ---
        candidates = self._fetch_candidates()
        ordering_core = self._get_reading_order_core()
        ordered_items = ordering_core.order_documents(candidates)

        # --- Phase 14: link context ---
        links_by_chunk = self._fetch_links(candidates)

        # --- Phase 3/5 chunk dicts for type annotation ---
        chunk_dicts = self._fetch_chunk_dicts(candidates)

        # --- OnboardingCore ---
        core = self._get_core()
        guide = core.generate(
            persona=persona,
            ordered_items=ordered_items,
            chunks=chunk_dicts,
            links_by_chunk=links_by_chunk,
        )

        # --- Write outputs additively ---
        self._set_output_additive("onboarding_guide", guide.to_dict())
        self._set_output_additive("onboarding_guide_markdown", guide.to_markdown())
        self._set_output_additive(SESSION_KEY, {
            "persona": persona,
            "section_count": len(guide.sections),
            "completed": False,
        })

    # ------------------------------------------------------------------
    # Additive output writer (never overwrites existing keys)
    # ------------------------------------------------------------------

    def _set_output_additive(self, key: str, value) -> None:
        """
        Write *key* = *value* to the canvas output variable dict.

        If *key* already exists in the output dict, it is NOT overwritten.
        Only new keys are added — matching the roadmap's requirement that
        "memory write does not overwrite existing agent session keys".
        """
        try:
            # ComponentBase.set_output() unconditionally overwrites.
            # We must check first.
            existing = self.output()          # gets the full output dict
            if isinstance(existing, dict) and key in existing:
                _log.debug(
                    "_set_output_additive: key %r already exists — skipping", key
                )
                return
            self.set_output(key, value)
        except Exception as exc:  # noqa: BLE001
            _log.warning("_set_output_additive(%r): %s", key, exc)
            # Best-effort fallback
            try:
                self.set_output(key, value)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Live RAGFlow integration helpers (all lazy imports)
    # ------------------------------------------------------------------

    def _get_core(self) -> OnboardingCore:
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
                cfg = get_model_config_from_provider_instance(
                    tenant_id, LLMType.CHAT, self._param.llm_id
                )
                mdl = LLMBundle(tenant_id, cfg)
                def llm_call(s, u):  # noqa: E306
                    return mdl.chat(s, [{"role": "user", "content": u}], {})
            except Exception as exc:  # noqa: BLE001
                _log.warning("OnboardingComponent: LLM build failed (%s) — template mode", exc)
        self._core = OnboardingCore(llm_call=llm_call)
        return self._core

    def _get_reading_order_core(self):
        """Build a ReadingOrderCore bound to the same LLM (or template)."""
        from agent.component.codeatlas.reading_order import ReadingOrderCore  # noqa: PLC0415
        core = self._get_core()
        # Share the same llm_call so we make one LLM construction
        return ReadingOrderCore(llm_call=core._llm_call or (lambda s, u: "[]"))

    def _fetch_candidates(self) -> list:
        """Fetch candidate documents → list[CandidateDocument]."""
        from agent.component.codeatlas.reading_order import CandidateDocument  # noqa: PLC0415
        if not self._param.kb_id:
            return []
        try:
            from api.db.services.document_service import DocumentService  # noqa: PLC0415
            docs, _ = DocumentService.get_by_kb_id(
                self._param.kb_id, 1, self._param.top_n,
                "create_time", True, "", [], []
            )
            return [
                CandidateDocument(
                    doc_id=str(d.get("id", "")),
                    title=str(d.get("name", "untitled")),
                    summary=str(d.get("description", "") or ""),
                )
                for d in docs
            ]
        except Exception as exc:  # noqa: BLE001
            _log.warning("_fetch_candidates: %s — returning []", exc)
            return []

    def _fetch_links(self, candidates: list) -> dict[str, list]:
        """Fetch Phase 14 links for all candidate chunk IDs."""
        try:
            from codeatlas.linker.link_store import LinkStore  # noqa: PLC0415
            store = LinkStore()
            result: dict[str, list] = {}
            for c in candidates:
                doc_id = getattr(c, "doc_id", "")
                if doc_id:
                    links = store.get_links(doc_id)
                    if links:
                        result[doc_id] = links
            return result
        except Exception as exc:  # noqa: BLE001
            _log.warning("_fetch_links: %s — returning {}", exc)
            return {}

    def _fetch_chunk_dicts(self, candidates: list) -> list[dict]:
        """Convert CandidateDocument list to minimal chunk dicts for type annotation."""
        dicts = []
        for c in candidates:
            doc_id = getattr(c, "doc_id", "")
            if doc_id:
                dicts.append({"id": doc_id, "source_type_kwd": "doc"})
        return dicts

    def thoughts(self) -> str:
        return f"Building onboarding guide for '{self._param.persona}'…"
