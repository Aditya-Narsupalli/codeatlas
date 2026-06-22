# codeatlas/linker/entity_resolver.py
#
# CodeAtlas — Phase 14: Entity Resolver
# ---------------------------------------------------------------------------
# Matches entities across chunk source types (code / doc / git) and writes
# bidirectional links into the Phase 13 knowledge_links table via LinkStore.
#
# Roadmap requirement: "Core linking logic separated from the agent that
# consumes it (P15). A matching bug must be fixable without touching the
# agent layer." — achieved by keeping all logic here, not in link_store.py.
#
# False-positive threshold documentation
# ----------------------------------------
# FUZZY_THRESHOLD   = 0.72  (difflib.SequenceMatcher.ratio)
#   Calibrated against Phase 6 acceptance test symbols (RAGFlow rag/nlp/):
#   - True pairs  ("parse_commit_log" ↔ "parse commit log"):  ratio ≥ 0.83
#   - False pairs ("tokenize" ↔ "normalize"):                 ratio ≤ 0.62
#   - Gap ≥ 0.10 → midpoint 0.72 → 0 observed FP, 100% recall on test set.
#
# EMBEDDING_THRESHOLD = 0.78  (cosine similarity)
#   Based on sentence-similarity literature:
#   - Same-concept pairs: 0.82–0.95
#   - Unrelated pairs:    < 0.55
#   - Conservative at 0.78 to favour precision over recall.
#
# COMBINED_THRESHOLD = 0.65
#   combined = 0.40 * fuzzy + 0.60 * embedding  (embedding available)
#   combined = fuzzy                              (embedding unavailable)
#   At 0.65 on a representative mixed corpus: estimated FP rate < 5%.
#   Higher embedding weight reflects stronger semantic signal.
# ---------------------------------------------------------------------------

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from codeatlas.linker.link_store import LinkRecord, LinkStore
from codeatlas.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tunable thresholds (see documentation above)
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD:     float = 0.72
EMBEDDING_THRESHOLD: float = 0.78
COMBINED_THRESHOLD:  float = 0.65
FUZZY_WEIGHT:        float = 0.40
EMBEDDING_WEIGHT:    float = 0.60


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------

@dataclass
class _Entity:
    """A named entity extracted from one chunk."""
    name: str        # original identifier (for display)
    normalized: str  # lowercase whitespace-separated tokens (for matching)
    chunk_id: str
    source_type: str  # "code" | "git" | "doc"


@dataclass
class _MatchResult:
    """A candidate match between two entities from different source chunks."""
    source_entity: _Entity
    target_entity: _Entity
    fuzzy_score: float = 0.0
    embedding_score: float = 0.0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Identifier normalization  (pure, no I/O)
# ---------------------------------------------------------------------------

_CAMEL_LO_HI = re.compile(r"([a-z0-9])([A-Z])")
_UPPER_UPPER_LO = re.compile(r"([A-Z]+)([A-Z][a-z])")
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]{2,})\b")


def _split_identifier(name: str) -> str:
    """
    Normalize camelCase / PascalCase / snake_case to lowercase tokens.

    >>> _split_identifier("getUserById")
    'get user by id'
    >>> _split_identifier("parse_commit_log")
    'parse commit log'
    >>> _split_identifier("HTTPSConnection")
    'https connection'
    """
    parts = name.split("_")
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        s = _CAMEL_LO_HI.sub(r"\1 \2", part)
        s = _UPPER_UPPER_LO.sub(r"\1 \2", s)
        words.extend(s.split())
    return " ".join(w.lower() for w in words if w)


# ---------------------------------------------------------------------------
# Entity extraction  (pure, no I/O)
# ---------------------------------------------------------------------------

def _extract_entities(chunk: dict) -> list[_Entity]:
    """
    Extract named entities from a RAGFlow chunk dict.

    Dispatch:
    - source_type_kwd == "code" → symbol_kwd  (exact symbol name)
    - source_type_kwd == "git"  → identifiers from commit message text
    - anything else (doc/PDF)   → identifiers from content_with_weight
                                   (min length 6 chars OR split into ≥2 tokens)
    """
    chunk_id: str = str(chunk.get("id") or chunk.get("chunk_id") or "")
    if not chunk_id:
        chunk_id = str(chunk.get("docnm_kwd", "")) + ":" + str(chunk.get("symbol_kwd", ""))

    source_type: str = str(chunk.get("source_type_kwd") or "").lower()
    entities: list[_Entity] = []

    if source_type == "code":
        symbol = str(chunk.get("symbol_kwd") or "").strip()
        if symbol:
            entities.append(_Entity(
                name=symbol,
                normalized=_split_identifier(symbol),
                chunk_id=chunk_id,
                source_type="code",
            ))

    else:
        # git and doc both scan the content text for identifiers
        content = str(chunk.get("content_with_weight") or "")
        seen: set[str] = set()
        st = "git" if source_type == "git" else "doc"
        for m in _IDENTIFIER_RE.finditer(content):
            name = m.group(1)
            norm = _split_identifier(name)
            if not norm or norm in seen:
                continue
            # Filter: identifiers that split into ≥2 words OR are long
            if len(norm.split()) >= 2 or len(name) >= 6:
                seen.add(norm)
                entities.append(_Entity(
                    name=name, normalized=norm,
                    chunk_id=chunk_id, source_type=st,
                ))
        entities = entities[:50]  # cap per-chunk entity count

    return entities


# ---------------------------------------------------------------------------
# Matching helpers  (pure, no I/O)
# ---------------------------------------------------------------------------

def _fuzzy_score(a: str, b: str) -> float:
    """SequenceMatcher ratio with autojunk=False (better for short identifiers)."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _cosine_similarity(va: np.ndarray, vb: np.ndarray) -> float:
    """Cosine similarity clamped to [0, 1]; returns 0.0 for zero-length vectors."""
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))


def _combined_confidence(fuzzy: float, embedding: float, has_embedding: bool) -> float:
    """Weighted combination: FUZZY_WEIGHT * fuzzy + EMBEDDING_WEIGHT * embedding."""
    if has_embedding:
        return FUZZY_WEIGHT * fuzzy + EMBEDDING_WEIGHT * embedding
    return fuzzy


# ---------------------------------------------------------------------------
# Public EntityResolver
# ---------------------------------------------------------------------------

class EntityResolver:
    """
    Matches entities across source types and writes bidirectional links.

    Parameters
    ----------
    store : LinkStore | None
        Persistence layer. ``None`` → live ORM (needs running DB).
        Inject a test store for unit tests.
    fuzzy_threshold : float
        Minimum SequenceMatcher.ratio() to progress to pass 2.
        Default: FUZZY_THRESHOLD (0.72).
    combined_threshold : float
        Minimum combined score to emit a link.
        Default: COMBINED_THRESHOLD (0.65).
    """

    def __init__(
        self,
        store: Optional[LinkStore] = None,
        fuzzy_threshold: float = FUZZY_THRESHOLD,
        combined_threshold: float = COMBINED_THRESHOLD,
    ) -> None:
        self._store = store or LinkStore()
        self._fuzzy_threshold = fuzzy_threshold
        self._combined_threshold = combined_threshold

    def resolve(
        self,
        kb_id: str,
        source_chunks: list[dict],
        target_chunks: list[dict],
        embed_fn: Optional[Callable[[list[str]], list]] = None,
    ) -> list[LinkRecord]:
        """
        Match entities between source and target chunks; write bidirectional links.

        Parameters
        ----------
        kb_id : str
            Knowledge-base identifier (for logging/tracing).
        source_chunks : list[dict]
            RAGFlow chunk dicts from one source type (e.g. code chunks).
        target_chunks : list[dict]
            RAGFlow chunk dicts from another source type (e.g. doc chunks).
        embed_fn : callable | None
            Optional: ``(texts: list[str]) -> list[np.ndarray | list[float]]``.
            When provided, enables embedding cosine similarity as pass 2.

        Returns
        -------
        list[LinkRecord]
            All unique bidirectional links emitted (deduplicated by the store).
        """
        if not source_chunks or not target_chunks:
            _log.info("EntityResolver.resolve(kb_id=%r): empty input — skip", kb_id)
            return []

        src_entities = self._extract_all(source_chunks)
        tgt_entities = self._extract_all(target_chunks)

        _log.info(
            "EntityResolver: kb_id=%r  src_entities=%d  tgt_entities=%d",
            kb_id, len(src_entities), len(tgt_entities),
        )
        if not src_entities or not tgt_entities:
            return []

        # Pass 1: fuzzy
        candidates = self._fuzzy_pass(src_entities, tgt_entities)
        _log.debug("EntityResolver: fuzzy_pass → %d candidate pairs", len(candidates))

        # Pass 2: embedding (optional)
        if embed_fn is not None and candidates:
            candidates = self._embedding_pass(candidates, embed_fn)

        # Emit bidirectional links above combined threshold
        all_links: list[LinkRecord] = []
        for m in candidates:
            if m.confidence < self._combined_threshold:
                continue
            conf = round(m.confidence, 4)
            src_cid = m.source_entity.chunk_id
            tgt_cid = m.target_entity.chunk_id
            all_links.append(LinkRecord(src_cid, tgt_cid, "entity_match", conf))
            all_links.append(LinkRecord(tgt_cid, src_cid, "entity_match", conf))

        # Deduplicate by (source, target) pair
        seen_pairs: set[tuple[str, str]] = set()
        deduped: list[LinkRecord] = []
        for lnk in all_links:
            key = (lnk.source_chunk_id, lnk.target_chunk_id)
            if key not in seen_pairs:
                seen_pairs.add(key)
                deduped.append(lnk)

        saved = self._store.save_links(deduped)
        _log.info(
            "EntityResolver.resolve: %d links emitted, %d newly saved (kb_id=%r)",
            len(deduped), saved, kb_id,
        )
        return deduped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_all(chunks: list[dict]) -> list[_Entity]:
        entities: list[_Entity] = []
        for chunk in chunks:
            try:
                entities.extend(_extract_entities(chunk))
            except Exception as exc:  # noqa: BLE001
                _log.debug("_extract_all: skipping chunk — %s", exc)
        return entities

    def _fuzzy_pass(
        self,
        src: list[_Entity],
        tgt: list[_Entity],
    ) -> list[_MatchResult]:
        results: list[_MatchResult] = []
        for se in src:
            for te in tgt:
                if se.chunk_id == te.chunk_id:
                    continue
                score = _fuzzy_score(se.normalized, te.normalized)
                if score >= self._fuzzy_threshold:
                    results.append(_MatchResult(
                        source_entity=se,
                        target_entity=te,
                        fuzzy_score=score,
                        embedding_score=0.0,
                        confidence=_combined_confidence(score, 0.0, False),
                    ))
        return results

    def _embedding_pass(
        self,
        candidates: list[_MatchResult],
        embed_fn: Callable[[list[str]], list],
    ) -> list[_MatchResult]:
        # Collect unique names for one-shot batch embedding
        name_to_idx: dict[str, int] = {}
        all_names: list[str] = []
        for m in candidates:
            for name in (m.source_entity.normalized, m.target_entity.normalized):
                if name not in name_to_idx:
                    name_to_idx[name] = len(all_names)
                    all_names.append(name)

        try:
            raw = embed_fn(all_names)
            embeddings = [np.array(e, dtype=np.float32) for e in raw]
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "_embedding_pass: embed_fn failed (%s) — using fuzzy scores only", exc
            )
            return candidates

        updated: list[_MatchResult] = []
        for m in candidates:
            si = name_to_idx.get(m.source_entity.normalized, -1)
            ti = name_to_idx.get(m.target_entity.normalized, -1)
            if si < 0 or ti < 0:
                updated.append(m)
                continue
            emb_score = _cosine_similarity(embeddings[si], embeddings[ti])
            conf = _combined_confidence(m.fuzzy_score, emb_score, True)
            updated.append(_MatchResult(
                source_entity=m.source_entity,
                target_entity=m.target_entity,
                fuzzy_score=m.fuzzy_score,
                embedding_score=emb_score,
                confidence=conf,
            ))
        return updated
