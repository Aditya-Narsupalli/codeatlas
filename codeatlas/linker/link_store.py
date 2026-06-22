# codeatlas/linker/link_store.py
#
# CodeAtlas — Phase 14: LinkStore
# ---------------------------------------------------------------------------
# Persistence layer for cross-source entity links.  Reads/writes the
# Phase 13 ``knowledge_links`` table via the KnowledgeLink ORM model.
#
# Design rules (roadmap §4)
# -------------------------
# - No duplicate rows: ``save_link()`` is idempotent — calling it twice for
#   the same (source_chunk_id, target_chunk_id, link_type) is a no-op.
# - Core resolver logic lives in entity_resolver.py — a matching bug must
#   be fixable without touching this file.
# - All ORM imports are lazy (inside methods), keeping the module importable
#   without a live RAGFlow environment — same pattern as arch_graph.py.
# - The ``_OrmAdapter`` interface makes the store testable with an in-memory
#   SQLite substitute, no Elasticsearch or DB server required.
# ---------------------------------------------------------------------------

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from codeatlas.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# LinkRecord — typed in-memory representation of one link
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkRecord:
    """
    Immutable representation of one cross-source entity link.

    Attributes
    ----------
    source_chunk_id : str
        RAGFlow chunk ID of the source side (opaque document-store string).
    target_chunk_id : str
        RAGFlow chunk ID of the target side.
    link_type : str
        Classification: ``"entity_match"``, ``"reference"``, or ``"other"``.
    confidence : float
        Combined resolver confidence score in ``[0.0, 1.0]``.
    """
    source_chunk_id: str
    target_chunk_id: str
    link_type: str = "entity_match"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# ORM adapter protocol — swapped out in tests
# ---------------------------------------------------------------------------

@runtime_checkable
class _OrmAdapterProtocol(Protocol):
    def save(self, link_id: str, rec: LinkRecord) -> None: ...
    def exists(self, link_id: str) -> bool: ...
    def get_by_chunk(self, chunk_id: str) -> list[LinkRecord]: ...


class _LiveOrmAdapter:
    """
    Wraps the Phase 13 KnowledgeLink ORM model.

    All imports are lazy so this class is importable without a running DB.
    """

    def save(self, link_id: str, rec: LinkRecord) -> None:
        from api.db.db_models import KnowledgeLink  # noqa: PLC0415
        KnowledgeLink.create(
            id=link_id,
            source_chunk_id=rec.source_chunk_id,
            target_chunk_id=rec.target_chunk_id,
            link_type=rec.link_type,
            confidence=rec.confidence,
        )

    def exists(self, link_id: str) -> bool:
        from api.db.db_models import KnowledgeLink  # noqa: PLC0415
        return KnowledgeLink.select().where(KnowledgeLink.id == link_id).exists()

    def get_by_chunk(self, chunk_id: str) -> list[LinkRecord]:
        from api.db.db_models import KnowledgeLink  # noqa: PLC0415
        rows = KnowledgeLink.select().where(
            (KnowledgeLink.source_chunk_id == chunk_id)
            | (KnowledgeLink.target_chunk_id == chunk_id)
        )
        return [
            LinkRecord(
                source_chunk_id=r.source_chunk_id,
                target_chunk_id=r.target_chunk_id,
                link_type=r.link_type,
                confidence=r.confidence,
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Public LinkStore
# ---------------------------------------------------------------------------

def _link_id(source_chunk_id: str, target_chunk_id: str, link_type: str) -> str:
    """
    Return a stable 32-char hex link ID, deterministic on its inputs.

    Using a content-hash guarantees idempotency: calling ``save_link()``
    twice for the same triple always produces the same ``id``, so the
    ``exists()`` pre-check prevents duplicate rows even if ``save_links()``
    is called repeatedly (e.g. on a resolver re-run).
    """
    key = f"{source_chunk_id}|{target_chunk_id}|{link_type}"
    return hashlib.md5(key.encode()).hexdigest()


class LinkStore:
    """
    Read/write interface for the Phase 13 ``knowledge_links`` table.

    Instantiate with no arguments to use the live ORM (requires a running
    database).  For tests, inject a ``_TestOrmAdapter`` via ``orm=``.

    Example
    -------
    ::

        store = LinkStore()
        store.save_link(
            LinkRecord("chunk_a", "chunk_b", "entity_match", 0.82)
        )
        links = store.get_links("chunk_a")
    """

    def __init__(self, orm: _OrmAdapterProtocol | None = None) -> None:
        self._orm: _OrmAdapterProtocol = orm or _LiveOrmAdapter()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_link(self, link: LinkRecord) -> bool:
        """
        Persist *link* if it does not already exist.

        Parameters
        ----------
        link : LinkRecord
            The link to save.

        Returns
        -------
        bool
            ``True`` if the link was inserted; ``False`` if it already
            existed (idempotent — not an error).
        """
        lid = _link_id(link.source_chunk_id, link.target_chunk_id, link.link_type)
        try:
            if self._orm.exists(lid):
                _log.debug(
                    "LinkStore.save_link: already exists id=%r — skip", lid
                )
                return False
            self._orm.save(lid, link)
            _log.debug(
                "LinkStore.save_link: saved id=%r  %r→%r  conf=%.3f",
                lid, link.source_chunk_id, link.target_chunk_id, link.confidence,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            _log.error("LinkStore.save_link failed: %s", exc)
            return False

    def save_links(self, links: list[LinkRecord]) -> int:
        """
        Persist a batch of links, skipping any that already exist.

        Returns
        -------
        int
            Number of links actually inserted (≤ ``len(links)``).
        """
        inserted = sum(1 for lnk in links if self.save_link(lnk))
        _log.info(
            "LinkStore.save_links: %d/%d inserted (duplicates skipped)",
            inserted, len(links),
        )
        return inserted

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_links(self, chunk_id: str) -> list[LinkRecord]:
        """
        Return all links where *chunk_id* is either source or target.

        Returns ``[]`` (never raises) if no links exist or the query fails.
        """
        try:
            return self._orm.get_by_chunk(chunk_id)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "LinkStore.get_links(%r) failed: %s — returning []", chunk_id, exc
            )
            return []

    def exists(self, source_chunk_id: str, target_chunk_id: str,
               link_type: str = "entity_match") -> bool:
        """
        Return ``True`` if a link between these two chunks already exists.
        """
        lid = _link_id(source_chunk_id, target_chunk_id, link_type)
        try:
            return self._orm.exists(lid)
        except Exception as exc:  # noqa: BLE001
            _log.warning("LinkStore.exists check failed: %s", exc)
            return False
