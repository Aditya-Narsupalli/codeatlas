# codeatlas/linker/__init__.py
#
# CodeAtlas — linker subpackage
# Phase 14 adds: EntityResolver, LinkStore
# Phase 15 will consume: LinkStore (read path for onboarding agent)
from codeatlas.linker.link_store import LinkStore, LinkRecord
from codeatlas.linker.entity_resolver import EntityResolver

__all__ = ["LinkStore", "LinkRecord", "EntityResolver"]
