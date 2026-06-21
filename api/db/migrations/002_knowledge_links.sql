-- api/db/migrations/002_knowledge_links.sql
--
-- CodeAtlas Phase 13 — Knowledge Links DB Schema
-- -----------------------------------------------------------------------
-- Creates the knowledge_links table that backs cross-source entity
-- linking: connecting a chunk from one source type (e.g. a doc chunk) to
-- a chunk from another source type (e.g. a code chunk) when an entity
-- resolver determines they refer to the same concept.
--
-- This is the SECOND and FINAL CodeAtlas schema migration in the entire
-- project. No further schema migrations are planned after this one.
--
-- Idempotency
-- -----------
-- CREATE TABLE uses IF NOT EXISTS, so running this script twice (or
-- against a DB that already has the table) is always a no-op.
-- CREATE INDEX uses IF NOT EXISTS (supported in MySQL 8.0+ and all
-- supported PostgreSQL versions), matching the style established in
-- 001_arch_graph.sql.
--
-- Compatibility
-- -------------
-- Written to work on both MySQL 8.0+ and PostgreSQL 12+, which are the
-- two database backends supported by RAGFlow (DB_TYPE=mysql / postgres).
--
-- No data writes
-- ---------------
-- This migration contains ONLY DDL (CREATE TABLE / CREATE INDEX).
-- No INSERT, UPDATE, or DELETE statements. Data is written later by
-- Phase 14's EntityResolver — never by this migration.
--
-- Rollback
-- --------
-- To reverse this migration entirely:
--   DROP TABLE IF EXISTS knowledge_links;
-- -----------------------------------------------------------------------


-- -----------------------------------------------------------------------
-- Table: knowledge_links
-- -----------------------------------------------------------------------
-- Mirrors KnowledgeLink in api/db/db_models.py.
-- Inherits the four BaseModel timestamp columns (create_time, create_date,
-- update_time, update_date) that every DataBaseModel carries, matching
-- the pattern used by arch_graph_nodes / arch_graph_edges in
-- 001_arch_graph.sql.
--
-- source_chunk_id / target_chunk_id store RAGFlow chunk IDs as they exist
-- in the document store (Elasticsearch/Infinity) — opaque strings, not a
-- SQL foreign key, since the document store is a separate system from
-- this relational database. This matches the existing
-- Task.chunk_ids / DialogTestCase.relevant_chunk_ids convention already
-- present in api/db/db_models.py.
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS knowledge_links (
    -- Primary key — 32-char UUID string, matches RAGFlow's CharField(max_length=32)
    id                  VARCHAR(32)     NOT NULL,

    -- Link endpoints (document-store chunk IDs — opaque strings, no SQL FK)
    source_chunk_id     VARCHAR(32)     NOT NULL,
    target_chunk_id     VARCHAR(32)     NOT NULL,

    -- Link classification and confidence
    link_type           VARCHAR(32)     NOT NULL    DEFAULT 'entity_match',
    confidence          FLOAT           NOT NULL    DEFAULT 0.0,

    -- BaseModel audit timestamps (BigInteger epoch ms + formatted date)
    create_time         BIGINT,
    create_date         DATETIME,
    update_time         BIGINT,
    update_date         DATETIME,

    PRIMARY KEY (id)
);

-- Indexes on knowledge_links
-- CREATE INDEX IF NOT EXISTS is MySQL 8.0.1+ / all PostgreSQL.
-- The IF NOT EXISTS guard makes re-runs safe.
CREATE INDEX IF NOT EXISTS idx_kl_source_chunk_id
    ON knowledge_links (source_chunk_id);

CREATE INDEX IF NOT EXISTS idx_kl_target_chunk_id
    ON knowledge_links (target_chunk_id);

CREATE INDEX IF NOT EXISTS idx_kl_link_type
    ON knowledge_links (link_type);
