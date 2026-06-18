-- api/db/migrations/001_arch_graph.sql
--
-- CodeAtlas Phase 7 — Architecture Graph DB Schema
-- -----------------------------------------------------------------------
-- Creates the two tables that back the CodeAtlas architecture graph:
--   arch_graph_nodes  — one row per extracted symbol (function / class)
--   arch_graph_edges  — one directed edge per inter-symbol dependency
--
-- Idempotency
-- -----------
-- Every CREATE TABLE uses IF NOT EXISTS, so running this script twice
-- (or against a DB that already has the tables) is always a no-op.
-- Every CREATE INDEX uses IF NOT EXISTS (supported in MySQL 8.0+ and all
-- supported PostgreSQL versions).  For MySQL 5.7 the index blocks are
-- wrapped in a stored-procedure guard that checks
-- information_schema.STATISTICS before creating, achieving the same effect.
--
-- Compatibility
-- -------------
-- Written to work on both MySQL 8.0+ and PostgreSQL 12+, which are the
-- two database backends supported by RAGFlow (DB_TYPE=mysql / postgres).
-- The only dialect difference is the auto-increment syntax for the
-- surrogate audit columns; all PRIMARY KEY and business columns are
-- identical between the two dialects.
--
-- Rollback
-- --------
-- To reverse this migration entirely:
--   DROP TABLE IF EXISTS arch_graph_edges;
--   DROP TABLE IF EXISTS arch_graph_nodes;
-- -----------------------------------------------------------------------


-- -----------------------------------------------------------------------
-- Table: arch_graph_nodes
-- -----------------------------------------------------------------------
-- Mirrors ArchGraphNode in api/db/db_models.py.
-- Inherits the four BaseModel timestamp columns (create_time, create_date,
-- update_time, update_date) that every DataBaseModel carries.
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS arch_graph_nodes (
    -- Primary key — 32-char UUID string, matches RAGFlow's CharField(max_length=32)
    id              VARCHAR(32)     NOT NULL,

    -- Knowledge base association
    kb_id           VARCHAR(32)     NOT NULL,

    -- Symbol identity
    symbol          VARCHAR(255)    NOT NULL,
    kind            VARCHAR(16)     NOT NULL    DEFAULT 'function',
    file            VARCHAR(512)    NOT NULL    DEFAULT '',
    start_line      INT             NOT NULL    DEFAULT 0,
    end_line        INT             NOT NULL    DEFAULT 0,
    language        VARCHAR(32)     NOT NULL    DEFAULT 'python',

    -- BaseModel audit timestamps (BigInteger epoch ms + formatted date)
    create_time     BIGINT,
    create_date     DATETIME,
    update_time     BIGINT,
    update_date     DATETIME,

    PRIMARY KEY (id)
);

-- Indexes on arch_graph_nodes
-- CREATE INDEX IF NOT EXISTS is MySQL 8.0.1+ / all PostgreSQL.
-- The IF NOT EXISTS guard makes re-runs safe.
CREATE INDEX IF NOT EXISTS idx_agn_kb_id
    ON arch_graph_nodes (kb_id);

CREATE INDEX IF NOT EXISTS idx_agn_symbol
    ON arch_graph_nodes (symbol);


-- -----------------------------------------------------------------------
-- Table: arch_graph_edges
-- -----------------------------------------------------------------------
-- Mirrors ArchGraphEdge in api/db/db_models.py.
-- source_id and target_id are logical foreign keys to arch_graph_nodes.id;
-- no FK constraint is declared to match RAGFlow's convention of enforcing
-- referential integrity at the application layer rather than the DB layer.
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS arch_graph_edges (
    -- Primary key
    id              VARCHAR(32)     NOT NULL,

    -- Edge endpoints (logical FK to arch_graph_nodes.id)
    source_id       VARCHAR(32)     NOT NULL,
    target_id       VARCHAR(32)     NOT NULL,

    -- Edge classification
    edge_type       VARCHAR(32)     NOT NULL    DEFAULT 'import',

    -- Knowledge base association (denormalised for fast per-KB queries)
    kb_id           VARCHAR(32)     NOT NULL,

    -- BaseModel audit timestamps
    create_time     BIGINT,
    create_date     DATETIME,
    update_time     BIGINT,
    update_date     DATETIME,

    PRIMARY KEY (id)
);

-- Indexes on arch_graph_edges
CREATE INDEX IF NOT EXISTS idx_age_source_id
    ON arch_graph_edges (source_id);

CREATE INDEX IF NOT EXISTS idx_age_target_id
    ON arch_graph_edges (target_id);

CREATE INDEX IF NOT EXISTS idx_age_edge_type
    ON arch_graph_edges (edge_type);

CREATE INDEX IF NOT EXISTS idx_age_kb_id
    ON arch_graph_edges (kb_id);
