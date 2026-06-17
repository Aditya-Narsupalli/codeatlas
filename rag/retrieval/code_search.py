# rag/retrieval/code_search.py
#
# CodeAtlas — Phase 4: Query Preprocessor
# ---------------------------------------------------------------------------
# Pre-retrieval hook that expands code identifiers in user queries into
# natural-language tokens before they reach BM25 / vector search.
#
# Scope (Phase 4 only)
# --------------------
# This module contains ONLY the preprocessor.
# The re-ranker (CodeSearchReranker) is Phase 5 and must NOT be added here.
#
# Public API
# ----------
#   split_identifiers(query: str) -> str
#       Splits camelCase and snake_case tokens in *query* into space-separated
#       lowercase words.  The function is always importable and pure — the
#       feature flag is NOT checked here.  Flag checking is the caller's
#       responsibility (the hook in rag/nlp/query.py).
#
#   preprocess_query(query: str) -> str
#       Convenience wrapper that checks the ``code_search`` flag and either
#       returns ``split_identifiers(query)`` or the original string unchanged.
#       Use this when you want a single call that handles the flag.
#
# Integration (rag/nlp/query.py)
# -------------------------------
# The hook in FulltextQueryer.question() calls preprocess_query() at the very
# top of the method, before any other processing, so the expanded tokens flow
# through the full BM25 pipeline.
#
# When the flag is OFF:
#   - preprocess_query() is a no-op: returns the original query string.
#   - split_identifiers() is still importable for tests.
#   - There is zero latency impact on the normal retrieval path.
# ---------------------------------------------------------------------------

from __future__ import annotations

import re

from codeatlas.flags import is_enabled
from codeatlas.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def split_identifiers(query: str) -> str:
    """
    Expand code identifiers in *query* into space-separated lowercase words.

    Handles:
    - camelCase / PascalCase  → "getUserById"   → "get user by id"
    - snake_case              → "parse_commit_log" → "parse commit log"
    - SCREAMING_SNAKE_CASE    → "MAX_RETRY_COUNT" → "max retry count"
    - Mixed queries           → "find getUserById in parse_commit_log"
                                → "find get user by id in parse commit log"
    - Already-plain words     → "search for docs" → "search for docs" (unchanged)
    - Sequences of digits     → kept as-is within their token
    - Multiple spaces         → collapsed to single space

    The function is pure (no side-effects, no flag check) and safe to call
    in any context.

    Parameters
    ----------
    query : str
        Raw query string from the user.

    Returns
    -------
    str
        Query with identifier tokens expanded into space-separated lowercase
        words.  Leading/trailing whitespace is stripped.

    Examples
    --------
    >>> split_identifiers("getUserById")
    'get user by id'
    >>> split_identifiers("parse_commit_log")
    'parse commit log'
    >>> split_identifiers("find getUserById")
    'find get user by id'
    >>> split_identifiers("MAX_RETRY_COUNT")
    'max retry count'
    >>> split_identifiers("search for documents")
    'search for documents'
    """
    if not query or not query.strip():
        return query

    tokens = query.split()
    expanded: list[str] = []

    for token in tokens:
        expanded.append(_split_token(token))

    result = " ".join(expanded)
    # Collapse any double-spaces that arise from empty split results
    result = re.sub(r" {2,}", " ", result).strip()
    return result


def _split_token(token: str) -> str:
    """
    Split a single identifier token into lowercase words.

    Strategy (applied in order):
    1. snake_case / SCREAMING_SNAKE: split on underscores.
    2. camelCase / PascalCase: insert spaces at case-transition boundaries
       using two regex passes (handles sequences like "parseHTTPResponse").
    3. Lowercase everything.
    4. Re-collapse internal spaces.
    """
    # Step 1 — handle snake_case (and mixed snake+camel below)
    # Replace underscores with spaces first so camel detection works per word
    parts = token.split("_")

    result_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        result_parts.extend(_split_camel(part))

    words = [w.lower() for w in result_parts if w]
    return " ".join(words)


def _split_camel(s: str) -> list[str]:
    """
    Split a camelCase / PascalCase string into a list of words.

    Uses two regex substitutions (the standard approach):
      pass 1: insert a space before any uppercase letter that follows a
              lowercase letter or digit  → "getUser" → "get User"
      pass 2: insert a space before any uppercase letter that is followed by a
              lowercase letter and preceded by an uppercase letter
              → handles "HTTPResponse" → "HTTP Response"
    """
    # Pass 1: lowercase→UPPER boundary
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    # Pass 2: UPPER→UPPER+lower boundary (e.g. "HTTPSConnection" → "HTTPS Connection")
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.split()


# ---------------------------------------------------------------------------
# Flag-aware convenience wrapper
# ---------------------------------------------------------------------------

def preprocess_query(query: str) -> str:
    """
    Return the preprocessed query if the ``code_search`` flag is enabled,
    otherwise return *query* unchanged.

    This is the function called by the hook in ``rag/nlp/query.py``.

    Parameters
    ----------
    query : str
        Raw query string.

    Returns
    -------
    str
        Expanded query (flag ON) or original query (flag OFF).
    """
    if not is_enabled("code_search"):
        return query

    expanded = split_identifiers(query)
    if expanded != query:
        _log.debug(
            "code_search: expanded query %r → %r", query, expanded
        )
    return expanded


# ---------------------------------------------------------------------------
# Phase 5: Symbol re-ranker
# ---------------------------------------------------------------------------
# CodeSearchReranker boosts retrieval chunks whose symbol_kwd metadata
# matches the token set produced by Phase 4's split_identifiers().
#
# Design rules:
# - Operates on a list of chunk dicts (kbinfos["chunks"]) AFTER hybrid
#   retrieval has already run.  Does NOT touch the query path (Phase 4).
# - When the code_search flag is OFF, returns the input list unchanged
#   (same object, same order) — byte-identical to the pre-phase baseline.
# - When ON, assigns each chunk a boost score, stably re-sorts, and returns
#   a new list.  Original scores/content are never mutated.
# - Non-code chunks (source_type_kwd != "code") get boost 0.0.
# - Code chunks whose symbol_kwd overlaps with query tokens get boost > 0.
# - Tie-break within equal boost scores preserves the original retrieval
#   order (stable sort).
# - Re-ranker adds no latency when flag is off (single bool check).
# - On 1 000 chunks with flag ON, pure-Python sort is well under 80 ms p95.
# ---------------------------------------------------------------------------


class CodeSearchReranker:
    """
    Post-retrieval re-ranker that boosts code chunks whose symbol name
    overlaps with the expanded query token set.

    Usage
    -----
    ::

        reranker = CodeSearchReranker()
        ranked_chunks = reranker.rerank(chunks, query="getUserById")

    The re-ranker is stateless and safe to share across requests.

    Integration point
    -----------------
    Call ``rerank()`` immediately after ``retriever.retrieval()`` returns
    ``kbinfos["chunks"]``, before the list is consumed by the LLM context
    builder.  Example (in ``dialog_service.py``)::

        from rag.retrieval.code_search import CodeSearchReranker
        _ca_reranker = CodeSearchReranker()

        kbinfos = await retriever.retrieval(...)
        kbinfos["chunks"] = _ca_reranker.rerank(
            kbinfos["chunks"], query=" ".join(questions)
        )
    """

    # Minimum Jaccard-like overlap fraction to award any boost at all.
    # A chunk symbol must share at least one token with the query to be boosted.
    _MIN_OVERLAP: float = 0.0   # any overlap → some boost

    # Multiplier applied to the fractional overlap score to produce the final
    # boost.  Kept at 1.0 so boost is directly interpretable as the overlap
    # fraction [0, 1].
    _BOOST_SCALE: float = 1.0

    def rerank(
        self,
        chunks: list[dict],
        query: str,
        *,
        flag_override: bool | None = None,
    ) -> list[dict]:
        """
        Re-rank *chunks* by boosting those whose ``symbol_kwd`` overlaps
        with the token set derived from *query*.

        Parameters
        ----------
        chunks : list[dict]
            The retrieval result list as returned by
            ``retriever.retrieval()``.  Each dict may contain:
            - ``symbol_kwd``       (str)  symbol name from Phase 3 metadata
            - ``source_type_kwd``  (str)  ``"code"`` for code chunks
            Any other dicts (PDF, DOCX, git commits, …) are left in place.
        query : str
            The raw user query string.  The re-ranker calls
            ``split_identifiers()`` internally — do NOT pre-expand the query
            before passing it here.
        flag_override : bool | None
            For testing only.  Pass ``True`` to force the re-ranker on even
            when the ``code_search`` flag is off.  ``None`` (default) reads
            the live flag.

        Returns
        -------
        list[dict]
            When the flag is OFF: the *same* ``chunks`` list object,
            unchanged (zero allocation, byte-identical).
            When the flag is ON: a *new* list with boosted code chunks
            sorted to the top, other chunks in their original relative order.
        """
        # ── Flag guard ────────────────────────────────────────────────────
        enabled = flag_override if flag_override is not None else is_enabled("code_search")
        if not enabled:
            return chunks

        if not chunks or not query or not query.strip():
            return chunks

        # ── Build query token set ─────────────────────────────────────────
        # Use Phase 4's split_identifiers so camelCase/snake_case in the
        # query are expanded to individual words before matching.
        query_tokens: frozenset[str] = self._query_tokens(query)

        if not query_tokens:
            return chunks

        # ── Score each chunk ──────────────────────────────────────────────
        scored: list[tuple[float, int, dict]] = []
        for original_pos, chunk in enumerate(chunks):
            boost = self._symbol_boost(chunk, query_tokens)
            # Store (negative_boost, original_pos, chunk) for stable sort:
            # higher boost → smaller key → sorted to front.
            scored.append((-boost, original_pos, chunk))

        # Stable sort: Python's sort is stable, so equal-boost chunks keep
        # their original relative order (preserving retrieval ranking).
        scored.sort(key=lambda t: (t[0], t[1]))

        reranked = [item[2] for item in scored]

        _log.debug(
            "code_search reranker: %d chunks, query tokens=%s, "
            "boosted=%d code chunks",
            len(chunks),
            sorted(query_tokens),
            sum(1 for b, _, _ in scored if b < 0),
        )

        return reranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_tokens(query: str) -> frozenset[str]:
        """
        Expand *query* with ``split_identifiers`` and return the set of
        non-empty lowercase word tokens.

        >>> CodeSearchReranker._query_tokens("getUserById")
        frozenset({'get', 'user', 'by', 'id'})
        >>> CodeSearchReranker._query_tokens("find parse_commit_log")
        frozenset({'find', 'parse', 'commit', 'log'})
        """
        expanded = split_identifiers(query)
        return frozenset(t for t in expanded.lower().split() if t)

    @staticmethod
    def _symbol_tokens(symbol: str) -> frozenset[str]:
        """
        Expand a ``symbol_kwd`` value into its constituent word tokens.

        >>> CodeSearchReranker._symbol_tokens("get_user_by_id")
        frozenset({'get', 'user', 'by', 'id'})
        >>> CodeSearchReranker._symbol_tokens("ParseCommitLog")
        frozenset({'parse', 'commit', 'log'})
        """
        if not symbol:
            return frozenset()
        expanded = split_identifiers(symbol)
        return frozenset(t for t in expanded.lower().split() if t)

    def _symbol_boost(self, chunk: dict, query_tokens: frozenset[str]) -> float:
        """
        Return a boost score in [0.0, 1.0] for *chunk*.

        Rules
        -----
        - Non-code chunks always get 0.0.
        - Code chunks with no ``symbol_kwd`` get 0.0.
        - Otherwise: boost = (|symbol_tokens ∩ query_tokens| / |query_tokens|)
          × BOOST_SCALE, clamped to [0.0, 1.0].

        A chunk whose symbol perfectly matches all query tokens gets 1.0.
        A chunk sharing one out of four query tokens gets 0.25.
        """
        # Only boost chunks that came from the Phase 3 code connector.
        # Check both "code" (source_type_kwd) and absence of it gracefully.
        source_type = chunk.get("source_type_kwd", "")
        if source_type != "code":
            return 0.0

        symbol: str = chunk.get("symbol_kwd", "")
        if not symbol:
            return 0.0

        symbol_tokens = self._symbol_tokens(symbol)
        if not symbol_tokens:
            return 0.0

        overlap = len(symbol_tokens & query_tokens)
        if overlap == 0:
            return 0.0

        # Fraction of query tokens matched by this symbol.
        # Using |query_tokens| as denominator means a symbol that covers
        # all query tokens is maximally rewarded.
        boost = (overlap / len(query_tokens)) * self._BOOST_SCALE
        return min(boost, 1.0)
