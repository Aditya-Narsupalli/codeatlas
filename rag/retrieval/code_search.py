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
