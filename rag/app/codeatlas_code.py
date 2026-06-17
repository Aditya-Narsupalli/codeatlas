# rag/app/codeatlas_code.py
#
# CodeAtlas — Phase 3: Source Code Connector
# ---------------------------------------------------------------------------
# Ingests source files into RAGFlow as one chunk-dict per top-level function
# or class, split at AST boundaries using tree-sitter.
#
# Integration note
# ----------------
# RAGFlow's task_executor dispatches to parsers via FACTORY[parser_id].chunk().
# To activate this connector, add to rag/svr/task_executor.py:
#
#   from rag.app import codeatlas_code          # (1) import
#   FACTORY[MIME_CODE] = codeatlas_code         # (2) register
#
# MIME_CODE is also exported from rag/app/__init__.py for clean imports.
#
# Chunk dict contract (matches RAGFlow tokenize() + tokenize_chunks()):
#   content_with_weight   str   full source text of the symbol — embedded/searched
#   docnm_kwd             str   source file path / name
#   title_tks             str   "symbol (file)" — keyword-indexed title
#   title_sm_tks          str   same as title_tks
#   source_type_kwd       str   "code"
#   symbol_kwd            str   qualified name, e.g. "MyClass" / "my_func"
#   language_kwd          str   "python"  (only Python in Phase 3)
#   start_line_int        int   1-based start line in the original file
#   end_line_int          int   1-based end line in the original file
#
# Phase scope
# -----------
# Python support only.  Other languages are scaffolded but raise
# UnsupportedLanguageError so future phases can add grammars without
# changing this file's structure.
#
# NOT in this phase:
#   - Symbol extractor (Phase 6)
#   - Graph builder    (Phase 8)
#   - DB writes
# ---------------------------------------------------------------------------

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from codeatlas.flags import is_enabled
from codeatlas.logger import get_logger

_log = get_logger(__name__)

# Parser identifier registered in RAGFlow's FACTORY dict.
# Must not collide with MIME_GIT = "git" from Phase 2.
MIME_CODE: str = "code"

# Top-level AST node types we treat as chunk boundaries.
_CHUNK_NODE_TYPES: frozenset[str] = frozenset(
    {"function_definition", "class_definition", "decorated_definition"}
)

# Map from file-extension → language tag.
# Only Python is supported in Phase 3; others are listed for future phases.
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsupportedLanguageError(ValueError):
    """Raised when a source file's language has no installed grammar."""


class MissingGrammarError(ImportError):
    """Raised when the tree-sitter grammar package is not installed."""


# ---------------------------------------------------------------------------
# Grammar loader (lazy, cached per language)
# ---------------------------------------------------------------------------

_grammar_cache: dict[str, Any] = {}   # language_tag → tree_sitter.Language


def _load_language(language: str) -> Any:
    """
    Return a ``tree_sitter.Language`` for *language*.

    Only ``"python"`` is supported in Phase 3.  Calling this for another
    language raises ``UnsupportedLanguageError`` so future phases can add
    grammars by registering them here.

    Raises
    ------
    UnsupportedLanguageError
        Language tag is not "python".
    MissingGrammarError
        tree-sitter or tree-sitter-python is not installed.
    """
    if language in _grammar_cache:
        return _grammar_cache[language]

    if language != "python":
        raise UnsupportedLanguageError(
            f"Language {language!r} is not supported in Phase 3. "
            "Only 'python' is available."
        )

    try:
        import tree_sitter          # noqa: PLC0415
        import tree_sitter_python   # noqa: PLC0415
    except ImportError as exc:
        raise MissingGrammarError(
            "tree-sitter and tree-sitter-python are required for the "
            "CodeAtlas source code connector. "
            "Install them with: pip install tree-sitter tree-sitter-python"
        ) from exc

    lang = tree_sitter.Language(tree_sitter_python.language())
    _grammar_cache[language] = lang
    return lang


def _get_parser(language: str) -> Any:
    """Return a fresh ``tree_sitter.Parser`` configured for *language*."""
    import tree_sitter  # noqa: PLC0415
    lang = _load_language(language)
    return tree_sitter.Parser(lang)


# ---------------------------------------------------------------------------
# AST walking helpers
# ---------------------------------------------------------------------------

def _resolve_language(filename: str, parser_config: dict) -> str | None:
    """
    Determine the language tag for *filename*.

    Checks ``parser_config["language"]`` first (explicit override), then
    falls back to the file extension.  Returns ``None`` if unknown.
    """
    if parser_config.get("language"):
        return str(parser_config["language"]).lower()

    ext = Path(filename).suffix.lower()
    return _EXT_TO_LANGUAGE.get(ext)


def _symbol_name(node: Any) -> str:
    """
    Extract the symbol name from a function_definition, class_definition,
    or decorated_definition node.

    For decorated_definition, we unwrap to the inner function/class.
    Falls back to the node type string if no name child exists.
    """
    target = node

    # Unwrap decorator wrapper
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                target = child
                break

    name_node = target.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8", errors="replace")

    return f"<{node.type}>"


def _extract_chunks(
    source_bytes: bytes,
    language: str,
    filename: str,
) -> list[dict]:
    """
    Parse *source_bytes* with tree-sitter and return one raw chunk dict per
    top-level function or class definition.

    Parameters
    ----------
    source_bytes:
        Raw source file content as bytes.
    language:
        Language tag, e.g. ``"python"``.
    filename:
        Original filename — stored in ``docnm_kwd``.

    Returns
    -------
    list[dict]
        Each dict has all required CodeAtlas + RAGFlow fields.
        Empty list if no chunk-boundary nodes found (e.g. script with only
        imports and assignments).
    """
    parser = _get_parser(language)
    tree = parser.parse(source_bytes)
    root = tree.root_node

    safe_name = re.sub(r"[^\w\-./]", "_", filename)[:120]
    results: list[dict] = []

    for node in root.children:
        if node.type not in _CHUNK_NODE_TYPES:
            continue

        symbol = _symbol_name(node)
        start_line = node.start_point.row + 1   # convert 0-based row → 1-based
        end_line = node.end_point.row + 1

        # Extract raw source text for this symbol
        node_text = source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        )

        # Human-readable content block: header + source
        content = (
            f"# {language} | {symbol} | {filename}:{start_line}-{end_line}\n"
            f"\n"
            f"{node_text}"
        )

        title = f"{symbol} ({Path(filename).name})"

        d: dict = {
            # ── Core RAGFlow fields ───────────────────────────────────────
            "content_with_weight": content,
            "docnm_kwd": safe_name,
            "title_tks": title,
            "title_sm_tks": title,
            # ── CodeAtlas metadata (keyword-indexed, filterable) ──────────
            "source_type_kwd": "code",
            "symbol_kwd": symbol,
            "language_kwd": language,
            # Store as int so downstream numeric filters work correctly.
            # RAGFlow Elasticsearch mapping treats int fields directly.
            "start_line_int": start_line,
            "end_line_int": end_line,
        }

        results.append(d)

    return results


# ---------------------------------------------------------------------------
# Public API — matches RAGFlow parser module interface
# ---------------------------------------------------------------------------

def chunk(
    filename: str,
    binary: bytes | None = None,
    lang: str = "English",
    callback=None,
    **kwargs,
) -> list[dict]:
    """
    RAGFlow parser entry-point for source code files.

    Parameters
    ----------
    filename:
        Path or logical name of the source file being ingested.
        Used to infer the programming language via file extension.
    binary:
        Raw file bytes.  If ``None`` and *filename* is a real path,
        the file is read from disk.  Raises ``ValueError`` if neither
        source is available.
    lang:
        RAGFlow language hint (``"English"`` / ``"Chinese"``).  Kept for
        interface compatibility; source code language is inferred from
        the file extension, not this field.
    callback:
        RAGFlow progress callback: ``callback(progress_float, message_str)``.
    **kwargs:
        ``parser_config`` dict (optional, forwarded by task_executor):
          - ``language`` (str)  explicit language override
        Other keys (tenant_id, kb_id, …) are accepted and ignored.

    Returns
    -------
    list[dict]
        One chunk per top-level function / class definition.
        Returns ``[]`` when the ``code_connector`` feature flag is disabled,
        when the language is unknown, or when no chunk-boundary nodes exist.
    """
    # ── Feature flag guard ────────────────────────────────────────────────
    if not is_enabled("code_connector"):
        _log.info(
            "Code connector is disabled (feature flag 'code_connector' is off). "
            "Set code_connector: true in conf/codeatlas.yaml to enable."
        )
        if callback:
            callback(1.0, "Code connector disabled via feature flag.")
        return []

    # ── Resolve source bytes ──────────────────────────────────────────────
    parser_config: dict = kwargs.get("parser_config", {}) or {}

    if binary is not None and len(binary) > 0:
        source_bytes = binary
    else:
        # Try reading from disk (local path)
        path = Path(filename)
        if path.is_file():
            source_bytes = path.read_bytes()
        else:
            _log.warning(
                "Code connector: no binary supplied and %r is not a readable file. "
                "Returning empty chunk list.",
                filename,
            )
            if callback:
                callback(1.0, f"Code connector: cannot read source for {filename!r}")
            return []

    if callback:
        callback(0.05, f"Code connector: parsing {filename!r}")

    # ── Detect language ───────────────────────────────────────────────────
    language = _resolve_language(filename, parser_config)
    if language is None:
        _log.info(
            "Code connector: unknown language for %r — skipping. "
            "Supported extensions: %s",
            filename,
            ", ".join(sorted(_EXT_TO_LANGUAGE)),
        )
        if callback:
            callback(1.0, f"Unsupported file type: {Path(filename).suffix!r}")
        return []

    # ── Parse and chunk ───────────────────────────────────────────────────
    try:
        results = _extract_chunks(source_bytes, language, filename)
    except UnsupportedLanguageError as exc:
        _log.warning("Code connector: %s", exc)
        if callback:
            callback(1.0, str(exc))
        return []
    except MissingGrammarError as exc:
        _log.error("Code connector: %s", exc)
        if callback:
            callback(-1, str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        _log.exception("Code connector: unexpected error parsing %r: %s", filename, exc)
        if callback:
            callback(-1, f"Parse error: {exc}")
        raise

    n = len(results)
    _log.info(
        "Code connector: %d chunk(s) from %r (%s)",
        n, filename, language,
    )

    if callback:
        callback(1.0, f"Code connector: {n} chunk(s) produced from {Path(filename).name}")

    return results
