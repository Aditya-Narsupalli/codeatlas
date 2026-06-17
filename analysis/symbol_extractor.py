# codeatlas/analysis/symbol_extractor.py
#
# CodeAtlas — Phase 6: Symbol Extractor
# ---------------------------------------------------------------------------
# Pure-Python module that reads Phase 3 code chunk metadata and/or source
# files on disk and returns a structured list of Symbol objects.
#
# Scope (Phase 6 only)
# --------------------
# - Symbol dataclass with: name, kind, file, start_line, end_line, imports
# - SymbolExtractor.extract(chunks)            ← from Phase 3 chunk dicts
# - SymbolExtractor.extract_from_files(paths)  ← from disk (.py files)
# - SymbolExtractor.extract_from_directory(dir_path, recursive=True)
#
# NOT in this phase:
# - DB writes of any kind (Phase 7 schema, Phase 8 graph builder)
# - API endpoints (Phase 9)
# - Graph construction or edge resolution (Phase 8)
# - Network calls of any kind
# - Agent integration (Phase 11+)
#
# Design rules
# ------------
# 1. No imports from RAGFlow internals (rag.*, api.*, common.*).
#    Only standard library + codeatlas.* + tree-sitter (optional).
# 2. Fully unit-testable without a running RAGFlow server or database.
# 3. tree-sitter is used when available; falls back to ast module if not.
# 4. Both extraction paths (from chunks / from files) produce identical
#    Symbol structures so downstream phases can use either interchangeably.
# ---------------------------------------------------------------------------

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeatlas.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Symbol dataclass
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """
    A single named code symbol extracted from a source file.

    Attributes
    ----------
    name : str
        The symbol's identifier as it appears in source, e.g. ``"MyClass"``
        or ``"parse_commit_log"``.
    kind : str
        ``"function"`` or ``"class"``.  Async functions are ``"function"``.
    file : str
        Source file path or logical name.  Matches ``docnm_kwd`` from the
        Phase 3 chunk dict when the symbol comes from a chunk.
    start_line : int
        1-based line number of the first line of the definition.
    end_line : int
        1-based line number of the last line of the definition.
    imports : list[str]
        Module-level import names referenced in the file containing this
        symbol.  Populated by best-effort static analysis; may be empty.
        Format: ``"os"``, ``"pathlib.Path"``, ``"typing.Optional"``.
        Local (function-level) imports are included too when extracting
        from chunk content.
    language : str
        Language tag, e.g. ``"python"``.  Defaults to ``"python"`` for
        Phase 6 (only language supported by Phase 3).
    """
    name: str
    kind: str                      # "function" | "class"
    file: str
    start_line: int
    end_line: int
    imports: list[str] = field(default_factory=list)
    language: str = "python"

    def __post_init__(self) -> None:
        if self.kind not in ("function", "class"):
            raise ValueError(
                f"Symbol.kind must be 'function' or 'class', got {self.kind!r}"
            )
        if self.start_line < 1:
            raise ValueError(f"Symbol.start_line must be ≥ 1, got {self.start_line}")
        if self.end_line < self.start_line:
            raise ValueError(
                f"Symbol.end_line ({self.end_line}) must be ≥ start_line "
                f"({self.start_line})"
            )


# ---------------------------------------------------------------------------
# Internal helpers — import extraction
# ---------------------------------------------------------------------------

def _imports_from_source(source: str) -> list[str]:
    """
    Extract all import names from *source* using ``ast``.

    Returns a flat list of dotted import strings:
      ``import os``               → ``["os"]``
      ``from pathlib import Path`` → ``["pathlib.Path"]``
      ``from typing import Any, Optional`` → ``["typing.Any", "typing.Optional"]``

    Silently returns ``[]`` if *source* cannot be parsed (e.g. syntax errors,
    incomplete snippet, Python 2 source).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.name
                results.append(f"{module}.{name}" if module else name)
    return results


def _kind_from_ast_node(node: ast.AST) -> str:
    """Return ``"class"`` or ``"function"`` for an ast definition node."""
    if isinstance(node, ast.ClassDef):
        return "class"
    return "function"   # FunctionDef, AsyncFunctionDef


# ---------------------------------------------------------------------------
# Tree-sitter extraction path (preferred — matches Phase 3 exactly)
# ---------------------------------------------------------------------------

def _ts_available() -> bool:
    """Return True if tree-sitter and tree-sitter-python are installed."""
    try:
        import tree_sitter          # noqa: F401
        import tree_sitter_python   # noqa: F401
        return True
    except ImportError:
        return False


def _ts_extract_symbols(
    source_bytes: bytes,
    filename: str,
    file_imports: list[str],
) -> list[Symbol]:
    """
    Extract symbols from *source_bytes* using tree-sitter.

    Mirrors the node walking in Phase 3 ``_extract_chunks`` so that the
    symbol boundaries are identical.  Captures ALL function and class
    definitions at every nesting level (top-level + methods inside classes).

    Parameters
    ----------
    source_bytes : bytes
        Raw UTF-8 source content.
    filename : str
        Logical file path stored in each Symbol.file.
    file_imports : list[str]
        Already-extracted module-level imports for this file.

    Returns
    -------
    list[Symbol]
        One Symbol per function_definition, class_definition, or
        decorated_definition node, at any nesting depth.
    """
    import tree_sitter
    import tree_sitter_python

    lang = tree_sitter.Language(tree_sitter_python.language())
    parser = tree_sitter.Parser(lang)
    tree = parser.parse(source_bytes)

    _SYMBOL_NODE_TYPES = frozenset(
        {"function_definition", "class_definition", "decorated_definition"}
    )

    results: list[Symbol] = []

    def _walk(node: Any) -> None:
        for child in node.children:
            if child.type in _SYMBOL_NODE_TYPES:
                sym = _ts_node_to_symbol(child, filename, file_imports)
                if sym is not None:
                    results.append(sym)
            # Always recurse — captures methods inside classes
            _walk(child)

    _walk(tree.root_node)
    return results


def _ts_node_to_symbol(
    node: Any,
    filename: str,
    file_imports: list[str],
) -> Symbol | None:
    """Convert a tree-sitter node to a Symbol, or return None on failure."""
    try:
        target = node

        # Unwrap decorated_definition → inner function/class
        if node.type == "decorated_definition":
            inner = next(
                (c for c in node.children
                 if c.type in ("function_definition", "class_definition")),
                None,
            )
            if inner is None:
                return None
            target = inner

        name_node = target.child_by_field_name("name")
        if name_node is None:
            return None

        name = name_node.text.decode("utf-8", errors="replace")
        kind = "class" if target.type == "class_definition" else "function"
        start_line = node.start_point.row + 1   # 0-based row → 1-based
        end_line = node.end_point.row + 1

        return Symbol(
            name=name,
            kind=kind,
            file=filename,
            start_line=start_line,
            end_line=end_line,
            imports=list(file_imports),
            language="python",
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("_ts_node_to_symbol: skipping node — %s", exc)
        return None


# ---------------------------------------------------------------------------
# AST fallback extraction path
# ---------------------------------------------------------------------------

def _ast_extract_symbols(
    source: str,
    filename: str,
    file_imports: list[str],
) -> list[Symbol]:
    """
    Extract symbols using Python's built-in ``ast`` module.

    Used as a fallback when tree-sitter is not installed.  Captures ALL
    function and class definitions at every nesting level.

    Parameters
    ----------
    source : str
        Source code as a string.
    filename : str
        Logical file path.
    file_imports : list[str]
        Module-level imports for this file.

    Returns
    -------
    list[Symbol]
        One Symbol per FunctionDef, AsyncFunctionDef, or ClassDef node
        at any depth.  Returns ``[]`` on parse failure.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _log.debug("ast fallback: cannot parse %r — %s", filename, exc)
        return []

    results: list[Symbol] = []

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                sym = _ast_node_to_symbol(child, filename, file_imports)
                if sym is not None:
                    results.append(sym)
            # Always recurse into every node
            _walk(child)

    _walk(tree)
    return results


def _ast_node_to_symbol(
    node: ast.AST,
    filename: str,
    file_imports: list[str],
) -> Symbol | None:
    """Convert an ast definition node to a Symbol."""
    try:
        name = getattr(node, "name", None)
        if not name:
            return None
        kind = _kind_from_ast_node(node)
        start_line = node.lineno          # already 1-based in ast
        end_line = getattr(node, "end_lineno", start_line)
        return Symbol(
            name=name,
            kind=kind,
            file=filename,
            start_line=start_line,
            end_line=end_line,
            imports=list(file_imports),
            language="python",
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("_ast_node_to_symbol: skipping — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SymbolExtractor:
    """
    Extracts Symbol objects from Phase 3 code chunks or source files.

    The extractor is stateless and safe to instantiate once and reuse.

    Extraction strategies
    ---------------------
    1. ``extract(chunks)``
       Reads Phase 3 chunk dicts directly — the fast path when chunks are
       already in memory after ingestion.  Uses ``symbol_kwd``,
       ``language_kwd``, ``start_line_int``, ``end_line_int``, and
       ``docnm_kwd`` from each chunk.  Imports are extracted from
       ``content_with_weight`` using ``ast``.

    2. ``extract_from_files(paths)``
       Reads ``.py`` files from disk and runs tree-sitter (with ``ast``
       fallback).  Captures ALL symbols including methods inside classes.
       Module-level imports are extracted per file and attached to every
       symbol from that file.

    3. ``extract_from_directory(dir_path, recursive=True)``
       Convenience wrapper over ``extract_from_files`` that discovers all
       ``.py`` files under a directory.

    Examples
    --------
    ::

        from codeatlas.analysis.symbol_extractor import Symbol, SymbolExtractor

        extractor = SymbolExtractor()

        # From Phase 3 chunks
        chunks = [...]   # list of dicts from codeatlas_code.chunk()
        symbols = extractor.extract(chunks)

        # From a directory
        symbols = extractor.extract_from_directory("rag/nlp/")

        for sym in symbols:
            print(sym.name, sym.kind, sym.file, sym.start_line, sym.end_line)
    """

    # ------------------------------------------------------------------
    # Path 1: from Phase 3 chunk dicts
    # ------------------------------------------------------------------

    def extract(self, chunks: list[dict]) -> list[Symbol]:
        """
        Build Symbol objects from Phase 3 code chunk dicts.

        Only chunks with ``source_type_kwd == "code"`` are processed.
        Other chunk types (git, PDF, …) are silently skipped.

        Parameters
        ----------
        chunks : list[dict]
            Output of ``codeatlas_code.chunk()`` or any list of RAGFlow
            chunk dicts that carry the Phase 3 metadata fields.

        Returns
        -------
        list[Symbol]
            One Symbol per code chunk that has the required metadata fields.
            Returned in the same order as *chunks*.
        """
        if not chunks:
            return []

        results: list[Symbol] = []
        skipped = 0

        for idx, chunk in enumerate(chunks):
            # Only process code chunks produced by Phase 3
            if chunk.get("source_type_kwd") != "code":
                continue

            sym = self._chunk_to_symbol(chunk, idx)
            if sym is not None:
                results.append(sym)
            else:
                skipped += 1

        if skipped:
            _log.debug(
                "SymbolExtractor.extract: skipped %d chunks (missing metadata)",
                skipped,
            )

        _log.info(
            "SymbolExtractor.extract: %d symbols from %d code chunks",
            len(results), sum(1 for c in chunks if c.get("source_type_kwd") == "code"),
        )
        return results

    def _chunk_to_symbol(self, chunk: dict, idx: int) -> Symbol | None:
        """Convert one Phase 3 chunk dict to a Symbol, or return None."""
        try:
            name: str = chunk.get("symbol_kwd", "").strip()
            if not name:
                _log.debug("chunk[%d]: missing symbol_kwd, skipping", idx)
                return None

            language: str = chunk.get("language_kwd", "python").strip() or "python"

            start_line = chunk.get("start_line_int")
            end_line   = chunk.get("end_line_int")
            if start_line is None or end_line is None:
                _log.debug("chunk[%d]: missing line fields, skipping", idx)
                return None

            start_line = int(start_line)
            end_line   = int(end_line)

            filename: str = chunk.get("docnm_kwd", "").strip() or "<unknown>"

            # Infer kind from the content_with_weight header or name heuristic.
            # The header line is: "# python | SymbolName | file:L1-L2"
            # We also parse the actual source text for a definitive answer.
            content: str = chunk.get("content_with_weight", "")
            kind = self._infer_kind(content, name)

            # Extract imports from the chunk's source text
            imports = _imports_from_source(content)

            return Symbol(
                name=name,
                kind=kind,
                file=filename,
                start_line=start_line,
                end_line=end_line,
                imports=imports,
                language=language,
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "SymbolExtractor._chunk_to_symbol: chunk[%d] error — %s", idx, exc
            )
            return None

    @staticmethod
    def _infer_kind(content: str, name: str) -> str:
        """
        Determine whether *name* refers to a function or class.

        Checks the source text in *content* for a ``class <name>`` or
        ``def <name>`` definition.  Falls back to ``"function"`` if
        the text is empty or ambiguous.
        """
        if not content:
            return "function"

        # Fast path: scan first non-blank, non-comment lines
        # The content_with_weight starts with the header comment, then the source
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Match optional decorators then class/def
            if re.match(rf"class\s+{re.escape(name)}\b", stripped):
                return "class"
            if re.match(r"(async\s+)?def\s+", stripped):
                return "function"
            # Might be a decorator line — keep looking
            if stripped.startswith("@"):
                continue
            # First real code line reached without match — stop
            break

        # Fallback: use ast on the entire content (strips the header line)
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == name:
                    return "class"
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name == name:
                    return "function"
        except SyntaxError:
            pass

        return "function"   # safe default

    # ------------------------------------------------------------------
    # Path 2: from disk files
    # ------------------------------------------------------------------

    def extract_from_files(self, paths: list[str | Path]) -> list[Symbol]:
        """
        Extract symbols from a list of source file paths on disk.

        Supports ``.py`` files only in Phase 6.

        Parameters
        ----------
        paths : list[str | Path]
            File paths to process.  Non-existent or non-``.py`` files are
            skipped with a debug log entry.

        Returns
        -------
        list[Symbol]
            All symbols found across all files, in file order.
        """
        results: list[Symbol] = []
        use_ts = _ts_available()

        for path in paths:
            path = Path(path)
            if not path.exists():
                _log.debug("extract_from_files: path not found: %s", path)
                continue
            if path.suffix.lower() not in (".py", ".pyi"):
                _log.debug("extract_from_files: skipping non-Python file: %s", path)
                continue

            syms = self._extract_one_file(path, use_ts)
            results.extend(syms)

        _log.info(
            "SymbolExtractor.extract_from_files: %d symbols from %d files",
            len(results), len(paths),
        )
        return results

    def extract_from_directory(
        self,
        dir_path: str | Path,
        *,
        recursive: bool = True,
    ) -> list[Symbol]:
        """
        Extract symbols from all ``.py`` files under *dir_path*.

        Parameters
        ----------
        dir_path : str | Path
            Root directory to search.
        recursive : bool
            If ``True`` (default), search subdirectories recursively.
            If ``False``, only process files directly inside *dir_path*.

        Returns
        -------
        list[Symbol]
            All symbols found, sorted by (file, start_line).
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            _log.warning(
                "extract_from_directory: %r is not a directory", str(dir_path)
            )
            return []

        pattern = "**/*.py" if recursive else "*.py"
        py_files = sorted(dir_path.glob(pattern))

        _log.info(
            "extract_from_directory: found %d .py files under %s",
            len(py_files), dir_path,
        )
        return self.extract_from_files(py_files)

    def _extract_one_file(self, path: Path, use_ts: bool) -> list[Symbol]:
        """Read *path* and extract symbols using tree-sitter or ast."""
        try:
            source_bytes = path.read_bytes()
        except OSError as exc:
            _log.warning("_extract_one_file: cannot read %s — %s", path, exc)
            return []

        filename = str(path)

        # Extract module-level imports (always use ast for this — reliable)
        try:
            source_str = source_bytes.decode("utf-8", errors="replace")
        except Exception:
            source_str = ""

        file_imports = _imports_from_source(source_str)

        if use_ts:
            try:
                syms = _ts_extract_symbols(source_bytes, filename, file_imports)
                if syms:
                    return syms
                # Empty can mean legitimate (no defs) or parser issue; fall through
            except Exception as exc:  # noqa: BLE001
                _log.debug(
                    "_extract_one_file: tree-sitter failed for %s (%s), "
                    "falling back to ast", path, exc
                )

        # AST fallback
        return _ast_extract_symbols(source_str, filename, file_imports)
