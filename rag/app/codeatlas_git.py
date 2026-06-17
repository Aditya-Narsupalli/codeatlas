# rag/app/codeatlas_git.py
#
# CodeAtlas — Phase 2: Git Connector
# ---------------------------------------------------------------------------
# Ingests a git repository into RAGFlow as one chunk-dict per commit.
#
# Integration note
# ----------------
# RAGFlow's task_executor dispatches to parsers via FACTORY[parser_id].chunk().
# To activate this connector, add one line to rag/svr/task_executor.py:
#
#   from rag.app import codeatlas_git           # (1) import
#   FACTORY["git"] = codeatlas_git              # (2) register
#
# The roadmap specifies the registration as a 1-line addition to
# rag/app/__init__.py (currently a bare licence header).  We expose the
# MIME_GIT constant there so other modules can import it without depending on
# this file directly.
#
# Chunk dict contract (matches RAGFlow's tokenize() + tokenize_chunks()):
#   content_with_weight  str   full text RAGFlow will embed and search
#   docnm_kwd            str   filename / document name keyword
#   title_tks            str   tokenised title (set by caller or us)
#   source_type_kwd      str   "git"  (CodeAtlas metadata, filterable)
#   commit_sha_kwd       str   40-char hex SHA
#   author_kwd           str   "Name <email>"
#   date_kwd             str   ISO-8601 UTC datetime
#   repo_url_kwd         str   remote origin URL (or local path)
#
# The _kwd suffix follows RAGFlow's own convention for keyword-indexed fields.
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Feature flag guard — checked at call time so the flag can be toggled in
# conf/codeatlas.yaml without restarting.
# ---------------------------------------------------------------------------
from codeatlas.flags import is_enabled
from codeatlas.logger import get_logger

_log = get_logger(__name__)

# Parser identifier registered in RAGFlow's FACTORY dict.
MIME_GIT: str = "git"

# Maximum number of commits to ingest per repository. Override via
# parser_config["max_commits"] in the KB settings.
_DEFAULT_MAX_COMMITS: int = 2000

# Maximum diff lines to include per commit (keeps chunks within token budget).
_MAX_DIFF_LINES: int = 150


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_git() -> Any:
    """Lazy-import gitpython and raise a clear error if missing."""
    try:
        import git  # noqa: PLC0415
        return git
    except ImportError as exc:
        raise ImportError(
            "gitpython is required for the CodeAtlas Git connector. "
            "Install it with: pip install gitpython"
        ) from exc


def _ensure_repo(git_mod: Any, source: str, workdir: str) -> tuple[Any, bool]:
    """
    Return (Repo, cloned) for *source*.

    *source* may be:
    - a local filesystem path that is already a git repo
    - a remote URL (https:// or git@…) — we clone into a temp dir under workdir
    """
    is_url = source.startswith(("http://", "https://", "git@", "ssh://", "git://"))

    if is_url:
        clone_path = os.path.join(workdir, "repo")
        _log.info("Cloning %s → %s", source, clone_path)
        repo = git_mod.Repo.clone_from(
            source,
            clone_path,
            depth=None,           # full history for complete commit walk
            no_single_branch=True,
        )
        return repo, True

    local_path = Path(source)
    if not local_path.exists():
        raise FileNotFoundError(f"Git source not found: {source!r}")
    repo = git_mod.Repo(str(local_path), search_parent_directories=True)
    return repo, False


def _diff_summary(commit: Any, max_lines: int) -> str:
    """Return a truncated unified diff for *commit* (first parent only)."""
    try:
        if not commit.parents:
            # Initial commit — diff against empty tree
            diff_text = commit.repo.git.show(
                commit.hexsha,
                "--stat",
                "--no-color",
            )
        else:
            diff_text = commit.repo.git.diff(
                commit.parents[0].hexsha,
                commit.hexsha,
                "--stat",
                "--no-color",
            )
    except Exception:  # noqa: BLE001
        diff_text = "(diff unavailable)"

    lines = diff_text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"… (truncated at {max_lines} lines)")
    return "\n".join(lines)


def _commit_to_chunk(commit: Any, repo_url: str, doc_base: dict) -> dict:
    """
    Build one RAGFlow chunk dict from a gitpython Commit object.

    We build the chunk dict ourselves (without calling rag.nlp.tokenize)
    because we do not have access to rag_tokenizer in this standalone module.
    The caller (task_executor via chunker.chunk) will pass the result list
    straight to the embedding pipeline, which only requires content_with_weight.
    """
    import copy

    # Authored date in UTC ISO-8601
    authored_dt: datetime = commit.authored_datetime
    if authored_dt.tzinfo is None:
        authored_dt = authored_dt.replace(tzinfo=timezone.utc)
    date_str = authored_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    author_str = f"{commit.author.name} <{commit.author.email}>"
    sha = commit.hexsha
    short_sha = sha[:12]
    message = (commit.message or "").strip()

    diff = _diff_summary(commit, _MAX_DIFF_LINES)

    # Human-readable content that will be embedded and searched.
    content = (
        f"Commit {short_sha}\n"
        f"Author: {author_str}\n"
        f"Date:   {date_str}\n"
        f"\n"
        f"{message}\n"
        f"\n"
        f"--- diff summary ---\n"
        f"{diff}"
    )

    d = copy.deepcopy(doc_base)

    # Core RAGFlow search fields
    d["content_with_weight"] = content

    # CodeAtlas metadata fields (keyword-indexed, filterable in Elasticsearch)
    d["source_type_kwd"] = "git"
    d["commit_sha_kwd"] = sha
    d["author_kwd"] = author_str
    d["date_kwd"] = date_str
    d["repo_url_kwd"] = repo_url

    return d


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
    RAGFlow parser entry-point for git repositories.

    Parameters
    ----------
    filename:
        The git repository source — either a filesystem path or a remote URL.
        RAGFlow passes the document's stored ``name`` field here; the KB
        ingest UI should store the URL/path in that field when parser_id="git".
    binary:
        Ignored for git sources (RAGFlow always passes file bytes; we accept
        and discard this to match the interface).
    lang:
        Language hint from the KB ("English" / "Chinese"). Kept for interface
        compatibility; git commit messages are typically mixed-language.
    callback:
        RAGFlow progress callback: ``callback(progress_float, message_str)``.
    **kwargs:
        parser_config, tenant_id, kb_id, etc. passed through by task_executor.

    Returns
    -------
    list[dict]
        One chunk dict per commit, newest-first. Returns [] when the
        ``git_connector`` feature flag is disabled.
    """
    # ── Feature flag guard ────────────────────────────────────────────────
    if not is_enabled("git_connector"):
        _log.info(
            "Git connector is disabled (feature flag 'git_connector' is off). "
            "Set git_connector: true in conf/codeatlas.yaml to enable."
        )
        if callback:
            callback(1.0, "Git connector disabled via feature flag.")
        return []

    # ── Resolve source ────────────────────────────────────────────────────
    # filename holds the git URL or path supplied in the KB source field.
    git_source = filename.strip()

    parser_config: dict = kwargs.get("parser_config", {}) or {}
    max_commits: int = int(parser_config.get("max_commits", _DEFAULT_MAX_COMMITS))

    if callback:
        callback(0.02, f"Git connector: opening {git_source!r}")

    git = _import_git()

    # Base doc dict — matches RAGFlow's minimal required fields.
    # We cannot call rag_tokenizer here (not available without the full RAGFlow
    # env), so we populate the keyword fields directly.
    _safe_name = re.sub(r"[^\w\-.]", "_", git_source)[:80]
    doc_base: dict = {
        "docnm_kwd": _safe_name,
        "title_tks": _safe_name,        # will be re-tokenised by embedder
        "title_sm_tks": _safe_name,
    }

    workdir = tempfile.mkdtemp(prefix="codeatlas_git_")
    try:
        repo, cloned = _ensure_repo(git, git_source, workdir)
        repo_url = git_source  # store the original URL / path

        if callback:
            callback(0.05, "Repository opened. Walking commit history…")

        commits = list(repo.iter_commits(max_count=max_commits))
        total = len(commits)
        _log.info("Git connector: %d commits to ingest from %s", total, git_source)

        if total == 0:
            if callback:
                callback(1.0, "No commits found in repository.")
            return []

        results: list[dict] = []
        for idx, commit in enumerate(commits):
            try:
                results.append(_commit_to_chunk(commit, repo_url, doc_base))
            except Exception as exc:  # noqa: BLE001
                _log.warning("Skipping commit %s: %s", commit.hexsha[:12], exc)

            if callback and (idx % 50 == 0 or idx == total - 1):
                progress = 0.05 + 0.90 * (idx + 1) / total
                callback(progress, f"Processed {idx + 1}/{total} commits")

        if callback:
            callback(1.0, f"Git ingestion complete: {len(results)} chunks produced.")

        _log.info(
            "Git connector finished: %d/%d commits ingested from %s",
            len(results), total, git_source,
        )
        return results

    except Exception as exc:
        _log.exception("Git connector failed for %r: %s", git_source, exc)
        if callback:
            callback(-1, f"Git connector error: {exc}")
        raise

    finally:
        # Clean up clone directory (skip for local repos we didn't clone)
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
