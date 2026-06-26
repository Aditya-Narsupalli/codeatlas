# api/apps/codeatlas/__init__.py
#
# CodeAtlas — Phase 9: Architecture Explorer API
# ---------------------------------------------------------------------------
# Package init for the CodeAtlas Quart blueprint sub-package.
#
# Registration
# ------------
# This blueprint is NOT auto-discovered by register_page() (which only scans
# for *_app.py, *sdk/*.py, and *restful_apis/*.py).  Instead, register it
# manually in api/apps/__init__.py alongside the backward_compat registration:
#
#   from api.apps.codeatlas import register_codeatlas_routes
#   register_codeatlas_routes(app)
#
# This keeps the activation isolated and easy to revert or feature-flag.
#
# URL prefix
# ----------
# All CodeAtlas routes are mounted at /api/codeatlas (no version prefix).
# This is intentional: CodeAtlas is an extension layer, not a versioned
# RAGFlow API endpoint.  When a versioned API is needed (Phase N+), the
# prefix can be changed here without touching arch_api.py.
# ---------------------------------------------------------------------------

from quart import Blueprint

from codeatlas.logger import get_logger

_log = get_logger(__name__)

# The blueprint object.  arch_api.py decorates its routes onto this.
blueprint = Blueprint("codeatlas", __name__)

# Import routes so their @blueprint.route decorators execute at import time.
from api.apps.codeatlas import arch_api as _arch_api  # noqa: E402, F401
from api.apps.codeatlas import flow_api as _flow_api  # noqa: E402, F401  Phase 16


def register_codeatlas_routes(app) -> None:
    """
    Register the CodeAtlas blueprint with the Quart *app* instance.

    Call this from ``api/apps/__init__.py`` after existing blueprint
    registrations, e.g.::

        from api.apps.codeatlas import register_codeatlas_routes
        register_codeatlas_routes(app)

    The blueprint is mounted at ``/api/codeatlas`` with no trailing slash.
    """
    app.register_blueprint(blueprint, url_prefix="/api/codeatlas")
    _log.info("CodeAtlas routes registered at /api/codeatlas")
