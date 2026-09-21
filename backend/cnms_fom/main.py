"""FastAPI application entry point.

    uvicorn cnms_fom.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cnms_fom import __version__
from cnms_fom.config import get_settings
from cnms_fom.routers import (
    bo,
    cards,
    fom,
    health,
    materials,
    modalfit,
    pilot,
    rag,
    research,
)

logger = logging.getLogger(__name__)

DESCRIPTION = """\
Physics-based engine for autonomous materials discovery at CNMS.

Analysis follows the auditable pathway **structure -> property -> function**, per
`FOM_PROOF`. A direct correlation between a structural descriptor and a composite
score is reported as a summary only; the scientific result is the mediated
pathway `S_j -> P_q -> F_a` (`POST /fom/mediate`).

Constraints the API enforces rather than documents:

* A missing property stays missing. Materials with an incomplete input set come
  back `not_scored`, never partially scored (Sec. 2.3, 6.2).
* Every correlation cell carries its own complete-case `n`. There is no
  run-level sample size anywhere in the API (Sec. 7.3).
* Permutation p-values are FDR-corrected within their block (Sec. 8).
* Normalization bounds, transforms, direction corrections, floors, and weights
  are versioned parts of a FOM definition. Changing one creates a new version
  (Sec. 5.3).
* Retrieval output is for reading, not data entry. There is no code path from a
  RAG answer into `property_values` (Sec. 15.2). Fitted values from ModalFit are a
  different case — instrument-derived, promoted through an explicit call that
  requires a material identity a person supplied (`POST /modalfit/fits/{id}/promote`).
* A ModalFit co-refinement is stored as a measurement record, not a file. Where two
  techniques determine the same quantity, both determinations are kept and the
  disagreement is reported. Nothing averages them (Sec. 2.1).
* The research assistant answers only from tool results. An answer produced without
  a retrieval is replaced by an explicit data gap before it is returned.
* A literature extraction is not a measurement. `/research` produces briefs, claims,
  and proposals; none of them can reach `property_values` (Sec. 15.2). Evidence
  influences the optimizer only through `/research/campaigns/{id}/context`, where a
  proposal must be accepted by a named person before it is applied, and where it may
  narrow a search space but never widen one.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger.info("CNMS Living FOM %s starting", __version__)

    #  Report, do not fail: the API should come up so /health/ready can explain
    #  what is wrong, rather than crash-looping behind a container restart.
    try:
        from sqlalchemy import text

        from cnms_fom.db.base import get_engine

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database reachable")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database unreachable at startup: %s", exc)

    yield
    logger.info("CNMS Living FOM shutting down")


app = FastAPI(
    title="CNMS Living FOM",
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, installed extras."},
        {"name": "materials", "description": "Materials, structures, descriptors (S), properties (P)."},
        {"name": "fom", "description": "Scores, correlations, mediated effects, integrity checks."},
        {
            "name": "rag",
            "description": "Retrieval over the synthesis corpus and the multi-step research "
            "assistant. Local via Ollama by default; Anthropic by explicit opt-in.",
        },
        {
            "name": "research",
            "description": "Evidence-grounded research loop: briefs, reviewed campaign context, "
            "experiment summaries, and the autoresearch benchmark.",
        },
        {
            "name": "cards",
            "description": "Knowledge cards: accumulated concept pages, source summaries, "
            "findings, and open questions, linked by typed edges.",
        },
        {
            "name": "modalfit",
            "description": "Multi-technique co-refinements (SE/SPR/QCM/XRR/NR) as measurement "
            "records, and cross-technique agreement.",
        },
        {"name": "bo", "description": "Bayesian optimization over growth recipes."},
        {"name": "pilot", "description": "The HfO2-on-Si worked example, end to end."},
    ],
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(materials.router)
app.include_router(fom.router)
app.include_router(rag.router)
app.include_router(modalfit.router)
app.include_router(cards.router)
app.include_router(research.router)
app.include_router(bo.router)
app.include_router(pilot.router)


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "name": "CNMS Living FOM",
        "version": __version__,
        "docs": "/docs",
        "protocol": "FOM_PROOF — structure -> property -> function",
    }
