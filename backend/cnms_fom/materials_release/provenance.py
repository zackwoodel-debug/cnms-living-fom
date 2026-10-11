"""How a materials-db evidence record may appear in CNMS, and what must never cross.

materials-db separates how a value was obtained (provenance), whether a person reviewed it (review), and which conditions it
states. CNMS has one field, ``ProvenanceTier``, that analysis filters on. This module is the single, tested place that maps
the first onto the second. It is a mapping and a dry-run preview: it writes nothing, it changes no CNMS table, and it does not
change the importer (``compare_with_importer`` only *reports* where the importer's tier differs).

Rules, each with a test:

* A literature value is somebody else's claim, not a CNMS measurement. ``measured``, ``fitted`` and ``reported`` get **no**
  tier here (``tier is None``) and are labelled ``literature_*``; only a CNMS experiment or an explicit promotion by a named
  operator makes a MEASURED value.
* ``calculated`` and ``modeled`` map to their CNMS tiers. ``unknown`` is UNAVAILABLE and blocked.
* A rejected or needs-clarification review, reviewers who disagree, or a required condition that is not stated all block.
* An approval changes the label (``reviewed``), never the tier or the destination.
* There is no field for a bound, a constraint or "verified": nothing here can become a fixed fit bound, and the fit code does
  not import this module.

CNMS does not import materials-db; the vocabulary below is a copy, compared with materials-db's when both are present.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cnms_fom.db.enums import ProvenanceTier
from cnms_fom.ingest.materials_db import DEFAULT_TIER as IMPORTER_DEFAULT_TIER

PROVENANCE_WORDS = ("measured", "fitted", "calculated", "modeled", "reported", "unknown")
#  "conflict" is not a review decision; it is how a caller reports two reviewers disagreeing.
REVIEW_WORDS = ("unreviewed", "approved", "rejected", "needs_clarification")
FIELD_STATES = ("stated", "not_reported", "not_recorded", "not_applicable")

_BLOCKING_REVIEWS = ("rejected", "needs_clarification")
_LITERATURE_LABEL = {"measured": "literature_measured", "fitted": "literature_fitted", "reported": "literature_reported"}


@dataclass(frozen=True)
class EvidenceInput:
    """What the mapper needs to know about one materials-db dataset or evidence row, as plain values."""

    dataset_id: int
    provenance: str
    review: str                                # one of REVIEW_WORDS, or "conflict" when review_conflict is true
    review_conflict: bool
    states: dict                               # condition key -> field state, for the conditions the caller reads
    required: tuple = ()                       # condition keys the analysis context needs to be stated
    basis: str = ""


@dataclass(frozen=True)
class MappingResult:
    tier: ProvenanceTier | None                # None = this mapper assigns no CNMS tier (a literature claim)
    destination: str                           # "reference" or "blocked"
    label: str
    reviewed: bool
    reasons: list = field(default_factory=list)
    dry_run: bool = True


def map_evidence(rec: EvidenceInput) -> MappingResult:
    """Map one record. Unknown words raise; nothing is defaulted."""
    if rec.provenance not in PROVENANCE_WORDS:
        raise ValueError(f"{rec.provenance!r} is not a provenance word; expected one of {PROVENANCE_WORDS}")
    if rec.review_conflict:
        review = "conflict"
    elif rec.review in REVIEW_WORDS:
        review = rec.review
    else:
        raise ValueError(f"{rec.review!r} is not a review word; expected one of {REVIEW_WORDS}")

    reasons: list[str] = []
    if review == "conflict":
        reasons.append("reviewers disagree; resolve before use")
    elif review in _BLOCKING_REVIEWS:
        reasons.append(f"review is {review}")
    for key in rec.required:
        state = rec.states.get(key)
        if state != "stated":
            reasons.append(f"required condition {key!r} is not stated ({state or 'absent'})")
    if rec.provenance == "unknown":
        reasons.append("provenance is unknown")

    if rec.provenance == "calculated":
        tier, label = ProvenanceTier.CALCULATED, "calculated"
    elif rec.provenance == "modeled":
        tier, label = ProvenanceTier.MODELED, "modeled"
    elif rec.provenance == "unknown":
        tier, label = ProvenanceTier.UNAVAILABLE, "unavailable"
    else:
        tier, label = None, _LITERATURE_LABEL[rec.provenance]
    reviewed = review == "approved"
    return MappingResult(tier=tier, destination="blocked" if reasons else "reference",
                         label=label + (" (reviewed)" if reviewed else ""), reviewed=reviewed, reasons=reasons)


def preview(records) -> list[MappingResult]:
    """Dry run over many records: pure, touches no session."""
    return [map_evidence(r) for r in records]


def _importer_tier(rec: EvidenceInput) -> ProvenanceTier:
    """The tier ``ingest/materials_db.py`` gives a record of this provenance: calculated and modeled keep their tier, everything
    else takes ``DEFAULT_TIER`` (documented in that module as MEASURED). This models the rule; it does not run the importer."""
    if rec.provenance == "calculated":
        return ProvenanceTier.CALCULATED
    if rec.provenance == "modeled":
        return ProvenanceTier.MODELED
    return IMPORTER_DEFAULT_TIER


def compare_with_importer(records) -> list[dict]:
    """Where the importer's current tier and this mapper's differ. A report only; nothing is changed."""
    rows = []
    for rec in records:
        mapped = map_evidence(rec)
        imported = _importer_tier(rec)
        rows.append(dict(dataset_id=rec.dataset_id, provenance=rec.provenance, importer_tier=imported.value,
                         mapper_tier=mapped.tier.value if mapped.tier else None, differs=imported is not mapped.tier,
                         note=("importer would store this as a CNMS measurement; the contract treats it as literature"
                               if imported is ProvenanceTier.MEASURED and mapped.tier is None else "")))
    return rows
