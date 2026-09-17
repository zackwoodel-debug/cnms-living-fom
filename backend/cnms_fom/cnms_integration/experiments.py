"""Experiment metadata: from a BO suggestion to a run, and from a run back to data.

The round trip is the point. A suggestion that is never linked to a run produces
no observation, and a run whose measured properties are not linked back to the
recipe cannot close the loop. Both links are explicit columns rather than
conventions, so a broken loop shows up as a NULL rather than as a campaign that
quietly stops improving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from cnms_fom.db.enums import ProvenanceTier


@dataclass
class ExperimentPlan:
    """A recipe about to be run, with its facility context."""

    recipe: dict
    instrument_id: str | None = None
    proposal_id: str | None = None
    operator: str | None = None
    sample_id: str | None = None
    bo_suggestion_id: int | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "recipe": self.recipe,
            "instrument_id": self.instrument_id,
            "proposal_id": self.proposal_id,
            "operator": self.operator,
            "sample_id": self.sample_id,
            "bo_suggestion_id": self.bo_suggestion_id,
            "notes": self.notes,
        }


@dataclass
class ExperimentOutcome:
    """Measured results from a completed run.

    Values land in ``property_values`` with ``provenance_tier=MEASURED`` and the
    full context the protocol requires. Anything without that context is not a
    usable measurement — see ``missing_context`` before writing.
    """

    experiment_id: int
    material_id: int | None = None
    #  {property_key: {"value": x, "units": "...", plus context fields}}
    measurements: dict[str, dict] = field(default_factory=dict)
    succeeded: bool = True
    failure_reason: str = ""

    def missing_context(self) -> dict[str, list[str]]:
        """Per property, which registry-required context fields are absent."""
        from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES

        out: dict[str, list[str]] = {}
        for key, payload in self.measurements.items():
            spec = PHYSICAL_PROPERTIES.get(key)
            if spec is None:
                out[key] = ["unknown property key"]
                continue
            absent = [f for f in spec.required_context if payload.get(f) in (None, "")]
            if absent:
                out[key] = absent
        return out


def plan_to_experiment_kwargs(plan: ExperimentPlan, instrument_pk: int | None = None) -> dict:
    """Flatten a plan into ``db.models.Experiment`` column values."""
    return {
        "instrument_id": instrument_pk,
        "proposal_id": plan.proposal_id,
        "operator": plan.operator,
        "sample_id": plan.sample_id,
        "bo_suggestion_id": plan.bo_suggestion_id,
        "recipe": plan.recipe,
        "status": "planned",
        "notes": plan.notes,
    }


def outcome_to_property_kwargs(outcome: ExperimentOutcome) -> list[dict]:
    """Flatten measurements into ``db.models.PropertyValue`` column values.

    Refuses to emit a value that is missing its declared context: a breakdown
    field with no thickness or failure criterion cannot be compared with any
    other, so storing it would only add an ineligible row (Sec. 16).
    """
    absent = outcome.missing_context()
    if absent:
        raise ValueError(
            f"Refusing to store measurements missing required context: {absent}. "
            "FOM_PROOF Sec. 16: context is what makes a value comparable."
        )

    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    for key, payload in outcome.measurements.items():
        rows.append(
            {
                "material_id": outcome.material_id,
                "experiment_id": outcome.experiment_id,
                "property_key": key,
                "provenance_tier": ProvenanceTier.MEASURED,
                "ingested_at": now,
                **{k: v for k, v in payload.items() if k != "property_key"},
            }
        )
    return rows


def fetch_proposal_metadata(proposal_id: str) -> dict:
    """Look up proposal/user-agreement metadata.

    TODO(CNMS): implement against ``CNMS_PROPOSAL_API_URL``. Needed:
      * proposal title, PI, and participating users;
      * approved techniques and instruments;
      * data-release and embargo terms — these govern what may appear in a
        published ranking, so they must be checked before any export;
      * allocation window and remaining tool time.
    """
    raise NotImplementedError(
        f"Proposal lookup for {proposal_id!r} is not implemented. Configure "
        "CNMS_PROPOSAL_API_URL and implement the client; do not hard-code proposal metadata."
    )
