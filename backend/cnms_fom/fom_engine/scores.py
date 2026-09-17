"""Composite application scores (FOM_PROOF Sec. 6.2).

    F_a = prod_q z_q ^ w_aq ,   sum_q w_aq = 1 ,   w_aq >= 0        (Eq. 29)
    ln F_a = sum_q w_aq ln z_q                                       (Eq. 30)

Computed in log space throughout: it is numerically better behaved, and Eq. (30)
is the form every downstream result (Gamma, the covariance identity, the
integrity checks) is actually expressed in.

Two rules from Sec. 6.2 are enforced here rather than left to the caller:

  * **Score completeness.**  A material missing a required input is reported
    ``not_scored``.  It never receives a partial or back-filled score.
  * **Modeled inputs are quarantined.**  A score built on any modeled value is
    labelled ILLUSTRATIVE and must be reported separately from measurement-based
    results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cnms_fom.db.enums import ProvenanceTier, ScoreStatus

from .normalization import NormalizationSpec, NormalizedValue, normalize_value

#  Weights are a simplex; allow only floating-point slack.
WEIGHT_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FomSpec:
    """An executable figure of merit — the in-memory twin of ``FomDefinition``."""

    name: str
    application: str
    version: int
    weights: dict[str, float]
    normalization: dict[str, NormalizationSpec]
    description: str = ""
    approved: bool = False
    floor_eps: float = 1e-3

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(f"FOM {self.name!r} declares no property weights.")
        negative = [q for q, w in self.weights.items() if w < 0]
        if negative:
            raise ValueError(f"FOM {self.name!r}: negative weights for {negative} (Eq. 29).")
        total = sum(self.weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"FOM {self.name!r}: weights sum to {total!r}, must be 1 (Eq. 29). "
                "Weights are an application-policy choice and must be stated explicitly."
            )
        missing = set(self.weights) - set(self.normalization)
        if missing:
            raise ValueError(
                f"FOM {self.name!r}: no normalization spec for weighted properties {sorted(missing)}. "
                "Sec. 5.3: bounds, transform, direction, and floor are part of the score definition."
            )

    @property
    def required_properties(self) -> tuple[str, ...]:
        """Properties with non-zero weight — the ones completeness is judged on."""
        return tuple(sorted(q for q, w in self.weights.items() if w > 0.0))

    @classmethod
    def from_definition(cls, definition) -> FomSpec:
        """Build from a ``db.models.FomDefinition`` row."""
        return cls(
            name=definition.name,
            application=definition.application,
            version=definition.version,
            weights={str(k): float(v) for k, v in definition.weights.items()},
            normalization={
                str(k): NormalizationSpec.from_dict(str(k), v)
                for k, v in definition.normalization.items()
            },
            description=definition.description or "",
            approved=bool(definition.approved),
            floor_eps=float(definition.floor_eps),
        )


@dataclass
class ScoreResult:
    """The outcome of scoring one material under one FOM."""

    material_key: str
    fom_name: str
    fom_version: int
    status: ScoreStatus
    value: float | None = None
    log_value: float | None = None
    missing_inputs: list[str] = field(default_factory=list)
    out_of_bounds: list[str] = field(default_factory=list)
    floored: list[str] = field(default_factory=list)
    components: dict[str, dict] = field(default_factory=dict)
    uses_modeled_inputs: bool = False
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "material_key": self.material_key,
            "fom_name": self.fom_name,
            "fom_version": self.fom_version,
            "status": self.status.value,
            "value": self.value,
            "log_value": self.log_value,
            "missing_inputs": self.missing_inputs,
            "out_of_bounds": self.out_of_bounds,
            "floored": self.floored,
            "components": self.components,
            "uses_modeled_inputs": self.uses_modeled_inputs,
            "note": self.note,
        }


def score_material(
    properties: dict[str, float | None],
    spec: FomSpec,
    *,
    material_key: str = "",
    provenance: dict[str, ProvenanceTier] | None = None,
) -> ScoreResult:
    """Evaluate Eq. (29) for one material.

    ``properties`` maps property key to raw value; ``None`` means NA (Eq. 4) and
    is never imputed.  ``provenance`` optionally maps property key to tier, which
    decides whether the result is a measurement-based score or an illustrative
    modeled scenario.
    """
    provenance = provenance or {}
    required = spec.required_properties

    missing = [
        q
        for q in required
        if properties.get(q) is None
        or (isinstance(properties.get(q), float) and not math.isfinite(properties[q]))  # type: ignore[index]
        or provenance.get(q) is ProvenanceTier.UNAVAILABLE
    ]
    if missing:
        # Sec. 6.2: report "not scored" rather than a manufactured score.
        return ScoreResult(
            material_key=material_key,
            fom_name=spec.name,
            fom_version=spec.version,
            status=ScoreStatus.NOT_SCORED,
            missing_inputs=missing,
            note=(
                "Required inputs are NA. FOM_PROOF Sec. 2.3/6.2: a missing property may not be "
                "replaced by a scaling law, another polymorph, or another specimen form."
            ),
        )

    components: dict[str, dict] = {}
    ln_f = 0.0
    out_of_bounds: list[str] = []
    floored: list[str] = []
    modeled = False

    for q in required:
        weight = spec.weights[q]
        normalized: NormalizedValue = normalize_value(float(properties[q]), spec.normalization[q])
        ln_z = math.log(normalized.z)
        ln_f += weight * ln_z

        if normalized.out_of_bounds:
            out_of_bounds.append(q)
        if normalized.floored:
            floored.append(q)
        if provenance.get(q) is ProvenanceTier.MODELED:
            modeled = True

        components[q] = {
            **normalized.as_dict(),
            "weight": weight,
            "ln_z": ln_z,
            "weighted_ln_z": weight * ln_z,
            "provenance_tier": provenance.get(q).value if provenance.get(q) else None,
        }

    notes: list[str] = []
    if out_of_bounds:
        notes.append(
            f"Outside declared normalization bounds: {out_of_bounds}. "
            "Re-freeze the bounds under a new FOM version before reporting a ranking."
        )
    if floored:
        notes.append(
            f"Floored at eps (Eq. 26): {floored}. The score is locally insensitive to these."
        )
    if modeled:
        notes.append(
            "ILLUSTRATIVE: modeled scenario. Report separately from measurement-based scores."
        )
    if not spec.approved:
        notes.append(
            "DRAFT weights: this FOM definition has not been formally approved (Sec. 6.2, Table 4)."
        )

    return ScoreResult(
        material_key=material_key,
        fom_name=spec.name,
        fom_version=spec.version,
        status=ScoreStatus.ILLUSTRATIVE if modeled else ScoreStatus.SCORED,
        value=math.exp(ln_f),
        log_value=ln_f,
        out_of_bounds=out_of_bounds,
        floored=floored,
        components=components,
        uses_modeled_inputs=modeled,
        note=" ".join(notes) or None,
    )


def score_population(
    population: dict[str, dict[str, float | None]],
    spec: FomSpec,
    *,
    provenance: dict[str, dict[str, ProvenanceTier]] | None = None,
) -> list[ScoreResult]:
    """Score every material in ``{material_key: {property_key: value}}``.

    Materials that cannot be scored are returned as NOT_SCORED entries rather
    than dropped, so a caller can report coverage honestly.
    """
    provenance = provenance or {}
    return [
        score_material(
            props, spec, material_key=key, provenance=provenance.get(key)
        )
        for key, props in population.items()
    ]
