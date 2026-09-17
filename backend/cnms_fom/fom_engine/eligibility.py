"""Eligibility and the missing-data rule (FOM_PROOF Sec. 2).

Two things happen here, and they are the difference between a defensible table
and a plausible-looking one:

  * **Context matching.**  Eq. (3) makes the unit of analysis
    (composition, polymorph, specimen form, temperature, direction, method).
    Values whose context does not match the analysis context are not eligible —
    a bulk single-crystal dielectric constant and an amorphous thin-film one are
    different records, and averaging them produces a number with no referent.
  * **Missing stays missing.**  Eq. (4): ``P_iq = NA`` when no context-matched
    value exists.  There is no imputation path in this module, by construction.

``build_analysis_table`` is the only supported way to get from the database to a
matrix the statistics modules will accept.  It returns ``None`` for absent
values and reports why each one is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cnms_fom.db.enums import ProvenanceTier, SpecimenForm
from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES


@dataclass
class ContextFilter:
    """The analysis context every eligible value must match (Eq. 3).

    ``None`` on a field means "do not filter on this", which is a real choice
    that gets recorded on the run — not a default to reach for absent-mindedly.
    """

    specimen_forms: tuple[SpecimenForm, ...] | None = None
    provenance_tiers: tuple[ProvenanceTier, ...] = (
        ProvenanceTier.MEASURED,
        ProvenanceTier.CALCULATED,
    )
    temperature_k: tuple[float, float] | None = None
    frequency_hz: tuple[float, float] | None = None
    tensor_component: str | None = None
    xc_functional: str | None = None
    #  Enforce the per-property required-context fields from the registry.
    require_declared_context: bool = True

    def as_dict(self) -> dict:
        return {
            "specimen_forms": [s.value for s in self.specimen_forms] if self.specimen_forms else None,
            "provenance_tiers": [t.value for t in self.provenance_tiers],
            "temperature_k": list(self.temperature_k) if self.temperature_k else None,
            "frequency_hz": list(self.frequency_hz) if self.frequency_hz else None,
            "tensor_component": self.tensor_component,
            "xc_functional": self.xc_functional,
            "require_declared_context": self.require_declared_context,
        }


@dataclass
class Exclusion:
    """Why one value did not make it into the analysis table."""

    material_key: str
    property_key: str
    reason: str


@dataclass
class AnalysisTable:
    """A context-consistent table, with its exclusions kept alongside it."""

    material_keys: list[str]
    columns: dict[str, list[float | None]]
    context: ContextFilter
    exclusions: list[Exclusion] = field(default_factory=list)
    value_provenance: dict[str, dict[str, str]] = field(default_factory=dict)

    def coverage(self) -> dict[str, int]:
        """Non-NA count per column.  Never collapse these into a single n (Sec. 7.3)."""
        return {
            key: sum(1 for v in values if v is not None) for key, values in self.columns.items()
        }

    def as_dict(self) -> dict:
        return {
            "material_keys": self.material_keys,
            "columns": self.columns,
            "context": self.context.as_dict(),
            "coverage": self.coverage(),
            "exclusions": [
                {"material_key": e.material_key, "property_key": e.property_key, "reason": e.reason}
                for e in self.exclusions
            ],
        }


def _in_range(value: float | None, bounds: tuple[float, float] | None) -> bool:
    if bounds is None:
        return True
    if value is None:
        return False
    return bounds[0] <= value <= bounds[1]


def missing_context_fields(property_value, property_key: str) -> list[str]:
    """Which registry-declared context fields are absent on this value.

    Implements the Sec. 16 checklist at the row level: a breakdown field with no
    thickness, electrode, area, or failure criterion cannot be compared with
    another one, so it is not eligible.
    """
    spec = PHYSICAL_PROPERTIES.get(property_key)
    if spec is None:
        return []
    return [
        field_name
        for field_name in spec.required_context
        if getattr(property_value, field_name, None) in (None, "")
    ]


def is_eligible(property_value, property_key: str, context: ContextFilter) -> tuple[bool, str]:
    """Decide whether one ``PropertyValue`` may enter the analysis table."""
    if property_value.value is None:
        return False, "value is NA (Eq. 4)"

    tier = property_value.provenance_tier
    if tier not in context.provenance_tiers:
        return False, f"provenance tier {tier.value} excluded by the analysis context"

    if context.specimen_forms is not None:
        form = getattr(property_value.material, "specimen_form", None)
        if form not in context.specimen_forms:
            form_name = form.value if form else "unknown"
            return False, f"specimen form {form_name} excluded by the analysis context"

    if not _in_range(property_value.temperature_k, context.temperature_k):
        return False, "temperature outside the analysis context"
    if not _in_range(property_value.frequency_hz, context.frequency_hz):
        return False, "frequency outside the analysis context"

    if context.tensor_component and property_value.tensor_component != context.tensor_component:
        return (
            False,
            f"tensor component {property_value.tensor_component!r} != "
            f"{context.tensor_component!r} (Sec. 3.2)",
        )
    if context.xc_functional and property_value.xc_functional != context.xc_functional:
        return False, f"XC functional {property_value.xc_functional!r} != {context.xc_functional!r}"

    if context.require_declared_context:
        absent = missing_context_fields(property_value, property_key)
        if absent:
            return False, f"missing required context fields {absent} (Sec. 16)"

    return True, ""


def build_analysis_table(
    materials,
    property_keys: list[str],
    descriptor_keys: list[str],
    context: ContextFilter,
) -> AnalysisTable:
    """Assemble a context-consistent S|P table from ORM ``Material`` rows.

    When several eligible values exist for the same material and property, the
    table refuses to choose: the cell is set to NA and an exclusion is recorded.
    Picking one silently — or averaging them — is exactly the aggregation Sec. 2.1
    forbids without an explicit, defensible rule.
    """
    material_keys: list[str] = []
    columns: dict[str, list[float | None]] = {
        key: [] for key in (*descriptor_keys, *property_keys)
    }
    exclusions: list[Exclusion] = []
    provenance: dict[str, dict[str, str]] = {}

    for material in materials:
        key = (
            f"{material.formula_reduced}|{material.polymorph}|{material.specimen_form.value}"
        )
        material_keys.append(key)
        provenance[key] = {}

        by_descriptor: dict[str, list] = {}
        for dv in material.descriptors:
            by_descriptor.setdefault(dv.descriptor_key, []).append(dv)

        for dkey in descriptor_keys:
            candidates = [
                d
                for d in by_descriptor.get(dkey, [])
                if d.value is not None and d.provenance_tier in context.provenance_tiers
            ]
            if len(candidates) == 1:
                columns[dkey].append(float(candidates[0].value))
                provenance[key][dkey] = candidates[0].provenance_tier.value
            else:
                columns[dkey].append(None)
                if len(candidates) > 1:
                    exclusions.append(
                        Exclusion(
                            key,
                            dkey,
                            f"{len(candidates)} eligible values; declare an aggregation rule "
                            "before merging (Sec. 2.1)",
                        )
                    )
                elif by_descriptor.get(dkey):
                    exclusions.append(Exclusion(key, dkey, "no value passed the context filter"))

        by_property: dict[str, list] = {}
        for pv in material.properties:
            by_property.setdefault(pv.property_key, []).append(pv)

        for pkey in property_keys:
            eligible = []
            for pv in by_property.get(pkey, []):
                ok, reason = is_eligible(pv, pkey, context)
                if ok:
                    eligible.append(pv)
                else:
                    exclusions.append(Exclusion(key, pkey, reason))

            if len(eligible) == 1:
                columns[pkey].append(float(eligible[0].value))
                provenance[key][pkey] = eligible[0].provenance_tier.value
            else:
                columns[pkey].append(None)
                if len(eligible) > 1:
                    exclusions.append(
                        Exclusion(
                            key,
                            pkey,
                            f"{len(eligible)} context-matched values; narrow the context or "
                            "declare an aggregation rule (Sec. 2.1)",
                        )
                    )

    return AnalysisTable(
        material_keys=material_keys,
        columns=columns,
        context=context,
        exclusions=exclusions,
        value_provenance=provenance,
    )
