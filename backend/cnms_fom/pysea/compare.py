"""Comparing a pySEA determination against a ModalFit one.

Two instruments, two forward models, one quantity. An electron microscope and an
X-ray reflectometer can both report a band gap or a density for the same film, and
when they agree the number is worth trusting. When they disagree, the disagreement
is the finding.

This module never averages. FOM_PROOF Sec. 2.1 forbids merging records without a
declared aggregation rule, and two techniques landing apart do not have a usable
mean: one of them is describing something the other is not. What comes back is
every determination with its own context, plus a verdict.

The verdict vocabulary matches ``modalfit.compare`` on purpose, so a caller
handling one can handle the other:

``no_determination``   nothing in the database determines this quantity
``single_determination``  one source, nothing to cross-check it against
``consistent``         the determinations agree within their stated uncertainties
``disagreement``       the spread exceeds what the uncertainties allow
"""

from __future__ import annotations

from dataclasses import dataclass

from cnms_fom.pysea.contract import enum_value

#  Relative spread above which two determinations are flagged when neither side
#  reports an uncertainty. A flag for a human, not a significance test, and the
#  summary says so rather than implying a test that was not performed.
DISAGREEMENT_FRACTION = 0.10


@dataclass(frozen=True)
class Determination:
    """One independent measurement or calculation of a quantity."""

    platform: str
    label: str
    value: float
    uncertainty: float | None
    units: str | None
    method: str
    provenance_tier: str
    source: str

    def as_dict(self) -> dict:
        return {
            "platform": self.platform,
            "label": self.label,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "units": self.units,
            "method": self.method,
            "provenance_tier": self.provenance_tier,
            "source": self.source,
        }


def _pysea_determinations(session, sample_id: str, quantity: str) -> list[Determination]:
    """Every pySEA scalar for this sample and quantity, promoted or not.

    Unpromoted scalars are included. A number that failed a promotion gate is
    still a determination someone made, and hiding it here would mean a
    disagreement went unseen because one side was refused for missing context.
    Each carries its promotion status in the label so the difference is visible.
    """
    from cnms_fom.db.models import PySeaDerivedScalar, PySeaRecord
    from cnms_fom.pysea.instrument_state import parse_instrument_state
    from cnms_fom.pysea.promote import _method_string, _tier_for

    rows = (
        session.query(PySeaDerivedScalar, PySeaRecord)
        .join(PySeaRecord, PySeaDerivedScalar.pysea_record_id == PySeaRecord.id)
        .filter(PySeaRecord.sample_id == sample_id)
        .filter(PySeaDerivedScalar.property_key == quantity)
        .filter(PySeaDerivedScalar.value.isnot(None))
        .all()
    )

    determinations: list[Determination] = []
    for scalar, record in rows:
        state = parse_instrument_state(record.instrument_state)
        tier = _tier_for(record.record_kind, scalar.derivation)
        determinations.append(
            Determination(
                platform="pySEA",
                label=(
                    f"pySEA {enum_value(record.record_kind)} #{record.id} "
                    f"({enum_value(scalar.promotion_status)})"
                ),
                value=float(scalar.value),
                uncertainty=scalar.uncertainty,
                units=scalar.units,
                method=_method_string(record, scalar, state),
                provenance_tier=tier.value,
                source=f"pysea_record:{record.id}",
            )
        )
    return determinations


def _modalfit_determinations(session, sample_id: str, quantity: str) -> list[Determination]:
    """Promoted ModalFit values for this sample and quantity.

    Read from ``property_values`` rather than from the fits, because that is where
    a ModalFit number lands once it has passed its own gates. A fit that was never
    promoted has not cleared them, and pretending otherwise here would let this
    comparison assert something ``modalfit.promote`` refused.
    """
    from cnms_fom.db.models import FitRecord, PropertyValue

    fit_ids = [
        row.id
        for row in session.query(FitRecord.id).filter(FitRecord.sample_id == sample_id).all()
    ]
    if not fit_ids:
        return []

    locators = {f"fit_record:{fit_id}" for fit_id in fit_ids}
    rows = (
        session.query(PropertyValue)
        .filter(PropertyValue.property_key == quantity)
        .filter(PropertyValue.value.isnot(None))
        .filter(PropertyValue.source_locator.in_(locators))
        .all()
    )
    return [
        Determination(
            platform="ModalFit",
            label=f"ModalFit {row.source_locator}",
            value=float(row.value),
            uncertainty=row.uncertainty,
            units=row.units,
            method=row.method or "method not recorded",
            provenance_tier=(
                row.provenance_tier.value
                if hasattr(row.provenance_tier, "value")
                else str(row.provenance_tier)
            ),
            source=row.source_locator or f"property_value:{row.id}",
        )
        for row in rows
    ]


def _verdict(determinations: list[Determination], quantity: str) -> dict:
    """Spread and verdict over a set of determinations. Never a merged value."""
    if not determinations:
        return {
            "verdict": "no_determination",
            "summary": f"Nothing in the database determines {quantity} for this sample.",
        }
    if len(determinations) == 1:
        only = determinations[0]
        return {
            "verdict": "single_determination",
            "summary": (
                f"{quantity} is determined once, by {only.label} ({only.value:.4g}"
                f"{' ' + only.units if only.units else ''}). Nothing cross-checks it: a "
                "second platform measuring the same sample would make this number "
                "falsifiable."
            ),
        }

    units = {d.units for d in determinations if d.units}
    if len(units) > 1:
        return {
            "verdict": "disagreement",
            "summary": (
                f"{quantity} is reported in more than one unit ({sorted(units)}). These are "
                "not comparable as stored, and converting them here would be this function "
                "asserting an equivalence nobody recorded."
            ),
        }

    values = [d.value for d in determinations]
    lo, hi = min(values), max(values)
    mid = (lo + hi) / 2.0
    spread = hi - lo
    relative = spread / abs(mid) if mid else float("inf")

    sigmas = [d.uncertainty for d in determinations if d.uncertainty is not None]
    if len(sigmas) >= 2:
        combined = sum(sigma**2 for sigma in sigmas) ** 0.5
        disagrees = spread > 2.0 * combined
        basis = (
            f"spread {spread:.4g} against combined 1σ {combined:.4g} over "
            f"{len(sigmas)} reported uncertainties"
        )
    else:
        disagrees = relative > DISAGREEMENT_FRACTION
        basis = (
            f"relative spread {relative:.1%} against a {DISAGREEMENT_FRACTION:.0%} review "
            "threshold. Fewer than two determinations reported an uncertainty, so this is a "
            "flag for a person rather than a significance test"
        )

    labels = ", ".join(f"{d.platform}={d.value:.4g}" for d in determinations)
    if disagrees:
        summary = (
            f"{quantity} determined {len(determinations)} times: {labels}. These disagree "
            f"({basis}). Independent platforms landing this far apart means one is "
            "describing something the other is not: a different sampled volume, an "
            "uncorrected instrument response, or a model that fits the data without being "
            "right. Do not average them."
        )
    else:
        summary = (
            f"{quantity} determined {len(determinations)} times: {labels}. These are "
            f"consistent ({basis}). Both determinations stand; the agreement is evidence "
            "about the number, not a reason to merge the records."
        )

    return {"verdict": "disagreement" if disagrees else "consistent", "summary": summary}


def compare_across_platforms(session, sample_id: str, *, quantity: str) -> dict:
    """Every determination of one quantity for one sample, across platforms.

    Returns the determinations and a verdict. No aggregation, no mean, no
    preferred value: two numbers that disagree are two findings.
    """
    determinations = [
        *_pysea_determinations(session, sample_id, quantity),
        *_modalfit_determinations(session, sample_id, quantity),
    ]
    determinations.sort(key=lambda d: (d.platform, d.value))

    result = _verdict(determinations, quantity)
    platforms = sorted({d.platform for d in determinations})

    return {
        "sample_id": sample_id,
        "quantity": quantity,
        "platforms": platforms,
        "n_determinations": len(determinations),
        "determinations": [d.as_dict() for d in determinations],
        **result,
        "note": (
            "No value here is an average. FOM_PROOF Sec. 2.1 forbids merging records "
            "without a declared aggregation rule, and two platforms disagreeing do not "
            "have a usable mean."
        ),
    }


def cross_platform_disagreements(session, sample_id: str) -> dict:
    """Every quantity this sample has more than one determination of.

    The overview worth looking at before trusting any single number: it names the
    quantities where two platforms have something to say and whether they agree.
    """
    from cnms_fom.db.models import PySeaDerivedScalar, PySeaRecord

    keys = {
        row[0]
        for row in session.query(PySeaDerivedScalar.property_key)
        .join(PySeaRecord, PySeaDerivedScalar.pysea_record_id == PySeaRecord.id)
        .filter(PySeaRecord.sample_id == sample_id)
        .filter(PySeaDerivedScalar.property_key.isnot(None))
        .distinct()
        .all()
    }

    compared = [compare_across_platforms(session, sample_id, quantity=key) for key in sorted(keys)]
    return {
        "sample_id": sample_id,
        "quantities": compared,
        "disagreements": [c["quantity"] for c in compared if c["verdict"] == "disagreement"],
        "unchecked": [c["quantity"] for c in compared if c["verdict"] == "single_determination"],
    }
