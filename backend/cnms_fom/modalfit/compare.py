"""Cross-technique comparison over stored ModalFit refinements.

This is the reason the fit records exist.  Five techniques fitting one shared
slab model means a film's thickness can be determined more than once by
unrelated physics — XRR's Parratt matrix method and SE's transfer-matrix
ellipsometry share no forward model, no instrument, and no systematic error.
When they agree the number is worth trusting; when they disagree, the
disagreement is the finding, and it is the only place in this platform where a
single measurement can be checked against anything but itself.

So the functions here refuse to average.  FOM_PROOF Sec. 2.1 forbids merging
records without a declared aggregation rule, and two techniques disagreeing by
40% on a thickness do not have a mean worth reporting — they have a problem
worth naming.  ``compare_parameter`` returns every determination alongside the
spread, and says which ones the evidence supports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#  Parameters worth comparing across techniques, with their units and the
#  techniques that can actually determine them.  A QCM fit does not measure
#  roughness at all, so an "absent" roughness from a QCM-only fit is not a
#  disagreement with an XRR roughness — it is silence, and the two must not be
#  compared as if they were rival claims.
COMPARABLE_PARAMETERS: dict[str, dict] = {
    "thickness": {
        "units": "Å",
        "determined_by": ("SE", "XRR", "NR", "QCM", "SPR"),
        "note": "QCM reports an acoustic (wet) thickness; SE/XRR/NR report an optical or "
        "nuclear one. A systematic offset between them is physics, not error.",
    },
    "roughness": {
        "units": "Å",
        "determined_by": ("SE", "XRR", "NR"),
        "note": "ModalFit's SPR path applies no roughness at all, so an SPR fit is silent on "
        "it rather than in agreement.",
    },
    "density": {
        "units": "g/cm³",
        "determined_by": ("XRR", "NR", "QCM"),
        "note": "XRR/NR reach density through SLD; QCM reaches it acoustically.",
    },
    "sld_real": {"units": "1e-6/Å²", "determined_by": ("XRR", "NR")},
    "sld_imag": {"units": "1e-6/Å²", "determined_by": ("XRR", "NR")},
    "n": {"units": "", "determined_by": ("SE", "SPR")},
    "k": {"units": "", "determined_by": ("SE", "SPR")},
}

#  Relative spread above which two determinations are called a disagreement,
#  used only when neither carries an uncertainty.  Not a significance test:
#  without uncertainties there is nothing to test, and this is a threshold for
#  "a human should look at this", which is what it is described as in the output.
DISAGREEMENT_FRACTION = 0.10


@dataclass
class Determination:
    """One fit's value for one parameter on one layer."""

    fit_record_id: int
    techniques: list[str]
    value: float
    uncertainty: float | None = None
    chi2: float | None = None
    algorithm: str | None = None
    was_free: bool = False
    at_bound: bool = False
    caveats: list[str] = field(default_factory=list)

    @property
    def is_single_technique(self) -> bool:
        return len(self.techniques) == 1

    @property
    def label(self) -> str:
        return "+".join(self.techniques)

    def as_dict(self) -> dict:
        return {
            "fit_record_id": self.fit_record_id,
            "techniques": self.techniques,
            "label": self.label,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "chi2": self.chi2,
            "algorithm": self.algorithm,
            "was_free": self.was_free,
            "at_bound": self.at_bound,
            "caveats": self.caveats,
        }


def _layer_caveats(record, layer, parameter: str) -> list[str]:
    """Everything about this determination that qualifies it."""
    caveats: list[str] = []
    techniques = set(record.techniques or [])

    if parameter not in (layer.free_parameters or []):
        caveats.append(
            "held fixed during refinement — an input to the fit, not a result of it"
        )
    bounds = (layer.bounds or {}).get(parameter) or {}
    value = (
        layer.thickness_ang
        if parameter == "thickness"
        else layer.roughness_ang
        if parameter == "roughness"
        else layer.density_g_cm3
        if parameter == "density"
        else _from_blocks(layer, parameter)
    )
    lower, upper = bounds.get("min"), bounds.get("max")
    if value is not None and lower is not None and upper is not None and upper > lower:
        tol = 1e-6 * (upper - lower)
        if value <= lower + tol or value >= upper - tol:
            caveats.append(f"sits on its fit bound [{lower}, {upper}] — clamped, not converged")

    if record.uses_placeholder_optical_constants and parameter in ("n", "k"):
        caveats.append("optical constants are ModalFit's placeholder n/k values")
    if techniques & {"XRR", "NR"} and record.resolution_smearing_applied is False:
        caveats.append(
            "XRR/NR fitted with dq=0: no angular-resolution smearing, so fringe contrast is "
            "sharper than the instrument measured"
        )
    if "SPR" in techniques and record.roughness_applied_to_spr is False and parameter == "roughness":
        caveats.append("the SPR forward model ignores roughness, so SPR did not constrain this")
    if record.chi2_total is None:
        caveats.append("no chi-squared recorded for this fit")
    return caveats


def _from_blocks(layer, parameter: str) -> float | None:
    """Read a parameter out of ``FitLayer.parameters`` regardless of its block."""
    for values in (layer.parameters or {}).values():
        if isinstance(values, dict) and parameter in values:
            value = values[parameter]
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _value_for(layer, parameter: str) -> float | None:
    direct = {
        "thickness": layer.thickness_ang,
        "roughness": layer.roughness_ang,
        "density": layer.density_g_cm3,
    }.get(parameter)
    return direct if direct is not None else _from_blocks(layer, parameter)


def fits_for_sample(session, sample_id: str, *, limit: int = 50) -> list:
    """Every stored refinement for one sample, newest first."""
    from cnms_fom.db.models import FitRecord

    return (
        session.query(FitRecord)
        .filter(FitRecord.sample_id == sample_id)
        .order_by(FitRecord.fitted_at.desc().nullslast(), FitRecord.id.desc())
        .limit(limit)
        .all()
    )


def layer_matches(layer, label: str | None) -> bool:
    """Whether a layer answers to ``label``, by its label or its material.

    Shared so that every tool taking a ``layer_label`` resolves it identically.
    They were written separately and drifted: ``compare_parameter`` accepted the
    material ("HfO2") while the plausibility check wanted only the label
    ("hfo2_film"), so the same argument produced a comparison from one and silence
    from the other. Silence that looks like "nothing to report" is the worst
    possible failure here — it reads as a clean result.
    """
    if not label:
        return True
    wanted = label.strip().lower()
    return wanted in {
        (layer.label or "").strip().lower(),
        (layer.material or "").strip().lower(),
    } - {""}


def available_layer_labels(record) -> list[str]:
    """Every name a caller could legitimately pass for this record's film layers."""
    names: list[str] = []
    for layer in record.layers:
        if layer.role != "layer":
            continue
        for candidate in (layer.label, layer.material):
            if candidate and candidate not in names:
                names.append(candidate)
    return names


def _match_layer(record, layer_label: str | None):
    """Pick the layer a comparison is about.

    With no label, the single film layer is used, and an ambiguous stack is
    refused rather than resolved by position: "the film layer" in a two-film
    stack is not a thing, and silently taking the first one would attribute one
    layer's thickness to another.
    """
    films = [layer for layer in record.layers if layer.role == "layer"]
    if layer_label:
        for layer in record.layers:
            if layer_matches(layer, layer_label):
                return layer
        return None
    if len(films) == 1:
        return films[0]
    return None


def compare_parameter(
    session,
    sample_id: str,
    *,
    parameter: str = "thickness",
    layer_label: str | None = None,
) -> dict:
    """Collect every determination of one parameter on one layer, across fits.

    Returns the determinations, the spread, and a verdict — never a merged value.
    Two techniques that disagree do not have a usable mean; they have a
    discrepancy that someone has to explain, and hiding it inside an average is
    how a 40% thickness error becomes a published number.
    """
    if parameter not in COMPARABLE_PARAMETERS:
        return {
            "sample_id": sample_id,
            "parameter": parameter,
            "error": f"{parameter!r} is not a cross-technique comparable parameter. "
            f"Available: {', '.join(sorted(COMPARABLE_PARAMETERS))}.",
            "determinations": [],
        }

    spec = COMPARABLE_PARAMETERS[parameter]
    records = fits_for_sample(session, sample_id)
    if not records:
        return {
            "sample_id": sample_id,
            "parameter": parameter,
            "determinations": [],
            "verdict": "no_fits",
            "summary": f"No ModalFit refinements are stored for sample {sample_id!r}.",
        }

    determinations: list[Determination] = []
    ambiguous: list[int] = []
    silent: list[dict] = []

    for record in records:
        layer = _match_layer(record, layer_label)
        if layer is None:
            ambiguous.append(record.id)
            continue
        value = _value_for(layer, parameter)
        if value is None:
            continue

        techniques = list(record.techniques or [])
        capable = [t for t in techniques if t in spec["determined_by"]]
        if not capable:
            #  Silence, not disagreement: none of this fit's techniques can
            #  determine this parameter, so its value is a carried-through input.
            silent.append({"fit_record_id": record.id, "techniques": techniques})
            continue

        chi2 = None
        if record.chi2_by_technique and len(capable) == 1:
            chi2 = record.chi2_by_technique.get(capable[0])
        chi2 = chi2 if chi2 is not None else record.chi2_total

        determinations.append(
            Determination(
                fit_record_id=record.id,
                techniques=capable,
                value=float(value),
                uncertainty=(layer.uncertainties or {}).get(parameter),
                chi2=chi2,
                algorithm=record.algorithm,
                was_free=parameter in (layer.free_parameters or []),
                at_bound=any("clamped" in c for c in _layer_caveats(record, layer, parameter)),
                caveats=_layer_caveats(record, layer, parameter),
            )
        )

    result = {
        "sample_id": sample_id,
        "parameter": parameter,
        "layer": layer_label,
        "units": spec["units"],
        "determined_by": list(spec["determined_by"]),
        "determinations": [d.as_dict() for d in determinations],
        "silent_fits": silent,
        "n_fits_considered": len(records),
    }
    if spec.get("note"):
        result["interpretation_note"] = spec["note"]
    if ambiguous:
        result["unresolved_layer_fits"] = ambiguous
        result["layer_hint"] = (
            "Some fits have more than one film layer and no layer_label was given; those were "
            "skipped rather than guessed. Pass layer_label to include them."
        )

    result.update(_assess(determinations, parameter))
    return result


def _assess(determinations: list[Determination], parameter: str) -> dict:
    """Spread and verdict over a set of determinations."""
    refined = [d for d in determinations if d.was_free]
    if not determinations:
        return {
            "verdict": "no_determination",
            "summary": f"No stored fit determines {parameter} for this layer.",
        }
    if len(refined) < 2:
        only = refined[0] if refined else determinations[0]
        return {
            "verdict": "single_determination",
            "summary": (
                f"{parameter} is determined once, by {only.label} "
                f"({only.value:.4g}). Nothing to cross-check it against: a second technique "
                "with data loaded would make this number falsifiable."
            ),
        }

    values = [d.value for d in refined]
    lo, hi = min(values), max(values)
    mid = (lo + hi) / 2.0
    spread = hi - lo
    relative = spread / abs(mid) if mid else float("inf")

    #  When both sides carry an uncertainty, ask whether the gap exceeds the
    #  combined 1-sigma; otherwise fall back to a flat relative threshold and
    #  say so, rather than implying a test that was not performed.
    sigmas = [d.uncertainty for d in refined if d.uncertainty is not None]
    if len(sigmas) >= 2:
        combined = (sum(s**2 for s in sigmas)) ** 0.5
        disagrees = spread > 2.0 * combined
        basis = (
            f"spread {spread:.4g} versus combined 1σ {combined:.4g} over "
            f"{len(sigmas)} reported uncertainties"
        )
    else:
        disagrees = relative > DISAGREEMENT_FRACTION
        basis = (
            f"relative spread {relative:.1%} against a {DISAGREEMENT_FRACTION:.0%} review "
            "threshold — no fit reported an uncertainty, so this is a flag for a human, not a "
            "significance test"
        )

    labels = ", ".join(f"{d.label}={d.value:.4g}" for d in refined)
    verdict = "disagreement" if disagrees else "consistent"
    summary = (
        f"{parameter} determined {len(refined)} times: {labels}. "
        + (
            f"These disagree ({basis}). Independent forward models landing this far apart means "
            "one of them is describing something the other is not — a wrong SLD, an unmodelled "
            "interlayer, or a fit that converged on a different minimum. Do not average them."
            if disagrees
            else f"These are consistent ({basis})."
        )
    )
    if len(sigmas) < 2:
        summary += (
            " Note that only DREAM (emcee) produces a posterior in ModalFit; the four scipy "
            "minimizers report no uncertainty, so most fits cannot support a statistical claim."
        )

    return {
        "verdict": verdict,
        "min": lo,
        "max": hi,
        "midpoint": mid,
        "spread": spread,
        "relative_spread": relative,
        "n_refined_determinations": len(refined),
        "summary": summary,
    }


def cross_technique_report(session, sample_id: str, *, layer_label: str | None = None) -> dict:
    """Run :func:`compare_parameter` over every comparable parameter."""
    comparisons = {
        parameter: compare_parameter(
            session, sample_id, parameter=parameter, layer_label=layer_label
        )
        for parameter in COMPARABLE_PARAMETERS
    }
    interesting = {
        name: block
        for name, block in comparisons.items()
        if block.get("determinations")
    }
    disagreements = [name for name, block in interesting.items() if block.get("verdict") == "disagreement"]
    return {
        "sample_id": sample_id,
        "layer": layer_label,
        "comparisons": interesting,
        "disagreements": disagreements,
        "n_parameters_compared": len(interesting),
        "summary": (
            f"{len(interesting)} parameter(s) have at least one determination for sample "
            f"{sample_id!r}"
            + (
                f"; {', '.join(disagreements)} disagree across techniques."
                if disagreements
                else "; no cross-technique disagreements found."
            )
        ),
    }


def fit_process_warnings(record) -> list[str]:
    """Findings about how a fit was *run*, as opposed to what it found.

    A clamped parameter and a fixed one are properties of the refinement, not of
    the physics, so they belong here rather than in the plausibility checker.

    Computed from the stored bounds rather than by reading ``describe_fit``'s
    prose: matching words in generated text is how a detector silently stops
    working when the wording is improved.
    """
    findings: list[str] = []
    for layer in record.layers:
        if layer.role != "layer":
            continue
        name = layer.label or layer.material or f"layer_{layer.layer_index}"
        free = set(layer.free_parameters or [])

        if not free:
            findings.append(
                f"On {name}, every parameter was held fixed. Those values are inputs to the fit, "
                "not results of it."
            )

        for parameter in sorted(free):
            bounds = (layer.bounds or {}).get(parameter) or {}
            low, high = bounds.get("min"), bounds.get("max")
            value = _value_for(layer, parameter)
            if value is None or low is None or high is None or high <= low:
                continue
            tolerance = 1e-6 * (high - low)
            if value <= low + tolerance or value >= high - tolerance:
                findings.append(
                    f"On {name}, {parameter}={value:g} finished on its bound [{low}, {high}]. It "
                    "has not converged — it has been clamped by a number somebody typed."
                )
    return findings


def describe_fit(record) -> str:
    """Render one fit as prose for a language model to read.

    Written for a reader that will otherwise fill gaps with plausible numbers, so
    the caveats are in the body rather than in a footnote: a model that is told
    "thickness 103.4 Å" and not told the parameter was held fixed will report a
    measurement that was never made.
    """
    techniques = "+".join(record.techniques or []) or "no technique recorded"
    lines = [
        f"ModalFit refinement #{record.id} — sample {record.sample_id or 'unrecorded'}, "
        f"stack {record.stack_id or 'unrecorded'}",
        f"Techniques co-refined: {techniques}"
        + (f" (weights {record.technique_weights})" if record.technique_weights else ""),
        f"Algorithm: {record.algorithm or 'not recorded'}",
    ]

    if record.chi2_total is not None:
        lines.append(f"Total chi-squared: {record.chi2_total:.4g}")
    else:
        lines.append(
            "Total chi-squared: NOT RECORDED — nothing distinguishes this from an unrefined "
            "starting model."
        )
    if record.chi2_by_technique:
        per = ", ".join(f"{k}={v:.4g}" for k, v in record.chi2_by_technique.items() if v is not None)
        if per:
            lines.append(f"Per-technique chi-squared: {per}")
    if record.n_free_parameters is not None:
        lines.append(f"Free parameters: {record.n_free_parameters}")

    lines.append("Stack (ambient first):")
    for layer in record.layers:
        name = layer.label or layer.material or layer.role
        bits = [f"  [{layer.layer_index}] {name} ({layer.role})"]
        if layer.thickness_ang is not None:
            bits.append(f"thickness {layer.thickness_ang:.4g} Å")
        if layer.roughness_ang is not None:
            bits.append(f"roughness {layer.roughness_ang:.4g} Å")
        if layer.density_g_cm3 is not None:
            bits.append(f"density {layer.density_g_cm3:.4g} g/cm³")
        if layer.formula:
            bits.append(f"formula {layer.formula}")
        free = layer.free_parameters or []
        bits.append(f"varied: {', '.join(free) if free else 'nothing (all parameters fixed)'}")
        lines.append(" — ".join(bits))

    if record.datasets:
        lines.append("Data fitted:")
        for dataset in record.datasets:
            parts = [f"  {dataset.technique.value}"]
            if dataset.n_points is not None:
                parts.append(f"{dataset.n_points} points")
            if dataset.x_min is not None and dataset.x_max is not None:
                parts.append(f"{dataset.x_min:.4g}–{dataset.x_max:.4g} {dataset.x_units or ''}".strip())
            if dataset.chi2 is not None:
                parts.append(f"chi2 {dataset.chi2:.4g}")
            if dataset.source_filename:
                parts.append(f"from {dataset.source_filename}")
            lines.append(", ".join(parts))

    caveats = []
    if record.uses_placeholder_optical_constants:
        caveats.append(
            "optical constants include ModalFit's placeholder n/k values, so SE- and "
            "SPR-derived numbers here are illustrative, not citable"
        )
    if set(record.techniques or []) & {"XRR", "NR"} and record.resolution_smearing_applied is False:
        caveats.append(
            "XRR/NR were fitted with dq=0 — no angular-resolution smearing, so fringe contrast "
            "is sharper than any real finite-divergence instrument measures"
        )
    if "SPR" in (record.techniques or []) and record.roughness_applied_to_spr is False:
        caveats.append("the SPR forward model applied no layer roughness")
    if record.algorithm and record.algorithm != "DREAM (emcee)":
        caveats.append(
            f"{record.algorithm} reports no posterior, so no parameter here has an uncertainty"
        )
    if caveats:
        lines.append("Caveats that qualify every number above:")
        lines.extend(f"  - {c}" for c in caveats)

    return "\n".join(lines)
