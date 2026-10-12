"""What a person should know before trusting a fit (workflow step 7). Pure: reads a FitRecord, writes nothing.

Each finding has a code, a severity (``blocks_promotion`` | ``caution`` | ``info``) and a sentence. Nothing here is a verdict about the physics; it lists what the record
does and does not support, so a held-fixed thickness is never read as a measurement.
"""
from __future__ import annotations

_ON_BOUND = 0.01   # a parameter within 1 % of the range from a bound is reported as sitting on it


def diagnose(record) -> list[dict]:
    out: list[dict] = []

    def add(code, severity, message):
        out.append(dict(code=code, severity=severity, message=message))

    if record.chi2_total is None:
        add("chi2_not_recorded", "blocks_promotion", "No total chi-squared was recorded: this cannot be told apart from an unrefined starting model.")
    if record.uses_placeholder_optical_constants:
        add("placeholder_optics", "blocks_promotion", "The model used placeholder optical constants; fitted values that depend on them are not measurements.")
    if not record.layers:
        add("no_layers", "blocks_promotion", "The record has no layers.")
    for layer in record.layers:
        name = layer.label or layer.material or f"layer {layer.layer_index}"
        free = set(layer.free_parameters or [])
        if layer.role == "layer" and not free:
            add("all_fixed", "caution", f"{name}: every parameter was held fixed, so nothing about this layer was determined by the fit.")
        if layer.role == "layer" and free and not (layer.uncertainties or {}):
            add("no_uncertainty", "caution", f"{name}: free parameters have no recorded uncertainty; do not quote them with error bars.")
        for param, bounds in (layer.bounds or {}).items():
            value = (layer.parameters or {}).get(param)
            if value is None or not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or bounds[0] is None or bounds[1] is None or bounds[1] <= bounds[0]:
                continue
            span = bounds[1] - bounds[0]
            if min(value - bounds[0], bounds[1] - value) <= _ON_BOUND * span:
                add("on_bound", "caution", f"{name}: {param} finished on its bound {list(bounds)}; the true value may lie outside the allowed range.")
    techniques = record.techniques or []
    if len(techniques) == 1:
        add("single_technique", "info", f"Only {techniques[0]} constrains this fit; thickness, roughness and density can trade off against each other.")
    if "XRR" in techniques and not record.resolution_smearing_applied:
        add("no_resolution_smearing", "caution", "XRR was fitted without instrument-resolution smearing; roughness may be overestimated.")
    if "SPR" in techniques and record.roughness_applied_to_spr:
        add("roughness_on_spr", "caution", "Interface roughness was applied to SPR, which does not model it that way.")
    for dataset in record.datasets:
        if dataset.chi2 is None:
            add("dataset_chi2_missing", "info", f"{dataset.technique.value} dataset {dataset.source_filename or ''}: no chi-squared recorded.")
    return out
