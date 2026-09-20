"""Parsing ModalFit slab-model JSON.

ModalFit describes a sample as an ordered stack of slabs — ambient first,
substrate last — where each slab carries the per-technique blocks its five
forward models read:

    structural    thickness / roughness, each ``{value, min, max}``
    optical       dispersion model + parameters          (SE, via refellips)
    xray          SLD real / imaginary                   (XRR, via refnx)
    neutron       SLD real / imaginary                   (NR, via refnx)
    viscoelastic  density / shear modulus / viscosity    (QCM)
    molecular     chemical formula + density             (composition-derived SLD)

Two shapes exist in the wild and both are accepted here: the nested
``{value, min, max}`` form that ``slab_model_builder.py`` emits, and the flat
scalar form used by ``substrates/library.json`` (``{"thickness": 2.0, "n": 1.46}``).
Normalising them in one place means nothing downstream has to branch on which
file it was handed.

Units, stated because getting this wrong is silent and expensive
----------------------------------------------------------------
The slab-model format does not record a length unit.  ModalFit's physics
backends are ``refnx``/``refellips``, which work in angstroms, but the bundled
substrate library is written in nanometres — its ``"thickness": 2.0`` is the
2 nm native oxide, and its ``"thickness": 50.0`` is the 50 nm gold film.  A
parser that guessed would be wrong by a factor of ten roughly half the time, so
:func:`parse_slab_model` takes the unit as an argument, defaults to the
angstrom convention of the fitting engine, and records what it used.  A file
that declares ``"length_units"`` or ``"units"`` overrides the argument.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#  Multiplier from the named unit to angstroms.
LENGTH_UNITS: dict[str, float] = {
    "angstrom": 1.0,
    "angstroms": 1.0,
    "a": 1.0,
    "ang": 1.0,
    "å": 1.0,
    "nanometre": 10.0,
    "nanometer": 10.0,
    "nanometres": 10.0,
    "nanometers": 10.0,
    "nm": 10.0,
}
DEFAULT_LENGTH_UNITS = "angstrom"

VALID_ROLES = ("ambient", "layer", "substrate")

#  Blocks copied through verbatim (values only) onto ``FitLayer.parameters``.
TECHNIQUE_BLOCKS = ("structural", "optical", "xray", "neutron", "viscoelastic", "molecular")

#  Keys that mark a parameter as free in the refinement.  ModalFit's UI calls it
#  a per-parameter checkbox; exports have been seen using each of these names.
_VARY_KEYS = ("vary", "fit", "free", "refine")

#  Flat-form aliases: substrates/library.json puts these at the top level of a
#  layer instead of inside a block.
_FLAT_ALIASES: dict[str, tuple[str, str]] = {
    "thickness": ("structural", "thickness"),
    "roughness": ("structural", "roughness"),
    "n": ("optical", "n"),
    "k": ("optical", "k"),
    "sld": ("xray", "sld_real"),
    "isld": ("xray", "sld_imag"),
    "density": ("molecular", "density"),
}


class SlabModelError(ValueError):
    """The JSON is not a slab model we can read.

    Raised rather than repaired.  A stack whose roles are unreadable is not a
    stack with a small problem; every thickness in it is unattributable.
    """


@dataclass
class Parameter:
    """One fittable number: its value, its bounds, and whether it was varied."""

    name: str
    block: str
    value: float | None
    lower: float | None = None
    upper: float | None = None
    vary: bool = False
    units: str | None = None
    uncertainty: float | None = None

    @property
    def at_bound(self) -> bool:
        """True when the value sits on a bound — clamped, not converged.

        Worth surfacing: a fit that reports a 500 Å film because 500 Å was the
        upper bound has not measured a thickness, it has hit a wall.  The
        tolerance is relative to the bound range rather than absolute, so it
        works the same for a 0-1 SLD and a 0-5000 Å thickness.
        """
        if self.value is None or self.lower is None or self.upper is None:
            return False
        span = self.upper - self.lower
        if span <= 0:
            return True
        tol = 1e-6 * span
        return self.value <= self.lower + tol or self.value >= self.upper - tol

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "block": self.block,
            "value": self.value,
            "min": self.lower,
            "max": self.upper,
            "vary": self.vary,
            "units": self.units,
            "uncertainty": self.uncertainty,
            "at_bound": self.at_bound,
        }


@dataclass
class SlabLayer:
    """One slab: a role, a label, and the parameters each technique reads."""

    index: int
    role: str
    label: str | None = None
    material: str | None = None
    formula: str | None = None
    parameters: dict[str, Parameter] = field(default_factory=dict)
    #  Everything not recognised, kept so a re-parse can improve without a
    #  re-import — the same contract as ``ExternalRecord.payload``.
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Parameter | None:
        return self.parameters.get(name)

    def value_of(self, name: str) -> float | None:
        param = self.parameters.get(name)
        return param.value if param else None

    @property
    def thickness_ang(self) -> float | None:
        return self.value_of("thickness")

    @property
    def roughness_ang(self) -> float | None:
        return self.value_of("roughness")

    @property
    def density_g_cm3(self) -> float | None:
        return self.value_of("density")

    @property
    def free_parameters(self) -> list[str]:
        return sorted(name for name, p in self.parameters.items() if p.vary)

    def parameters_by_block(self) -> dict[str, dict[str, float | None]]:
        """``{block: {name: value}}`` — the shape ``FitLayer.parameters`` stores."""
        out: dict[str, dict[str, float | None]] = {}
        for name, param in self.parameters.items():
            out.setdefault(param.block, {})[name] = param.value
        return out

    def bounds_dict(self) -> dict[str, dict[str, float | None]]:
        return {
            name: {"min": p.lower, "max": p.upper}
            for name, p in self.parameters.items()
            if p.lower is not None or p.upper is not None
        }

    def uncertainties_dict(self) -> dict[str, float]:
        return {
            name: p.uncertainty for name, p in self.parameters.items() if p.uncertainty is not None
        }

    @property
    def clamped_parameters(self) -> list[str]:
        return sorted(name for name, p in self.parameters.items() if p.vary and p.at_bound)


@dataclass
class SlabModel:
    """A parsed slab model: identifiers, the stack, and what the file claimed."""

    stack_id: str | None
    sample_id: str | None
    layers: list[SlabLayer]
    length_units: str
    #  Fit metadata when the export carried any — see ``read_fit_metadata``.
    fit: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def film_layers(self) -> list[SlabLayer]:
        """The deposited layers — everything that is neither ambient nor substrate."""
        return [layer for layer in self.layers if layer.role == "layer"]

    @property
    def substrate(self) -> SlabLayer | None:
        for layer in reversed(self.layers):
            if layer.role == "substrate":
                return layer
        return None

    @property
    def total_film_thickness_ang(self) -> float | None:
        """Summed film thickness, or None if any film layer is missing one.

        None rather than a partial sum: a stack total that quietly omits a layer
        is worse than no total at all.
        """
        films = self.film_layers
        if not films:
            return None
        values = [layer.thickness_ang for layer in films]
        if any(v is None for v in values):
            return None
        return float(sum(values))  # type: ignore[arg-type]

    @property
    def free_parameter_count(self) -> int:
        return sum(len(layer.free_parameters) for layer in self.layers)

    def describe(self) -> str:
        """One-line stack summary, e.g. ``air / film_1 (103.4 Å) / silicon``."""
        parts = []
        for layer in self.layers:
            name = layer.label or layer.material or layer.role
            if layer.role == "layer" and layer.thickness_ang is not None:
                parts.append(f"{name} ({layer.thickness_ang:.1f} Å)")
            else:
                parts.append(name)
        return " / ".join(parts)


def _as_float(value: Any) -> float | None:
    """Coerce to float, treating anything unparseable as absent.

    Absent rather than zero: a thickness that failed to parse is unknown, and
    zero is a specific physical claim.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)  # drop NaN
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def _parse_parameter(name: str, block: str, payload: Any, *, scale: float) -> Parameter | None:
    """Read one parameter from either the nested or the flat form.

    ``scale`` converts a length into angstroms and is applied to the value and
    both bounds together — scaling a value without its bounds would silently
    turn a converged parameter into one that looks clamped.
    """
    if isinstance(payload, dict):
        value = _as_float(payload.get("value"))
        lower = _as_float(payload.get("min", payload.get("lower")))
        upper = _as_float(payload.get("max", payload.get("upper")))
        vary = any(_as_bool(payload[key]) for key in _VARY_KEYS if key in payload)
        units = payload.get("units") or payload.get("unit")
        uncertainty = _as_float(
            payload.get("uncertainty", payload.get("stderr", payload.get("sigma")))
        )
    else:
        value = _as_float(payload)
        lower = upper = None
        vary = False
        units = None
        uncertainty = None

    if value is None and lower is None and upper is None:
        return None

    if scale != 1.0:
        value = None if value is None else value * scale
        lower = None if lower is None else lower * scale
        upper = None if upper is None else upper * scale
        uncertainty = None if uncertainty is None else uncertainty * scale

    #  Bounds the wrong way round are a builder bug; swapping them silently
    #  would hide it, and refusing the whole stack over it is worse than
    #  recording it. Normalise and let ``at_bound`` stay meaningful.
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower

    return Parameter(
        name=name,
        block=block,
        value=value,
        lower=lower,
        upper=upper,
        vary=vary,
        units=units,
        uncertainty=uncertainty,
    )


#  Parameters measured in a length, and therefore subject to the unit scale.
_LENGTH_PARAMETERS = {"thickness", "roughness", "thickness_ang", "roughness_ang"}


def _normalise_name(name: str) -> str:
    """``sld_real``, ``sldReal``, ``SLD_im`` all name one parameter.

    The camel-case split fires only at a lower-to-upper boundary and at the end
    of an acronym. Splitting on every capital would turn ``SLD_im`` into
    ``s_l_d_im``, which matches no alias and silently drops the parameter — and a
    dropped SLD is a layer that quietly stops contributing to the fit record.
    """
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", str(name)).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return {
        "thickness_ang": "thickness",
        "thick": "thickness",
        "d": "thickness",
        "roughness_ang": "roughness",
        "rough": "roughness",
        "sigma": "roughness",
        "real_sld": "sld_real",
        "sld_re": "sld_real",
        "imag_sld": "sld_imag",
        "sld_im": "sld_imag",
        "rho": "density",
        "density_g_cm3": "density",
    }.get(text, text)


def _parse_layer(index: int, payload: dict, *, scale: float) -> SlabLayer:
    role = str(payload.get("role", "")).strip().lower()
    if role not in VALID_ROLES:
        raise SlabModelError(
            f"Layer {index} has role {payload.get('role')!r}; expected one of {VALID_ROLES}. "
            "Without a role the stack order is unattributable and no thickness in it can be "
            "assigned to a film rather than a substrate."
        )

    layer = SlabLayer(
        index=index,
        role=role,
        label=payload.get("label") or payload.get("name"),
        material=payload.get("material"),
    )

    molecular = payload.get("molecular")
    if isinstance(molecular, dict):
        layer.formula = molecular.get("formula") or molecular.get("chemical_formula")

    consumed = {"role", "label", "name", "material"}

    for block in TECHNIQUE_BLOCKS:
        block_payload = payload.get(block)
        consumed.add(block)
        if not isinstance(block_payload, dict):
            continue
        for raw_name, raw_value in block_payload.items():
            name = _normalise_name(raw_name)
            if name in ("formula", "chemical_formula"):
                continue
            param_scale = scale if name in _LENGTH_PARAMETERS else 1.0
            param = _parse_parameter(name, block, raw_value, scale=param_scale)
            #  First block to define a name wins; TECHNIQUE_BLOCKS is ordered so
            #  that ``structural`` claims thickness before anything else can.
            if param is not None and name not in layer.parameters:
                layer.parameters[name] = param

    for flat_name, (block, name) in _FLAT_ALIASES.items():
        if flat_name not in payload or name in layer.parameters:
            continue
        consumed.add(flat_name)
        param_scale = scale if name in _LENGTH_PARAMETERS else 1.0
        param = _parse_parameter(name, block, payload[flat_name], scale=param_scale)
        if param is not None:
            layer.parameters[name] = param

    layer.extra = {key: value for key, value in payload.items() if key not in consumed}
    return layer


def _resolve_scale(payload: dict, length_units: str) -> tuple[float, str]:
    declared = payload.get("length_units") or payload.get("units")
    unit = str(declared or length_units).strip().lower()
    if unit not in LENGTH_UNITS:
        raise SlabModelError(
            f"Unknown length unit {unit!r}. Expected one of {sorted(set(LENGTH_UNITS))}. "
            "The slab-model format does not record a unit, so an unrecognised one has to be "
            "refused rather than assumed — a wrong guess is a factor-of-ten error in every "
            "thickness in the stack."
        )
    return LENGTH_UNITS[unit], unit


def parse_slab_model(
    payload: dict, *, length_units: str = DEFAULT_LENGTH_UNITS
) -> SlabModel:
    """Parse a slab-model dict into a :class:`SlabModel`.

    ``length_units`` names the unit the file's thicknesses and roughnesses are
    written in; everything is converted to angstroms.  A file carrying its own
    ``length_units`` key overrides the argument.
    """
    if not isinstance(payload, dict):
        raise SlabModelError(f"Expected a JSON object at the top level, got {type(payload).__name__}.")

    stack = payload.get("stack")
    if not isinstance(stack, list) or not stack:
        raise SlabModelError(
            "No 'stack' array found. A ModalFit slab model is {stack_id, sample_id, stack: [...]}."
        )

    scale, unit = _resolve_scale(payload, length_units)
    layers = [
        _parse_layer(index, entry, scale=scale)
        for index, entry in enumerate(stack)
        if isinstance(entry, dict)
    ]
    if not layers:
        raise SlabModelError("'stack' contained no layer objects.")

    return SlabModel(
        stack_id=payload.get("stack_id"),
        sample_id=payload.get("sample_id"),
        layers=layers,
        length_units=unit,
        fit=read_fit_metadata(payload),
        raw=payload,
    )


def load_slab_model(
    path: Path | str, *, length_units: str = DEFAULT_LENGTH_UNITS
) -> SlabModel:
    """Read and parse a slab-model JSON file."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SlabModelError(f"{path.name} is not valid JSON: {exc}") from exc
    return parse_slab_model(payload, length_units=length_units)


#  Where fit metadata has been observed to live in an export.  ModalFit's HTTP
#  layer was not part of the source we were given, so this reads the documented
#  export rather than guessing at an API: every field is optional, and anything
#  absent can be supplied explicitly at import instead of inferred.
_FIT_CONTAINERS = ("fit", "fit_result", "refinement", "fit_meta", "meta")


def read_fit_metadata(payload: dict) -> dict[str, Any]:
    """Pull whatever refinement metadata an export happens to carry.

    Deliberately forgiving and deliberately non-inventive: a missing chi-squared
    comes back absent, never zero, and a missing technique list comes back empty
    rather than guessed from which blocks are populated.  A stack having an
    ``xray`` block does not mean XRR data was ever loaded against it.
    """
    container: dict[str, Any] = {}
    for key in _FIT_CONTAINERS:
        value = payload.get(key)
        if isinstance(value, dict):
            container = {**value, **container}

    def pick(*names: str) -> Any:
        for name in names:
            for source in (container, payload):
                if name in source and source[name] is not None:
                    return source[name]
        return None

    techniques = pick("techniques", "active_techniques", "fitted_techniques")
    if isinstance(techniques, str):
        techniques = [techniques]
    elif isinstance(techniques, dict):
        techniques = [name for name, on in techniques.items() if _as_bool(on)]
    elif not isinstance(techniques, list):
        techniques = []

    out: dict[str, Any] = {
        "techniques": [str(t).strip().upper() for t in techniques if str(t).strip()],
        "algorithm": pick("algorithm", "optimizer", "method"),
        "chi2_total": _as_float(pick("chi2", "chi2_total", "chisq", "chi_squared")),
        "chi2_by_technique": pick("chi2_by_technique", "chi2_per_technique", "chi2s"),
        "weights": pick("weights", "technique_weights"),
        "settings": pick("tech_settings", "technique_settings", "settings"),
        "fitted_at": pick("fitted_at", "timestamp", "finished_at"),
        "operator": pick("operator", "user"),
        "notes": pick("notes", "comment"),
    }
    return {key: value for key, value in out.items() if value not in (None, [], {})}
