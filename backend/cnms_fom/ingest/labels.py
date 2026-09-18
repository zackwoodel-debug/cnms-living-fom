"""Recover structured fields from packed ``dataset_label`` strings.

The CNMS oxide database stores phase, literature source, optical axis, and
sometimes the quantity itself inside one pipe-delimited free-text column:

    "corundum/sapphire | Malitson1972 | o-ray"
    "Pestryakov1997 | alpha-axis"
    "hcp | xray_sld_real | periodictable_CuKalpha"
    "density_MP_DFT"

That is exactly the encoding FOM_PROOF Sec. 2.1 and 3.2 warn about: the
polymorph and the tensor direction — the two things that decide whether two
numbers may be compared — are invisible to any query.

Positional parsing does not work here: the field count varies from one to three
and the same slot means different things in different rows.  So tokens are
classified by *vocabulary* instead, and anything unrecognised is returned in
``unclassified`` rather than guessed at.  A token nobody can classify is a
reason to quarantine the record, not to assume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#  Optical/crystallographic axes.  Uniaxial crystals use ordinary/extraordinary;
#  biaxial ones use the three principal axes.
AXIS_TOKENS: dict[str, str] = {
    "o-ray": "o-ray",
    "oray": "o-ray",
    "ordinary": "o-ray",
    "e-ray": "e-ray",
    "eray": "e-ray",
    "extraordinary": "e-ray",
    "alpha-axis": "alpha",
    "beta-axis": "beta",
    "gamma-axis": "gamma",
    "a-axis": "a",
    "b-axis": "b",
    "c-axis": "c",
    "isotropic": "isotropic",
    "iso": "isotropic",
}

#  Quantity tokens that name what was measured rather than the material.
QUANTITY_TOKENS: dict[str, str] = {
    "xray_sld_real": "sld_xray",
    "xray_sld_imag": "sld_xray_imag",
    "neutron_sld_real": "sld_neutron",
    "neutron_sld_imag": "sld_neutron_imag",
    "density": "rho",
}

#  Author-year citation keys, e.g. "Malitson1972", "Edwards1991a".
_CITATION = re.compile(r"^[A-Z][A-Za-z.\-]*\d{4}[a-z]?$")
#  Computational or tabulated provenance, e.g. "periodictable_CuKalpha", "MP_DFT".
_METHOD = re.compile(r"^(periodictable|MP|ICSD|COD|NIST|DFT)[_A-Za-z0-9]*$", re.IGNORECASE)


@dataclass
class ParsedLabel:
    """What a label yielded. Fields are ``None`` when the label did not say."""

    raw: str
    phase: str | None = None
    source_tag: str | None = None
    axis: str | None = None
    quantity: str | None = None
    method: str | None = None
    unclassified: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "phase": self.phase,
            "source_tag": self.source_tag,
            "axis": self.axis,
            "quantity": self.quantity,
            "method": self.method,
            "unclassified": list(self.unclassified),
        }


def _classify(token: str) -> tuple[str, str]:
    """Map one token to a ``(field, value)`` pair."""
    lowered = token.lower()

    if lowered in AXIS_TOKENS:
        return "axis", AXIS_TOKENS[lowered]
    if lowered in QUANTITY_TOKENS:
        return "quantity", QUANTITY_TOKENS[lowered]

    #  "density_MP_DFT" packs quantity and method into one token.
    for prefix, quantity in QUANTITY_TOKENS.items():
        if lowered.startswith(prefix + "_"):
            return "quantity+method", f"{quantity}|{token[len(prefix) + 1:]}"

    if _CITATION.match(token):
        return "source_tag", token
    if _METHOD.match(token):
        return "method", token
    return "phase", token


def parse_dataset_label(label: str | None) -> ParsedLabel:
    """Split a packed label into explicit fields.

    Never raises and never guesses: unrecognised tokens come back in
    ``unclassified`` so the caller can decide whether the record is usable.
    """
    if not label or not label.strip():
        return ParsedLabel(raw=label or "")

    parsed = ParsedLabel(raw=label)
    unclassified: list[str] = []

    for token in (part.strip() for part in label.split("|")):
        if not token:
            continue
        kind, value = _classify(token)
        if kind == "quantity+method":
            quantity, method = value.split("|", 1)
            parsed.quantity = parsed.quantity or quantity
            parsed.method = parsed.method or method
            continue
        current = getattr(parsed, kind, None)
        if current is None:
            setattr(parsed, kind, value)
        elif current != value:
            #  A second value for a slot that is already filled: two phases, or
            #  two citations. Recording it as unclassified keeps the ambiguity
            #  visible instead of silently keeping whichever came first.
            unclassified.append(token)

    parsed.unclassified = tuple(unclassified)
    return parsed


def axis_to_tensor_component(axis: str | None) -> str | None:
    """Map a recovered axis onto a dielectric-tensor component (Sec. 3.2).

    For a uniaxial crystal the ordinary ray samples the field perpendicular to
    the optic axis (eps_xx = eps_yy) and the extraordinary ray samples it
    parallel (eps_zz).  For a biaxial crystal the three principal axes map onto
    the diagonal components in order.

    Returns ``None`` for an unknown axis, so the caller stores nothing rather
    than a component it cannot justify.
    """
    return {
        "o-ray": "xx",
        "e-ray": "zz",
        "alpha": "11",
        "beta": "22",
        "gamma": "33",
        "a": "xx",
        "b": "yy",
        "c": "zz",
        "isotropic": "iso",
    }.get(axis or "")
