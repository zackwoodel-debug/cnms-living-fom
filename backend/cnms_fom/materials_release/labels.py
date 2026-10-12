"""The release's label grammar, parsed the way the release itself parses it.

``optical_dispersion.dataset_label`` is '[variant or phase |] source tag [| axis]'
and ``physical_properties.dataset_label`` is '[phase |] quantity[_origin] [| method]'
(DATA_DICTIONARY.md, "Label grammar").

The optical parser is a transcription of ``parse_label`` in materials-db's
``scripts/release_validation.py``, the function that decides which datasets are
comparable and which vote in ``consensus_properties``. It has to be the same
rule, not a similar one: two datasets the release keeps apart (different phase,
different axis) must never be grouped here, and the atlas checks that it
reproduces every stored consensus row from these groups.

``ingest.labels`` is a different parser on purpose. It classifies tokens by
vocabulary because the older CNMS oxide export packed labels inconsistently; a
release label has a grammar, and a grammar is parsed, not guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#  The release's optical axes (release_validation.AXES). "absent = isotropic".
AXES: frozenset[str] = frozenset(
    {"o-ray", "e-ray", "a-axis", "b-axis", "c-axis", "alpha-axis", "beta-axis", "gamma-axis"}
)

#  Physical quantities named in physical_properties.dataset_label
#  (DATA_DICTIONARY.md, vocabulary "physical_quantities").
PHYSICAL_QUANTITIES: tuple[str, ...] = (
    "density",
    "xray_sld_real",
    "xray_sld_imag",
    "neutron_sld_real",
    "neutron_sld_imag",
    "dielectric_static_total",
    "dielectric_electronic",
)

_DENSITY = re.compile(r"^density(?:_(?P<origin>.+))?$")


class LabelError(ValueError):
    """A label the release grammar cannot account for. Never guessed around."""


@dataclass(frozen=True)
class OpticalLabel:
    """One optical dataset label, split into the fields the release compares on."""

    raw: str
    phase: str | None
    tag: str
    axis: str | None


@dataclass(frozen=True)
class PhysicalLabel:
    """One physical-property label: which quantity, of which phase, from where."""

    raw: str
    phase: str | None
    quantity: str
    #  For a density, what follows "density_" ("MP_DFT", "literature (manufacturer
    #  datasheet)", ...). None for the other quantities, whose origin is the method.
    origin: str | None
    method: str | None


def parse_optical_label(label: str | None) -> OpticalLabel:
    """(phase, source tag, axis) from 'phase | tag | axis', both ends optional.

    Same rule as the release: the last segment is an axis only when it is one of
    ``AXES`` and is not the only segment; what remains is either the tag alone
    or a phase followed by the tag.
    """
    raw = label or ""
    segments = [part.strip() for part in raw.split(" | ")]
    axis = segments.pop() if len(segments) > 1 and segments[-1] in AXES else None
    if len(segments) > 2:
        raise LabelError(f"optical label {raw!r} has more than phase | tag | axis")
    phase = segments[0] if len(segments) == 2 else None
    tag = segments[-1]
    if not tag:
        raise LabelError(f"optical label {raw!r} has no source tag")
    return OpticalLabel(raw=raw, phase=phase, tag=tag, axis=axis)


def parse_physical_label(label: str | None) -> PhysicalLabel:
    """Split '[phase |] quantity[_origin] [| method]'.

    Raises ``LabelError`` for anything the grammar does not cover. A build that
    meets such a label stops; an importer quarantines the row. Neither assumes.
    """
    raw = label or ""
    segments = [part.strip() for part in raw.split(" | ")]
    position = None
    quantity = origin = None
    for index, segment in enumerate(segments):
        if segment in PHYSICAL_QUANTITIES and segment != "density":
            position, quantity = index, segment
            break
        match = _DENSITY.match(segment)
        if match:
            position, quantity, origin = index, "density", match.group("origin")
            break
    if position is None:
        raise LabelError(f"physical label {raw!r} names no known quantity")
    before, after = segments[:position], segments[position + 1 :]
    if len(before) > 1 or len(after) > 1:
        raise LabelError(f"physical label {raw!r} has more than phase | quantity | method")
    return PhysicalLabel(
        raw=raw,
        phase=before[0] if before else None,
        quantity=quantity,
        origin=origin,
        method=after[0] if after else None,
    )
