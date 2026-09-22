"""Material identity: one reduced formula, computed the same way by every writer.

Eq. (3) identifies a material by composition *plus* polymorph *plus* specimen form,
and ``materials.uq_material_identity`` enforces that triple. Which makes
``formula_reduced`` an identity field rather than a display string, and it was being
computed three different ways:

* ``routers/materials.py`` tried ``pymatgen``, and on ``ImportError`` fell through to
  the formula exactly as typed. ``pymatgen`` is the optional ``descriptors`` extra, so
  whether ``Hf2O4`` and ``HfO2`` are one material or two depended on which packages
  the machine happened to have. A corpus written with the extra installed and then
  added to without it accumulates duplicates that the unique constraint cannot see.
* ``ingest/materials_db.py`` assigned ``formula_reduced=formula`` with no reduction at
  all, so whatever an external source wrote became the identity.
* Nothing recorded which of the two had been applied.

None of that is a guess a machine should be making on its own. The protocol's rule is
that identity is not inferred, and an identity that varies with the environment is
inferred by definition.

So: ``pymatgen`` canonicalises when it is present, because it is the domain's
convention and existing rows were written with it. When it is absent, a formula that
is *already* canonical is accepted unchanged — ``HfO2`` reduces to itself, and the
check needs no dependency — and anything else is **refused** with a message naming
what would fix it. The two environments therefore never disagree silently: one
accepts, the other says it cannot.
"""

from __future__ import annotations

import logging
from math import gcd

logger = logging.getLogger(__name__)


class FormulaNotCanonical(ValueError):
    """The formula could not be reduced to an identity on this installation."""


def _already_reduced(formula: str) -> bool:
    """Whether integer element counts share no common factor above one.

    ``HfO2`` is reduced; ``Hf2O4`` is not. Fractional occupancies — ``Hf0.5Zr0.5O2``,
    a real and common case here — have no meaningful GCD, so they are taken as given:
    reducing them is not defined, and refusing them would reject legitimate alloys.
    """
    from cnms_fom.fom_engine.plausibility import parse_formula

    counts = parse_formula(formula)
    if not counts:
        return False
    values = list(counts.values())
    if any(abs(v - round(v)) > 1e-9 for v in values):
        return True  # fractional occupancies: nothing to reduce
    integers = [int(round(v)) for v in values]
    common = 0
    for value in integers:
        common = gcd(common, value)
    return common == 1


def reduced_formula(formula: str) -> str:
    """The identity form of ``formula``, or raise :class:`FormulaNotCanonical`.

    Raises rather than returning the input unchanged, because returning it unchanged
    is what let two spellings of one material become two materials.
    """
    text = (formula or "").strip()
    if not text:
        raise FormulaNotCanonical("A material needs a formula.")

    try:
        from pymatgen.core import Composition
    except ImportError:
        if _already_reduced(text):
            return text
        raise FormulaNotCanonical(
            f"{text!r} is not in reduced form, and pymatgen is not installed to reduce "
            "it canonically. Supply the reduced formula, or install the 'descriptors' "
            "extra. Storing it as given would make it a second material distinct from "
            "the one it actually is."
        ) from None

    try:
        return Composition(text).reduced_formula
    except Exception as exc:  # noqa: BLE001 - pymatgen raises several types here
        raise FormulaNotCanonical(
            f"pymatgen could not parse {text!r} as a composition: {exc}"
        ) from exc
