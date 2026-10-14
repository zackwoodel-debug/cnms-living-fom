"""Words that name a crystal phase or an optical axis, for checking claims about data.

Used two ways: the atlas flags a phase that appears in a material's *name* but in none
of its datasets ("Titanium dioxide (rutile / anatase)" has rutile data only), and the
answer checks flag a phase or axis an answer asserts that no returned dataset carries.
A fixed list, tested, never inferred. Words that are also ordinary English in this
setting ("solid", "liquid", "gas", "alpha") are left out on purpose: a check that
fires on "solid evidence" would teach people to ignore it.
"""

from __future__ import annotations

PHASE_WORDS: frozenset[str] = frozenset(
    {
        # structure states
        "amorphous", "polycrystalline", "vitreous",
        # crystal systems
        "cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal", "rhombohedral", "monoclinic",
        "triclinic",
        # polymorphs and structure types
        "rutile", "anatase", "brookite", "quartz", "cristobalite", "tridymite", "coesite", "stishovite",
        "corundum", "sapphire", "spinel", "perovskite", "fluorite", "calcite", "aragonite", "wurtzite",
        "zincblende", "sphalerite", "rocksalt", "greenockite", "cuprite", "tenorite", "chrysoberyl",
        "berlinite", "anhydrite", "diamond", "graphite", "lonsdaleite", "fcc", "bcc", "hcp",
    }
)

#  Two-word structure names, matched as phrases.
PHASE_PHRASES: frozenset[str] = frozenset({"rock salt", "zinc blende", "zinc-blende"})

AXIS_WORDS: frozenset[str] = frozenset(
    {"o-ray", "e-ray", "a-axis", "b-axis", "c-axis", "alpha-axis", "beta-axis", "gamma-axis"}
)


def phases_in(text: str) -> set[str]:
    """Phase words and phrases that occur in ``text`` (lower-cased, whole words)."""
    import re

    lowered = text.casefold()
    found = {w for w in PHASE_WORDS if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", lowered)}
    found |= {p for p in PHASE_PHRASES if p in lowered}
    return found
