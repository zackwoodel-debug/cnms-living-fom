"""Read-only access to the materials-atlas index: reference values for the assistant.

    reader.py   open ``atlas_index.sqlite`` read-only, refuse a missing or stale index
    tools.py    search, read, lookup, compare, coverage — what the assistant's atlas tools return

The atlas holds materials-db reference data: published n and k, densities, SLDs and
calculated dielectric constants, each with its provenance and an evidence grade. It is
*reference*, not platform data. Nothing in this package writes anywhere — it imports no
database session and opens its one file with SQLite ``mode=ro`` — and no code path leads
from an atlas value into ``property_values``, a FOM score or the optimizer (FOM_PROOF
Sec. 15.2). ``tests/test_atlas.py`` checks both statically.
"""

from .reader import AtlasIndex, AtlasUnavailable, open_atlas  # noqa: F401
