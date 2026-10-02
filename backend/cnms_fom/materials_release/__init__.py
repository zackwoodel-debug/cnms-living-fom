"""Read a materials-db release the one way this platform reads it.

    release.py  open the release read-only and prove it is the release its MANIFEST describes
    labels.py   the release's label grammar (optical and physical), parsed exactly as the release builds it
    kinds.py    measured vs model fit, and what kind of number a physical value is

Two consumers share this package and nothing else: the materials-atlas generator (reference
notes and a search index for the assistant) and ``ingest.materials_db`` (the gated path into the
analysis tables). They must agree on what a value *is* — a DFT density read as a measurement in
one place and as a calculation in the other is the failure this package exists to prevent — so
neither keeps its own copy of these rules.

Nothing here writes anywhere: the release is opened ``mode=ro`` with ``query_only`` on, and the
materials-db helpers are imported with bytecode writing switched off.
"""

from .kinds import (  # noqa: F401
    DENSITY_KINDS,
    DatasetKindHelper,
    PhysicalKind,
    density_kind,
    physical_kind,
)
from .labels import (  # noqa: F401
    AXES,
    LabelError,
    OpticalLabel,
    PhysicalLabel,
    parse_optical_label,
    parse_physical_label,
)
from .release import RegistryEntry, Release, ReleaseError  # noqa: F401
