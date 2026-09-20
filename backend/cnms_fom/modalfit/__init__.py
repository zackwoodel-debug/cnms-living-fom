"""ModalFit integration: multi-technique co-refinements as measurement records.

ModalFit (https://github.com/agauer/modalfit) fits one shared slab model against
up to five characterization techniques — SE, SPR, QCM, XRR, NR — delegating each
forward model to a purpose-built package (``refellips``, ``PyMoosh``, ``refnx``)
rather than reimplementing the physics.

What this package adds is not fitting.  It is the observation that a
co-refinement is the only place this platform ever measures the same quantity
twice by independent means: an XRR thickness and an SE thickness for one film
share no forward model, no instrument, and no systematic error.  Stored as
records rather than as exported files, those determinations become comparable,
citable, and — through :mod:`promote`, under explicit gates — analysable.

Layout
------
``slab_model``  parse slab-model JSON (both the nested and flat forms)
``records``     import a refinement as ``FitRecord`` + layers + datasets
``compare``     cross-technique agreement, and prose for a model to read
``promote``     fitted values -> ``property_values`` / ``descriptor_values``

The interface targeted is ModalFit's documented **exported model JSON**, not its
Flask API — see :mod:`records` for why that is the right seam rather than a
compromise.
"""

from .compare import (  # noqa: F401
    COMPARABLE_PARAMETERS,
    compare_parameter,
    cross_technique_report,
    describe_fit,
    fits_for_sample,
)
from .promote import PromotionRefused, promote_fit, promotion_plan  # noqa: F401
from .records import FitImportError, import_directory, import_fit  # noqa: F401
from .slab_model import (  # noqa: F401
    SlabLayer,
    SlabModel,
    SlabModelError,
    load_slab_model,
    parse_slab_model,
)
