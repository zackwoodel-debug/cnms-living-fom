"""Structural descriptors (S) and the physical-property vocabulary (P).

FOM_PROOF Sec. 3 defines the S matrix (Eq. 5-6) and the P matrix (Eq. 9-10);
Sec. 13.1 requires a descriptor dictionary to ship with every analysis.  The
dictionary is code, not documentation, so it cannot drift from what was
computed: see ``registry.py``.
"""

from .registry import (  # noqa: F401
    PHYSICAL_PROPERTIES,
    STRUCTURAL_DESCRIPTORS,
    DescriptorSpec,
    PropertySpec,
    descriptor_dictionary,
)
from .tensors import dielectric_anisotropy, directional_component, isotropic_average  # noqa: F401
