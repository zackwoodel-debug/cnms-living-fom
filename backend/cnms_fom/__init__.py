"""CNMS Living FOM — physics-based figure-of-merit engine for materials discovery.

The package is organised along the auditable pathway defined in FOM_PROOF:

    structure (S)  ->  property (P)  ->  function (F)

    descriptors/        computes S from structures (pymatgen + matminer)
    fom_engine/         normalises P, builds F, and estimates dln(F)/dS
    rag_backend/        retrieval over the synthesis corpus (MBE/PLD/ALD/CNMS)
    bo_engine/          proposes the next experiment (BoTorch)
    cnms_integration/   binds proposals to real instruments and constraints
"""

__version__ = "0.1.0"
