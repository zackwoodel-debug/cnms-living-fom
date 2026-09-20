"""Controlled vocabularies.

These are deliberately closed sets.  FOM_PROOF Sec. 2.2 requires that specimen
form and provenance tier be recorded explicitly for every value; free text
would make the eligibility filter unenforceable.
"""

from __future__ import annotations

from enum import Enum


class SpecimenForm(str, Enum):
    """FOM_PROOF Sec. 2.1 — records with different specimen form are different rows."""

    BULK_SINGLE_CRYSTAL = "bulk_single_crystal"
    CERAMIC = "ceramic"
    AMORPHOUS_FILM = "amorphous_film"
    CRYSTALLINE_FILM = "crystalline_film"
    COMPUTATIONAL = "computational"
    OTHER = "other"


class ProvenanceTier(str, Enum):
    """FOM_PROOF Sec. 2.2 — measured / calculated / modeled values never merge silently."""

    MEASURED = "measured"
    CALCULATED = "calculated"
    MODELED = "modeled"
    UNAVAILABLE = "unavailable"


class Transform(str, Enum):
    """FOM_PROOF Sec. 5.1 — the transform is part of the stored analysis run."""

    NONE = "none"
    LOG10 = "log10"


class Direction(str, Enum):
    """FOM_PROOF Sec. 5.3 — is the property beneficial or detrimental when larger?"""

    BENEFIT = "benefit"
    COST = "cost"


class ScoreStatus(str, Enum):
    """FOM_PROOF Sec. 6.2 score-completeness rule."""

    SCORED = "scored"
    NOT_SCORED = "not_scored"
    ILLUSTRATIVE = "illustrative"  # modeled inputs; never mixed with measured results


class HypothesisOutcome(str, Enum):
    """FOM_PROOF Table 6 — result versus the pre-registered sign."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INCONCLUSIVE = "inconclusive"


class CorrelationBlock(str, Enum):
    """FOM_PROOF Sec. 7.4 / Table 5."""

    SS = "R_SS"
    SP = "R_SP"
    PP = "R_PP"
    FF = "R_FF"
    SF = "R_SF"  # direct structure-function; secondary by Sec. 1


class SynthesisTechnique(str, Enum):
    """Corpus partition for the RAG backend."""

    MBE = "mbe"
    PLD = "pld"
    ALD = "ald"
    SPUTTERING = "sputtering"
    CVD = "cvd"
    SOLUTION = "solution"
    CNMS_USER_DOC = "cnms_user_doc"
    OTHER = "other"


class FitTechnique(str, Enum):
    """The five characterization techniques ModalFit co-refines on one slab model.

    Distinct from :class:`SynthesisTechnique` on purpose: one is how a film was
    *grown*, the other is how it was *measured*. Collapsing them would make
    "which technique produced this thickness?" unanswerable, and that question is
    the whole point of a co-refinement record.
    """

    SE = "SE"      # spectroscopic ellipsometry
    SPR = "SPR"    # surface plasmon resonance
    QCM = "QCM"    # quartz crystal microbalance
    XRR = "XRR"    # X-ray reflectometry
    NR = "NR"      # neutron reflectometry


class FitAlgorithm(str, Enum):
    """The five refinement algorithms ModalFit exposes (README Sec. 6 step 7)."""

    LBFGSB = "L-BFGS-B"
    NELDER_MEAD = "Nelder-Mead"
    DIFFERENTIAL_EVOLUTION = "Differential Evolution"
    BASIN_HOPPING = "Basin-Hopping"
    DREAM_EMCEE = "DREAM (emcee)"


class ChatRole(str, Enum):
    """Message roles in an assistant conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
