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


class CardType(str, Enum):
    """What kind of thing a knowledge card is about.

    The type decides what the card is *for*, which decides what its links mean:
    a CONCEPT fed by a SOURCE is a different relationship from a CONCEPT that
    depends on another CONCEPT, and the graph is only useful if it knows which.
    """

    CONCEPT = "concept"      # a mechanism, a material family, a parameter window
    SOURCE = "source"        # one paper or document, summarised
    METHOD = "method"        # a procedure: how a thing is grown or measured
    FINDING = "finding"      # something this lab observed, with its evidence
    QUESTION = "question"    # an open question, with what would answer it


class CardStatus(str, Enum):
    """Whether a card has been read by a person yet.

    The distinction is load-bearing, not bureaucratic. A card is where a language
    model's synthesis of the corpus gets written down, and FOM_PROOF Sec. 15.2
    means that synthesis is not evidence until someone has checked it against the
    sources. PROPOSED cards are returned and clearly labelled rather than hidden —
    hiding them would defeat the point — but nothing may be built on one.
    """

    PROPOSED = "proposed"      # written by the assistant; unreviewed
    REVIEWED = "reviewed"      # a person checked it against its sources
    SUPERSEDED = "superseded"  # replaced by a later card, kept for the audit trail


class CardRelation(str, Enum):
    """Typed edges between cards.

    Typed rather than a bare link, for the reason the llm-wiki design gives: a
    graph that only knows *that* two pages are related cannot answer "what does
    this rest on?" — and ``CONTRADICTS`` is the one this platform most needs,
    because an unresolved contradiction between two sources is a finding that a
    flat link would bury.
    """

    FED_BY = "fed_by"            # concept <- source document
    RELATES_TO = "relates_to"    # general association
    DEPENDS_ON = "depends_on"    # this card's claim rests on that one
    CONTRADICTS = "contradicts"  # the two cannot both be right
    MEASURED_BY = "measured_by"  # concept <- a ModalFit fit or a property value
    ANSWERS = "answers"          # finding -> question
