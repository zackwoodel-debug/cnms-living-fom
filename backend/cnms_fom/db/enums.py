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


class ClaimTier(str, Enum):
    """How a *literature* claim came to be, as the source describes it.

    Deliberately **not** :class:`ProvenanceTier`, and the separation is the whole
    point.  ``ProvenanceTier`` labels a value that has entered the analysis tables
    and is eligible for a score; this labels a number an extraction pulled out of
    a PDF.  A paper reporting "k = 25" gives us ``REPORTED`` — we know a human
    wrote it down, and nothing more.  Collapsing the two vocabularies would let a
    literature value inherit ``MEASURED`` on a type coercion, which is precisely
    the confusion FOM_PROOF Sec. 2.2 exists to prevent.

    ``MEASURED`` and ``FITTED`` appear here only to record what the *source*
    claimed about its own number.  They confer nothing: a claim at any tier is
    still a candidate, and the route into ``property_values`` runs through a
    person, not through this enum.
    """

    REPORTED = "reported"        # the source states it; no method recorded
    MEASURED = "measured"        # the source says it measured it
    CALCULATED = "calculated"    # the source says it computed it analytically
    MODELED = "modeled"          # the source says it came from a simulation
    FITTED = "fitted"            # the source says it came from fitting a model to data
    UNKNOWN = "unknown"          # the source does not say


class ClaimStatus(str, Enum):
    """What has happened to an extracted claim since it was pulled out of a paper.

    There is no ``accepted`` member, and that is not an omission.  Acceptance means
    entering the analysis tables, which happens through ``PropertyValue`` with a
    DOI, a page, and a human, and leaves this record behind as the provenance of
    that decision rather than being promoted in place.
    """

    CANDIDATE = "candidate"        # extracted, unreviewed
    CORROBORATED = "corroborated"  # a second independent source agrees
    DISPUTED = "disputed"          # another source disagrees; both are kept
    SUPERSEDED = "superseded"      # a better extraction of the same claim exists
    REJECTED = "rejected"          # a reviewer found the extraction wrong


class BriefStatus(str, Enum):
    """Review state of a research brief."""

    PROPOSED = "proposed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    #  The campaign or the corpus moved underneath it, so its conclusions describe
    #  a state that no longer exists.
    STALE = "stale"


class ContextStatus(str, Enum):
    """Lifecycle of a proposed change to a BO campaign's configuration.

    ``APPLIED`` is reachable only from ``REVIEWED``.  Nothing a language model
    produces can reach it directly, which is the single invariant this enum
    exists to make representable.
    """

    PROPOSED = "proposed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    APPLIED = "applied"
    SUPERSEDED = "superseded"
    #  Approved, but the card that justified it has since been edited or the
    #  campaign has changed, so the approval no longer covers what it approved.
    STALE = "stale"


class CardCategory(str, Enum):
    """What a knowledge card is *about*, orthogonal to :class:`CardType`.

    ``CardType`` says what shape the page is — a concept, a source summary, a
    question.  This says which job it does in the research loop, and the two are
    independent: a process window is a ``CONCEPT`` in shape and a
    ``PROCESS_WINDOW`` in purpose.

    A closed set rather than a tag, for the reason the rest of this module is
    closed: the BO context bridge selects cards by category, and a free-text tag
    would make that selection unenforceable at the storage layer.
    """

    PROCESS_WINDOW = "process_window"                # a growth window with its context
    PROPERTY_PRIOR = "property_prior"                # an expected range for a property
    MEASUREMENT_CAVEAT = "measurement_caveat"        # a reason to distrust a measurement
    OPTIMIZATION_CONSTRAINT = "optimization_constraint"  # a region to avoid or prefer
    HYPOTHESIS = "hypothesis"                        # a mechanism proposed, not shown
    CONTRADICTION = "contradiction"                  # two sources that cannot both hold
    EXPERIMENT_SUMMARY = "experiment_summary"        # what one run actually showed


class StatementKind(str, Enum):
    """Epistemic status of one sentence in generated output.

    An experiment summary mixes three things that must not be read alike: what was
    recorded, what someone thinks it means, and what to do next.  Labelling them
    is cheaper than asking a reader to infer the boundary, and it is the boundary
    people get wrong when a summary is quoted onward.
    """

    EVIDENCE = "evidence"              # traceable to a record or a cited passage
    INTERPRETATION = "interpretation"  # a reading of that evidence
    PROPOSAL = "proposal"              # a suggested action, resting on the above


class PySeaRecordKind(str, Enum):
    """Whether a pySEA record came off an instrument or out of a simulation.

    Closed vocabulary, and never inferred. A multislice spectrum and a measured
    one can look identical once they are both arrays of counts against energy
    loss; the difference lives in this field and nowhere else, and promotion
    reads it to decide which provenance tier a derived number may claim.
    """

    EXPERIMENTAL = "experimental"
    SIMULATION = "simulation"
    #  One envelope carrying both, such as a measured spectrum shipped with the
    #  simulation it is being compared against. Treated as SIMULATION for tier
    #  purposes: the safe reading of a mixed record is the weaker one.
    HYBRID = "hybrid"


class PySeaAxisKind(str, Enum):
    """What a signal axis indexes.

    MOMENTUM is listed because momentum-resolved vEELS is the case this
    integration exists for; an energy-loss spectrum resolved along q is a
    three-axis signal, and collapsing q into "other" would lose the axis that
    makes the measurement a dispersion rather than a spectrum.
    """

    SPATIAL = "spatial"
    ENERGY = "energy"
    MOMENTUM = "momentum"
    TIME = "time"
    OTHER = "other"


class PySeaDerivation(str, Enum):
    """How a scalar was obtained from its source signals.

    Separate from :class:`ProvenanceTier`. Derivation describes the arithmetic,
    the tier describes what the platform is allowed to claim. A fitted number
    from an experimental record is MEASURED; the same fit applied to a simulated
    spectrum is MODELED, and only the record kind decides that.
    """

    MEASURED = "measured"
    FITTED = "fitted"
    CALCULATED = "calculated"
    SIMULATED = "simulated"


class PySeaValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


class PySeaPromotionStatus(str, Enum):
    """Where a derived scalar sits on the path into the analysis tables."""

    UNEXAMINED = "unexamined"
    ELIGIBLE = "eligible"
    REFUSED = "refused"
    PROMOTED = "promoted"
