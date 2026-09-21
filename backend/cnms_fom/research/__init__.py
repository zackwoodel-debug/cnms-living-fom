"""The research loop: evidence in, reviewed understanding out, optimizer last.

```
corpus + records
   -> ResearchBrief          evidence, claims, contradictions, gaps  (read-only)
   -> proposed KnowledgeCard unreviewed, non-citable
   -> human review           the only way anything becomes citable
   -> ProposedBOContext      propose -> review -> apply, each by a named person
   -> BO suggestion
   -> experiment
   -> ModalFit + plausibility
   -> measured observation   through the existing validated path, never from here
   -> ExperimentSummary      labelled evidence / interpretation / proposal
```

Layout
------
``contracts``   the typed shapes, with their invariants enforced in construction
``campaign``    a deterministic snapshot of a BO campaign, and the warnings it earns
``brief``       assembling a brief from cards, corpus, records, and the campaign
``extract``     passages to typed claims; contradictions found arithmetically
``bo_context``  the propose/review/apply bridge — the only path to the optimizer
``experiment``  outcome summaries, assembled from records and then explained
``policy``      the knobs an autoresearch experiment may turn
``store``       persistence for briefs, claims, and proposals
``benchmark``   fixed cases, honest metrics, reproducible runs

The boundary this package exists to hold: **a literature claim is not a
measurement.** Nothing here writes to ``property_values``, ``descriptor_values``,
``fom_scores``, or ``fit_records``, and ``bo_context`` touches only a campaign's
``constraints`` and only after a person has accepted the change by name.
"""

from .campaign import CampaignSnapshot, snapshot  # noqa: F401
from .contracts import (  # noqa: F401
    BoundProposal,
    Contradiction,
    DataGap,
    EvidenceItem,
    ExtractedClaim,
    LabelledStatement,
    ProposedBOContext,
    ResearchBrief,
    ResearchContractError,
)
from .policy import BASELINE, CANDIDATES, ResearchPolicy, get_policy  # noqa: F401
