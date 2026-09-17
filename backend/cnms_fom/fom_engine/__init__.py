"""The FOM engine — FOM_PROOF made executable.

Module map, by protocol section:

    eligibility.py    Sec. 2       context matching, missing-data rule
    normalization.py  Sec. 5       transform, standardize, min-max + floor
    scores.py         Sec. 6.2     weighted geometric score F_a
    definitions.py    Table 4      draft application scores (unapproved)
    correlations.py   Sec. 7       Pearson/Spearman, pairwise complete case
    inference.py      Sec. 8       permutation p-values, Benjamini-Hochberg
    regression.py     Sec. 9       multivariable fit, VIF
    mediation.py      Sec. 10      B, Gamma, M = B Gamma  (the main result)
    integrity.py      Sec. 11      covariance identity, null overlap, leakage
    hypotheses.py     Sec. 4.2     pre-registered signs
    physics.py        Sec. 4.1/6.1 eps_static, delta_eps_m, C/A, EOT
"""

from .definitions import all_draft_foms, draft_fom  # noqa: F401
from .eligibility import AnalysisTable, ContextFilter, build_analysis_table  # noqa: F401
from .hypotheses import hypothesis_map, registry_fingerprint  # noqa: F401
from .mediation import mediated_effect, mediated_effect_from_theory  # noqa: F401
from .normalization import NormalizationSpec, normalize_value  # noqa: F401
from .scores import FomSpec, ScoreResult, score_material, score_population  # noqa: F401
