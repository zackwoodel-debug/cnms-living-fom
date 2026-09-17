# FOM_PROOF → code

Where each part of the protocol lives in this repository. Section and equation
numbers refer to *A Non-Local Fractional Operator Formulation for
Structure–Property–Function Transduction* (Woodel, 2026).

The point of this table is that a reviewer can go from a claim in the paper to
the code that implements it without reading the whole codebase.

## The pathway

```
        S                    P                     F
structural descriptors → physical properties → functional descriptors
   descriptors/            descriptors/          fom_engine/
   registry.py             registry.py           scores.py
   structural.py           (ingested with        physics.py
   tensors.py               provenance)
                    └──────── fom_engine/mediation.py ────────┘
                               M = B Γ   (the result)
```

## Section-by-section

| § | Requirement | Implementation |
|---|---|---|
| 1 | Mediated pathway is the result; direct S→F is a summary | `fom_engine/mediation.py`; `CorrelationBlock.SF` is labelled secondary |
| 2.1 | Unit of analysis = composition + polymorph + specimen form + context | `db/models.py::Material` (+ context on `PropertyValue`); `eligibility.build_analysis_table` |
| 2.2 | Required provenance on every value | `db/models.py::PropertyValue` — every field of Table 1 |
| 2.3 | Missing stays missing; no imputation | `eligibility.py` (no imputation path); `scores.score_material` → `not_scored` |
| 3.1 | Structural descriptor matrix, Eq. (6) | `descriptors/registry.py::STRUCTURAL_DESCRIPTORS` |
| 3.2 | Tensor reduction is never silent, Eqs. (7)–(8) | `descriptors/tensors.py` — every result carries its rule |
| 3.3 | Property matrix, Eq. (10) | `descriptors/registry.py::PHYSICAL_PROPERTIES` |
| 4.1 | Lattice-polarization mechanism, Eqs. (11)–(17) | `fom_engine/physics.py`; `regression.fit_log_linear_lattice_model` |
| 4.2 | Pre-registered signs, Table 3 | `fom_engine/hypotheses.py` + `registry_fingerprint()` |
| 5.1 | Declared log transform, Eq. (18) | `normalization.apply_transform`; stored on the run |
| 5.2 | Standardization, Eqs. (20)–(22) | `normalization.standardize` (ddof = 1) |
| 5.3 | Normalization + direction + floor, Eqs. (23)–(26) | `normalization.normalize_value`; `bounds_from_population` refuses small samples |
| 6.1 | Direct physical functions, Eqs. (27)–(28) | `physics.capacitance_density`, `physics.equivalent_oxide_thickness` |
| 6.2 | Weighted geometric score, Eqs. (29)–(30) | `fom_engine/scores.py` |
| 7.1–7.2 | Pearson and Spearman together, Eqs. (31)–(32) | `correlations.correlation_cell` |
| 7.3 | Per-pair complete case, Eq. (33) | `correlations.complete_case_mask`; no run-level `n` exists in the API |
| 7.4 | Correlation blocks, Eqs. (35)–(38) | `correlations.correlation_block`; `db/enums.py::CorrelationBlock` |
| 8.1 | Permutation p-value, Eqs. (41)–(42) | `inference.permutation_p_value` (B = 10,000 default) |
| 8.2 | Benjamini–Hochberg, Eq. (44) | `inference.benjamini_hochberg` |
| 9.1 | Multivariable regression, Eq. (45) | `regression.ols` |
| 9.2 | Collinearity control, Eq. (46) | `regression.vif` |
| 10.1 | B = ∂P/∂S, Eqs. (47)–(48), (50)–(53) | `mediation.sensitivity_from_elasticities`, `THEORY_ELASTICITIES` |
| 10.2 | Γ = w ∂ln z/∂P, Eq. (54) | `mediation.gamma_matrix` + `normalization.dlnz_dproperty` |
| 10.3 | M = BΓ, Eqs. (55)–(57), (64)–(65) | `mediation.mediated_effect` |
| 11.1 | Cov(ln F) = W Cov(ln z) Wᵀ, Eqs. (58)–(59) | `integrity.reconstruct_log_score_covariance` |
| 11.2 | Null score correlation, Eqs. (60)–(61) | `integrity.null_score_correlation`, `excess_correlation` |
| 11.3 | Reconstruction and leakage, Eqs. (62)–(63) | `integrity.check_reconstruction`, `integrity.leakage_check` |
| 13.1 | Descriptor dictionary | `descriptors/registry.descriptor_dictionary()`; `GET /materials/dictionary` |
| 13.2 | Correlation output table, Table 6 | `db/models.py::CorrelationResult`; `correlations.CorrelationCell` |
| 13.3 | Mediated-effect table | `db/models.py::MediationResult` |
| 15 | Supported and prohibited claims | enforced structurally — see below |
| 16 | Reproducibility checklist | `cnms_integration/provenance.py`; `scripts/check_protocol_compliance.py` |

## Prohibited claims, and what stops them

Sec. 15.2 lists claims the protocol does not support. Several are blocked by
construction rather than by documentation:

| Prohibited claim | What prevents it |
|---|---|
| "A two-material comparison proves a population correlation." | `correlation_cell` returns no coefficient below n = 3; `bounds_from_population` refuses fewer than 5 materials |
| "All polymorphs of a formula have the same properties." | Polymorph is required and part of the uniqueness constraint on `Material` |
| "A missing value may be filled by a plausible number from memory." | No imputation path exists; `rag_backend.chains.assert_not_property_ingestion` raises |
| "A theoretical scaling law is equivalent to a measured property." | `ProvenanceTier` separates them; a modeled input forces `ScoreStatus.ILLUSTRATIVE` |
| "A loss value without frequency and temperature can rank RF materials." | `PropertySpec.required_context` is enforced on write and on eligibility |
| "A FOM–FOM correlation establishes physical mechanism." | `POST /fom/integrity` returns the Eq. (60) null alongside the observed value |

## Two places where implementation had to make a choice

**Storage normalisation.** Eq. (3) puts temperature, direction, and method in
the unit of analysis. Storing a separate `Material` row per temperature would
duplicate composition and polymorph across thousands of rows. Instead,
composition/polymorph/specimen-form live on `Material` and the remaining context
lives on each `PropertyValue`; an eligible analysis row is a `Material` joined to
a *context-matched* set of values. The filter is implemented once, in
`fom_engine/eligibility.py`, so the rule is enforced in one auditable place. The
protocol's semantics are preserved; only the table layout differs.

**Elasticities target ε_ionic, scores weight k.** Sec. 4.1 derives the polar-mode
mechanism for the lattice term, but Table 4 writes application scores against the
measured dielectric constant. Eq. (11) bridges them: at fixed ε_∞,
∂k/∂S = ∂ε_ionic/∂S, so

    ∂ln k/∂ln S = (∂ln ε_ionic/∂ln S) · (ε_ionic / k)

The ε_ionic/k factor is the fraction of the dielectric response the mechanism can
reach, and dropping it would overstate the structural lever by exactly that
ratio. Implemented in `mediation.propagate_ionic_elasticities_to_k`, which states
the fixed-ε_∞ approximation on the result rather than burying it.

## What is deliberately not implemented

- **Empirical B through the API.** `POST /fom/mediate` supports `source="theory"`
  only. Estimating B from regression needs decisions the protocol requires a human
  to make — which confounders (Sec. 9.1), which descriptors survive the
  collinearity check (Sec. 9.2) — and a default here would be a default about the
  science. Use `regression.ols` plus `mediation.sensitivity_from_regressions`
  directly.
- **Approved FOM weights.** Table 4 is explicitly a list of examples. The
  built-in definitions use uniform weights so the absence of a real policy
  decision is visible rather than hidden behind plausible numbers.
- **Real normalization bounds.** `DRAFT_BOUNDS` are literature screening ranges,
  not the bounds of any study population. Re-derive them from the frozen eligible
  set before reporting anything.
