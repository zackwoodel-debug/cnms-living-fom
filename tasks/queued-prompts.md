# Queued prompts for the suite integration

Stored 2026-10-11. Run strictly in order, one at a time. Nothing starts until the user approves that prompt.
Each follows the audit in `docs/suite-integration-audit.md`.

| # | Prompt | Status |
|---|---|---|
| 1 | Scientific provenance contract | pending approval |
| 2 | Harden the materials-db to ModalFit export path | pending approval |
| 3 | Source-backed catalog expansion pipeline | pending approval |
| 4 | First integrated suite workflow (vertical slice) | pending approval |

---

## Prompt 1: Scientific provenance contract

Use the completed suite audit to implement a scientific provenance contract
in materials-db, with a compatible mapping into CNMS.

Extend existing structures where practical. Do not create a parallel
database or silently replace the authoritative schema.

Separate these dimensions:
- Provenance: measured, fitted, calculated, modeled, reported, unknown.
- Review: unreviewed, approved, rejected, needs clarification.
- Quality findings: structural errors, physical inconsistencies,
  applicability problems, and heuristic warnings.
- Readiness: suitable for a requested technique and measurement window.

Each property record or dataset must be able to carry:
- Stable material and dataset identifiers.
- Composition and source-supported phase/structure information.
- Specimen form, substrate, processing, and thickness when stated/relevant.
- Property value or curve, units, and validity domain.
- Temperature, frequency, wavelength/energy, direction/polarization,
  isotope assumptions, and method when applicable.
- Uncertainty, or an explicit "not reported" state.
- Source citation and exact locator: table, figure, page, dataset/file/key.
- Raw-source checksum and retrieval/import timestamp.
- Importer version and transformation history.
- Review decision, reviewer, rationale, and timestamp.

Requirements:
- A DOI resolving is not proof that a value is correct.
- Source metadata validation and numeric/source verification are separate.
- An LLM may propose an extraction, never invent or approve a value.
- Inaccessible sources remain explicitly unverified.
- Preserve original values and conflicting determinations.
- Do not infer room temperature, phase, zero absorption, or uncertainty.
- Literature claims must not silently become CNMS measurement records.
- Do not transfer the label "verified" into fixed fit bounds automatically.

Add non-destructive migrations, compatibility handling, and tests.
Do not relabel existing records as approved without an actual review.
Report every new field, migration decision, and remaining limitation.

---

## Prompt 2: Harden the materials-db to ModalFit export path

Harden the existing materials-db → ModalFit export path.
Do not implement a second exporter unless the audit demonstrates a need.

Implement:
1. Strict scientific mode:
   - Missing required data produces actionable errors.
   - Missing k must not automatically become k=0.
   - Ambiguous phase, dataset, optical axis, or property context must
     require an explicit selection.
2. Explicit exploratory mode:
   - Permits named, user-selected approximations.
   - Records each assumption in the model and run metadata.
   - Never modifies the source scientific dataset.
3. Portable model bundles:
   - Model JSON plus required optical assets and a manifest.
   - Asset checksums, dataset release, source identifiers, units,
     selected conditions, software versions, and assumptions.
   - Inspect current ModalFit code and implement a tested loading path
     for its web application rather than assuming one exists.
   - Reject path traversal, unsafe archive members, oversized assets,
     and unresolved asset references.
4. Stable layer identifiers:
   - Repeated materials in a multilayer stack remain independently
     addressable and independently fittable.
5. Technique-specific readiness:
   - Check the selected forward model's actual required inputs.
   - Do not require optical data for an XRR-only model unnecessarily.
   - Reject unsupported anisotropic use rather than silently averaging axes.
   - Check requested wavelength/frequency/energy coverage.
   - Do not extrapolate by default.

Add tests for:
- Missing absorption data.
- Out-of-range spectra.
- Ambiguous datasets.
- Repeated material layers.
- Moving a bundle to another directory/machine context.
- Missing/corrupt assets.
- Unit conversion of values, bounds, and uncertainties.
- Round-trip preservation of source identifiers.

Document any compatibility-breaking changes.

---

## Prompt 3: Source-backed catalog expansion pipeline

Implement a reproducible, source-backed catalog expansion pipeline using
the existing materials-db ingestion and release infrastructure.

First generate a coverage report from the actual authoritative database:
- Distinct material identities and datasets.
- Coverage by material family and property.
- Review/provenance status.
- SE, SPR, XRR, NR, and QCM readiness at explicitly supplied conditions.
- Missing inputs and reasons.
- Unsupported contexts and source-access blockers.

Build a prioritized acquisition backlog:
- Existing materials missing inputs needed for actual planned workflows.
- Substrates, electrode layers, ambients, and calibration materials.
- Additional materials that expand useful technique-ready coverage.

For each proposed source:
- Identify its authority and underlying primary references.
- Check access terms, licensing, redistribution permissions,
  authentication requirements, and rate limits.
- Record which properties and conditions it actually supplies.
- Do not assume a chemistry identity service supplies film properties.
- Do not call computational data experimentally measured.

Pipeline:
discover → fetch/cache → preserve raw → parse → normalize → validate
→ stage → review → release.

Requirements:
- Idempotent ingestion and stable source/dataset identifiers.
- Pinned upstream versions or content checksums.
- Offline fixtures and reproducible release builds.
- Preserve unknowns and source disagreements.
- No end-member interpolation unless explicitly stored as modeled.
- No automatic pooling across temperature, phase, specimen form,
  optical axis, composition, or measurement regime.
- No "verified" status based solely on multiple matching websites.
- Deduplicate sources that reproduce the same underlying experiment.

Implement and demonstrate one small expansion batch first.
If network/source access is unavailable, implement the pipeline and
fixtures, but do not fabricate acquired data or successful verification.

---

## Prompt 4: First integrated suite workflow (vertical slice)

Implement the first integrated suite workflow using the tested contracts:

1. Browse materials and inspect property-specific evidence.
2. Select a dataset and measurement conditions.
3. Build a sample stack.
4. See technique-readiness and missing-input explanations.
5. Run a forward simulation.
6. Load experimental data and refine.
7. Inspect fit diagnostics and scientific limitations.
8. Import the fit into CNMS.
9. Preview eligible property promotion.
10. Analyze or score only when required inputs are present.

Use existing application components where possible.
Inspect available frontend implementations and choose one explicitly;
do not build another disconnected dashboard.

CNMS owns samples, experiments, provenance, analysis, and optimization.
materials-db owns versioned reference datasets and their evidence.
ModalFit owns characterization predictions and refinement.

The interface must distinguish:
- Reference/literature values.
- User-entered assumptions.
- Calculated values.
- Synthetic predictions.
- Experimentally fitted determinations.
- Reviewed versus unreviewed records.

Do not silently push reference values into the measured-property store.
Keep promotion explicit, with a dry-run preview and recorded operator.

Deliver one working vertical slice before adding all techniques.
Include setup instructions and an executable end-to-end smoke test.
