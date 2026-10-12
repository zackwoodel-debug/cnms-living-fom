# SPEC: first integrated suite workflow (Prompt 4, vertical slice)

Status: **approved by the maintainer 2026-10-12; slice implemented** (see `docs/SUITE_WORKFLOW.md`). Decisions: `frontend/` is the home; Au on SiO2/Si for XRR/NR and a 2D material for SE; the CNMS backend reads reference data over HTTP from a materials-db service (local, or the full database on a remote host through an SSH tunnel), not from the release file. Inputs: `docs/suite-integration-audit.md`, `tasks/provenance-capability-map.md`, the materials-db contracts (provenance 0.23/0.24, strict ModalFit export + bundles, readiness).

## Objective
One working path, in one existing application, from a reference material to a promoted fit, in which every number says where it came from and nothing becomes a measured property unless a person previews and confirms it.

## Frontend decision (explicit)
Extend **`frontend/`** (CNMS's React/Vite app: `MaterialsPage`, `MaterialDetailPage`, `FomPage`, `DictionaryPage`, `MediationPage`, `TierBadge`, `StatusBadge`). Reasons: CNMS owns samples, experiments, provenance, analysis and promotion, and this app already talks to those routers.
Not used: `quark/` (Lovable-generated chat assistant; it stays the question-answering surface and gets a link into the workflow, nothing more); `materials-atlas/` (static graded vault); ModalFit's own web app (stays the place where simulation and refinement run). No new dashboard.

## Ownership (unchanged)
CNMS: samples, stacks, experiments, provenance, analysis, promotion. materials-db: versioned reference datasets, evidence, readiness. ModalFit: prediction and refinement. The slice calls each through its existing contract; it does not copy logic across.

## Slice scope (what is built first)
| # | Step | Where it runs | Existing piece reused |
|---|---|---|---|
| 1 | Browse materials, see per-property evidence | CNMS `frontend/`, `GET /materials` | `materials_release` reader, `provenance.py` tier mapper, `TierBadge` |
| 2 | Choose dataset + conditions (technique, window, temperature) | CNMS | materials-db readiness rules (read from the release; no second implementation) |
| 3 | Build a stack (layers, thickness, substrate) | CNMS | `pilot/stack.py`, `GET /pilot/.../stack` |
| 4 | Technique readiness and missing-input explanations | CNMS | readiness reason codes from the release |
| 5 | Forward simulation | **ModalFit** (opened with a strict bundle) | materials-db bundle export |
| 6 | Load experimental data, refine | **ModalFit** | none (external) |
| 7 | Fit diagnostics and limitations | CNMS (after import) | `FitRecord`, `modalfit/compare.py` |
| 8 | Import the fit | CNMS | `POST /modalfit/import`, `modalfit/records.py` |
| 9 | Promotion preview (dry run) then explicit confirm with operator | CNMS | `GET /modalfit/fits/{id}/promotion-plan`, `POST .../promote` |
| 10 | Score only when inputs are present | CNMS | `fom_engine` eligibility |

Steps 5 and 6 are not re-implemented: CNMS produces the bundle and a launch instruction, ModalFit does the work, and the fit JSON comes back through step 8. A later slice may embed more.

## The interface must distinguish (one badge vocabulary on every number)
reference/literature value; user-entered assumption; calculated value; synthetic prediction; experimentally fitted determination; reviewed vs unreviewed. Mapping: materials-db `provenance` x `review` -> CNMS tiers via `materials_release/provenance.py` (measured/fitted/reported have no tier there and are shown as such, not forced into one). User-entered assumptions are a separate badge and are listed in the stack summary.

## Hard rules
- No silent push of reference values into the measured-property store. Promotion = dry-run preview, then confirm with a recorded operator name; the dry run writes nothing (tested).
- A strict export refuses missing inputs; the UI shows the refusal reason, it never offers "fill with defaults" except as a named exploratory mode that labels the result.
- Unreviewed datasets are shown as unreviewed everywhere; only `approved` by a human shows as approved.

## Commands (to be confirmed at plan time)
Backend: `cd backend && pytest -q`; frontend: `cd frontend && npm test` / `npm run build`; smoke test: `make smoke-suite` (new; runs backend + a fixture release + a recorded fit JSON, no network, no ModalFit install).
Setup doc: `docs/SUITE_WORKFLOW.md` (new).

## Success criteria
- [ ] From a fresh checkout, `make smoke-suite` runs steps 1-4 and 7-10 end to end against a small fixture release and a recorded ModalFit fit, with assertions at each step (including: dry run writes nothing; promote without operator is refused; an unreviewed dataset is labelled unreviewed in the API payload).
- [ ] Steps 5-6 are demonstrated once by hand against real ModalFit (throwaway venv) and recorded in a short transcript in the doc; not part of CI.
- [ ] The workflow page uses existing components and one badge vocabulary; no second dashboard exists.
- [ ] A user can see, for each layer, which dataset, which release, which provenance and review state, and which assumptions were entered.
- [ ] `docs/SUITE_WORKFLOW.md` lets someone else run it.

## Boundaries
Always: tests first for each endpoint; read-only access to the release; show reasons for refusal. Ask first: schema migrations in CNMS, new dependencies, CI changes. Never: edit ModalFit; push reference values to measured properties without confirmation; commit your uncommitted work in this repo.

## Plan (tasks, in order)
S1 backend: release reader endpoints for evidence + readiness (reuse, thin). S2 backend: stack + bundle export endpoint wrapping materials-db strict export. S3 backend: fit diagnostics endpoint over `FitRecord`. S4 frontend: Workflow page (browse, evidence, conditions, stack, readiness). S5 frontend: bundle download + fit import + diagnostics + promotion preview/confirm. S6 smoke test + Makefile target. S7 setup doc + manual ModalFit transcript.

## Open questions
1. Is `frontend/` the right home? (my recommendation: yes.)
2. Which materials/technique should the fixture and the hand demo use? (suggestion: Au on SiO2/Si, XRR/NR, since those have full density/SLD coverage; SE with a TMD from batch A as the second demo.)
3. (decided: HTTP service, see above) Should materials-db be a Python dependency of the CNMS backend (import the readiness code) or read through the release file only? (recommendation: release file only for the slice; the readiness result is already stored in the release's provenance tables? -- **to verify**: today readiness is computed by materials-db code, not stored.)
