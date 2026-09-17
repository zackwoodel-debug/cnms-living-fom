# CNMS Living FOM

A physics-based research platform for autonomous materials discovery, built
around a living, tunable figure of merit.

Analysis follows the auditable pathway from `FOM_PROOF`:

> **structure (S) → property (P) → function (F)**

A direct correlation between a structural descriptor and a composite score is
reported as a summary only. The scientific result is the mediated pathway
`Sⱼ → P_q → F_a`, because a direct correlation does not identify the
intermediate property pathway that produced it.

---

## What makes it defensible

The protocol's rules are enforced in code rather than described in documentation.
Concretely:

- **A missing value stays missing.** There is no imputation path in the codebase.
  A material missing a required input comes back `not_scored`, never partially
  scored. (Sec. 2.3, 6.2)
- **A chemical formula is not a material identifier.** `TiO₂` may be rutile,
  anatase, brookite, amorphous, a doped film, or a ceramic. Polymorph and specimen
  form are required and part of the uniqueness constraint. (Sec. 2.1)
- **Context is required on write.** A breakdown field without thickness,
  electrode, area, and failure criterion is rejected — not because of style, but
  because it cannot be compared with any other breakdown field. (Sec. 16)
- **Every correlation cell carries its own `n`.** There is no run-level sample
  size anywhere in the API, because a matrix computed over different complete-case
  sets does not have one. (Sec. 7.3)
- **Signs are pre-registered, and the table is fingerprinted.** A result that
  contradicts its prediction is reported as a contradiction. Editing a sign after
  seeing results changes the fingerprint stamped on every run. (Sec. 4.2)
- **Tensors are never collapsed silently.** Every scalar derived from a tensor
  carries the reduction rule that produced it. (Sec. 3.2)
- **A FOM–FOM correlation is not evidence of shared physics.** `POST /fom/integrity`
  returns the correlation implied by weight overlap alone, so you interpret the
  excess rather than the raw value. (Sec. 11)
- **Retrieval is for reading, not data entry.** No code path leads from a RAG
  answer into `property_values`; the guard raises if one is attempted. A language
  model is an extremely efficient source of plausible numbers, which is precisely
  what Sec. 15.2 prohibits.

`docs/FOM_PROTOCOL.md` maps every section and equation of the paper to the code
that implements it.

---

## Quickstart

### Docker (full stack)

```bash
cp .env.example .env            # edit POSTGRES_PASSWORD
docker compose up -d db ollama

# One-time: pull the local models (a few GB)
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text

docker compose run --rm api cnms-fom init-db
docker compose run --rm api cnms-fom seed
docker compose up -d api

docker compose --profile frontend up -d frontend
```

API at <http://localhost:8000/docs>, UI at <http://localhost:5173>.

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # FOM engine + API, no heavy extras
pip install -e ".[all]"                 # everything: pymatgen, torch, langchain

export DATABASE_URL=postgresql+psycopg2://cnms:cnms@localhost:5432/cnms_fom
cnms-fom init-db && cnms-fom seed
cnms-fom serve --reload

cd frontend && npm install && npm run dev
```

Tests need nothing beyond the base install — no database, no network, no model
server:

```bash
pytest                                  # 101 tests
```

---

## Layout

```
backend/cnms_fom/
  descriptors/        S from structures (pymatgen + matminer); tensor reduction
  fom_engine/         the protocol, made executable — see below
  rag_backend/        retrieval over MBE/PLD/ALD/CNMS docs, local via Ollama
  bo_engine/          BoTorch loop over growth recipes
  cnms_integration/   instruments, experiments, run provenance  [placeholders]
  db/                 SQLAlchemy models with full provenance columns
  routers/            /materials  /fom  /rag  /bo  /health
frontend/             React + Vite + TypeScript
docs/                 FOM_PROTOCOL.md (equation → code), ARCHITECTURE.md
scripts/              example loader, Sec. 16 compliance checker
```

### The FOM engine

`fom_engine/` depends on numpy, scipy, and the enums — nothing else. That is what
makes the protocol testable without a database or a network.

| Module | Protocol section |
|---|---|
| `eligibility.py` | Sec. 2 — context matching, missing-data rule |
| `normalization.py` | Sec. 5 — transform, standardize, min-max + floor |
| `scores.py` | Sec. 6.2 — `F = ∏ z^w` |
| `correlations.py` | Sec. 7 — Pearson/Spearman, pairwise complete case |
| `inference.py` | Sec. 8 — permutation p-values, Benjamini–Hochberg |
| `regression.py` | Sec. 9 — multivariable fit, VIF |
| `mediation.py` | Sec. 10 — `M = BΓ`, the primary result |
| `integrity.py` | Sec. 11 — covariance identity, null overlap, leakage |
| `hypotheses.py` | Sec. 4.2 — pre-registered signs + fingerprint |
| `physics.py` | Sec. 4.1, 6.1 — ε_static, Δε_m, C/A, EOT |

---

## Endpoints

| Method | Path | Does |
|---|---|---|
| `GET` | `/materials` | Browse material-context records |
| `POST` | `/materials/{id}/properties` | Add a value **with its full context** |
| `POST` | `/materials/{id}/descriptors/compute` | Geometric descriptors from the CIF |
| `GET` | `/materials/dictionary` | The Sec. 13.1 descriptor dictionary |
| `POST` | `/fom/score` | `F_a` per material, or `not_scored` |
| `POST` | `/fom/correlate` | A correlation block + permutation + FDR |
| `POST` | `/fom/mediate` | `M = BΓ` with the dominant channel per descriptor |
| `POST` | `/fom/integrity` | Observed vs. weight-overlap-implied correlation |
| `GET` | `/fom/hypotheses` | Pre-registered signs and their fingerprint |
| `POST` | `/rag/query` | Synthesis Q&A with page-level citations |
| `POST` | `/bo/run` · `/bo/run/{id}/suggest` | Campaign, then next recipes |

---

## Before you report anything

The scaffold runs end to end, but three things are placeholders by design, and
all three must be replaced before a result leaves the building:

1. **FOM weights are uniform and unapproved.** Table 4 of `FOM_PROOF` is a list of
   examples. Weights are an application-policy choice with a named approver, not a
   measurement. Uniform placeholders make that gap visible instead of hiding an
   arbitrary choice behind plausible numbers.
   → `fom_engine/definitions.py`

2. **Normalization bounds are literature screening ranges.** They are not the
   bounds of your population. Re-derive them from the *frozen* eligible set with
   `normalization.bounds_from_population`, record `bounds_basis` and
   `n_materials_in_bounds`, then freeze the definition.
   → `fom_engine/definitions.py::DRAFT_BOUNDS`

3. **Instrument envelopes are invented.** They are typical of the technique, not
   of any CNMS tool. `cnms_integration.instruments.assert_real_registry` blocks
   outward-facing actions until the real registry is wired in.
   → `cnms_integration/instruments.py`

Every `TODO(FOM_PROOF)` and `TODO(CNMS)` in the source marks one of these. To
check where you stand:

```bash
python scripts/check_protocol_compliance.py
grep -rn "TODO(CNMS)\|TODO(FOM_PROOF)" backend/
```

`scripts/load_example_data.py` will populate the database for a demo. Every value
it writes is tagged `modeled`, so every score built on it returns status
`illustrative` — which is the Sec. 2.3 quarantine working, not a bug.

---

## Reference

Woodel, Z. *A Non-Local Fractional Operator Formulation for Structure–Property–
Function Transduction: Reproducible Documentation for Correlating Structural and
Functional Material Descriptors.* 2026.

Center for Nanophase Materials Sciences, Oak Ridge National Laboratory.

## License

MIT — see `LICENSE`.
