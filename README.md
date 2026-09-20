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
- **The research assistant cannot answer without evidence.** An answer produced
  without a single retrieval is discarded and replaced with an explicit data gap
  before it is returned. That is enforced in the loop, not requested in the prompt.
- **An impossible number is distinguished from a surprising one.** Plausibility
  reports in three tiers — violations (a permittivity below 1), inconsistencies (an
  X-ray SLD that contradicts its own density), and heuristic flags. A heuristic
  never votes on whether something is physical and is never grounds to exclude a
  value, because a surprising result that survives scrutiny is the point of the
  exercise. (Sec. 2.3)
- **A knowledge card is not evidence until a person says so.** Cards accumulate
  what has been worked out, and an assistant-written one is `proposed` and not
  citable: review needs a named reviewer and a resolved source, and editing a
  reviewed card makes the review stale automatically. (Sec. 15.2, 2.2)
- **Two techniques measuring one quantity are two measurements.** A ModalFit
  co-refinement is stored as a measurement record, and where XRR and SE determine
  the same thickness, both determinations are kept and the disagreement is
  reported. Nothing averages them. (Sec. 2.1)
- **A fitted parameter that was held fixed is not a measurement.** Promotion of a
  ModalFit value into the analysis tables refuses a fixed parameter, one clamped on
  its fit bound, a fit with no chi-squared, and a technique that never constrained
  the quantity.

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
cnms-fom init-db && cnms-fom seed      # init-db runs Alembic migrations
cnms-fom serve --reload

cd frontend && npm install && npm run dev
```

Tests need nothing beyond the base install — no database, no network, no model
server:

```bash
pytest                                  # 398 tests
```

### The pilot loop

One complete experimental cycle for HfO₂ on Si, meant as the template a real
CNMS study is cut from:

```bash
cnms-fom init-db && cnms-fom seed
cnms-fom serve --reload

curl -sX POST localhost:8000/pilot/hfo2_logic_run -H 'content-type: application/json' \
     -d '{"random_seed": 20260823}'
curl -sX POST localhost:8000/pilot/hfo2_logic_run/1/iterate -H 'content-type: application/json' \
     -d '{"q": 2, "persist": true, "seed": 1}'
```

`iterate` suggests recipes, exports the layer stack, simulates X-ray
reflectivity, derives properties, scores the FOM, and logs the observations so
the surrogate retrains. `ingest_experiment` is the identical pipeline with the
simulation replaced by real measurements — which is the only change needed to go
from pilot to production.

Simulated properties are stored `modeled`, so every score from `iterate` comes
back `illustrative`. Measured ones score normally. See `docs/PILOT_WORKFLOW.md`
for the physics, what is real versus heuristic, and the CI pipeline.

### Migrations

The schema carries CHECK constraints, a backfilled context digest, and an
enum-storage convention that `create_all` cannot apply to a database that
already has rows, so schema changes go through Alembic.

```bash
cnms-fom init-db                    # fresh database → head
cnms-fom init-db --stamp-baseline   # database created before Alembic existed
cnms-fom migrate current            # what revision is this database at?
cnms-fom migrate up | down
```

### Importing an external materials database

Two-phase by design: everything is staged verbatim, and only rows carrying a
phase, a specimen form, and the context their property requires are promoted.
The default mode writes nothing.

```bash
# What is in it, and what would stop it being used?
python scripts/ingest_materials_db.py survey path/to/materials_oxide_test.db

python scripts/ingest_materials_db.py stage  path/to/materials_oxide_test.db
python scripts/ingest_materials_db.py promote path/to/materials_oxide_test.db \
    --specimen-form bulk_single_crystal --operator "Your Name"

# eps_inf = n² − k² from a transparent window (feeds Eq. 11)
python scripts/ingest_materials_db.py derive-eps-inf --window 1000 2000
```

`--specimen-form` is required and is recorded on every material it creates: the
external source does not record it, and asserting it is a human judgement that
belongs on the record rather than in the importer.

See `docs/DB_PROTOCOL.md` for the full mapping, the constraint table, and what a
survey of the CNMS oxide database actually found.

---

## Layout

```
backend/cnms_fom/
  descriptors/        S from structures (pymatgen + matminer); tensor reduction
  fom_engine/         the protocol, made executable — see below
  rag_backend/        hybrid retrieval + the research assistant; local via Ollama
  knowledge/          knowledge cards: typed concept pages with a review gate
  modalfit/           ModalFit co-refinements as measurement records
  bo_engine/          BoTorch loop over growth recipes
  cnms_integration/   instruments, experiments, run provenance  [placeholders]
  db/                 SQLAlchemy models, constraints, context identity
  ingest/             external materials-DB import (label parsing, staging)
  pilot/              HfO2-on-Si loop: stack export, XRR, property model
  routers/            /materials /fom /rag /cards /modalfit /bo /pilot /health
migrations/           Alembic revisions
frontend/             React + Vite + TypeScript
docs/                 COSCIENTIST.md ← start here · FOM_PROTOCOL.md · DB_PROTOCOL.md
                      PILOT_WORKFLOW.md · RESEARCH_ASSISTANT.md · MODALFIT_INTEGRATION.md
scripts/              example loader, external ingester, compliance checker
.github/workflows/    CI: science on SQLite, migrations on Postgres, pilot loop
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
| `plausibility.py` | Is this number possible? Violations, inconsistencies, heuristics |

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
| `POST` | `/rag/ingest/upload` | **Upload PDFs** and index them in one call |
| `POST` | `/rag/search` | Retrieval only, with per-retriever diagnostics |
| `POST` | `/rag/chat` | The research assistant: multi-step, with its evidence trail |
| `GET` | `/rag/sessions/{key}` | A conversation transcript and the evidence behind it |
| `GET` | `/cards` · `/cards/graph` · `/cards/stats` | Knowledge cards and their typed graph |
| `POST` | `/cards/{slug}/review` | Sign a card off — the only way it becomes citable |
| `POST` | `/modalfit/import` | Import a ModalFit export as a measurement record |
| `POST` | `/modalfit/compare` | One parameter across every technique that determined it |
| `POST` | `/modalfit/fits/{id}/promote` | Fitted values → `property_values`, gated |
| `POST` | `/bo/run` · `/bo/run/{id}/suggest` | Campaign, then next recipes |
| `POST` | `/pilot/hfo2_logic_run` | Create the worked HfO₂-on-Si campaign |
| `POST` | `/pilot/hfo2_logic_run/{id}/iterate` | Suggest → simulate → score → observe |
| `POST` | `/pilot/hfo2_logic_run/{id}/ingest_experiment` | The same, with measured data |

---

## The research assistant

A co-scientist over everything the platform knows: the document corpus, ModalFit
co-refinements, property values, FOM scores, and the optimizer's own history.
**`docs/COSCIENTIST.md` is the walkthrough** — where PDFs go, what happens to them,
and how each piece connects.

```bash
# Give it the literature (or POST /rag/ingest/upload from /docs in a browser)
curl -X POST localhost:8000/rag/ingest/upload \
     -F 'files=@~/papers/kim-2024.pdf' -F 'technique=ald'

# Give it your measurements
cnms-fom import-fits data/fits

# Ask it something it can only answer by combining the two
cnms-fom ask "XRR and SE disagree on the HfO2 thickness for PILOT-07 by 38%. \
Is either fit internally consistent, and what measurement would settle it?" \
  --sample-id HFO2-PILOT-07 --technique ald
```

Retrieval is hybrid — dense embeddings for paraphrase, Postgres full-text for the
rare exact tokens a synthesis corpus is mostly made of (`TMA`, `Nevot-Croce`,
`HfO2`) — fused by reciprocal rank, then graded for whether each passage actually
answers the question rather than merely sharing its vocabulary. If too little
survives, the query is rewritten once and retried; if it still fails, the answer is
`[DATA GAP: explicitly unresolved]`, which is a correct answer here.

The assistant chooses and chains its own retrievals through nine read-only tools.
Every tool call and its full result are returned and persisted, so "where did that
number come from?" is answerable six months later. Structured records never come
back through similarity search: an embedding of `103.4` sits close to one of
`130.4`, and a number is exactly what a model reports without hedging.

It runs twenty read-only tools and picks its own path through them, so "is this
thickness trustworthy?" becomes: list the sample's fits, compare thickness across
techniques, check whether either fit is internally consistent, then search the
literature for the mechanism. Every tool call and its full result are returned and
persisted, so "where did that number come from?" is answerable six months later.

Three things it does that a search box does not:

- **Judges whether a number is possible.** Three tiers, kept apart: a permittivity
  below 1 is a *violation*; an X-ray SLD that contradicts its own density is an
  *inconsistency* (and catches the XRR degeneracy that makes a co-refinement
  disagree with itself); high k *and* a wide gap together is a *heuristic* flag. A
  heuristic never overrides data.
- **Reads the optimizer.** `evaluations_since_best_improved`, whether suggestions
  still carry large uncertainty, whether they cluster on a bound — a search space
  whose optimum lies outside its own bounds looks exactly like a converged campaign.
  It also flags an unapproved FOM definition: uniform placeholder weights mean the
  campaign is optimising toward a policy choice nobody has made.
- **Writes what it worked out down.** Knowledge cards, opt-in per request, landing
  `proposed` and not citable until someone signs them off.

Local by default — `RAG_LLM_PROVIDER=ollama` keeps every excerpt on the machine,
because the corpus is unpublished CNMS work. `anthropic` is available and opt-in;
the trade-off is stated at the switch, logged at WARNING, and reported by
`/health/ready`.

→ `docs/COSCIENTIST.md` · `docs/RESEARCH_ASSISTANT.md`

## Knowledge cards

Plain retrieval re-derives every answer and keeps nothing: ask the same question
twice and the model reasons from scratch. Cards do the integration work when a
source arrives instead — concept pages, source summaries, findings, and open
questions, joined by typed edges.

```
concepts/ald-window-hfo2
  ├─ fed_by      → sources/kim-2024
  ├─ measured_by → findings/pilot07-thickness
  └─ contradicts → concepts/ald-window-alt
                     "0.98 vs 1.4 Å/cycle over the same 200-300 °C range"
```

Typed edges rather than plain links, because a graph that only knows *that* two
pages are related cannot say what one rests on. `contradicts` is the one this
platform needs most — an unresolved disagreement between sources is a finding — and
it requires a note saying which claims conflict.

What keeps an LLM-written page safe to keep is the review gate. A card is a
synthesis, and Sec. 15.2 says a synthesis is not evidence: an assistant-written
card is `proposed` and `citable` is false until a named person has checked it
against a *resolved* source. Editing a reviewed card makes the review stale
automatically, so nothing can be approved and then quietly rewritten. There is no
path from a card into `property_values`.

```bash
cnms-fom cards list --status proposed     # the backlog that gates citability
cnms-fom cards review concepts/ald-window-hfo2 --reviewed-by "Z. Woodel"
cnms-fom cards stats                      # unresolved_contradictions is the one to watch
```

## ModalFit co-refinements

[ModalFit](https://github.com/agauer/modalfit) fits one shared slab model against
SE, SPR, QCM, XRR, and NR at once. That makes it the only place this platform
measures the same quantity twice by independent physics — an XRR thickness and an
SE thickness share no forward model, no instrument, and no systematic error.

```bash
cnms-fom import-fits data/fits/hfo2_xrr_fitted.json --technique XRR
cnms-fom compare-fits HFO2-PILOT-07 --parameter thickness
```

Imports target the documented **exported JSON**, not the Flask API, and are
idempotent by content hash. The length unit is an argument and never a guess:
ModalFit's physics backends use angstroms, its bundled substrate library is written
in nanometres, and guessing would be wrong by a factor of ten about half the time.
The technique list is never inferred from which slab-model blocks are populated —
ModalFit fills every block a technique *could* read.

Comparison returns each determination with its caveats plus a verdict, and never
an average. The third outcome matters most: a technique that *cannot* determine a
parameter is silent, not in disagreement, and collapsing those two turns a
non-result into a finding.

→ `docs/MODALFIT_INTEGRATION.md`

---

## Before you report anything

The scaffold runs end to end, but four things are placeholders by design, and
all four must be replaced before a result leaves the building:

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

4. **ModalFit's bundled optical constants are placeholders.** Its own README says
   the Si/Au/Cr/Ti n,k tables are not digitized literature values. An imported fit
   resting on them is marked `uses_placeholder_optical_constants`, the import warns,
   and promotion refuses to export an optical property from it — so an SE- or
   SPR-derived number from such a fit is illustrative until the tables are replaced
   with literature or measured n,k.
   → `modalfit/records.py::PLACEHOLDER_NK_MATERIALS`

Every `TODO(FOM_PROOF)` and `TODO(CNMS)` in the source marks one of these. To
check where you stand:

```bash
python scripts/check_protocol_compliance.py   # includes DB-level checks
grep -rn "TODO(CNMS)\|TODO(FOM_PROOF)" backend/
```

The protocol's rules are also enforced in storage, so they hold for ingestion
scripts and hand-written SQL, not just for the API: a `not_scored` row cannot
carry a number, a modeled input cannot become a measurement-based score, a
frozen FOM definition cannot be edited, and the same measurement cannot be
stored twice. `docs/DB_PROTOCOL.md` has the full table.

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
