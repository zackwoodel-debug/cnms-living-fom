# Architecture

## Shape

```
┌─────────────┐     ┌────────────────────────────────────────────────┐
│  frontend/  │────▶│  FastAPI  (backend/cnms_fom)                   │
│  React+Vite │     │                                                │
└─────────────┘     │  routers/  materials · fom · rag · modalfit    │
                    │            bo · pilot · health                 │
                    │  ─────────────────────────────────────────────  │
                    │  descriptors/   S from structures              │
                    │  fom_engine/    S→P→F, the protocol            │
                    │  rag_backend/   hybrid retrieval + assistant   │
                    │  modalfit/      co-refinements as records      │
                    │  bo_engine/     next experiment                │
                    │  cnms_integration/  facility binding           │
                    └───────┬──────────────────┬─────────────────────┘
                            │                  │
                   ┌────────▼────────┐  ┌──────▼───────────────┐
                   │ Postgres        │  │ Ollama (local)       │
                   │ (+pgvector,     │  │ or Anthropic (opt-in)│
                   │  +tsvector FTS) │  └──────────────────────┘
                   └─────────────────┘
```

## Module boundaries

`fom_engine` is the core and depends on **numpy, scipy, and the enums only** — not
on SQLAlchemy, FastAPI, pymatgen, torch, or LangChain. That is what makes the
protocol testable: `backend/tests/` exercises every equation without a database,
a network, or a model server. If a change would make `fom_engine` import a heavy
dependency, the logic belongs in a router or a service instead.

Everything heavy is imported lazily inside the function that needs it, so the API
starts and `/health/ready` reports honestly even when `pymatgen`, `botorch`, or
`langchain-ollama` are absent.

| Module | Depends on | Imported lazily |
|---|---|---|
| `fom_engine/` | numpy, scipy, `db.enums` | — |
| `descriptors/` | numpy, `db.enums` | pymatgen, matminer |
| `rag_backend/` | numpy, SQLAlchemy | langchain, pypdf, Ollama client, anthropic |
| `modalfit/` | stdlib, SQLAlchemy | — |
| `bo_engine/` | numpy | torch, gpytorch, botorch |
| `cnms_integration/` | stdlib | — |
| `routers/` | FastAPI, SQLAlchemy | all of the above |

## Data flow, one full loop

1. **Ingest.** A structure (CIF) and context-complete property values land on a
   `Material`. Values missing their declared context are rejected on write.
2. **Descriptors.** `POST /materials/{id}/descriptors/compute` derives the
   geometric entries of Table 2. Z*, ω_TO, S_osc, A_ε are ingested separately with
   their own DFPT provenance.
3. **Score.** `POST /fom/score` normalises each property against the versioned
   bounds and forms F = ∏ z^w. Incomplete materials return `not_scored`.
4. **Explain.** `POST /fom/mediate` returns M = BΓ and names the dominant property
   channel per descriptor — the mechanistic answer.
5. **Propose.** `POST /bo/run/{id}/suggest` fits a GP on ln F over the recipe
   space, intersected with the instrument envelope, and proposes the next runs.
6. **Execute.** The suggestion becomes an `Experiment` on a real tool; its
   measured results come back as `PropertyValue` rows at tier `measured`, and the
   loop closes.

`rag_backend` sits beside this loop, not inside it. It answers process questions
with page-level citations, and there is deliberately no code path from a
retrieval answer into `property_values`.

`modalfit` sits *inside* step 6, and the distinction from `rag_backend` is the
whole reason they are separate packages. A ModalFit refinement is
instrument-derived — a photon or neutron bounced off the film, and a forward model
with no interpretive freedom reproduced the curve — so it may enter
`property_values` at tier `measured`, through `modalfit.promote`, under explicit
gates and with a material identity a person supplied. A retrieval answer may not
enter at all. Keeping the two in separate packages means nothing can drift into
treating them alike.

It also gives step 6 something it otherwise lacks: a second, independent
determination of the same quantity. Five techniques on one shared slab model means
a thickness can be measured twice by unrelated physics, and
`modalfit.compare` reports the two determinations and their disagreement rather
than a mean.

## Where the assistant sits

```
question ──► agent loop ──┬──► search_corpus        document_chunks (hybrid)
                          ├──► list_sample_fits     fit_records  (SQL)
                          ├──► compare_fit_techniques
                          ├──► lookup_property_values
                          └──► lookup_fom_scores
                                     │
                          all read-only, all evidence recorded
```

Prose is retrieved by similarity; every stored number is retrieved by a typed
query. That split is deliberate — an embedding of `103.4` sits close to one of
`130.4`, so finding a fitted thickness by cosine similarity would be silently
wrong about exactly the kind of value a model reports without hedging.

## Why the objective is ln F, not F

F is a weighted *geometric* mean of terms bounded in [ε, ~1], so it is strongly
skewed and its residuals are nowhere near Gaussian — which is exactly what a GP
assumes. Eq. (30) gives the additive form, ln F = Σ w ln z, and that is the scale
the surrogate models. ln is monotone, so the argmax is unchanged: it is a
better-behaved parameterisation of the same problem, and improvements in ln F
correspond to multiplicative improvements in F, which is usually what "10%
better" means for a score.

## Extending it

- **A new structural descriptor**: add a `DescriptorSpec` to
  `descriptors/registry.py` (formula, units, interpretation, transform), compute
  it in `structural.py`, and add a pre-registered sign in `fom_engine/hypotheses.py`
  if it carries a directional prediction.
- **A new application score**: `POST /fom/definitions` with weights summing to 1
  and a normalization spec per weighted property. Do not edit a frozen version.
- **A new synthesis technique for RAG**: add it to `SynthesisTechnique` and to the
  filename heuristics in `rag_backend/ingest.infer_technique`.
- **A new assistant tool**: add a `Tool` to `rag_backend/tools.TOOLS` with its
  schema and a read-only implementation. Keep it read-only — Sec. 15.2 is enforced
  by there being no writing tool, not by asking the model nicely.
- **A new comparable fit parameter**: add it to
  `modalfit/compare.COMPARABLE_PARAMETERS` with the techniques that can actually
  determine it. The `determined_by` list is what separates a disagreement from a
  technique being silent, and getting it wrong turns a non-result into a finding.
- **A new LLM provider**: implement the `ChatProvider` protocol in
  `rag_backend/providers.py`. The seam is small on purpose — grading, reranking,
  the correction loop, and the no-evidence guard all sit above it, so the
  guarantees are identical whichever provider is selected.
- **A new instrument**: extend `cnms_integration/instruments.py` — and replace the
  placeholder registry with the real one before any suggestion leaves the building.
