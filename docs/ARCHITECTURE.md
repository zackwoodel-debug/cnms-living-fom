# Architecture

## Shape

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│  frontend/  │────▶│  FastAPI  (backend/cnms_fom)             │
│  React+Vite │     │                                          │
└─────────────┘     │  routers/   materials · fom · rag · bo   │
                    │  ───────────────────────────────────────  │
                    │  descriptors/   S from structures        │
                    │  fom_engine/    S→P→F, the protocol      │
                    │  rag_backend/   retrieval w/ citations   │
                    │  bo_engine/     next experiment          │
                    │  cnms_integration/  facility binding     │
                    └───────┬──────────────────┬───────────────┘
                            │                  │
                   ┌────────▼────────┐  ┌──────▼──────┐
                   │ Postgres        │  │ Ollama      │
                   │ (+pgvector)     │  │ local LLM   │
                   └─────────────────┘  └─────────────┘
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
| `rag_backend/` | numpy, SQLAlchemy | langchain, pypdf, Ollama client |
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
- **A new instrument**: extend `cnms_integration/instruments.py` — and replace the
  placeholder registry with the real one before any suggestion leaves the building.
