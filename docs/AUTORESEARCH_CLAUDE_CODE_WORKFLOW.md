# Autoresearch Integration: Claude Code Implementation Workflow

This document is the implementation prompt and operating procedure for adding an
 evidence-grounded autoresearch capability to CNMS Living FOM.

It is written to be pasted into Claude Code at the start of an implementation
session. Claude Code must follow the phases in order and must not skip validation
because a later phase depends on the contracts established by an earlier one.

## Copy-paste prompt for Claude Code

You are implementing an evidence-grounded autoresearch capability in the CNMS
Living FOM repository.

The platform already contains:

- `backend/cnms_fom/fom_engine/`: the scientific S -> P -> F protocol.
- `backend/cnms_fom/rag_backend/`: PDF ingestion, hybrid retrieval, grading,
  tool-calling assistant, citations, and conversation evidence.
- `backend/cnms_fom/knowledge/`: knowledge cards with proposed/reviewed/citable
  gates.
- `backend/cnms_fom/modalfit/`: instrument-derived co-refinement records.
- `backend/cnms_fom/bo_engine/`: Bayesian optimization over synthesis recipes.
- `backend/cnms_fom/routers/`: API boundaries.
- `backend/tests/`: the existing scientific and API test suite.

The goal is to create a controlled research loop in which:

1. The RAG LLM finds and interprets literature evidence.
2. Knowledge cards preserve reviewed, provenance-backed understanding.
3. Bayesian optimization chooses the next experiment within approved bounds.
4. Experimental and ModalFit results update BO only through existing validated
   measurement paths.
5. An autoresearch benchmark can tune document selection, retrieval, extraction,
   interpretation, and research-brief policies without changing the FOM protocol
   or inventing measurements.

The system must remain auditable, provenance-preserving, and safe for scientific
use. Treat this as a production-quality backend feature, not a demo chatbot.

### Non-negotiable scientific boundaries

- Never allow an LLM output, RAG answer, knowledge card, or literature value to
  write directly into `property_values`, `descriptor_values`, FOM scores, or
  ModalFit records.
- Never treat a literature value as a measured value.
- Never allow an unreviewed or stale card to automatically alter hard BO bounds.
- Never change the FOM equations, frozen FOM definitions, hypothesis signs, or
  protocol constraints as part of this feature.
- Never average disagreeing measurements. Preserve each determination and report
  disagreement.
- Preserve document hash, document id, page, chunk, source quote, model/provider,
  prompt/version, and timestamps for every extracted claim.
- A missing value remains missing. The system must abstain and produce an explicit
  data gap when evidence is insufficient.
- Violations, inconsistencies, and heuristic flags must remain distinct.
- Existing optional dependency behavior must remain intact: core API and FOM
  tests must not require LangChain, Ollama, Anthropic, Torch, BoTorch, or a live
  database.
- Do not send unpublished corpus content to an external provider unless the
  configured provider explicitly permits it and the existing settings/reporting
  path says so.

### Before editing

1. Inspect the repository and current git status.
2. Read:
   - `docs/ARCHITECTURE.md`
   - `docs/RESEARCH_ASSISTANT.md`
   - `docs/COSCIENTIST.md`
   - `docs/FOM_PROTOCOL.md`
   - `backend/cnms_fom/rag_backend/agent.py`
   - `backend/cnms_fom/rag_backend/tools.py`
   - `backend/cnms_fom/rag_backend/ingest.py`
   - `backend/cnms_fom/knowledge/`
   - `backend/cnms_fom/bo_engine/`
   - relevant routers, schemas, models, migrations, and tests
3. Identify the exact existing card status/review contract and BO campaign
   persistence contract. Reuse them; do not create competing concepts.
4. Run the existing focused tests before making changes. Record the baseline.
5. Produce a short implementation plan in the terminal, then execute it. Do not
   ask for approval between phases unless a destructive or ambiguous migration is
   required.

## Target architecture

Implement a controlled three-layer capability:

```text
RAG LLM
  -> ResearchBrief proposal
  -> proposed KnowledgeCards
  -> human review
  -> approved cards and approved soft priors
  -> BO campaign context
  -> BO suggestion
  -> experiment / simulation
  -> ModalFit and plausibility validation
  -> measured observation
  -> BO update
  -> RAG experiment summary
```

RAG is the evidence analyst. Knowledge cards are reviewed memory. BO is the
numerical decision engine. A new orchestration layer may coordinate them, but it
must not collapse their responsibilities.

Prefer a small service module under:

```text
backend/cnms_fom/research/
```

If an existing local abstraction is a better owner, use it instead and explain why
in the final report. Keep orchestration separate from `fom_engine/`.

## Phase 1: define contracts

Create typed schemas/dataclasses for the following concepts. Follow existing
Pydantic and SQLAlchemy conventions.

### ResearchBrief

Must include:

- campaign id or run id when applicable
- research question
- material and specimen context
- target property or FOM objective
- evidence items
- extracted candidate claims
- known data gaps
- contradictions
- proposed actions
- proposed BO context changes
- citations and provenance
- model/provider and policy version
- status: proposed, reviewed, rejected, stale

### EvidenceItem

Must include:

- document id and content hash
- document title
- page number
- chunk id when available
- exact supporting quote
- retrieval method and rank
- grading result
- citation validity

### ExtractedClaim

Must include:

- typed field/property name
- value and unit, if present
- normalized value only when conversion is explicit and auditable
- material, technique, substrate, process, temperature, pressure, frequency, and
  other required context fields when available
- provenance tier: reported, calculated, modeled, fitted, measured, or unknown
- confidence as a model output, never as scientific validation
- evidence references
- candidate status, never measured status

### ProposedBOContext

Must distinguish:

- hard constraints
- soft priors
- recommended bounds
- categorical exclusions
- rationale
- evidence references
- approval status
- reviewer and review timestamp

Do not allow a proposal to mutate a live BO campaign until an explicit approval
operation succeeds.

## Phase 2: research brief generation

Add a RAG capability that assembles a campaign-aware research brief.

It should:

1. Read the BO campaign objective and definition status.
2. Read the current BO history and suggestions.
3. Read relevant reviewed cards first.
4. Read proposed/stale cards only as non-authoritative context.
5. Search the corpus for missing evidence, disagreements, and process windows.
6. Read relevant property, fit, plausibility, and FOM records through typed tools.
7. Produce a structured ResearchBrief plus a human-readable explanation.
8. Record every tool call and evidence item using the existing assistant audit
   pattern.
9. Abstain when the evidence is insufficient.

The assistant must explicitly warn when:

- the FOM definition is unapproved
- the campaign is flat but suggestions remain uncertain
- suggestions cluster on a search-space boundary
- observations are mostly infeasible
- measurements disagree across techniques
- a value is held fixed or clamped in ModalFit
- a source lacks required experimental context
- a proposed card is being used

Add only read-only RAG tools unless a narrowly scoped proposed-card writer is
needed. Proposed-card writes must be quarantined and must never write analysis
records.

## Phase 3: knowledge-card integration

Reuse the existing knowledge-card review system.

Add typed card categories or metadata only if the current model cannot express
these concepts:

- process_window
- property_prior
- measurement_caveat
- optimization_constraint
- hypothesis
- contradiction
- experiment_summary

Generated cards must land as `proposed` and non-citable. A card becomes useful to
BO only after the existing review pathway marks it reviewed/citable according to
repository rules.

When a reviewed card is edited, ensure the existing stale-review behavior remains
correct. Do not create an alternate approval flag.

Add tests proving:

- proposed cards are visible but non-authoritative
- stale cards are not used as approved context
- reviewed cards carry source references
- cards cannot promote values into measurement tables
- contradictions remain represented rather than silently merged

## Phase 4: BO context bridge

Add a narrow bridge from approved card context to BO configuration.

The bridge may provide:

- approved soft priors
- approved recommended bounds
- approved process-window hints
- known infeasible categorical combinations
- evidence-based uncertainty notes

The bridge must not:

- overwrite the FOM objective
- rewrite frozen FOM definitions
- replace measured observations
- alter hard constraints without explicit approval
- accept proposed or stale cards as authoritative
- convert literature values into BO observations

Use explicit operations such as:

```text
propose campaign context
review campaign context
apply approved campaign context
```

Record who approved the change, which card/version supplied it, and the campaign
configuration fingerprint before and after application.

Add tests for approval, rejection, stale-card rejection, provenance, and campaign
fingerprint changes.

## Phase 5: experiment outcome summarization

After a simulation or real experiment completes, create an audited summary input
containing:

- recipe and campaign id
- BO prediction before execution
- measured or modeled result tier
- ModalFit records and fit warnings
- plausibility report
- FOM result and status
- cross-technique comparisons
- whether the result was feasible
- relevant literature and cards

The RAG layer may produce:

- a factual experiment summary
- possible explanations
- unresolved contradictions
- a proposed next question
- a proposed experiment summary card

It must label every statement as evidence, interpretation, or proposal. It must
not silently update BO or scientific records.

## Phase 6: autoresearch benchmark and runner

Create a benchmark-driven autoresearch capability for policy improvement.

The benchmark should contain fixed cases covering:

- direct process parameter lookup
- table and figure-caption retrieval
- material and substrate disambiguation
- technique disambiguation
- numeric extraction with units
- cross-paper comparisons
- cross-technique disagreement
- insufficient-evidence questions
- plausibility violations versus heuristic flags
- BO campaign interpretation
- proposed versus reviewed cards

Each case should define expected evidence, acceptable answer fields, required
context, and whether abstention is correct. Do not make the benchmark depend on a
live external LLM or network.

The evaluator should measure at least:

- document recall@k
- page or passage recall@k
- citation validity
- numeric extraction accuracy or tolerance accuracy
- unit normalization accuracy
- context completeness
- contradiction detection
- correct abstention/data-gap rate
- unsupported claim rate
- latency and token/call cost when available

Create a policy/configuration abstraction for candidate experiments, covering
things such as:

- chunk size and overlap
- dense/lexical fusion settings
- candidate count
- grader threshold
- query rewrite behavior
- table/caption weighting
- document-selection features
- extraction prompt/version
- interpretation prompt/version
- abstention thresholds

The runner must:

1. load a fixed benchmark
2. run one candidate policy
3. produce machine-readable metrics
4. produce per-case failure diagnostics
5. compare against a baseline
6. mark keep/discard/crash
7. preserve the candidate configuration and git commit
8. never modify benchmark truth data during an experiment

Follow the spirit of the attached autoresearch project, but adapt it to this
repository. Do not copy its GPU training loop or make a five-minute wall-clock
training assumption. Here the objective is evidence quality, not neural-network
loss.

Use a tab-separated results format with these columns:

```text
commit | overall_score | doc_recall | citation_accuracy | extraction_f1 | abstention_f1 | unsupported_claim_rate | latency_ms | status | description
```

The persisted file may use literal tabs; the pipe-delimited line above is only a
Markdown-safe representation of the schema.

## Phase 7: API and CLI

Expose the capability through repository-native interfaces.

Prefer endpoints shaped like:

```text
POST /research/campaigns/{run_id}/brief
POST /research/campaigns/{run_id}/context/propose
POST /research/campaigns/{run_id}/context/{proposal_id}/review
POST /research/campaigns/{run_id}/context/{proposal_id}/apply
POST /research/experiments/{experiment_id}/summary
GET  /research/benchmarks/{benchmark_id}/results
```

Use existing authentication and dependency patterns if present. Avoid introducing
an API that bypasses existing BO, card, ModalFit, or RAG services.

Add CLI commands if the project convention supports them, for example:

```text
cnms-fom research brief --run-id 1
cnms-fom research benchmark --case-set baseline
cnms-fom research audit --run-id 1
```

## Phase 8: tests and validation

After every implementation slice, run the narrowest relevant tests immediately.
Then run the full validation sequence:

```bash
make test
make lint
make typecheck
make compliance
```

Also run targeted API tests for:

- research brief generation with fake providers
- no-network/core operation
- evidence and citation persistence
- proposed card creation
- card review and stale-card handling
- BO context proposal/review/apply
- rejection of unreviewed context
- rejection of literature-to-measurement writes
- experiment summary generation
- benchmark scoring
- deterministic baseline evaluation

Use fake providers and fixture data. Do not make tests depend on Ollama,
Anthropic, Postgres, a PDF download, or a live BO model unless the repository
already has an explicit integration-test marker.

Run migration tests if schema changes are made. Inspect the migration SQL and
verify upgrade and downgrade behavior where supported.

For frontend/API changes, start the service using the repository's documented
commands and smoke test:

- `/health/live`
- `/health/ready`
- OpenAPI generation
- research brief endpoint
- card review endpoint
- BO campaign read endpoint
- existing RAG search/chat endpoint

Do not claim a smoke test passed if the required optional service was unavailable;
record it as blocked and test the fallback path instead.

## Final audit required before completion

Produce `docs/AUTORESEARCH_AUDIT.md` containing:

1. implemented files and purpose
2. schema and migration changes
3. API and CLI changes
4. data-flow diagram
5. all scientific safety boundaries checked
6. all test commands and results
7. smoke-test commands and HTTP results
8. optional dependency behavior
9. provider and data-egress behavior
10. benchmark baseline and candidate results
11. known limitations and deferred application-specific work
12. exact commands needed to run the feature locally

The final audit must answer these questions explicitly:

- Can an LLM write a measured property? It must be no.
- Can an unreviewed card change a live BO campaign? It must be no.
- Can a stale reviewed card be treated as current? It must be no.
- Are page-level citations retained for extracted claims? It must be yes.
- Are ModalFit and literature values distinguishable? It must be yes.
- Are disagreements preserved? It must be yes.
- Does a data gap remain a data gap? It must be yes.
- Can core tests run without heavy RAG/BO dependencies? It must be yes.
- Are all autonomous policy experiments reproducible? It must be yes.

Do not commit changes unless explicitly instructed. Leave a clean, reviewable diff
except for user-owned unrelated changes. Do not reset or discard changes that were
not made during this implementation.

At the end, report:

- what was implemented
- what was intentionally not implemented
- exact validation results
- any blocked checks
- the first three recommended application-specific extensions

## Suggested execution order for Claude Code

```text
1. baseline inspection and tests
2. typed ResearchBrief and provenance contracts
3. campaign-aware RAG read path
4. proposed knowledge-card integration
5. approved BO context bridge
6. experiment outcome summaries
7. fixed benchmark and evaluator
8. API/CLI surfaces
9. migrations and integration tests
10. full audit and smoke test
```

The first milestone is complete only when a campaign can produce an audited,
read-only ResearchBrief from RAG, cards, ModalFit, plausibility, and BO data.
The second milestone is complete only when a reviewed proposal can safely become
BO context. The autoresearch benchmark comes after those contracts are stable.

## Operator checklist before launching Claude Code

- Start Claude Code in the repository root.
- Confirm the correct branch and cleanly understood worktree state.
- Ensure the current database can be backed up or recreated for migration tests.
- Ensure local RAG credentials/provider settings are appropriate for unpublished
  documents.
- Install the repository's development dependencies.
- Have the baseline test output available.
- Tell Claude Code not to commit unless you explicitly want commits.
- Keep the first run focused on infrastructure and generic research capability;
  defer application-specific objectives, new FOM definitions, and instrument
  integrations until the audit passes.

## First prompt after implementation

After the implementation and audit pass, use a separate prompt for application
specific work:

> Read `docs/AUTORESEARCH_AUDIT.md` and the implemented research contracts. Do not
> modify the core FOM protocol or provenance gates. Design one application-specific
> benchmark for [MATERIAL / TECHNIQUE / PROPERTY / FOM OBJECTIVE]. Add reviewed
> fixture cases, define the expected evidence and abstention behavior, run the
> baseline, and report what the current system cannot establish. Do not tune the
> application objective until the benchmark has a human-reviewed truth set.
