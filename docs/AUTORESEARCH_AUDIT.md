# Autoresearch integration — implementation audit

What was built, what it refuses to do, how that was verified, and where it is weak.

Baseline before this work: commit `d7acf71`, **404 tests passing**, ruff clean.
After: **585 tests passing**, ruff clean, no new mypy errors in the added code.

---

## 1. The boundary this feature exists to hold

> **A literature extraction is not a measurement.**

Everything below follows from that. A brief, a claim, a card, and a proposal are all
*evidence about evidence*. They inform a person. They do not enter the analysis
tables, and the code path by which they could does not exist — which is a stronger
guarantee than a rule saying they must not.

```
corpus + records
   → ResearchBrief           evidence, claims, contradictions, gaps    READ-ONLY
   → proposed KnowledgeCard  unreviewed, non-citable
   → human review            the only way anything becomes citable
   → ProposedBOContext       propose → review → apply, each named
   → BoRun.constraints       narrowed only, fingerprint recorded
   → BO suggestion → experiment → ModalFit + plausibility
   → measured observation    through the existing validated path, never from here
   → ExperimentSummary       labelled evidence / interpretation / proposal
```

---

## 2. Files added

### Contracts and services — `backend/cnms_fom/research/`

| File | Purpose |
|---|---|
| `contracts.py` | `EvidenceItem`, `ExtractedClaim`, `Contradiction`, `DataGap`, `ProposedBOContext`, `BoundProposal`, `LabelledStatement`, `ResearchBrief`. Dataclasses, not ORM rows: the invariants are enforced in `__post_init__` and are testable without a database. |
| `campaign.py` | `CampaignSnapshot` — a deterministic read of a BO campaign and **the warnings it earns**, computed with no model in the loop. |
| `brief.py` | Assembles a brief: campaign → reviewed cards → corpus → records → extraction → contradictions → gaps → interpretation. Also the table/caption reweighting. |
| `extract.py` | Passages → typed claims (model work). Contradictions (arithmetic, never model judgement). |
| `bo_context.py` | The propose/review/apply bridge. The only path to the optimizer. |
| `experiment.py` | `collect_outcome` (records only) then `summarise` (labelled narrative), plus `propose_summary_card`. |
| `policy.py` | `ResearchPolicy` — every knob an autoresearch experiment may turn, with a version hash. |
| `store.py` | Persistence for briefs and claims, and the `claims_for_field` query. |
| `benchmark/cases.py` | 24-document fixture corpus and 12 cases with known answers. |
| `benchmark/evaluate.py` | Metrics, including the ones that keep a score honest. |
| `benchmark/runner.py` | Throwaway-database runs, keep/discard verdicts, TSV results. |

### Other additions

- `backend/cnms_fom/fom_engine/plausibility.py` — pre-existing this session; unchanged.
- `backend/cnms_fom/routers/research.py`, `schemas/research.py` — the API surface.
- `migrations/versions/0005_research_loop.py`.
- `backend/tests/test_research_{contracts,campaign,brief,bo_context,experiment,benchmark}.py`,
  `test_api_research.py` — 181 new tests.

### Files changed

`db/enums.py` (six new vocabularies), `db/models.py` (three tables + one column),
`knowledge/cards.py` (category, `citable_cards_for_bo`), `modalfit/compare.py`
(`fit_process_warnings`, shared `layer_matches`), `config.py`, `cli.py`, `main.py`,
`rag_backend/vectorstore.py` + four call sites (the `citation` fix, §9).

---

## 3. Schema and migration

Migration **0005**, revises 0004. Three tables and one column.

| Table | Holds | Notable constraints |
|---|---|---|
| `research_briefs` | one audited brief | `ck_brief_reviewed_has_reviewer` |
| `research_claims` | one value a source stated | `ck_claim_has_a_source`, `ck_claim_has_a_quote`, `ck_claim_normalisation_explained` |
| `campaign_context_proposals` | a requested campaign change | `ck_context_reviewed_has_reviewer`, `ck_context_applied_has_applier`, `ck_context_applied_has_timestamp` |
| `knowledge_cards.category` | what job a card does | closed enum, nullable |

**The claim tier is `claim_tier`, not `provenance_tier`.** Deliberate and load-bearing:
`ProvenanceTier` labels a value eligible to be scored; `ClaimTier` labels a number an
extraction pulled out of a PDF. Sharing the vocabulary would let a literature value
inherit `MEASURED` on a type coercion — the confusion FOM_PROOF Sec. 2.2 exists to
prevent. `ClaimTier` has a `reported` member that no analysis tier has, and
`research_claims` has no `provenance_tier`, no `context_digest`, and no relationship
to `fom_scores`.

`ClaimStatus` has **no `accepted` member**. Acceptance means entering the analysis
tables, which happens through `PropertyValue` with a DOI, a page, and a person, and
leaves the claim row behind as provenance rather than promoting it.

Verification: `test_migrations.py::test_migrated_schema_matches_the_orm_metadata`
proves `alembic upgrade head` and `metadata.create_all` produce the same schema.
Upgrade and downgrade both exercised on SQLite (8 migration tests pass). **Postgres
migration not exercised** — see §11.

---

## 4. API and CLI

```
POST   /research/campaigns/{run_id}/brief                    read-only
POST   /research/brief                                        read-only
GET    /research/briefs · /research/briefs/{id}
POST   /research/briefs/{id}/review                           records a reader
GET    /research/claims/{field_name}                          no aggregation
GET    /research/campaigns/{run_id}/snapshot                  deterministic
GET    /research/campaigns/{run_id}/context                   the change log
POST   /research/campaigns/{run_id}/context/propose           writes a row only
POST   /research/campaigns/{run_id}/context/{id}/review       needs a named person
POST   /research/campaigns/{run_id}/context/{id}/apply        the only path to BO
POST   /research/campaigns/{run_id}/context/{id}/revert       most-recent only
POST   /research/experiments/{id}/summary                     updates nothing
GET    /research/policies
POST   /research/benchmarks/run                               throwaway database
GET    /research/benchmarks/{case_set}/results
```

```bash
cnms-fom research brief "<question>" --run-id 1 --technique ald   # exit 2 = abstained
cnms-fom research benchmark --case-set hard
cnms-fom research benchmark --compare no_grading table_biased
cnms-fom research policies
cnms-fom research audit --run-id 1
cnms-fom research context review|apply|revert <id> --by "Name"
```

There is **no endpoint that writes a property value, a descriptor value, a FOM
score, or a ModalFit record.**

---

## 5. Scientific safety boundaries — checked

Each row names the test that holds it.

| Boundary | How it is held | Test |
|---|---|---|
| An LLM cannot write a measured property | No such code path. `research` imports nothing that writes those tables. | `test_a_brief_is_read_only`, `test_a_brief_writes_nothing_scientific`, `test_reviewing_a_brief_promotes_nothing` |
| A literature value is not a measured value | `ClaimTier` ≠ `ProvenanceTier`; `is_measurement: False` in every payload | `test_claim_tier_is_not_the_analysis_provenance_tier`, `test_a_claim_never_serialises_as_a_measurement` |
| An unreviewed card cannot change a live campaign | `apply` requires `status is REVIEWED`; cards re-validated at propose, review **and** apply | `test_an_unreviewed_proposal_cannot_be_applied`, `test_a_proposal_resting_on_an_unreviewed_card_is_refused` |
| A stale reviewed card is not current | body hash recorded at propose, re-checked at apply; proposal marked `STALE` | `test_a_card_edited_between_review_and_apply_marks_the_proposal_stale`, `test_a_card_unreviewed_between_review_and_apply_also_blocks` |
| Page-level citations retained | `quote` is NOT NULL and the quote must appear in the passage | `test_a_quote_not_present_in_the_passage_is_rejected`, `test_a_fabricated_quote_is_counted_as_an_unsupported_claim` |
| Disagreements preserved, never averaged | `Contradiction` has no `resolved_value`; detection is arithmetic | `test_a_contradiction_keeps_both_claims_and_has_no_resolved_value`, `test_two_documents_disagreeing_produce_a_contradiction_not_an_average` |
| A data gap stays a data gap | abstention is an outcome; `DataGap` must name what resolves it | `test_a_question_the_corpus_cannot_answer_abstains_with_a_route_out`, `test_a_data_gap_must_say_what_would_resolve_it` |
| No FOM change | nothing in `research` imports `fom_engine.definitions`; no weights, bounds, or signs touched | `test_applying_touches_only_the_constraints` |
| A search space may narrow, never widen | `widening_violations` at propose, review and apply | `test_a_widening_bound_is_refused_with_the_reason`, `test_a_widening_proposal_is_refused_at_propose_time` |
| Violations / inconsistencies / heuristics stay distinct | three tiers; heuristics do not vote on `physical` | `test_the_k_eg_tradeoff_flags_the_corner_that_is_actually_rare` |
| Context is reported, never filled | `missing_context` computed and stored; nothing invented | `test_missing_required_context_is_reported_never_filled`, `test_an_incomparable_expectation_is_satisfied_by_recording_it_as_such` |
| Core tests need no heavy deps | contracts, campaign, bo_context, benchmark all run on SQLite with scripted providers | whole suite offline |

**One narrow write exception, pre-existing:** `write_card` / `link_cards` in
`rag_backend/tools.py` reach knowledge cards only, are withheld unless
`allow_card_writes=True`, and land cards `proposed`/non-citable. Nothing that
touches an analysis table is writable.

---

## 6. Validation results

| Command | Result |
|---|---|
| `pytest` | **604 passed, 1 warning** (404 baseline → 604) |
| `ruff check backend/ scripts/ migrations/` | **All checks passed** |
| `mypy backend/cnms_fom` | 43 errors, **0 in the added code** (all pre-existing; `make typecheck` is `\|\| true` by project convention) |
| `python scripts/check_protocol_compliance.py` | Runs. Fails on the documented placeholders (unapproved FOM weights, draft bounds, no frozen eligible set) and on the dirty working tree. **Pre-existing; not caused by this work.** |
| `cnms-fom init-db` on an empty database | `0001 → 0005`, 32 tables |
| `cnms-fom migrate current` | revision `0005` |

Test counts by area: contracts 34, campaign 18, brief 32, bo_context 31, experiment
17, benchmark 28, research API 21.

---

## 7. Smoke tests — live HTTP

Server: `uvicorn cnms_fom.main:app --port 8077`, SQLite at head, corpus of two
ingested PDFs, two ModalFit fits, one campaign with six observations, one reviewed
process-window card. Ollama on localhost with `nomic-embed-text` + `qwen3:14b`.

| Request | Status | Observed |
|---|---|---|
| `GET /health` | 200 | |
| `GET /health/ready` | 200 | |
| `GET /openapi.json` | 200 | 64 paths, 16 under `/research` |
| `GET /research/policies` | 200 | 9 policies with diffs |
| `GET /research/campaigns/1/snapshot` | 200 | fingerprint + 3 computed warnings |
| `GET /research/campaigns/999/snapshot` | **404** | |
| `POST …/context/propose` | **201** | `status=proposed`, campaign untouched |
| `POST …/context/{id}/apply` *(before review)* | **409** | "only a REVIEWED proposal can be applied" |
| `POST …/context/{id}/review` *(blank name)* | **422** | refused at the schema |
| `POST …/context/{id}/review` | 200 | `status=reviewed by=Z. Woodel` |
| `POST …/context/{id}/apply` | 200 | fingerprint `7530e9c5c696 → d999c0bbd8c7`, bounds narrowed to `[200, 300]`, advisory content in notes stamped with proposal id and person |
| `POST /research/campaigns/1/brief` *(1B grader)* | 200 | **10m 28s**, 23 claims, 19 comparable, 3 gaps, **0 contradictions** — see §11.1 |
| `POST /research/campaigns/1/brief` *(14B grader, `narrow_pool`)* | 200 | 17 claims, 12 comparable, **1 contradiction found** |
| `POST /rag/search` *(grade=true)* | 200 | 7 candidates graded |
| `GET /rag/assistant`, `/rag/corpus`, `/cards`, `/bo/runs`, `/modalfit/samples` | 200 | existing surfaces unaffected |

The campaign snapshot correctly flagged, without any model involvement: the
**unapproved objective**, a **stall of 5 evaluations with an uncertain surrogate**,
and **every pending suggestion sitting on `substrate_temp_c`'s upper bound**.

### The end-to-end result that matters

The corpus contains a deliberate cross-paper disagreement: 0.98 Å/cycle in a
hot-wall reactor, 1.42 in a cross-flow one. With an adequate grader the loop found
it, through the live API, over real PDFs, with a local model:

```
CONTRADICTION on growth_per_cycle_ang
  basis: 37% relative spread between 1.42 angstrom and 0.98 angstrom; context
         differs in material, oxidant, precursor, pressure_torr, technique,
         temperature_k, which is the likely explanation and should be checked
         before treating this as a conflict
  left : 1.42 angstrom | synthetic_ald_hfo2_crossflow, p. 1
  right: 0.98 angstrom | synthetic_ald_hfo2_hotwall, p. 1
```

Both values kept, each wired to its page, the likely explanation named, and no
average anywhere. That is the behaviour the whole feature exists to produce.

---

## 8. Optional dependency and provider behaviour

**Core operation with no extras.** Contracts, campaign snapshots, the BO bridge,
the benchmark and the schema all run with no LangChain, no Ollama, no Anthropic, no
torch, no BoTorch, no Postgres. A scripted provider (`tests/fakes.py`) covers the
model-dependent paths, and `StubExtractor` makes the benchmark runnable in CI.

**Retrieval unavailable is not an empty literature.** If the `rag` extra is absent
or the embedder is unreachable, `brief` records
*"Corpus retrieval did not run … its data gaps must not be read as gaps in the
literature"* and continues on cards and records. Saying the corpus is silent about a
search that never ran is a wrong scientific conclusion wearing a clean answer's
clothes. Test: `test_retrieval_failing_is_not_reported_as_an_empty_literature`.

**Data egress.** Unchanged and still opt-in. `RAG_LLM_PROVIDER=ollama` is the
default and keeps every excerpt local; `anthropic` logs at WARNING on selection and
is reported by `/health/ready`. No new egress path was added — `/research` uses the
same provider seam.

**New settings.** `RAG_EXTRACTION_MODEL` (§10), alongside the existing
`RAG_GRADER_MODEL`.

---

## 9. Bugs found and fixed during this work

1. **`list_proposals` omitted `bo_run_id`** — the response model required it, so the
   context change log 500'd. Found by an API test.
2. **A single-space reviewer passed `min_length=1`** — `" "` reached the service and
   produced a 409 instead of a 422. Now a validator strips and rejects at the
   schema, because "reviewed by nobody" is exactly the state Sec. 15.2 is about.
3. **`ChunkHit.citation` was a method while `EvidenceItem.citation` is a property** —
   a bare `.citation` on the former formats a bound method into the string,
   producing a citation that reads like a bug report. I tripped on it writing a
   diagnostic. Aligned as a property, four call sites updated, regression test added.
4. **`table_weight` / `caption_weight` were declared and did nothing** — a policy knob
   that changes nothing measurable is worse than no knob. Implemented as a
   post-grading reweight (a weight can promote a passage the grader accepted, never
   smuggle in one it rejected), and it now measurably improves the tabular case.
5. **The benchmark had no discriminating power** — see §10.

The next five were found only by running against a real provider and a real Postgres,
which is the argument for having done both rather than trusting the offline suite.

6. **A correct abstention scored 0.000, exactly like a failed one.** `SCORE_WEIGHTS`
   listed `abstention_f1`, a *run-level* metric that no single case carries — F1 needs
   a population. `CaseResult.score` looked it up with a defaulting `getattr`, found
   nothing, and since an abstention case computes no other metric the weighted set
   came out empty and hit the `return 0.0` floor. Two cases that abstained correctly
   were reported as failures with no diagnostic, `insufficient_evidence` read 0.0000
   when it was really 2/3, and the headline score was 0.6131 instead of 0.7899. The
   damaging part is not the number: **no policy change could be rewarded for getting
   abstention right**, while one that destroyed abstention paid only the 0.07
   run-level weight. Invisible offline because the stub grader marks abstention cases
   unexercised and drops them. Fixed with an explicit `PER_CASE_METRIC` mapping to
   `abstention_correct`, plus a test asserting every weight resolves to a real case
   attribute — the silent `getattr` default is what hid it.
7. **An unreachable model was indistinguishable from an empty corpus.** A failed grade
   fails closed to grade 1, which drops the passage; a failed extraction yields no
   claims. `grade_and_rerank` computed a `failed` count and the caller discarded it.
   So an Ollama outage produced a brief that abstained and said "the evidence did not
   clear the policy's threshold" — a claim about the literature, made by a pipeline
   that never reached a model. In the benchmark it was worse: every case abstained and
   every abstention case scored *correct*, so a total outage looked like perfect
   judgement. `ResearchBrief` now carries `degraded_reason`, the cost report counts
   `grading_failed` and `extraction_failed` separately from content problems, the
   warning says "absence of evidence here is not evidence of absence", and the
   benchmark scores abstention as `None` — not exercised — when the brief is degraded.
8. **The model-call cache was bypassed in the one workload it was built for.** The
   benchmark builds a throwaway corpus database per run and the cache lived inside it,
   so every run started cold. A sweep of eight policies paid full model cost eight
   times over the same passages — and extraction is keyed on passage text alone, so
   nearly all of it was reusable. The cache now lives in its own persistent database
   (`data/cache/`, gitignored, holds only `llm_cache`), threaded through as `cache_db`
   and shared across a sweep.
9. **Making the cache persistent immediately broke test isolation** — and not just by
   writing 139 rows into the repo. A test asserting that a model call *fails* got a
   cache hit from a different test's successful stub call, so the failure never
   happened and the test passed for the wrong reason. Fixed with an autouse fixture
   redirecting the cache per test. The first attempt at that fixture did nothing,
   because `DEFAULT_CACHE_PATH` was a **default argument value**, bound at import, so
   no monkeypatch could reach it; the path is now resolved at call time.
10. **A corrected document title never reached the lexical index.** Migration 0007
    materialises `documents.title` onto `document_chunks.search_title`, because a
    functional index cannot span two tables. Its trigger fired on `document_chunks`
    insert/update — nothing fired when a *document* was retitled, though the comment
    claimed the column "cannot drift". Verified on real Postgres: after correcting a
    title from "LSMO on SrTiO3" to "LSMO on NdGaO3", every chunk still matched
    `SrTiO3` and none matched `NdGaO3`. The retriever vouched for the **retracted**
    substrate and refused the corrected one, and substrate identity is a descriptor
    here — a wrong answer, not a stale cache. Fixed with an `AFTER UPDATE OF title`
    trigger on `documents`, guarded on `OLD.title IS DISTINCT FROM NEW.title`.
11. **`_should_abstain` took `claims` as an argument and never looked at it.** The dead
    parameter was the bug. A brief could retrieve passages, extract **zero** claims
    from them, and still not abstain — so it went on to interpret passages it had
    found nothing in. That is the precise shape of answering from a near-miss source:
    the benchmark's GaAs-on-Ge question retrieves GaAs-on-GaAs passages, which are
    close enough to pass grading, and extraction correctly finds nothing in them that
    answers the question. Now `extraction_ran and not claims` abstains, guarded so a
    brief assembled without a provider is not read as a statement about the corpus.
12. **The scorer compared unit strings literally**, so `'angstrom per cycle'` counted as
    wrong against an expected `'A/cycle'`. The extractor is told to copy the passage's
    units verbatim — deliberately, because rewriting them would destroy the verbatim
    record that makes a claim checkable — so comparing spellings measured which
    spelling the source happened to use. Normalisation added on the scoring side only
    (`normalise_units`), with a test that it does not normalise so hard that a length
    matches a rate: `angstrom` and `angstrom per cycle` must still differ, as must
    `mTorr` and `Torr`.
13. **`unit_accuracy` was computed since the first version of this benchmark and never
    printed.** Surfacing it immediately exposed a second bug in it: `_match_claims`
    returned `unit_correct / correct if correct else 0.0`, so a case that matched no
    claims at all contributed a hard zero. The metric read **0.7778** while every unit
    it had actually checked was correct — 7 cases perfect, 2 cases with nothing to
    check averaged in as failures. Now `None`, which is the rule this document states
    for every other metric ("an unavailable metric is written `n/a`, never `0.0000`")
    and was not following here. Reads 1.0000, and `overall_score` is unchanged at
    0.8257 because `unit_accuracy` carries no weight — the fix corrected a report, not
    a score. A metric nobody can read is a metric nobody maintains.

---

## 10. Benchmark: baseline and candidates

Offline run, lexical retrieval, stub grader, `commit d7acf71-dirty`:

```
policy           score    verdict   note
table_biased     0.9079   keep      score 0.859 → 0.908
baseline         0.8592   —
strict_grading   0.8592   keep      unchanged
no_rewrite       0.8592   keep      unchanged
wide_pool        0.8524   discard   score fell 0.859 → 0.852
no_grading       0.7394   discard   score fell 0.859 → 0.739
```

**Finding: table/caption weighting is worth adopting.** On the `hard` case set it
moves the tabular case from 0.711 to 0.886 and its reciprocal rank from 0.750 to
1.000 — the table page goes from rank ~1.3 to rank 1.

**Finding: grading buys 12 points of score.** `no_grading` is the largest regression
in the sweep, which independently confirms what the earlier live run suggested.

**Two benchmark design failures I had to fix before any of that was meaningful:**

- The first corpus had **11 pages against a retrieval window of 6**, so recall was
  trivially 1.0 and *every policy scored 1.000*. A benchmark whose corpus is smaller
  than its retrieval window measures nothing. Now 24 documents / 47 pages, with
  distractors that share the corpus's vocabulary and answer different questions.
- **Recall alone still scored everything at 1.000**, because it asks only whether the
  right page *appeared*. Added `reciprocal_rank` (where the answer ranked) and
  `page_precision` (how much of the window was wasted), which is what actually
  separates policies.

Metrics: `doc_recall`, `page_recall`, `reciprocal_rank`, `page_precision`,
`citation_accuracy`, `extraction_precision/recall/f1`, `unit_accuracy`,
`context_completeness`, `contradiction_detected`, `abstention_f1`,
`unsupported_claim_rate`, `latency_ms`.

`unsupported_claim_rate` is **subtracted** from the score rather than weighted into
it, so a policy cannot buy a better number with confident fabrication
(`test_fabrication_cannot_buy_a_better_score`).

**An unexercisable capability is skipped, not failed.** With the stub grader the
three abstention cases are marked `exercised=False` and left out of the aggregate,
because separating "GaAs on GaAs(001)" from "GaAs on germanium" is grading work the
term-overlap stub cannot do. Scoring the pipeline zero for the stub's blindness
would blame the wrong component.

Results file: `data/exports/autoresearch_results.tsv`, tab-separated, columns
`commit · overall_score · doc_recall · citation_accuracy · extraction_f1 ·
abstention_f1 · unsupported_claim_rate · latency_ms · status · description`. An
unavailable metric is written `n/a`, never `0.0000`. A dirty tree is recorded as
`<sha>-dirty`, because a row from an uncommitted tree does not identify the code
that ran.

---

## 10b. Optimisation round

Three changes, each measured before and after.

### The per-passage call cache (migration 0006)

Grading and extraction are **18 of the 19 model calls a brief makes**; retrieval is
about a second. Both are now cached, and the design follows from one observation:

> The extraction prompt contains the passage and **not the question**. So an
> extraction is a pure function of `(passage, model, prompt version)`, and a passage
> needs extracting *once ever*.

Keys are a **hash of the content**, not a chunk id, so a re-ingest that renumbers
chunks cannot serve a stale answer and an edited passage misses automatically —
correctness is a property of the key rather than of an invalidation rule somebody has
to remember. `prompt_version` is part of the key, which is why bumping
`EXTRACTION_PROMPT_VERSION` to `extract-v2` (below) invalidated the extraction cache
and nothing else.

A transport failure is **never cached**; a refusal or an unparseable reply is, because
both are deterministic properties of that passage and model and re-asking buys
nothing. Caching a connection error would let one unreachable server poison every
later run.

Verified exact rather than merely fast: `test_a_cached_brief_produces_the_same_claims_as_an_uncached_one`
runs a brief twice and asserts an **identical fingerprint with zero model calls** on
the second pass.

The one table in this schema that is infrastructure rather than science — it holds no
claim, no measurement, nothing citable, and clearing it costs time and nothing else.
That is why it is one generic table rather than a typed one per stage.

Measured against one local Ollama instance, `qwen3:14b`, 7 grading calls:

| | total | per call |
|---|---|---|
| serial | 196.9 s | 28.1 s |
| 4 workers | 179.2 s | 25.6 s |
| **cached** | **0.011 s** | 0 calls, 7 hits |

### Parallelism: I was wrong about this one

I projected roughly 4× from running the independent per-passage calls concurrently.
**It gives 9%.**

The reason is that I had the wrong model of the bottleneck. A single local model
instance is *compute*-bound, so the server time-slices concurrent requests rather than
overlapping them — four workers do not get four times the throughput out of one
saturated GPU. Parallelism pays where *latency* dominates: a remote API, or several
models. It is left enabled because 9% is still free and it becomes a real win the
moment the provider is remote, but the configuration comment now states the
measurement rather than the projection.

The cache is what matters locally, and by a margin no projection was needed for:
196.9 s → 0.011 s on the warm path.

### Scoring the document title as well as the chunk text (migration 0007)

Diagnosed from the benchmark's worst case. `technique_disambiguation` scored 0.554
with the right page at **rank 4**, beaten by a passage reading *"The material is LSMO,
not SrTiO3"* — bag-of-words scored that as matching "SrTiO3".

The fixable half: the winning passage is a **table row**, and it contains none of
`pulsed`, `deposition` or `SrTiO3`. All three are in its document's *title*, which the
chunk text does not include. **A table row cannot be scored on its own.** The lexical
leg now scores `title + text`; the title is used for scoring only, so the text
returned, the quote stored and the citation are unchanged.

Migration 0007 adds a matching functional GIN index. A functional index cannot span
two tables, so the title is kept as a trigger-maintained column on the chunk row and
the query reads *that* — an expression that did not match the index would have left
Postgres ignoring it, which was the entire point.

| | before | after |
|---|---|---|
| benchmark baseline | 0.8592 | **0.8884** |
| `reciprocal_rank` | 0.750 | **0.900** |
| `table_retrieval` | 0.711 | **0.886** |
| `technique_disambiguation` | 0.554 | **0.642** |

This subsumed table weighting on the *tabular* case — the baseline's
`table_retrieval` rose to match `table_biased`'s 0.886, because both fixes address the
same cause. Table weighting still earns its place on `technique_disambiguation`
(0.642 → 0.818), where the competing passages are prose and the title does not
separate them. A test that asserted the old relationship was rewritten to assert the
new one rather than deleted.

### A better extraction prompt (`extract-v2`)

The live run left `chamber` unset on every claim although both passages name their
reactor, and reported `material` and `technique` as *differing* between two passages
that both concern HfO2 by ALD — which made `differing_context` noisier than the design
intends. The prompt now enumerates each context field, states that **a reactor
geometry is a chamber**, and requires consistency within a passage, noting why: two
claims from one passage disagreeing about their own context makes two sources look as
though they differ when they do not.

Re-measured against a real extractor on the cross-flow HfO2 passage, `chamber` is now
filled (`cross-flow`) on both claims, which was the defect. One new problem surfaced
that the prompt did not cause and cannot fix: the model put the string
`"200 to 300 degC"` into `temperature_k`. That is worse than a missing temperature,
because the field name asserts kelvin and anything trusting it reads 200–300 K. A
source stating a range in Celsius is ordinary, so this is a guard the contract needs
rather than a prompt bug — see `UNIT_BEARING_CONTEXT` in
`research/contracts.py`, which moves a non-numeric value to `<field>_as_stated`,
leaves the unit-bearing field absent so `missing_context` reports it honestly, and
notes the reason on the claim. A bare numeric string is coerced; a real number passes
through untouched. Five tests in `test_research_contracts.py` hold this.

### Choosing the local models

Extraction and grading are separate roles (`RAG_EXTRACTION_MODEL`,
`RAG_GRADER_MODEL`), and grading dominates the wall clock: 12 candidates × 12 cases =
144 calls per benchmark against one extraction per retained passage. Both were
measured rather than assumed.

**Grading** — four passages with known correct grades, from a target passage down to
an unrelated cryostat distractor:

```
model              total   target/3  related/2  weak/1  distractor/0
qwen2.5-coder:7b    6.3s      3          3        0         0
llama3.1:8b         6.6s      3          3        1         0
qwen3:14b          64.9s      3          2        1         0
```

All three put the target at 3 and the distractor at 0, and — the property that
actually decides retrieval — all three agree on which passages clear
`MIN_USEFUL_GRADE = 2`, retaining `{target, related}` and dropping `{weak,
distractor}`. The 10× slower model buys nothing here. This is not a claim that grader
size never matters: a 1B grader graded the discriminating 1.42 Å/cycle passage as 1
and dropped it, which is what made the cross-paper contradiction invisible (§9). The
finding is narrower — the 7B is already past that cliff.

**Extraction** — one passage, both models correct on the two numbers:

```
qwen2.5-coder:7b   13.4s   carried `chamber: cross-flow` onto both claims
qwen3:14b          50.1s   dropped `chamber` on the density claim, switched to `xrr`
```

The coder model is 3.7× faster. The consistency difference deserves a caveat rather
than a victory: the 14B's `technique: xrr` on the density claim is arguably the *more
faithful* reading, since the passage does attribute density to X-ray reflectometry
while the growth rate is the ALD measurement. The extract-v2 consistency rule is
therefore slightly too strong — a single passage may legitimately report two
quantities measured by different techniques. The 7B obeys the rule as written and the
14B follows the physics. Uniform context is the safer failure for *disagreement
detection*, which is what these claims feed, so the rule stands for now, but it is
recorded here as a known imprecision rather than settled.

### Verifying against real Postgres

Previously deferred as blocked on Docker. It was not: Homebrew `postgresql@17` with
pgvector 0.8.4 was already installed and running, and the block was my assumption that
Docker was the only route. Worth stating plainly, because the deferral hid the one bug
in this work that produces a *wrong scientific answer* rather than a slow or misreported
one (§9.10).

What was checked on PostgreSQL 17.10, in a throwaway database:

- All seven migrations apply from empty, and `0007` down-migrates and re-applies
  cleanly. The re-apply backfills `search_title` from the current titles, so the
  column is self-healing rather than only correct at creation.
- Both GIN indexes exist and the new expression index is **used**: the exact
  expression `hybrid.py` emits plans as a `Bitmap Index Scan on
  ix_document_chunks_fts_title_text` in 5.7 ms.
- The "must match exactly" claim is not folklore. Dropping one space from the
  concatenation — `coalesce(search_title,'') || text` instead of `|| ' ' ||` — falls
  back to a sequential scan at 99 ms over 10,001 chunks, **17x slower**, for a query
  that is character-for-character equivalent in intent.
- Keeping the 0003 text-only index was right: it still serves a text-only match, at
  0.9 ms. That query also demonstrates 0007's whole purpose in one number — it returns
  **0 rows** for `DyScO3` where the title+text index returns 1, because the term exists
  only in the title.
- The retitle bug in §9.10, found and then fixed here: a non-title `UPDATE` does not
  fire the trigger, and a title `UPDATE` to the same value is a no-op.

One thing noted and not pursued: `document_chunks.embedding` is a `json` column, not a
pgvector `vector`, so `pgvector_enabled` does not currently buy an ANN index. On a 47-page
fixture corpus that is irrelevant; on a real corpus it is the next thing to measure, and
it is recorded in §11 rather than guessed at here.

### The resulting policy sweep

```
policy           score    verdict
focused          0.9282   keep      ← narrow pool + table weighting
narrow_pool      0.9087   keep
table_biased     0.9079   keep
baseline         0.8884   —
strict_grading   0.8884   keep      unchanged
no_rewrite       0.8884   keep      unchanged
wide_pool        0.8817   discard
no_grading       0.7947   discard
```

**`focused` is the recommendation: 0.888 → 0.928 at half the extraction cost.** It
combines the two improvements that fix *different* cases — table weighting lifts
`technique_disambiguation` to 0.848, the narrow pool lifts precision from 0.596 to
0.685 — and neither subsumes the other.

The check that mattered: a tighter window could have won the aggregate by failing the
multi-source case. It does not. `cross_paper_disagreement` *improves* (0.848 → 0.924)
and both recall metrics stay at 1.000. On a corpus much larger than 47 pages a narrow
pool would eventually pay for itself in recall, and this benchmark cannot say where
that boundary is.

---

## 10c. The first real-provider run

Everything in §10 and §10b was measured with a stub extractor, which meant four of the
nine metrics were reported as `n/a`: `extraction_f1`, `unit_accuracy`,
`context_completeness` and `abstention_f1`. Running against a real model unlocked them
and changed the conclusion of the whole optimisation round.

Configuration: `qwen2.5-coder:7b` for answering, grading and extraction (one model, so
Ollama never swaps weights mid-run), lexical retrieval, 12 cases, 24 documents /
47 pages. Model choice measured in §10b, not assumed.

### What it showed: retrieval was never the bottleneck

```
                         stub run   real run
doc_recall                  1.0000    1.0000
page_recall                 1.0000    1.0000
reciprocal_rank             0.9444    0.9444
page_precision              0.8704    0.8704
citation_accuracy           1.0000    1.0000
extraction_f1                  n/a    0.1790   <-- 
context_completeness           n/a    0.7500
abstention_f1                  n/a    0.8000
unsupported_claim_rate      0.0000    0.0000
```

Retrieval is essentially saturated on this corpus and extraction was scoring 0.179.
**The policy sweep in §10b — the work that moved the headline score from 0.888 to
0.928 — was tuning the leg that already worked.** That is not wasted (precision and
rank did improve, and on a larger corpus retrieval will bind again) but it is a
correction to the previous section's framing: the ranked policy table answered a
question that was not the limiting one.

### Why extraction scored 0.179

Two causes, separated by reading the diagnostics rather than by assuming:

1. **Vocabulary, not reading comprehension.** `_match_claims` requires exact
   `field_name` equality, and the registry keys are deliberately terse — `k` for
   permittivity, `rho` for density. The prompt listed the keys but glossed none of
   them and said "use these when they fit, otherwise invent a descriptive name". A 7B
   model has no way to know `rho` means density, so it dutifully invented
   `film_density_g_cm3`. The claim was extracted correctly and filed under a name
   nothing searches for.
2. **A rule that made correct extraction impossible.** `REQUIRED_CONTEXT` demands
   `temperature_k` for every growth claim; the prompt forbade converting units, and
   there was no `temperature_c` field. So a source stating "250 degC" — which is
   essentially every ALD and PLD paper — had nowhere legitimate to put its
   temperature, and its growth claims could never become comparable. The extractor's
   workaround was to write the string `"200 to 300 degC"` into `temperature_k`, a
   field whose name asserts kelvin.

`extract-v3` glossed every registry key with its synonyms and required their use;
`extract-v4` added `temperature_c` and moved the conversion into code
(`_derive_kelvin_from_celsius`), which is exact, one line, and auditable — the model
reads, code does arithmetic. The mislabelled-unit case is caught separately by
`UNIT_BEARING_CONTEXT`, which quarantines a non-numeric value to
`<field>_as_stated` rather than letting a Celsius range be read as kelvin.

Measured effect of the glosses alone, with everything else held constant:

```
                       extract-v2   extract-v3
extraction_f1              0.1790       0.3972    +122%
overall_score              0.7899       0.8124
missed claims                   6            3
```

### Where it ended up

```
                    v2       v3       v4       v5     final
overall_score     0.7899   0.8124   0.8257   0.8174   0.8257
extraction_f1     0.1790   0.3972   0.3972   0.3823   0.3972
unit_accuracy        n/a      n/a      n/a      n/a   1.0000
context_compl     0.7500   0.7143   1.0000   0.8571   1.0000
abstention_f1     0.8000   0.8000   0.8000   0.8000   0.8000
run time (s)         240      100       81       61        6
```

`unit_accuracy` is `n/a` in the first four columns because it was computed and never
printed until the end of this round (§9.13) — not because it was unmeasurable. The
`final` column is the run with `extract-v4`, `grade-v1`, unit normalisation, the
zero-claims abstention and the `unit_accuracy` fix.

`v5` is in the table because it was **reverted**: two plausible rules (prefer the
passage's own label; emit one claim per tabulated quantity) cost `context_completeness`
1.0000 → 0.8571 and fixed none of the three misses they were written for. On a 7B model
a new rule spends attention the existing rules were using, and that is only visible by
measuring it.

The `final` column adds the unit normalisation (§9.12) and the zero-claims abstention
(§9.11) on top of `v4`. Its 6-second run time is the whole cache warm — 46x faster than
the cold run, on identical inputs, producing an identical score.

Category scores, final run:

```
cross_paper_disagreement           0.9500
material_disambiguation            0.9500
numeric_extraction_with_context    0.9500
direct_parameter_lookup            0.8975
table_retrieval                    0.8833
process_window                     0.8370
technique_disambiguation           0.8370
incomparable_value                 0.7062
insufficient_evidence              0.6667   <-- 2 of 3; see §11.8c
```

### Dense vs lexical, finally measured

This was §11.7, the largest remaining gap: every number above came from lexical-only
retrieval, while RRF over two retrievers is the design. Run with `--embed-corpus` so the
fixture chunks carry embeddings, all three configurations confirmed active by the
`retrievers` line (`dense`, `lexical`, `dense+lexical`):

```
metric                  dense   lexical   both
doc_recall             1.0000    1.0000  1.0000
page_recall            1.0000    1.0000  1.0000
reciprocal_rank        0.9444    0.9444  0.9444
page_precision         0.8704    0.8704  0.8704
citation_accuracy      1.0000    1.0000  1.0000
extraction_f1          0.3972    0.3972  0.3972
context_completeness   1.0000    1.0000  1.0000
abstention_f1          0.8000    0.8000  0.8000
```

Identical to four decimal places on every metric. The tempting read — "the two
retrievers are equivalent" — is wrong, and checking it was worth the effort. Measuring
the overlap of the pages they actually return:

**Mean Jaccard overlap: 0.485.** The two retrievers disagree about more than half of
what they return, per case, ranging from 0.33 to 0.62. They are genuinely
complementary. The benchmark cannot see it because both put the answer page in the
window and the grader discards the rest — at depth 12 over 47 pages the window is a
quarter of the whole corpus, so "the answer appeared" is nearly free.

Sweeping the depth down finds where the benchmark starts having something to say:

```
depth    dense  lexical   RRF     page_recall, 9 answerable cases
   12    1.000    1.000  1.000
    8    1.000    1.000  1.000
    6    1.000    1.000  1.000
    4    1.000    1.000  1.000
    3    1.000    1.000  0.900   <-- RRF loses one both legs find
    2    0.800    0.900  0.900
    1    0.500    0.800  0.800
```

Three findings:

1. **The default depth is three times past saturation.** Nothing about retrieval is
   measurable above depth 4 on this corpus. That is the same defect as §10's original
   "no discriminating power", one level down: fixed for *policy* comparison, still
   present for *retriever* comparison.
2. **Lexical is the stronger single leg here** — 0.800 vs 0.500 at depth 1. Expected,
   and it is a property of the corpus rather than of the retrievers: the fixtures are
   keyword-dense synthetic technical text full of exact formulae and snake_case
   parameter names, which is the best possible ground for lexical matching and the
   worst for embeddings. **This is not a reason to drop the dense leg.** On real PDFs
   with prose, paraphrase and OCR noise the ordering could easily reverse, and this
   corpus cannot say.
3. **RRF is not always at least as good as its best leg.** At depth 3 it loses
   `technique_disambiguation`, which both single legs find:

   ```
   dense    tabular p1 at rank 3   ✓
   lexical  tabular p2 at rank 3   ✓
   rrf      tabular absent         ✗
   ```

   Both legs rank the right *document* third but on **different pages**. RRF fuses at
   chunk level, so `tabular p1` gets one rank-3 contribution from dense and
   `tabular p2` one from lexical, while two wrong documents each keep a single rank-2
   contribution and take the window. Fusion split the evidence for the right document
   across two of its pages and lost to two wrong ones.

   Latent rather than live: it bites only at depth ≤ 3, and the shipped policies use
   6 (`narrow_pool`), 8 (`focused`) and 12 (`baseline`). Recorded, not fixed — a
   document-level bonus or de-duplicating to document level before truncating would
   plausibly address it, and changing the fusion rule on the strength of one case at a
   depth nothing uses would be exactly the unmeasured churn this benchmark exists to
   prevent.

### The abstention/recall trade, measured

§11.8c recorded abstention at 2 of 3 and said the fix was "stated as an open item rather
than patched, because making the grader stricter trades against recall on the nine
answerable cases and that trade has not been measured." It has now been measured.

`grade-v2` added one rule: the question's *system* is all of it, so a passage about GaAs
on GaAs is a 1 at most for a question about GaAs on germanium, however well it answers
the property.

```
                        grade-v1   grade-v2
insufficient_evidence     0.6667     1.0000   <-- 3 of 3, the thing it was for
abstention_f1             0.8000     0.8571
overall_score             0.8257     0.8342   <-- up, and misleading
doc_recall                1.0000     0.8889
page_recall               1.0000     0.8889
reciprocal_rank           0.9444     0.8889
page_precision            0.8704     0.7917
extraction_f1             0.3972     0.3354
table_retrieval           0.8833     0.0000   <-- broke completely
```

It did what it was written to do and **broke an answerable case doing it**:
`table_retrieval_pld` now abstains on a question the corpus answers, with "no expected
page appeared in the retrieved window at all". The headline score *rose* — which is the
most useful thing about this experiment, because it shows the aggregate hiding a bad
trade: the 0.07-weighted abstention metric moved up while recall and rank, carrying 0.24
between them, moved down, and one capability went to zero.

**Reverted to `grade-v1`.** The reasoning, beyond the numbers: a false abstention on an
answerable question is not a safe failure. It produces a brief that looks disciplined
and is useless, and §9.7 exists precisely to stop "we found nothing" from being said
when it is not true. Trading a correct answer for a correct refusal at 1:1 is not an
improvement for a platform whose corpus is process tables.

Why the naive rule fails is the more useful finding: **a table row does not restate its
system.** `tabular p.2` is a row of PLD process parameters whose material and substrate
are in the document title and the table header, not in the row. A strict
system-matching rule therefore penalises exactly the passages whose context lives above
them — the same underlying problem that migration 0007 and the title-scoring change in
§10b addressed from the retrieval side. A workable version of this rule would have to
exempt a passage whose system is stated in its title or caption, which is a larger
change than one prompt line and would need its own measurement. Recorded, not attempted.

### What the cache is worth on a real workload

The same run, twice, with the persistent cache from §9.8 — and the second run under a
*new* prompt version, which is the interesting case:

```
cold (extract-v2, empty cache)      279 s
second run (extract-v3)             100 s   118 grades reused, 14 extractions re-run
```

Grading is the expensive leg and the grader prompt did not change, so all 118 grades
came back free; extraction correctly *missed* because the prompt version changed. That
is the cache behaving as designed in both directions at once — reuse where the inputs
are identical, a miss where the prompt is not. The alternative would have been a cache
that served v2 answers under a v3 prompt, which is worse than no cache.

---

## 10d. First run on real PDFs through real Postgres

Everything up to here was measured on the synthetic fixture corpus inside a throwaway
SQLite database. This is the first pass through the path a user actually takes: ingest
PDFs into Postgres, ask a question, read a brief. It found four bugs, three of which
produced **fabricated scientific conclusions** — which is the failure mode this whole
design exists to prevent, so they are worth stating plainly.

Setup: `cnms-fom ingest data/pdfs --technique ald` into a migrated PostgreSQL 17.10
database, two synthetic ALD papers, 7 chunks, all embedded with `nomic-embed-text`.

What worked first time: ingestion, the 0007 `search_title` trigger (all 7 chunks in step
with their document titles on the real insert path), embeddings, the dense leg,
extraction units (`A/cycle`, from `extract-v4`), the missing-context guard, and the
quote-verification guard, which caught the extractor inventing three quotes and marked
those claims low-confidence instead of trusting them.

### Bug 14: the lexical leg did not work on Postgres at all

`NotImplementedError` with an **empty message**, surfacing as the warning "Corpus
retrieval failed: ." — a report that something broke with no way to find out what.
Cause: `.label()` on a `text()` clause, which SQLAlchemy 2.0 does not support. The
lexical leg had only ever run on the SQLite term-overlap fallback path, so no test and
no benchmark run had ever exercised it.

Two things made it worse than a one-line bug:

- **The documented fallback was unreachable.** The `try` wrapped only execution, while
  the failure was at statement *construction*. The comment above it promised "the
  caller below turns into a fallback to term overlap with a warning. That is the right
  failure: slower, not wrong." It was not the right failure; it was a crash. Statement
  construction now happens inside the `try`.
- **A crashed query was reported as a finding about the literature.** With no passages,
  the brief said "the evidence did not clear the policy's threshold" — §9.7's exact
  failure mode in a path §9.7 did not cover. Retrieval failure now sets
  `degraded_reason`, and `_reason()` guarantees a non-empty description.

### Bug 15: the grader marked down partial answers to compound questions

The question was "what growth per cycle **and** film density are reported, **and** do
the sources agree?". The second paper's passage was graded **1** and dropped, with the
reason "discusses growth per cycle for HfO2 ALD but does not address film density or
agreement between sources."

That reasoning is accurate and the grade is wrong. The rubric it was given says
`2 = useful: contains part of what the question asks for`, and the passage contains a
part. The grader was marking a passage down for the parts it did not cover, and no
single passage can ever "address agreement between sources" — that comparison happens
downstream, over the passages it keeps. `grade-v3` states this, enforcing the existing
rubric rather than adding a rule.

This also qualifies §10b's conclusion that the 7B grader is "past the cliff" where a 1B
grader drops the discriminating passage. On a two-document real corpus it dropped it
too, for a different reason.

### Bug 16: a dose time was filed as a growth rate, and reported as a disagreement

With the passage retrieved, extraction produced this:

```
growth_per_cycle_ang   0.2 s     <- a TDMAH dose time
growth_per_cycle_ang   6 s       <- an N2 purge time
growth_per_cycle_ang   0.1 s
growth_per_cycle_ang   6 s
```

and the interpretation step then wrote: *"growth per cycle values vary widely (0.2 s,
6.0 s, 0.1 s, 6.0 s), indicating a lack of consistency in the literature."* A fabricated
disagreement, assembled out of purge timings, in a field whose name declares Ångström.

The same bug class as `temperature_k` holding `"200 to 300 degC"` (§10b), which was
guarded for *context* fields while a claim's own field/unit pairing had no guard at all.
`FIELD_DIMENSION` plus `classify_unit` now reject a registry-keyed claim whose units
have a recognised and incompatible dimension. It **raises**, so `extract.py` records the
rejection as a visible problem on the brief rather than dropping it silently.

Deliberately a dimension check rather than a list of acceptable spellings: sources write
"A/cycle", "Å/cy" and worse, and rejecting a legitimate value over an unrecognised
spelling would lose data. A unit is rejected only when its dimension is *recognised* and
wrong; an unclassifiable unit is kept. Act on knowledge, abstain on ignorance.

Writing the classifier produced its own instructive bug: `"s" in "angstrom"` is true, so
substring matching classified every length as a time and would have rejected exactly the
claims the guard protects. Bare unit markers now require whole-string equality after a
leading magnitude is stripped; only compound markers (containing `/`) match as
substrings. Twenty-five classifier cases are pinned in tests.

### Bug 17: a category with a number invented a second disagreement

After bug 16 was fixed the narrative still said *"a growth per cycle of **2.0** was
reported for the hot-wall chamber"*. The source was an extraction of `material = 2` —
a categorical context field with a numeric value — which the interpretation step read as
a growth rate. `CATEGORICAL_CONTEXT_FIELDS` now rejects a numeric claim under a field
that names a category. Carefully scoped: `thickness_nm`, `temperature_c`,
`pressure_torr` and `frequency_hz` are legitimately both context and quantity, and are
untouched.

### The one change kept despite measuring worse, and why

`grade-v3` costs benchmark score:

```
                      grade-v1   grade-v3
overall_score           0.8257     0.8078
page_precision          0.8704     0.7407
extraction_f1           0.3972     0.3549
doc_recall              1.0000     1.0000
page_recall             1.0000     1.0000
unit_accuracy           1.0000     1.0000
context_completeness    1.0000     1.0000
abstention_f1           0.8000     0.8000
```

Kept anyway, which needs justifying because the discipline everywhere else in this
document is to revert what measures worse (`extract-v5`, `grade-v2`).

The reason the two disagree: **`grade-v3` only changes behaviour on compound questions,
and the benchmark has none.** All 12 cases ask for a single fact. So the benchmark sees
the cost — a more permissive grader keeps more passages, which dilutes precision and
feeds extraction more chances to produce a non-matching claim — and cannot see the
benefit, because no case exercises it.

The benefit is categorical rather than incremental. Without `grade-v3`, a compound
question drops the only passage from the second source, so a cross-source disagreement
cannot be *attempted*, let alone found. FOM_PROOF Sec. 2.1 and this project's own
boundary require preserving each determination and reporting disagreement; a grader that
discards one side makes that impossible rather than merely harder. And real questions are
compound — the question that exposed this was the obvious one to ask of a two-paper
corpus.

It also pairs with the guards from bugs 16 and 17. "Keep more passages, then reject the
claims that are dimensionally or categorically incoherent" is a defensible architecture.
"Drop passages early and trust whatever survives" is the one that produced a fabricated
literature disagreement out of purge timings.

**The actual defect here is in the benchmark, not in either grader.** It needs a
compound-question case. Not added in this round on purpose: every comparison table above
is stated against the current 12-case baseline, and changing the case set would
re-baseline all of them at once. It is the first thing to do next, and until it exists
the 0.8078 above should be read as "measured against a case set that cannot evaluate this
change".

### Where it stands

Both fabrications are gone. Every claim in the final brief traces to a real number in a
real passage with units that can belong to its field, and the narrative makes only
statements the evidence supports.

**The real disagreement is still not found.** The second paper reports `0.98 angstrom
per cycle` and extraction never picks it up, so the 1.42-vs-0.98 contradiction — the
flagship capability — does not appear. That is extraction *recall*, consistent with
`extraction_f1 = 0.397` on the benchmark, and it is now the top open problem rather than
a suspicion. The guards added here make a wrong answer much harder; they do not make a
missing one appear.

---

## 11. Known limitations and deferred work

**Measured, and significant:**

1. **The cheap grader drops relevant pages.** With `gemma3:1b` grading, the passage
   containing the cross-flow `1.42 Å/cycle` figure scored **grade 1 and was dropped**
   — so the live brief found **zero contradictions** on a corpus built to contain
   one. Nothing scored above 2 at all. Re-run with `qwen3:14b` grading, that passage
   scores **3**, is kept, and the contradiction is found (§7). The 6× speedup from a
   1B grader costs the headline result.
   **Recommendation: do not use a 1B grader for questions that span sources.**
   This is a measured causal chain, not an inference: weak grader → dropped passage
   → missed contradiction, and reversing the grader reverses the outcome.
2. ~~**Per-passage extraction is slow on a reasoning model.**~~ **Resolved.** It was
   90–250 s per passage with `qwen3:14b`. `qwen2.5-coder:7b` does it in 13.4 s and
   grades in 1.6 s, and §10b shows the grades it keeps are the same ones. A full
   12-case real-provider run is 81 s warm, 279 s cold. The remaining caveat is the
   one in §10b: the 7B is *more* internally consistent about context but slightly
   less faithful about which technique measured which quantity.
3. **The extractor over-extracts.** It returned 23 claims for one question, most of
   them recipe parameters nobody asked for. Harmless — each is provenance-backed —
   but it inflates the extraction cost and dilutes a brief.
4. ~~**The extractor fills context fields inconsistently.**~~ **Resolved, and it was
   two separate problems.** `chamber` unset on every claim was a prompt gap, fixed in
   `extract-v2` ("a reactor geometry *is* a chamber") and confirmed against a real
   extractor. The rest was not the extractor's fault at all: `REQUIRED_CONTEXT`
   demanded `temperature_k` while the prompt forbade converting and no
   `temperature_c` field existed, so a source stating °C could never produce a
   comparable claim. With `temperature_c` and the in-code conversion,
   `context_completeness` is **1.0000**. See §10c.

   Remaining and unresolved: **prompt length has a measurable cost on a 7B model.**
   `extract-v5` added two plausible rules — prefer the passage's own label, and emit
   one claim per tabulated quantity — and made things *worse*: `overall_score`
   0.8257 → 0.8174 and `context_completeness` 1.0000 → 0.8571, with the model
   starting to drop `frequency_hz`. It did not fix the three misses it was written
   for. Reverted to `extract-v4`. Recorded because the negative result is the useful
   part: on a small model, adding a rule spends attention that the existing rules
   were using.

**Now exercised — the two that were listed as blocked:**

5. ~~**Postgres.**~~ **Done on PostgreSQL 17.10 with pgvector 0.8.4.** All seven
   migrations apply from empty; 0007 down-migrates and re-applies; the expression
   index is confirmed *used* by the planner. The earlier "blocked on Docker" was
   wrong — Homebrew `postgresql@17` was already installed and running. Details and
   numbers in §10b. This is where §9.10 was found, which is the argument for not
   trusting a deferral.
6. ~~**Benchmark with a real provider.**~~ **Done.** All four metrics unlocked, and
   they overturned the previous section's conclusion: retrieval was saturated and
   extraction was at 0.179. See §10c.

**Now the top open problem:**

6b. **Extraction recall, not retrieval, is the binding constraint.** `extraction_f1` is
    0.397 after the prompt work in §10c, and on real PDFs (§10d) the extractor misses
    `0.98 angstrom per cycle` outright, so the one genuine cross-source disagreement in
    a two-paper corpus is never reported. Everything added in §10d makes a *wrong*
    extraction much harder to produce; none of it makes a *missing* one appear. Three
    benchmark claims are also still missed (`decomposition_onset_c`,
    `substrate_temperature`, `oxygen_pressure`), all of them present in the retrieved
    page. `extract-v5` tried to fix exactly this and measured worse, so the next attempt
    should probably not be another prompt rule — a larger extraction model, or a second
    pass over a page that a claim was expected on and not found, are the untried options.

**Still not exercised:**

7. ~~**`dense_only` / `lexical_only` policies against the benchmark.**~~ **Done, and it
   produced the most useful negative result in this document.** All three
   configurations score identically on every metric, *and* they retrieve substantially
   different pages (mean Jaccard 0.485). The benchmark cannot discriminate retrievers
   above depth 4 on a 47-page corpus, and RRF loses one case at depth 3 that both
   single legs find. See §10c. **What remains genuinely unexercised: retrieval quality
   on a real corpus.** Every retrieval number here is saturated, so it measures that
   the plumbing works, not that the retrieval is good.
8. **Anthropic provider on the research path.** Structurally identical to the chat
   path (same seam) but not run.
8b. **pgvector is not actually used for the embedding column.** `document_chunks.embedding`
    is a `json` column, so `pgvector_enabled` buys no ANN index today. Irrelevant on a
    47-page fixture corpus; the first thing to measure on a real one.
8c. **Abstention is 2 of 3, not 3 of 3 — and the obvious fix is worse.** The
    `absent_gaas_on_ge` case still fails. §9.11 fixed the half that was a logic gap
    (zero claims no longer means "carry on interpreting"). The remaining half is a
    grading judgement, and the trade is now measured rather than assumed: a stricter
    grader takes abstention to 3 of 3 and breaks `table_retrieval` to 0.000, because a
    table row does not restate its own system. Reverted; see §10c. A version that
    exempts a passage whose system is in its title or caption is the next thing to
    try, and has not been tried.

**Deliberately not implemented:**

9. **No application-specific benchmark cases.** Per the brief's own instruction,
   deferred until the infrastructure audit passes. The fixture corpus is synthetic
   and marked `SYNTHETIC` on every page.
10. **No new FOM definitions, weights, bounds, or hypothesis signs.** Out of scope by
   the non-negotiable boundaries.
11. **No instrument-registry integration.** The placeholder envelopes are unchanged.
12. **No GPU training loop or wall-clock training budget** from the reference
    project. The objective here is evidence quality, not neural-network loss.

---

## 12. Running it locally

```bash
# Schema and seed
cnms-fom init-db && cnms-fom seed

# Models. Use a capable grader — a 1B model drops relevant pages (§11).
ollama pull nomic-embed-text
ollama pull qwen3:14b
export RAG_GRADER_MODEL=qwen3:14b          # or the chat model
export RAG_EXTRACTION_MODEL=<fast model>   # optional; extraction is per passage

# Corpus and measurements
python scripts/make_synthetic_corpus.py     # or POST /rag/ingest/upload
cnms-fom ingest data/pdfs --technique ald
cnms-fom import-fits data/fits

# The benchmark — offline, no model server needed
cnms-fom research benchmark --case-set baseline
cnms-fom research benchmark --compare no_grading table_biased wide_pool

# A brief (exit 2 means it abstained)
cnms-fom research brief \
  "What is the growth per cycle for HfO2 ALD from TDMAH and water, and does the literature agree?" \
  --run-id 1 --technique ald --policy narrow_pool

# The gated path to the optimizer
curl -X POST localhost:8000/research/campaigns/1/context/propose -H 'content-type: application/json' -d '{
  "recommended_bounds":[{"parameter":"substrate_temp_c","lower":200,"upper":300,
                          "rationale":"the reported ALD window"}],
  "supporting_card_slugs":["concepts/ald-window-hfo2"]}'
cnms-fom research context review 1 --by "Your Name"
cnms-fom research context apply  1 --by "Your Name"

# The evidence trail
cnms-fom research audit --run-id 1
```

---

## The nine questions, answered

| Question | Answer | Where |
|---|---|---|
| Can an LLM write a measured property? | **No.** No code path exists. | §5 row 1 |
| Can an unreviewed card change a live BO campaign? | **No.** `apply` requires `REVIEWED`; cards re-validated three times. | §5 row 3 |
| Can a stale reviewed card be treated as current? | **No.** Body hash re-checked at apply; proposal marked `STALE`. | §5 row 4 |
| Are page-level citations retained for extracted claims? | **Yes.** `page` + NOT NULL `quote`, and the quote must appear in the passage. | §5 row 5 |
| Are ModalFit and literature values distinguishable? | **Yes.** Different tables, different tier enums, and a record warrant reads `experiment:12`, not `Kim 2024, p. 7`. | §3, §5 row 2 |
| Are disagreements preserved? | **Yes.** No `resolved_value` anywhere; detection is arithmetic. | §5 row 6 |
| Does a data gap remain a data gap? | **Structurally yes; behaviourally 2 of 3.** Abstention is a first-class outcome, every gap names what resolves it, and nothing fills a gap with a guess — `unsupported_claim_rate` is 0.0000 on every run. But measured against a real model, one of three unanswerable questions is *not* abstained on: the corpus has GaAs on GaAs and the question asks about GaAs on germanium. Written as a qualified answer because the first real-provider run is what turned this from an assertion into a measurement. | §5 row 7, §10c, §11.8c |
| Can core tests run without heavy RAG/BO dependencies? | **Yes.** The whole suite runs on SQLite with scripted providers. | §8 |
| Are all autonomous policy experiments reproducible? | **Partly, and more so than before.** Every row records the commit, policy fingerprint, case set, retrievers and whether extraction was real, and a dirty tree is marked. A run against a live model is only as reproducible as that model — but the content-addressed cache now makes a re-run over unchanged inputs bit-identical (0.8257 reproduced exactly, 279 s → 6 s), and it misses rather than lies when the prompt or model changes. | §10, §10c, §11 |
