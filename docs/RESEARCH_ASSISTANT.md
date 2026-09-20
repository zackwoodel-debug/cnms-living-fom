# The research assistant

A retrieval assistant over two kinds of evidence: the **document corpus** (MBE /
PLD / ALD / sputtering / CVD process notes and CNMS user guides) and the
platform's **own records** — ModalFit co-refinements, property values, FOM
scores.

It answers only from what it retrieves. That sentence is doing real work: an
answer produced without a single tool call is discarded and replaced with an
explicit data gap before it reaches the caller. FOM_PROOF Sec. 15.2 says a
missing value may not be filled by a plausible number from memory, and a language
model is an extremely efficient source of plausible numbers. Enforcing that in
code rather than in a prompt is the difference between this and a chatbot with a
materials vocabulary.

---

## 1. Three entry points

| Endpoint | Shape | Use it when |
|---|---|---|
| `POST /rag/query` | one retrieval, one answer | The question maps onto a single search: "what ALD window does this paper report for HfO2 on Si?" |
| `POST /rag/search` | retrieval only, no generation | You want to read the passages yourself, or find out why a document you know is indexed did not come back. |
| `POST /rag/chat` | multi-step assistant | Anything needing more than one lookup. "Is this thickness trustworthy?" needs the fit records, then the literature on why two techniques disagree. |

There is also `cnms-fom ask "<question>"` on the CLI, which prints the evidence
trail and exits **2** on a data gap — so a script can tell *answered* from *could
not answer*.

Documents get in through `POST /rag/ingest/upload` (multipart, several files at
once, indexed immediately) or by putting them where the API can see them and
calling `POST /rag/ingest` with the path. `docs/COSCIENTIST.md` walks the whole
path from a PDF on disk to a cited answer.

---

## 2. The retrieval pipeline

```
question
   │
   ├─ dense retrieval      embeddings (Ollama) over document_chunks, pgvector or numpy
   ├─ lexical retrieval    Postgres to_tsvector / websearch_to_tsquery, ts_rank_cd
   │
   ├─ reciprocal rank fusion            score = Σ 1/(60 + rank)
   ├─ relevance grading                 each candidate 0–3, against the question
   ├─ corrective rewrite (once)         if too little survives
   │
   └─ generation, or an explicit data gap
```

### Why both retrievers

Embedding search is good at paraphrase and bad at rare exact tokens. A synthesis
corpus is mostly rare exact tokens.

`TMA` and `trimethylaluminum` are the same precursor and `nomic-embed-text` knows
it — that is the dense leg earning its place. But a question about the
*Nevot-Croce* roughness factor retrieves passages about roughness in general while
the one passage that names the factor ranks eleventh, and `HfO2` / `HfO_2` /
`hafnia` are three spellings a 768-dimension vector will happily place next to
`ZrO2`. Lexical search inverts both failure modes: useless for paraphrase, exact
on tokens.

### Why rank fusion rather than a weighted sum

Cosine similarity and `ts_rank_cd` are not on the same scale, do not have
comparable distributions, and a weighted blend of them is a free parameter
pretending to be a method. Reciprocal Rank Fusion uses only the ranks, needs no
calibration, cannot be broken by one retriever's scores drifting, and rewards a
chunk both retrievers liked over one either loved alone. `found_by` on every hit
tells you which retriever surfaced it.

### Why grading, on top of a similarity threshold

A similarity threshold cannot distinguish an on-topic passage that *answers* the
question from an on-topic passage that merely *shares its vocabulary*, and in a
corpus of process recipes those look nearly identical to an embedding. So each
candidate is graded:

| Grade | Meaning |
|---|---|
| 0 | irrelevant — does not concern the question's subject |
| 1 | related — same subject area, does not address the question |
| 2 | useful — contains part of what was asked |
| 3 | directly answers — the specific fact, with the context that makes it meaningful |

Only grade ≥ 2 reaches the answer. A numerical parameter stripped of its chamber,
substrate, and precursor is capped at 2 — a number without its context is not an
answer.

An unparseable grader reply becomes **grade 1**, deliberately. Failing to 3 would
turn every grader hiccup into a fabricated basis for an answer; failing to 0 would
empty the context and report a data gap the corpus does not have. Grade 1 keeps
the passage out of the answer and visible in the evidence.

### Why one rewrite

Most retrieval failures are vocabulary failures: the user asked about "growth
temperature", the paper says "substrate setpoint". One rewrite recovers a large
fraction of those. Failing *after* it is a finding rather than a phrasing
accident.

### A failed search is an error, not a data gap

If retrieval could not run — Ollama unreachable, the extra not installed — and the
lexical leg found nothing either, the call fails with a 503. Reporting "the corpus
does not contain this" about a search that never ran is a wrong scientific
conclusion wearing a clean answer's clothes.

When the dense leg is down but lexical *does* find something, retrieval degrades
to lexical-only, logs it loudly, and says so in `found_by`. Half a retriever beats
none.

---

## 3. The assistant's tools

Twenty tools. Everything that touches an **analysis table is read-only** — there is
no tool that writes a property value, edits a fit, or creates a material, and that
is the design rather than an unimplemented feature. The way to forbid a
model-mediated path into the analysis tables is not to build one.

| Tool | Returns |
|---|---|
| `search_cards` | knowledge cards matching a query, each with its `citable` flag |
| `read_card` | one card with its typed links in both directions |
| `card_graph` | nodes and typed edges; `orphans` names cards nothing links to |
| `card_stats` | review backlog, stale reviews, unresolved contradictions |
| `search_corpus` | graded, citable passages; `sufficient_evidence=false` when the corpus does not answer |
| `corpus_coverage` | documents and chunks by technique — what a gap is actually caused by |
| `list_samples_with_fits` | samples with stored refinements |
| `list_sample_fits` | one sample's fits: stack, techniques, chi², which parameters were varied, and every caveat |
| `compare_fit_techniques` | one parameter across every technique that determined it |
| `fit_disagreements` | every cross-technique disagreement on a sample |
| `check_physical_plausibility` | violations, inconsistencies, heuristic flags on a set of values |
| `check_fit_plausibility` | the same checks over a sample's stored fit layers |
| `lookup_bo_campaign` | search space, constraints, acquisition, objective, counts |
| `lookup_bo_history` | best-so-far trajectory and `evaluations_since_best_improved` |
| `lookup_bo_suggestions` | pending proposals with acquisition value and `predicted_std` |
| `lookup_property_values` | stored values with full measurement context and source |
| `descriptor_dictionary` | valid registry keys, units, transforms, caveats |
| `lookup_fom_scores` | scores with status and definition version |
| `write_card` *(opt-in)* | records a synthesis as a `proposed`, non-citable card |
| `link_cards` *(opt-in)* | a typed edge; `contradicts` requires a note |

`descriptor_dictionary` exists so the assistant uses the platform's vocabulary
instead of inventing a plausible-looking key: "the band gap" resolves to `Eg`
with its declared units, not to free text.

### The two writers, and why they are the exception

`write_card` and `link_cards` write, and only to knowledge cards. The exception is
narrow and deliberate. A card is explicitly a *reading aid* — the assistant's
synthesis, written down so it accumulates instead of evaporating — and it is
quarantined by construction: it lands `proposed`, `citable` is false until a named
person has checked it against a resolved source, and no code path leads from a card
to a stored property value.

They are also withheld by default. `tool_specs()` and `agent.ask` omit them unless
`allow_card_writes=True`, and `run_tool` refuses them even if the model invents the
name — so the default surface is entirely read-only.

### Judging whether a number is physical

`check_physical_plausibility` reports in three tiers, and keeping them apart is the
whole value of it:

- **violation** — the number cannot be true. A permittivity below 1 polarizes
  against the field; a band offset above the gap puts the conduction band below the
  valence band; an optical permittivity above the static one contradicts the fact
  that the static response contains everything the optical one does.
- **inconsistency** — two values that must agree and do not. The X-ray SLD versus
  mass density check is the one that earns its keep: `SLD = rₑ·ρ·N_A·(Z/A)`, so for a
  fixed composition they are one measurement. XRR is nearly degenerate in the two,
  so a co-refinement that leaves both free will trade one against the other and
  reproduce the curve while disagreeing with itself.
- **heuristic** — a domain expectation with real exceptions. High k *and* a wide gap
  together; a band offset under ~1 eV on silicon; an ALD growth-per-cycle above one
  monolayer.

A heuristic does not vote on `physical`, and the report says outright that it is
never grounds to exclude a value. Sec. 2.3 governs exclusion, and "a heuristic
disliked it" is not on that list.

### Reading an optimization campaign

The three BO tools exist because the raw tables do not answer the question people
actually ask. `evaluations_since_best_improved` is not a column anywhere; neither is
"are the suggestions still uncertain, or has the surrogate given up?".

Two failure modes the assistant is told to watch, because both look like success
from the inside: a **flat best-so-far** is not convergence if the suggestions still
carry large `predicted_std` and cluster on a bound — a search space whose optimum
lies outside its own bounds looks exactly like a converged campaign. And
**infeasible observations carry information**: they bound the feasible region, so a
mostly-infeasible campaign has a constraint problem, not a search problem.

One scale trap the tools state explicitly: the surrogate models **ln F**, not F
(Eq. 30), so a predicted mean of −0.7 against −1.4 is a factor of two in F.

### Structured records are never retrieved by similarity

A fitted thickness of 103.4 Å must not be found by cosine similarity. An
embedding of `103.4` sits close to an embedding of `130.4`, the retrieval would be
silently wrong, and a number is exactly the kind of thing a model reports without
hedging. So prose goes through the vector index and every stored number goes
through a typed query with its provenance attached.

---

## 4. The loop

```
question ──► model ──► tool calls ──► results ──► model ──► … ──► answer
                                                      │
                                    bounded by ASSISTANT_MAX_STEPS (default 6)
```

Three rules hold it together.

**Evidence is mandatory.** No tool call, no answer — the reply is replaced with a
data gap. Enforced in `agent.ask`, not requested in the prompt.

**Every step is recorded.** `steps[]` carries each tool call, its arguments, its
full result, and its duration, and all of it is persisted on the turn. "Where did
that number come from?" is answerable six months later.

**The budget is finite.** At the limit the model is asked once more *with tools
withheld*, so it has to commit to an answer from what it already has rather than
returning the user a bare "step limit reached".

A model that searched, found nothing, and answered anyway is also caught: if every
retrieval in the turn came back empty, the answer is marked
`insufficient_context` regardless of how confident it sounded.

---

## 5. Conversations

Two things are stored, serving opposite needs.

**History** is replayed so follow-ups work — "and how thick was the interlayer?"
needs the previous turn to mean anything. Only the *text* of each turn is
replayed. Feeding raw tool results back would spend the context window re-reading
fit records the assistant already summarised, and by turn four there is no room
left for the retrieval instructions — which are the first thing to fall out and
the last thing you want to lose.

**Evidence** is stored and never replayed: `chat_messages.tool_calls` and
`.evidence` hold the exact chunk ids, tool arguments, and results.

`n_data_gaps` on a conversation summary is a corpus-coverage metric, not a failure
log. A conversation that is mostly gaps is telling you what to ingest next.

`DELETE /rag/sessions/{key}` removes a conversation and its evidence trail. It
exists because a conversation can contain a user's unpublished work and they are
entitled to remove it; it takes the audit trail with it, which is why it is an
explicit call and not a retention policy.

---

## 6. Providers

| Provider | Default | Trade-off |
|---|---|---|
| `ollama` | yes | Nothing leaves the machine. A local 8B model follows the retrieval discipline less reliably. |
| `anthropic` | no | Better instruction-following on exactly the constraints that matter — declining when evidence is thin, carrying a parameter's context, refusing to average disagreeing sources. Retrieved excerpts, including CNMS user documents, are sent to a remote API. |

Ollama is the default because the corpus is unpublished CNMS process
documentation. Switching is a per-corpus decision, not a one-time setting:

```bash
RAG_LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-opus-5
pip install -e '.[anthropic]'
```

Selecting the Anthropic provider logs at WARNING, and `/health/ready` says so
under `checks.assistant.warning`. Leave the API key blank to let the SDK resolve
credentials itself (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an
`ant auth login` profile).

The provider seam is deliberately small — system prompt in, messages in, optional
tools in, text and tool calls out. Grading, reranking, the correction loop, and the
loop guardrails all sit above it, which is the only way the guarantees are
identical on both paths.

Notes on the Anthropic path: adaptive thinking is on and `effort` is `high`
(temperature is rejected on that model family, so determinism comes from the
prompt, not from `temperature=0`); the system prompt carries a `cache_control`
breakpoint while retrieved excerpts go in the *messages*, since caching is a
prefix match and putting excerpts in the system block would invalidate the cache
on every question; tools are declared `strict`, so a hallucinated argument name
fails at the schema rather than in the SQL; server-side refusal fallback is on by
default, and a refusal is surfaced as a refusal rather than recorded as a data gap.

---

## 7. Operational notes

**The lexical index.** Migration 0003 creates a functional GIN index over
`to_tsvector('english', document_chunks.text)`. Without it lexical retrieval still
works and sequentially scans, which looks fine on a test corpus and falls over at
50,000 chunks. `/health/ready` reports `checks.retrieval.fts_index`.

**Cost.** Grading costs one model call per candidate. `grade=false` on
`/rag/search` turns it off, which is right for exploratory search a human will read
and wrong when a model is about to write an answer from the results.

**Diagnosing a miss.** `POST /rag/search` with `diagnostics: true` returns each
retriever's own ranking beside the fused one, plus the tokenised query. Run it
before concluding the corpus is missing a document.

---

## 8. Worked example

```bash
# Index the corpus and import the fits.
cnms-fom ingest data/pdfs --technique ald
cnms-fom import-fits data/fits --technique XRR

# Where do two techniques disagree?
cnms-fom compare-fits HFO2-PILOT-07 --parameter thickness

# Ask about it.
cnms-fom ask "XRR and SE disagree on the HfO2 thickness for PILOT-07. \
What does the literature say causes that in a high-k oxide on Si?" \
  --sample-id HFO2-PILOT-07 --technique ald
```

The assistant lists the sample's fits, compares thickness across techniques,
finds the 38% spread, and then searches the corpus for the mechanism — reporting
both determinations and never their average, because two techniques that far apart
do not have a mean worth reporting. They have a discrepancy someone has to explain.
