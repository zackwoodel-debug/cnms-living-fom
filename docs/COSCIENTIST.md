# How this works, end to end

Start here. This is the walkthrough: where documents go, what happens to them,
how the assistant reaches them, and how it helps with the optimizer.

---

## 1. Where the PDFs go

Two ways in. Both end up in the same place.

**Upload them over HTTP.** This is the one you want day to day.

```bash
curl -X POST localhost:8000/rag/ingest/upload \
     -F 'files=@~/papers/kim-2024-ald-hfo2.pdf' \
     -F 'files=@~/papers/robertson-2006-high-k.pdf' \
     -F 'technique=ald'
```

Or from `/docs` in a browser — the endpoint shows up as a file picker, which is
usually the easiest way to drop in a stack of papers.

**Or put the files somewhere the API can see and point at the directory.**

```bash
cp ~/papers/*.pdf data/pdfs/
cnms-fom ingest data/pdfs --technique ald
```

Either way the files land in `CORPUS_DIR` (default `data/pdfs`, mounted at
`/app/data/pdfs` under docker-compose) and get indexed immediately.

Worth knowing:

- **Set `technique` when you know it.** Without it the corpus partition is guessed
  from the filename, and a wrong guess files the paper in a slice that a
  technique-filtered search will never look in.
- **Re-uploading the same paper is free.** Deduplication is by content hash, not
  filename, so the same PDF arriving twice under two names is one document.
- **A scanned PDF with no text layer will fail.** It has no text to chunk. The
  file is kept, the error says so, and OCR then re-ingest works.
- **Only PDFs.** Every passage is stored with a page number so it can be cited;
  a format without pages cannot produce a citation, so it is refused rather than
  ingested without one.

Check what landed:

```bash
curl -s localhost:8000/rag/corpus | jq
```

---

## 2. What happens to a PDF after it lands

```
paper.pdf
   │
   ├─ split into ~1000-character passages, 200 overlapping
   │    (small, with generous overlap, so a parameter and its units stay together)
   ├─ each passage keeps its page number          ← this is what makes citation work
   ├─ each passage gets an embedding vector       ← via Ollama, locally
   └─ rows in documents + document_chunks
```

The page number is the point. An answer this platform can use is one you can trace
to a page, so the locator is attached at chunk time and travels with every
retrieval afterwards.

---

## 3. How a question gets answered

```
your question
   │
   ├─ dense retrieval    embeddings — good at paraphrase ("TMA" ≈ "trimethylaluminum")
   ├─ lexical retrieval  Postgres full-text — good at exact rare tokens
   │                     ("Nevot-Croce", "HfO2", "0.98 A/cycle")
   │
   ├─ fused by rank      a passage both retrievers liked outranks one either liked alone
   ├─ graded 0-3         does this passage ANSWER the question, or just share its words?
   ├─ one rewrite        if too little survives — most misses are vocabulary misses
   │
   └─ answer with citations, or [DATA GAP: explicitly unresolved]
```

Both retrievers, because each fails where the other works. An embedding knows TMA
and trimethylaluminum are the same precursor; it does not reliably find the one
passage that names the Nevot-Croce factor, and it will happily put `HfO2` next to
`ZrO2`. Lexical search is the reverse: exact on tokens, useless on paraphrase.

The grading step is the one people skip. A similarity threshold cannot tell a
passage that answers your question from one that merely shares its vocabulary, and
in a corpus of process recipes those look nearly identical to an embedding. A
number stripped of its chamber, substrate, and precursor is capped at grade 2 — it
is not an answer.

If nothing graded useful, you get a data gap naming what would resolve it. That is
a correct answer here, not a failure.

**One distinction that matters:** if retrieval could not *run* — Ollama down, the
extra not installed — you get a 503, not a data gap. Reporting "the corpus does
not contain this" about a search that never happened is a wrong scientific
conclusion wearing a clean answer's clothes.

---

## 4. The co-scientist: `POST /rag/chat`

`/rag/query` does one retrieval and answers. `/rag/chat` is the collaborator: it
picks its own retrievals, chains them, and reaches everything the platform knows.

```
"XRR and SE disagree on the HfO2 thickness for PILOT-07. Is that real,
 and what should I do about it?"

  step 1  list_sample_fits            → two fits: XRR 103.4 Å, SE 152 Å
  step 2  compare_fit_techniques      → 38% spread, no uncertainties reported
  step 3  check_fit_plausibility      → SLD and density disagree by 40% on the XRR fit
  step 4  search_corpus               → what causes an SE/XRR thickness discrepancy
  → answer, with every number traced to a fit record or a page
```

Its twenty tools, by what they are for:

| Purpose | Tools |
|---|---|
| What we already worked out | `search_cards`, `read_card`, `card_graph`, `card_stats` |
| The literature | `search_corpus`, `corpus_coverage` |
| Measured structure | `list_samples_with_fits`, `list_sample_fits`, `compare_fit_techniques`, `fit_disagreements` |
| Is this physical? | `check_physical_plausibility`, `check_fit_plausibility` |
| The optimizer | `lookup_bo_campaign`, `lookup_bo_history`, `lookup_bo_suggestions` |
| Stored records | `lookup_property_values`, `lookup_fom_scores`, `descriptor_dictionary` |
| Writing knowledge down | `write_card`, `link_cards` *(opt-in)* |

**It cannot answer without using them.** An answer produced with zero tool calls
is thrown away and replaced with a data gap before you see it. That is checked in
code, not requested in the prompt, and it is the whole difference between this and
a chatbot that has read about materials science.

```bash
cnms-fom ask "Is the HfO2 thickness on PILOT-07 trustworthy?" \
  --sample-id HFO2-PILOT-07
# exit code 2 means it reported a data gap, so a script can tell the difference
```

---

## 5. Judging whether a number is physical

`check_physical_plausibility` is the tool that makes it a scientist rather than a
search box. It reports in three tiers, and keeping them apart is the point —
collapsing them is how a real finding gets thrown out for being surprising.

**Violation — the number is wrong.** Not surprising. Wrong.

- A relative permittivity below 1 would mean the material polarizes against the
  field.
- A band offset larger than the band gap puts the conduction band below the
  valence band.
- An optical permittivity above the static one: the static response contains
  everything the optical one does, plus the ionic part, so it cannot be smaller.
- `Ebd = 4e6`: that is V/cm entered as MV/cm. The message says so, because that is
  almost always the cause.

**Inconsistency — two values that must agree, and do not.** The one that earns its
keep:

> X-ray SLD and mass density are the same measurement for a fixed composition.
> `SLD = rₑ · ρ · N_A · (Z/A)`. HfO₂ at 9.1 g/cm³ *has* an SLD of 64.6e-6 Å⁻². If a
> fit reports 40, one of the two numbers is wrong.

This catches a real and common failure: XRR is nearly degenerate in density and
SLD, so a co-refinement that leaves both free will trade one against the other and
produce a fit that reproduces the curve beautifully while disagreeing with itself.
(It caught exactly that in this repo's own example data during development.)

**Heuristic — look again.** Domain expectations with real exceptions:

- High permittivity *and* a wide gap together. Both come from the same
  polarizability, so they trade off: SiO₂ is k≈3.9/Eg≈9, HfO₂ is k≈25/Eg≈5.7,
  TiO₂ is k≈80/Eg≈3.1. The top-right corner is nearly empty, and it is where a
  swapped pair of numbers lands.
- A conduction-band offset below ~1 eV on silicon — thermionic leakage over a
  barrier that low dominates regardless of thickness.
- An ALD growth-per-cycle above one monolayer. Self-limiting chemistry cannot do
  that; it is CVD-like behaviour, usually an unpurged precursor.

**A heuristic never overrides data.** It does not vote on whether something is
physical, and it is never grounds to exclude a value. A surprising number that
survives scrutiny is the most interesting kind of result there is, and FOM_PROOF
Sec. 2.3 does not list "a heuristic disliked it" among the reasons to drop one.

---

## 6. Where Bayesian optimization fits

The loop the platform runs:

```
suggest a recipe → grow it → measure it (ModalFit) → derive properties
    → score the FOM → feed the observation back → suggest again
```

The assistant reads that loop; it does not drive it. Three tools, and each answers
a question the raw tables do not:

**`lookup_bo_campaign`** — the search space, the instrument constraints, the
acquisition function, the objective. Read this *before* any result, because the
search space and the objective definition are what a result means. It also warns
when the FOM definition is unapproved: uniform placeholder weights with no named
owner means the campaign is optimising toward a policy choice nobody has made, and
its ranking is not yet a result.

**`lookup_bo_history`** — the best-so-far trajectory and, crucially,
`evaluations_since_best_improved`. "Is this working?" is a question about the
trend, and nothing else in the schema exposes it.

**`lookup_bo_suggestions`** — the pending proposals with `predicted_std`. Large
spread means it is exploring; small spread with small acquisition values means it
thinks it is done.

Two things the assistant is told to watch, because both look like success from the
inside:

- **A flat best-so-far is not convergence.** If the suggestions still carry large
  uncertainty and keep clustering on a bound, the optimum is probably *outside*
  the search space, not inside it. A campaign whose bounds clip the answer looks
  exactly like a converged one.
- **Infeasible observations are information.** They bound the feasible region. A
  campaign that is mostly infeasible has a constraint problem, not a search
  problem.

And one scale trap: the surrogate models **ln F**, not F (Eq. 30), because F is a
weighted geometric mean of terms in (0, 1] and its residuals are nothing like the
Gaussian a GP assumes. So a predicted mean of −0.7 against −1.4 is a *factor of
two* in F.

---

## 7. Knowledge cards: making it compound

The problem with plain retrieval: every answer is disposable. Ask the same question
twice and the model reasons from scratch, having learned nothing in between.

Cards are the fix, and the pattern is a **Dynamic Knowledge Repository** — do the
integration work when a source arrives, not when a question is asked. A card is a
markdown page with typed links:

```
concepts/ald-window-hfo2
  ├─ fed_by      → sources/kim-2024
  ├─ fed_by      → sources/puurunen-2005
  ├─ measured_by → findings/pilot07-thickness
  ├─ contradicts → concepts/ald-window-hfo2-alt
  │                  "0.98 vs 1.4 Å/cycle over the same 200-300 °C range"
  └─ depends_on  → concepts/self-limiting-growth
```

Typed edges rather than plain links, because a graph that only knows *that* two
pages are related cannot answer "what does this rest on?". And `contradicts` is the
one this platform most needs: an unresolved disagreement between two sources is a
finding, and a flat link would bury it. A `contradicts` link **requires a note**
saying which claims conflict — recording that two cards disagree while discarding
what they disagree about helps nobody.

### The review gate

This is the part that makes an LLM-written page safe to keep.

A card is where a language model's synthesis gets written down, and Sec. 15.2 says
a synthesis is not evidence. So:

| State | `citable` | Meaning |
|---|---|---|
| `proposed` | no | The assistant's draft. Read it, check it, correct it. |
| `reviewed` | yes | A named person checked it against resolved sources. |
| reviewed, then edited | **no** | The review no longer covers what the card says. |
| `superseded` | no | Replaced, kept for the audit trail. |

Review needs a named person and at least one *resolved* source — a free-text
"Kim 2024, Table 2" is not enough, because Sec. 2.2 wants provenance a reader can
follow. And editing a reviewed card makes the review stale automatically, so a card
cannot be approved and then quietly rewritten.

Proposed cards are still returned, clearly labelled. Hiding a draft helps nobody;
what it may not do is be built on.

```bash
# The assistant records what it worked out (opt-in per request)
curl -X POST localhost:8000/rag/chat -H 'content-type: application/json' \
  -d '{"question": "What is the ALD window for HfO2 on Si?", "allow_card_writes": true}'

# You review it
curl -s localhost:8000/cards?status=proposed | jq '.cards[].slug'
curl -X POST localhost:8000/cards/concepts/ald-window-hfo2/review \
     -H 'content-type: application/json' -d '{"reviewed_by": "Z. Woodel"}'

# Corpus health — the two numbers that matter
curl -s localhost:8000/cards/stats | jq '{awaiting_review, unresolved_contradictions}'
```

`awaiting_review` decides whether this is knowledge or just model output written
down. `unresolved_contradictions` is the most valuable number in the system: each
one is a disagreement between sources that somebody has to settle.

### What a card can never be

There is no code path from a card into `property_values`. Cards are a reading aid.
The only route from a measurement into the analysis tables is
`POST /modalfit/fits/{id}/promote`, which takes instrument-derived fits, refuses a
parameter that was held fixed or clamped, and requires a material identity a person
supplied.

---

## 8. Which model is doing this

**Local by default.** `RAG_LLM_PROVIDER=ollama` keeps every retrieved excerpt on
the machine, which is the right default when the corpus is unpublished CNMS work.

```bash
docker compose up -d db ollama
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

**Claude is available and opt-in.** The honest trade: the retrieval discipline this
platform needs — declining when the evidence is thin, carrying a parameter's full
context, refusing to average disagreeing sources, keeping the three plausibility
tiers apart — is instruction-following, and an 8B local model follows those rules
less reliably than a frontier one. The cost is that retrieved excerpts leave the
machine.

```bash
RAG_LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-opus-5
pip install -e '.[anthropic]'
```

Selecting it logs at WARNING and `/health/ready` says so under
`checks.assistant.warning`. Decide it per corpus, not once. Either way the
guardrails are identical — they sit above the provider seam, not inside it.

---

## 9. Start to finish

```bash
# 1. Bring it up
cnms-fom init-db && cnms-fom seed
cnms-fom serve --reload

# 2. Give it the literature
curl -X POST localhost:8000/rag/ingest/upload \
     -F 'files=@~/papers/kim-2024.pdf' -F 'technique=ald'

# 3. Give it your measurements
cnms-fom import-fits data/fits          # ModalFit exports

# 4. Ask it something it can only answer by combining the two
cnms-fom ask "XRR and SE disagree on the HfO2 thickness for PILOT-07 by 38%. \
Is either fit internally consistent, and what measurement would settle it?" \
  --sample-id HFO2-PILOT-07 --technique ald

# 5. Keep what it worked out
cnms-fom cards list --status proposed
cnms-fom cards review concepts/ald-window-hfo2 --reviewed-by "Z. Woodel"
```

---

## Further reading

- [RESEARCH_ASSISTANT.md](RESEARCH_ASSISTANT.md) — retrieval internals, the loop,
  conversations, providers
- [MODALFIT_INTEGRATION.md](MODALFIT_INTEGRATION.md) — co-refinements, cross-technique
  comparison, promotion gates
- [FOM_PROTOCOL.md](FOM_PROTOCOL.md) — every section and equation of `FOM_PROOF`
  mapped to the code
- [PILOT_WORKFLOW.md](PILOT_WORKFLOW.md) — the HfO₂-on-Si loop, and what is real
  versus heuristic in it
