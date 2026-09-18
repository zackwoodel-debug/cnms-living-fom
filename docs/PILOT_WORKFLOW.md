# Pilot workflow — HfO₂ on Si

One complete loop, end to end, meant as the template a real CNMS study is cut
from.

```
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   ▼                                                             │
 design ──▶ suggest ──▶ export ──▶ simulate ──▶ score ──▶ observe ┘
 (space)    (BoTorch)   (stack)    (XRR +       (FOM       (BO
                        JSON+CSV)   properties)  engine)    campaign)
                            │
                            └──▶  or: run it on a real tool, and
                                  POST the measurements instead
```

The two ways round the loop are the **same pipeline with one stage swapped**:

| | `iterate` | `ingest_experiment` |
|---|---|---|
| Properties from | the forward model | a real measurement |
| Provenance tier | `modeled` | `measured` |
| Resulting score | `illustrative` | `scored` |
| Objective | ln F, logged | ln F, logged |

Going from pilot to production means calling the second endpoint instead of the
first. Nothing else changes — which is the point of the exercise.

---

## Run it locally

```bash
cnms-fom init-db && cnms-fom seed
cnms-fom serve --reload
```

```bash
# 1. Create the campaign.
curl -sX POST localhost:8000/pilot/hfo2_logic_run \
     -H 'content-type: application/json' \
     -d '{"random_seed": 20260823}'

# 2. Suggest two recipes, simulate them, log the observations.
curl -sX POST localhost:8000/pilot/hfo2_logic_run/1/iterate \
     -H 'content-type: application/json' \
     -d '{"q": 2, "persist": true, "seed": 1}'

# 3. Export a stack for an external simulator.
curl -s 'localhost:8000/pilot/hfo2_logic_run/stack?thickness_ang=150&roughness_ang=3'

# 4. Close the loop with real data.
curl -sX POST localhost:8000/pilot/hfo2_logic_run/1/ingest_experiment \
     -H 'content-type: application/json' \
     -d @experiment.json
```

Interactive docs at <http://localhost:8000/docs#/pilot>.

**Expected behaviour.** The first four or five calls return a Sobol design, not
GP suggestions — a GP fitted on three points is reporting its prior, and the
response says so in `notes`. Past that threshold it switches to `qLogEI`, which
needs the `bo` extra (`pip install -e '.[bo]'`); without it the endpoint returns
a clear 501 rather than failing obscurely.

Every score from `iterate` comes back `illustrative`. That is correct and not a
configuration problem — see *Provenance* below.

---

## The search space

| Parameter | Range | Why it is in the space |
|---|---|---|
| `thickness_ang` | 50–400 Å | Sets capacitance directly, and how much the interfacial layer dominates |
| `roughness_ang` | 1–10 Å | Field enhancement at asperities derates breakdown; hurts thin films most |
| `dopant_fraction` | 0–0.30 | Trades permittivity against bandgap — the reason there is an interior optimum |

Objective is **ln F**, maximised. The FOM is a weighted geometric mean, so its
log is the additive quantity (Eq. 30) and the scale a GP should model on; `ln`
is monotone, so the argmax is unchanged. See `bo_engine/surrogate.py`.

---

## What each stage does

### Suggest — `bo_engine`

Standard ask/tell. The search space is intersected with the instrument envelope
*before* optimisation, so the acquisition budget is never spent on recipes the
tool cannot run.

### Export — `pilot/stack.py`

Builds `air / HfO₂ / SiO₂ / Si` and emits it as JSON plus an n,k CSV.

The **native SiO₂ interlayer is always present**, and leaving it out would make
the pilot optimistic in a way that matters: it is what caps achievable EOT.

Scattering length densities come from the database when the import supplied
them, and are otherwise computed from composition and density:

```
SLD = r_e · (ρ N_A / M) · Σ Z
```

Every layer records which route was used. The calculation neglects anomalous
dispersion, which is right away from an absorption edge and a reason to prefer a
measured value — hafnium's L3 edge at 9.56 keV sits above Cu Kα at 8.05 keV, so
f′ is a few percent here.

### Simulate — `pilot/simulate.py`

**Real physics.** Specular X-ray reflectivity by Parratt recursion with
Névot–Croce roughness:

```
k_j = sqrt((q/2)² − 4π(ρ_j − ρ_ambient))
r_j = (k_j − k_{j+1})/(k_j + k_{j+1}) · exp(−2 k_j k_{j+1} σ_j²)
R_j = (r_j + R_{j+1} e^{2i k_{j+1} d_{j+1}}) / (1 + r_j R_{j+1} e^{2i k_{j+1} d_{j+1}})
```

Verified in the test suite: silicon's critical edge lands at q_c = 0.0315 Å⁻¹
(literature 0.0317), reflectivity is unity below it and decays above, roughness
damps the high-q tail, and the Kiessig fringes encode the thickness that went in.

Implemented here rather than pulled in as a dependency so CI is deterministic
and the pilot runs with no extra install. `refnx` is used instead when available
and `prefer_refnx` is set — the path to take when fitting real data.

### Derive properties — `pilot/properties.py`

**This is the layer to read sceptically**, and it separates two kinds of relation:

**Derived physics** — exact given its inputs, no calibration:

```
t_total / k_eff = t_film / k_film + t_IL / k_IL        (series capacitance)
EOT = t_film · (3.9 / k_film) + t_IL
```

This is why a thinner high-k film does not keep improving EOT: the SiO₂
interlayer sets a floor.

**Heuristics** — stated functional forms with plausible coefficients, chosen so
the pilot has a non-trivial landscape. **Not fits to any dataset.** Each is
marked `HEURISTIC` and every coefficient lives in one `PilotConstants` table, so
replacing them with a fit to CNMS data is a single edit:

| Relation | Form | Stands in for |
|---|---|---|
| k(dopant) | linear mixing + a Gaussian bump near x ≈ 0.06 | Tetragonal-phase stabilisation before the low-k dopant oxide dominates |
| Eg, ΔEc(dopant) | linear mixing toward Al₂O₃ | Wider gap, larger offset — the reliability half of the trade |
| Ebd | ∝ Eg², derated by 1/(1 + βσ/t) | Gap scaling; field enhancement at asperities |
| tan δ | linear in dopant and roughness | Disorder-driven loss |
| κ_th | constant | No recipe dependence modelled |

Sanity check on the resulting landscape:

| t (Å) | σ (Å) | x | k_eff | EOT (Å) | E_bd | tan δ |
|---|---|---|---|---|---|---|
| 50 | 2 | 0.00 | 14.19 | 16.49 | 3.23 | 2.3e−3 |
| 400 | 2 | 0.00 | 25.83 | 61.90 | 3.88 | 2.3e−3 |
| 100 | 2 | 0.06 | 20.96 | 20.47 | 3.63 | 2.8e−3 |
| 100 | 8 | 0.00 | 18.67 | 22.97 | 2.70 | 3.3e−3 |
| 100 | 2 | 0.30 | 14.64 | 29.31 | 3.88 | 4.7e−3 |

Light doping improves both k_eff and EOT; heavy doping trades k away for
breakdown; roughness costs breakdown and loss.

### Score — `fom_engine`

Unchanged from the rest of the platform. Weighted geometric mean, missing inputs
give `not_scored`, modeled inputs give `illustrative`.

### Observe — `bo_engine`

`ln F` becomes the objective. An unscoreable run is logged **infeasible**, not
given a bad score: a failed growth carries no information about the objective
surface there, and pretending otherwise teaches the GP something untrue.

---

## Provenance, and why every simulated score is ILLUSTRATIVE

Three rules, enforced structurally rather than by convention:

1. **Simulated properties are stored `modeled`**, so `score_material` returns
   `ScoreStatus.ILLUSTRATIVE` and the `ck_score_modeled_is_illustrative`
   database constraint rejects anything else.
2. **The recipe is the measurement context, not a new material.** Eq. (3) puts
   processing in the context; composition, polymorph and specimen form are
   identical across the whole search space. So every recipe shares one
   `Material` and differs by `processing_route`, which is part of the context
   digest — two recipes are two contexts, and the uniqueness constraint holds.
3. **Each simulated recipe gets its own `AnalysisRun`**, because it genuinely is
   a separate analysis over different inputs.

A simulated loop exercises the machinery. It does not produce a ranking, and
FOM_PROOF Sec. 2.3 requires the two to stay separable.

---

## Stubs versus real physics

| Component | Status | To make it real |
|---|---|---|
| Parratt XRR forward model | **Real physics** | Swap in `refnx` to *fit* measured curves rather than simulate |
| Series-capacitance k_eff, EOT | **Real physics** | Nothing — exact given its inputs |
| SLD from composition | **Real calculation** | Add f′/f″ for work near an absorption edge |
| k(dopant), Eg(dopant), E_bd, tan δ | **Heuristic** | Fit `PilotConstants` to CNMS measurements |
| κ_th | **Constant** | Measure, or model against composition |
| Interfacial layer thickness | **Fixed at 10 Å** | Fit from XRR, or measure by TEM |
| Instrument envelope | **Placeholder** | CNMS instrument registry — it has no thickness/roughness limits yet |
| Proposal / user-agreement binding | **Not implemented** | `CNMS_PROPOSAL_API_URL`; embargo terms govern what may be published |

Every one is marked `TODO(CNMS)` or `HEURISTIC` in the source:

```bash
grep -rn "HEURISTIC\|TODO(CNMS)" backend/cnms_fom/pilot/
```

---

## How this maps to the protocol

| Loop stage | FOM_PROOF |
|---|---|
| Search space | Processing context of Eq. (3) |
| Export / simulate | Produces P, not S — the descriptors are unchanged across the space |
| Score | Sec. 6.2, Eq. (29)–(30) |
| Modeled quarantine | Sec. 2.3 |
| Required context on ingest | Sec. 16, items 4–9 |
| `not_scored` on incomplete input | Sec. 6.2 score-completeness rule |
| ln F as objective | Eq. (30) |

Note what the pilot does **not** exercise: the mediated effect M = BΓ (Sec. 10).
Every recipe here shares one structure, so S does not vary and there is no
structure→property relationship to estimate. Mediation needs a population of
different materials — `POST /fom/mediate`, and the ingested materials database.
Recipe optimisation and mechanistic attribution are different questions, and the
pilot answers only the first.

---

## CI

`.github/workflows/ci.yml`, three jobs:

* **test** — lint plus the full suite on SQLite. No services, no heavy extras,
  because `fom_engine` depends only on numpy/scipy.
* **postgres** — the same suite against Postgres + pgvector, plus a full
  migration up → down → up and a check that the migrated schema matches the ORM
  metadata. SQLite tolerates things Postgres does not, so this is where a
  production-only schema bug surfaces.
* **pilot** — the loop end to end through the CLI and API: migrate, seed, create
  a campaign, iterate twice, ingest an experiment, and assert the campaign state
  moved.

All seeded (`RANDOM_SEED=20260823`, `PYTHONHASHSEED=0`), so a green run today
means a green run tomorrow.

The compliance check runs with `|| true` on purpose: it is *expected* to report
the deliberate placeholders. It must run cleanly, not pass — passing would mean
somebody quietly filled in the decisions it exists to surface.
