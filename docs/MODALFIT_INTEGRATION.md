# ModalFit integration

[ModalFit](https://github.com/agauer/modalfit) fits one shared slab model against
up to five characterization techniques at once — SE, SPR, QCM, XRR, NR —
delegating each forward model to a purpose-built package (`refellips`, `PyMoosh`,
`refnx`) rather than reimplementing the physics.

What this integration adds is not fitting. It is the observation that **a
co-refinement is the only place this platform measures the same quantity twice by
independent physics.** An XRR thickness and an SE thickness for one film share no
forward model, no instrument, and no systematic error. When they agree the number
is worth trusting. When they disagree, the disagreement is the finding — and it is
the only place a single measurement here can be checked against anything but
itself.

---

## 1. The seam: exported JSON, not the HTTP API

The integration targets ModalFit's **exported model JSON** (its README Sec. 4),
not its Flask routes. Three reasons, in order of weight:

1. The export format is documented and stable; the HTTP layer was not part of the
   source we had, and a client written against an API nobody has seen produces
   code that looks integrated and is not.
2. ModalFit's own `session_store` is explicitly in-process memory keyed by a
   cookie, and its README names that as the thing to fix before any shared
   deployment. Building against it would inherit that.
3. An export is a file with a content hash. Import is idempotent, re-runnable, and
   works the same whether the JSON arrived by hand, from `data/fits`, or out of
   DataFed.

DataFed pulling stays ModalFit's job — it already scans a sample's experiment tree
by `{"instrument": ...}` metadata and unwraps NXS containers. Re-implementing that
here would be two copies of one integration. The `datafed_record_id` columns on
`fit_records` and `fit_datasets` are there to carry the provenance across.

---

## 2. Units, stated because getting this wrong is silent

The slab-model format does not record a length unit.

ModalFit's physics backends are `refnx`/`refellips`, which work in **angstroms**.
Its bundled substrate library is written in **nanometres** — `"thickness": 2.0` is
the 2 nm native oxide, and `"thickness": 50.0` is the 50 nm gold film.

A parser that guessed would be wrong by a factor of ten roughly half the time. So
the unit is an argument (`--length-units`, `length_units`), it defaults to the
angstrom convention of the fitting engine, a file declaring its own
`length_units` overrides it, and an unrecognised unit is refused rather than
assumed. Values and bounds are scaled together — scaling a value without its
bounds turns a converged parameter into one that looks clamped.

Everything is stored in angstroms (`thickness_ang`, `roughness_ang`), so the
column name carries the convention.

---

## 3. What the importer refuses to infer

**The technique list.** ModalFit populates every slab-model block a technique
*could* read, whether or not data was ever loaded for it. So a stack having an
`xray` block does not mean XRR was ever fitted, and the technique list comes from
the export's fit metadata or from the caller — never from which blocks are
populated. A fit with no technique is refused outright: a refinement with no
technique is not a measurement of anything.

**A chi-squared.** Absent stays absent. Zero would read as a perfect fit.

**A material identity.** A slab-model layer carries a formula and nothing else,
and FOM_PROOF Eq. (3) makes identity composition + polymorph + specimen form.
Inventing "amorphous" to fill the polymorph column is exactly the fabrication the
protocol forbids, so promotion requires a `material_id` someone supplied.

---

## 4. What the importer warns about

The `warnings` list on an import is the part worth reading.

| Warning | Why it matters |
|---|---|
| A claimed technique has no matching slab-model block | The fit describes something that did not happen. Either the technique list is wrong or the stack is incomplete. |
| A parameter finished on its fit bound | Clamped, not converged. The optimizer wanted to go further and was stopped by a number someone typed. |
| Optical constants look like the bundled placeholders | ModalFit's README says the Si/Au/Cr/Ti n,k tables are not digitized literature values. SE- and SPR-derived numbers resting on them are illustrative. |
| No chi-squared recorded | Nothing distinguishes the fit from an unrefined starting model. |

Two known limitations are recorded as columns rather than assumed away, because
both bias the fitted values and a downstream consumer that does not know cannot
correct for them:

- `resolution_smearing_applied` — ModalFit builds its `refnx` models with
  `dq=0.0`, so XRR/NR fits carry no angular-resolution smearing and show sharper
  Kiessig fringe contrast than a real finite-divergence instrument measures.
- `roughness_applied_to_spr` — the SPR/PyMoosh path applies no layer roughness at
  all, so roughness set on a layer has no effect on an SPR prediction.

---

## 5. Schema

```
fit_records ──┬── fit_layers     one slab each: role, label, fitted parameters,
              │                  free_parameters, bounds, uncertainties
              └── fit_datasets   per technique: file, point count, fitted window
                                 with units, chi², weight
```

`fit_layers.free_parameters` is the load-bearing field. A thickness held fixed
during refinement is an **input** to the fit, and reporting it as a measured
thickness would be fabrication dressed up as instrument data. Everything
downstream — comparison, promotion, the assistant's prose — branches on it.

Parameters live in a JSON dict rather than in columns because the set is
technique-dependent: a QCM-only fit has a shear modulus and no SLD, an XRR/NR fit
has SLD and no dispersion model. Columns for the union would be mostly NULL and
would still need extending for the next technique.

`fit_datasets` exists because a chi-squared is otherwise unfalsifiable: 1.8 over
40 points in a narrow Q-range and 1.8 over 400 points across two decades are not
the same claim.

---

## 6. Cross-technique comparison

`POST /modalfit/compare` returns every determination of one parameter, with its
caveats, plus the spread and a verdict. **It never averages.** FOM_PROOF Sec. 2.1
forbids merging records without a declared aggregation rule, and two techniques
40% apart on a thickness do not have a mean worth reporting.

Three outcomes, and the third is the one usually collapsed by mistake:

- **consistent** — the determinations agree.
- **disagreement** — they do not. One forward model is describing something the
  other is not: a wrong SLD, an unmodelled interlayer, a fit that landed in a
  different minimum.
- **silence** — the technique *cannot determine this parameter*. A QCM fit carries
  a roughness value through, but QCM never constrained it, so it is not a rival
  claim to an XRR roughness. Treating silence as disagreement turns a non-result
  into a finding.

How disagreement is judged depends on what the fits reported. With two
uncertainties, it is the spread against the combined 1σ. Without them — and only
`DREAM (emcee)` produces a posterior; the four scipy minimizers do not — it falls
back to a 10% relative spread, and says in the summary that this is a flag for a
human rather than a significance test. Claiming a test that was not performed is
worse than having no test.

Ambiguity is refused rather than resolved: a comparison on a two-film stack with
no `layer_label` skips those fits instead of taking the first film, because
attributing one layer's thickness to another is not recoverable afterwards.

---

## 7. Promotion into the analysis tables

`POST /modalfit/fits/{id}/promote` writes fitted values into `property_values` /
`descriptor_values` with the **MEASURED** tier.

That is allowed because a fitted SLD is instrument-derived: a photon or neutron
bounced off the film, a forward model with no interpretive freedom reproduced the
curve, and the number came out. It is a different path from retrieval —
`rag_backend.chains.assert_not_property_ingestion` exists to keep a language
model's summary of a paper out of these tables, and the two live in separate
packages so nothing drifts into treating them alike.

| Fitted | Becomes | Needs |
|---|---|---|
| `xray.sld_real` / `sld_imag` | `sld_xray` / `sld_xray_imag` | XRR among the techniques |
| `neutron.sld_real` / `sld_imag` | `sld_neutron` / `sld_neutron_imag` | NR among the techniques |
| `molecular`/`viscoelastic.density` | descriptor `rho` | XRR, NR, or QCM |

Thickness and roughness are deliberately **not** promoted. They are not properties
of a material — they are *context* for one, and Table 1 gives `PropertyValue` a
`thickness_nm` column for exactly that. A 103 Å film and a 1030 Å film of the same
oxide are the same material measured under different conditions, so thickness
rides along on each promoted row instead of becoming a row of its own. Roughness
lands in `interface` as a Nevot-Croce note.

Every refusal, and why none is pedantry:

| Refused | Because |
|---|---|
| a parameter held fixed | It is an input to the fit. Promoting it reports the operator's starting guess as a measurement. |
| a parameter on its fit bound | Clamped, not converged. |
| a fit with no chi-squared | Nothing separates it from an unrefined model. |
| a technique that cannot determine it | An SLD carried through a QCM-only fit was never constrained by anything. |
| an ambient or substrate layer | Promoting the wafer's SLD would file it against the film's material. |
| a material identity we were not given | Sec. 2.1. |
| an ambiguous film layer | A misattributed SLD is not recoverable. |

`GET /modalfit/fits/{id}/promotion-plan` is a dry run with no side effects, and
`promote` itself defaults to `dry_run: true`. Run it first every time: the refusal
list is the actionable half — it names the parameters to free, the bounds to widen,
and the fits to re-run.

The `method` string on a promoted row is built from what the fit actually
recorded, because the registry requires it and says why: X-ray SLD is
energy-dependent and neutron SLD is isotope-dependent, so "XRR" alone does not
identify the quantity. An unrecorded beam energy is named as unrecorded rather
than defaulted to Cu Kα because Cu Kα is the common case.

---

## 8. Usage

```bash
# One export, or a directory of them. Idempotent by content hash.
cnms-fom import-fits data/fits/hfo2_xrr_fitted.json --technique XRR
cnms-fom import-fits data/fits --length-units nm

# Cross-technique agreement.
cnms-fom compare-fits HFO2-PILOT-07 --parameter thickness
cnms-fom compare-fits HFO2-PILOT-07 --parameter all
```

```
POST   /modalfit/import
GET    /modalfit/samples
GET    /modalfit/samples/{sample_id}/fits
GET    /modalfit/samples/{sample_id}/disagreements
GET    /modalfit/fits/{id}
POST   /modalfit/compare
GET    /modalfit/fits/{id}/promotion-plan
POST   /modalfit/fits/{id}/promote
```

The research assistant reaches all of this through its own tools — see
[RESEARCH_ASSISTANT.md](RESEARCH_ASSISTANT.md). `modalfit.compare.describe_fit`
renders a fit as prose with the caveats *in the body* rather than in a footnote: a
model told "thickness 103.4 Å" and not told the parameter was held fixed will
report a measurement that was never made.
