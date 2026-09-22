# pySEA integration

## What is provisional, and what is not

**`pysea-canonical/0.1` is a contract we wrote, not pySEA's format.** It was
derived from the published abstracts of Walker, Pfeifer, Lupini, Hachtel,
Pantelides and Hoglund (*Microscopy & Microanalysis*, 2026). We have not seen
pySEA's container schema, its field names, or its API.

Nothing in `backend/cnms_fom/pysea/` should be described to the pySEA authors as
a format we require of them. Every field name is ours, chosen to hold the
information the protocol needs; the mapping onto theirs is the next piece of work,
and it needs a conversation and a sample container.

Three properties make that mapping survivable:

1. **Every stored row carries `contract_version`.** Rows written under a guessed
   mapping stay findable once the guess is corrected.
2. **Unknown fields are kept, not dropped.** A container's unmapped keys land
   under `extensions._unmapped`, because the fields we failed to anticipate are
   exactly the ones worth having when the real schema arrives.
3. **Major versions are refused rather than approximated.** A
   `pysea-canonical/1.x` envelope is not parsed by a 0.x build. Reading it would
   produce plausible rows from fields that may have been redefined.

What is *not* provisional is the discipline. Units are never inferred. A
simulation is MODELED. Material identity is supplied by a person. Those rules come
from FOM_PROOF and hold whatever pySEA's container turns out to look like.

## The seam

pySEA runs from instrument configuration to analysed signal. This platform runs
from a determined property to a figure of merit to the next recipe. The two meet
at one object: **a scalar with its measurement context attached.**

```
pySEA                                    CNMS Living FOM
─────                                    ───────────────
instrument configuration
  ray-optics digital twin
multislice + MLIP + MD
  TACAW vEELS prediction
FAIR signal storage
  calibration, provenance
        │
        └── derived scalar ──────────▶ validate
             + instrument state         promote (gated)
             + calibration              PropertyValue
             + DataFed locator          FOM score
                                        Bayesian optimizer
                                            │
            next recipe ◀───────────────────┘
```

## Fields we need to accept a value as MEASURED

Each entry names what breaks without it. Anything absent from this list is
welcome and unused.

1. **`sample.sample_id`** — ties the acquisition to a specimen. Without it nothing
   can be compared against a second determination of the same film.
2. **`record_id`** — a stable acquisition identifier, carried verbatim. It is how
   a promoted number is traced back to pySEA and to DataFed after the file moves.
3. **`record_kind`** — `experimental`, `simulation` or `hybrid`, stated in a
   field. A multislice spectrum and a measured one are both arrays of counts
   against energy loss; this field is the only thing that separates them, and it
   is never inferred. A `hybrid` record is treated as a simulation, because the
   safe reading of a mixed envelope is the weaker one.
4. **`instrument.instrument_id` and `instrument.technique`** — the measurement
   context cannot be reproduced without knowing which column produced it.
5. **A twin-reconstructed `instrument_state`**, carrying:
   - `twin_reconstructed: true` as an explicit boolean. A truthy string does not
     count. Treating a nominal state as reconstructed files a wrong number as a
     measurement; the reverse is a refusal that better metadata overturns.
   - `beam_energy_kev` — inelastic cross-sections and the relativistic correction
     both depend on it.
   - `collection_semi_angle_mrad` — the solid angle scattered into is part of the
     quantity's identity. Two acquisitions at different collection angles measure
     different numbers and both are correct.
   - `convergence_semi_angle_mrad` — needed to interpret the probe.
   - `lens_strength_source` — `twin`, `nominal`, `user-entered` or `unknown`.
     Only `twin` supports a quantitative claim.
6. **Calibration identity and validity window** — `calibration_id` plus the dates
   it applied. Validity is checked against the **acquisition date**, not the
   import date: a calibration that expired last month was still valid when the
   data was taken.
7. **`dispersion_ev_per_channel`**, on the state or on a referenced calibration.
   Without it the energy axis is detector channels and a peak position read off it
   is not in eV.
8. **Axes and units on every signal.** One axis per dimension, in dimension order,
   with sizes matching the shape. Energy, momentum, spatial and time axes must
   carry units. An unlabelled energy axis is not an eV axis with the label
   missing, and this importer will not supply one. A caller may pass `unit_hints`
   at import; the axis then records `units_source: "caller"`, because a unit typed
   by an operator is weaker evidence than one the instrument wrote.
9. **For each derived scalar**: `name`, `property_key` (a key in
   `descriptors/registry.py`), `value`, `units`, `uncertainty`, `derivation`
   (`measured` / `fitted` / `calculated` / `simulated`), `method`, and
   `source_signal_ids` naming signals that travel in the same record.
   - **Uncertainty is required for promotion.** A number without one cannot be
     cross-checked against a second determination, and cross-platform comparison
     is what this integration exists for.
   - **A scalar with no registry key is refused**, not invented. A property this
     platform has no specification for has no required context, no direction, and
     no place in a FOM.
10. **`datafed.record_id`** — the promoted row carries a pointer back to the
    managed copy of the data.

## What we hand back

- **A `PropertyValue`** with its full measurement context, provenance tier, and a
  `source_locator` of `pysea_record:{id}` plus the DataFed record id.
- **A FOM score**, or `not_scored` with the reason. A value whose required context
  is incomplete does not silently score.
- **A provenance tier**: MEASURED for an experimental record whose scalar was
  measured or fitted; MODELED for everything else, including a simulation that
  agrees with the measurement exactly.
- **A cross-platform verdict**: every determination of the quantity, each with its
  own method and tier, and one of `no_determination`, `single_determination`,
  `consistent`, `disagreement`. Never a mean.
- **A next recipe**, when the value feeds a Bayesian optimisation campaign, with
  the bounds it was suggested under.

## The promotion gate

`instrument_state.is_quantitative` returns True only when all three hold:

```python
state.twin_reconstructed                      # the column, not the setpoint
and state.collection_semi_angle_mrad is not None
and state.dispersion_ev_per_channel is not None
```

It judges whether the measurement context was recorded, not whether the science is
good. A record failing it is still stored, still readable, and still comparable;
what it cannot do is put a number in `property_values` as MEASURED.

Promotion additionally requires a valid record, an explicitly supplied
`material_id`, and every field the registry names in `required_context` for that
property key.

## Refusals you should expect

These are quoted verbatim from `pysea/promote.py`, and each names something to go
and fetch rather than something to argue with.

- *"instrument state was not twin-reconstructed; values rest on nominal lens
  strengths rather than the column's reconstructed optical configuration"*
- *"collection semi-angle is missing; the solid angle scattered into is part of
  the quantity's identity and a cross-section cannot be identified without it"*
- *"no energy dispersion is recorded; the energy axis is detector channels, so a
  position read off it is not in eV"*
- *"the scalar reports no uncertainty; it could not be cross-checked against a
  second determination, which is what this integration exists to do"*
- *"the scalar maps onto no registry key, so the platform has no specification for
  it: no required context, no direction, no place in a FOM"*

## Data volume

**No bulk arrays reach Postgres.** A 4D-STEM scan is gigabytes. What is stored is
shape, axes, calibration and a locator; the array stays with pySEA and DataFed.
`contract.strip_bulk` removes array payloads before hashing and before storage, so
two exports differing only in their arrays are recognised as one acquisition.

## DataFed required-context schema

A JSON-Schema fragment for the fields a record must carry before a value derived
from it can be promoted. Offered as a starting point for discussion with the pySEA
team, not as a published schema.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CNMS Living FOM: required context for a MEASURED promotion",
  "type": "object",
  "required": ["sample_id", "record_id", "record_kind", "instrument", "instrument_state"],
  "properties": {
    "sample_id": { "type": "string", "minLength": 1 },
    "record_id": { "type": "string", "minLength": 1 },
    "record_kind": { "enum": ["experimental", "simulation", "hybrid"] },
    "instrument": {
      "type": "object",
      "required": ["instrument_id", "technique"],
      "properties": {
        "instrument_id": { "type": "string" },
        "technique": { "type": "string" }
      }
    },
    "instrument_state": {
      "type": "object",
      "required": [
        "twin_reconstructed",
        "beam_energy_kev",
        "collection_semi_angle_mrad",
        "lens_strength_source"
      ],
      "properties": {
        "twin_reconstructed": { "type": "boolean" },
        "beam_energy_kev": { "type": "number", "exclusiveMinimum": 0 },
        "collection_semi_angle_mrad": { "type": "number", "exclusiveMinimum": 0 },
        "convergence_semi_angle_mrad": { "type": "number", "exclusiveMinimum": 0 },
        "dispersion_ev_per_channel": { "type": "number", "exclusiveMinimum": 0 },
        "lens_strength_source": {
          "enum": ["twin", "nominal", "user-entered", "unknown"]
        },
        "calibration_id": { "type": "string" },
        "calibration_validity_window": {
          "type": "array",
          "items": { "type": "string", "format": "date-time" },
          "minItems": 2,
          "maxItems": 2
        }
      }
    }
  }
}
```

## Questions for the pySEA team

Each of these is a place where we guessed and would rather not have.

1. What does a pySEA container look like on disk, and what are its top-level
   sections? Our `contract.KNOWN_SECTIONS` is an invention.
2. How does pySEA record that a state came from the ray-optics twin rather than
   from the control software's setpoints? This is the single most load-bearing
   field in the integration and we have no idea what it is called.
3. Are calibrations first-class records with their own identifiers and validity
   windows, and can a signal reference one?
4. How is a momentum axis labelled and what units does it carry? We assumed
   `1/angstrom`.
5. Does TACAW output carry its interatomic potential, MD ensemble and supercell in
   the container, or are those in a sibling record?
6. Is there a DataFed record id on the container at export time, or is deposition
   a later step?
7. Does pySEA compute derived scalars with uncertainties, and if so, what is the
   basis for them?

## Running it

```bash
# Validate, import, plan, promote and compare, end to end, offline.
python scripts/pysea_demo.py
```

The demo uses an in-memory database and the fixtures under
`backend/tests/fixtures/pysea/`. It needs no network, no pySEA install and no
DataFed account.
