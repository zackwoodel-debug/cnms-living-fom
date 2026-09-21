"""The fixed benchmark: a corpus, and the questions whose answers we know.

Self-contained and offline.  The corpus is defined here as text, seeded into a
throwaway database by the runner, and never modified by a run — so a score
compares policies rather than corpora, and two runs a month apart are comparable.

The documents are written to make specific failures detectable:

* Two of them **disagree** on the same quantity under different reactor conditions,
  so a pipeline that averages them, or reports one without its reactor, scores badly
  on a case built for exactly that.
* One states a number in a **table-like line** rather than a sentence, because
  process parameters live in tables and a retriever tuned only on prose buries them.
* Two materials with **similar names** (HfO2 and HfSiOx) appear, so a question about
  one can be answered from the other and be wrong in a way that looks right.
* One paper reports a value with **no measurement context**, which must be recorded
  as incomparable rather than compared.
* Several questions have **no answer in the corpus at all**, and abstaining is the
  only correct response.

Every number here is invented. The banner in each document says so, and the seeded
documents carry a ``SYNTHETIC`` title prefix so they cannot be mistaken for
literature if they ever escape a test database.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cnms_fom.db.enums import SynthesisTechnique


@dataclass(frozen=True)
class FixturePage:
    """One page of a fixture document."""

    page: int
    text: str


@dataclass(frozen=True)
class FixtureDocument:
    """A document the benchmark seeds. ``key`` is how cases refer to it."""

    key: str
    title: str
    technique: SynthesisTechnique
    pages: tuple[FixturePage, ...]
    doi: str | None = None

    @property
    def content_sha256(self) -> str:
        blob = "\n".join(page.text for page in self.pages)
        return hashlib.sha256(f"{self.key}|{blob}".encode()).hexdigest()


@dataclass(frozen=True)
class ExpectedClaim:
    """A value a correct extraction finds, and how close counts."""

    field_name: str
    value: float | None = None
    units: str | None = None
    #  Relative tolerance. 0 means exact.
    tolerance: float = 0.02
    #  Context keys the source does state, which an extraction should carry.
    required_context: tuple[str, ...] = ()
    #  True when the source states the value without the context its field needs, so
    #  a correct extraction records it as incomparable.
    expect_incomparable: bool = False


@dataclass(frozen=True)
class ForbiddenClaim:
    """A claim shape that must NOT appear in the brief.

    The suite could only ever assert what *should* be found, so the §10d fabrications —
    four purge times filed as growth per cycle, a material read as a growth rate — could
    not become regression cases. A capability the benchmark cannot see is one a future
    change can silently remove.

    ``units_dimension`` reuses ``contracts.classify_unit`` rather than matching unit
    spellings, so "6 s" and "6 seconds" are one rule.
    """

    field_name: str
    #  Any claim under this field whose units are of this physical dimension.
    units_dimension: str | None = None
    #  Any claim under this field carrying a number at all.
    any_numeric_value: bool = False
    why: str = ""


@dataclass(frozen=True)
class BenchmarkCase:
    """One question with a known-correct outcome."""

    case_id: str
    question: str
    #  What the case is testing. Used to report scores per category, because an
    #  aggregate hides which capability a policy change traded away.
    category: str
    #  Document keys a correct retrieval surfaces, in no particular order.
    expected_documents: tuple[str, ...] = ()
    #  (document key, page) pairs a correct retrieval surfaces.
    expected_pages: tuple[tuple[str, int], ...] = ()
    expected_claims: tuple[ExpectedClaim, ...] = ()
    #  Fields on which the corpus contains a genuine disagreement.
    expected_contradiction_fields: tuple[str, ...] = ()
    #  True when the corpus does not answer the question and abstaining is correct.
    should_abstain: bool = False
    techniques: tuple[str, ...] | None = None
    notes: str = ""
    #  Documents that look relevant and are not. Retrieving one is not an error;
    #  extracting a claim from it and presenting it as the answer is.
    distractor_documents: tuple[str, ...] = field(default_factory=tuple)
    #  Claim shapes whose presence is a failure, however good the rest of the brief is.
    forbidden_claims: tuple[ForbiddenClaim, ...] = field(default_factory=tuple)
    #  Fields on which the corpus does NOT disagree, so reporting a contradiction would
    #  be manufacturing one. Distinct from simply omitting the field from
    #  ``expected_contradiction_fields``, which asserts nothing either way.
    forbid_contradiction_fields: tuple[str, ...] = field(default_factory=tuple)


BANNER = (
    "SYNTHETIC BENCHMARK DOCUMENT. Invented numbers, written to exercise a retrieval "
    "pipeline. Not a publication and not citable."
)


CORPUS: tuple[FixtureDocument, ...] = (
    FixtureDocument(
        key="hotwall",
        title="SYNTHETIC ALD of HfO2 on Si(100) from TDMAH and water (hot-wall)",
        technique=SynthesisTechnique.ALD,
        doi="10.0000/synth-hotwall",
        pages=(
            FixturePage(1, (
                f"{BANNER} Films were deposited in a Beneq TFS-200 hot-wall reactor on Si(100) "
                "after an HF-last clean. The hafnium precursor was tetrakis(dimethylamido)hafnium "
                "(TDMAH) held at 75 degC, with deionised water vapour as the oxidant, nitrogen "
                "purge, and a chamber pressure of 1.5 Torr."
            )),
            FixturePage(2, (
                "Between 200 and 300 degC the growth per cycle was constant at 0.98 angstrom per "
                "cycle, which we take as the ALD window for this precursor combination. Below 200 "
                "degC the growth per cycle rose to 1.3 angstrom per cycle, consistent with "
                "precursor condensation. Above 320 degC it fell to 0.71 angstrom per cycle as "
                "TDMAH begins to decompose thermally."
            )),
            FixturePage(3, (
                "X-ray reflectometry at Cu K-alpha gave a mass density of 9.1 g/cm3 for films "
                "grown at 250 degC. Capacitance-voltage measurements on Pt/HfO2/Si capacitors at "
                "10 kHz and 300 K gave a relative permittivity of 18.5 for the as-deposited "
                "amorphous films."
            )),
        ),
    ),
    FixtureDocument(
        key="crossflow",
        title="SYNTHETIC HfO2 ALD in a cross-flow reactor: a re-examination",
        technique=SynthesisTechnique.ALD,
        doi="10.0000/synth-crossflow",
        pages=(
            FixturePage(1, (
                f"{BANNER} Depositions were performed in a custom cross-flow reactor at 0.3 Torr "
                "on Si(100) with the native oxide left intact. TDMAH was held at 80 degC and the "
                "oxidant was deionised water vapour."
            )),
            FixturePage(2, (
                "Over the range 200 to 300 degC the growth per cycle saturated at 1.42 angstrom "
                "per cycle, substantially above the value near 1 angstrom per cycle reported for "
                "hot-wall reactors. We attribute the difference to the reactor geometry and to the "
                "native oxide retained on our substrates."
            )),
        ),
    ),
    FixtureDocument(
        key="tabular",
        title="SYNTHETIC PLD of SrTiO3: deposition parameter survey",
        technique=SynthesisTechnique.PLD,
        doi="10.0000/synth-pld",
        pages=(
            FixturePage(1, (
                f"{BANNER} Table 2. Deposition conditions and resulting film properties for "
                "pulsed laser deposition of SrTiO3 on MgO(001)."
            )),
            #  Deliberately tabular: a retriever tuned on prose ranks this poorly,
            #  and the parameter it holds appears nowhere else in the corpus.
            FixturePage(2, (
                "Table 2 (continued). substrate_temperature 700 degC | oxygen_pressure 100 mTorr "
                "| laser_fluence 2.0 J/cm2 | repetition_rate 5 Hz | film_thickness 45 nm | "
                "out_of_plane_lattice_parameter 3.905 angstrom | surface_roughness 0.4 nm"
            )),
        ),
    ),
    FixtureDocument(
        key="hfsiox",
        title="SYNTHETIC ALD of HfSiOx alloy dielectrics",
        technique=SynthesisTechnique.ALD,
        doi="10.0000/synth-hfsiox",
        pages=(
            FixturePage(1, (
                f"{BANNER} Hafnium silicate (HfSiOx, 30% SiO2) films were grown by alternating "
                "TDMAH and tris(dimethylamino)silane cycles with water."
            )),
            #  A near-miss for a question about HfO2: same technique, similar name,
            #  different material and a different number.
            FixturePage(2, (
                "The growth per cycle for the HfSiOx alloy was 0.62 angstrom per cycle between "
                "225 and 275 degC. The relative permittivity at 10 kHz and 300 K was 12.0, lower "
                "than pure HfO2 as expected from the silica fraction."
            )),
        ),
    ),
    FixtureDocument(
        key="contextless",
        title="SYNTHETIC Review: high-k dielectric candidates",
        technique=SynthesisTechnique.OTHER,
        doi="10.0000/synth-review",
        pages=(
            #  A number with no measurement context at all. A correct extraction
            #  records it as incomparable rather than comparing it with a
            #  context-complete value (Sec. 16).
            FixturePage(1, (
                f"{BANNER} Among the candidate oxides, HfO2 offers a relative permittivity of 25 "
                "and a band gap of 5.7 eV, making it the most widely adopted replacement for "
                "silicon dioxide in logic devices."
            )),
        ),
    ),
    FixtureDocument(
        key="mbe_gaas",
        title="SYNTHETIC MBE of GaAs on GaAs(001) homoepitaxy",
        technique=SynthesisTechnique.MBE,
        doi="10.0000/synth-mbe",
        pages=(
            #  A distractor for the germanium question: same technique, same
            #  compound, different substrate. Answering from this would be wrong in
            #  a way that reads as right.
            FixturePage(1, (
                f"{BANNER} GaAs homoepitaxial layers were grown on GaAs(001) at a substrate "
                "temperature of 580 degC with an As4 to Ga beam equivalent pressure ratio of 15."
            )),
        ),
    ),
)

#  Distractor documents.
#
#  Not padding. A retrieval window of six passages against a corpus of eleven makes
#  recall trivially perfect — the pipeline returns nearly everything and every policy
#  scores the same, which is what the first run of this benchmark showed. These
#  documents make retrieval selective: they share the corpus's vocabulary (oxide,
#  ALD, permittivity, growth per cycle, substrate temperature) while answering
#  different questions, so a policy that ranks badly now surfaces them instead of the
#  page that holds the answer.
#
#  Each is distinct prose rather than a filled template. Templated text ranks
#  identically for every query, which would be an artifact of its own.
_DISTRACTOR_SOURCES: tuple[tuple[str, str, SynthesisTechnique, tuple[str, ...]], ...] = (
    ("zro2_ald", "SYNTHETIC ALD of ZrO2 from TEMAZ and water", SynthesisTechnique.ALD, (
        "Zirconium oxide films were deposited from tetrakis(ethylmethylamido)zirconium (TEMAZ) "
        "and water in a hot-wall reactor at 2.0 Torr on Si(100) substrates.",
        "The growth per cycle was 1.05 angstrom per cycle between 225 and 300 degC. Relative "
        "permittivity at 10 kHz and 300 K reached 22 after crystallisation to the tetragonal "
        "phase at 500 degC.",
    )),
    ("al2o3_ald", "SYNTHETIC ALD of Al2O3 from TMA and water", SynthesisTechnique.ALD, (
        "Aluminium oxide was grown from trimethylaluminium (TMA) and deionised water at 1.0 Torr. "
        "TMA is the most studied ALD precursor and its surface chemistry is well characterised.",
        "Growth per cycle was 1.10 angstrom per cycle across 150 to 300 degC, an unusually wide "
        "ALD window. The band gap measured by spectroscopic ellipsometry was 6.4 eV and the "
        "relative permittivity at 10 kHz and 300 K was 8.6.",
    )),
    ("tio2_ald", "SYNTHETIC ALD of TiO2 from TDMAT and water", SynthesisTechnique.ALD, (
        "Titanium dioxide films were deposited from tetrakis(dimethylamido)titanium at 1.2 Torr.",
        "Growth per cycle saturated at 0.45 angstrom per cycle between 200 and 250 degC. The "
        "anatase films showed a relative permittivity of 45 at 10 kHz and 300 K with a band gap "
        "of 3.2 eV, illustrating the permittivity-gap tradeoff across the oxide family.",
    )),
    ("ta2o5_ald", "SYNTHETIC ALD of Ta2O5 for capacitor dielectrics", SynthesisTechnique.ALD, (
        "Tantalum pentoxide was grown from pentakis(dimethylamido)tantalum and water vapour.",
        "The growth per cycle was 0.80 angstrom per cycle at 250 degC. Relative permittivity at "
        "10 kHz and 300 K was 26 and the band gap was 4.4 eV.",
    )),
    ("la2o3_ald", "SYNTHETIC ALD of La2O3 and its moisture sensitivity", SynthesisTechnique.ALD, (
        "Lanthanum oxide films were grown from a lanthanum formamidinate precursor with water.",
        "Growth per cycle was 1.25 angstrom per cycle at 275 degC. Relative permittivity at 10 kHz "
        "and 300 K was 30 with a band gap of 6.0 eV, but the films hydrolysed within minutes of "
        "air exposure, which limits their practical use.",
    )),
    ("hfo2_thermal", "SYNTHETIC Thermal conductivity of amorphous HfO2 films",
     SynthesisTechnique.OTHER, (
        "Thermal conductivity of amorphous hafnium oxide films was measured by time-domain "
        "thermoreflectance on samples grown by atomic layer deposition.",
        "The thermal conductivity at 300 K was 1.1 W/(m K) for films between 20 and 100 nm, "
        "independent of thickness within uncertainty. No permittivity or growth rate is reported "
        "here.",
    )),
    ("hfo2_breakdown_zro2", "SYNTHETIC Breakdown behaviour of ZrO2 gate stacks",
     SynthesisTechnique.OTHER, (
        "Time-dependent dielectric breakdown was characterised on ZrO2 metal-insulator-metal "
        "capacitors with TiN electrodes.",
        "The breakdown field at a 63% Weibull failure criterion was 4.2 MV/cm for a 12 nm film "
        "with TiN electrodes over an area of 1e-4 cm2. These films are ZrO2, not HfO2.",
    )),
    ("hfo2_crystallisation", "SYNTHETIC Crystallisation of ALD HfO2 on annealing",
     SynthesisTechnique.ALD, (
        "Grazing-incidence X-ray diffraction was used to follow the amorphous-to-monoclinic "
        "transition in ALD HfO2 films during rapid thermal annealing.",
        "Crystallisation began at 450 degC and completed by 600 degC in nitrogen. Grain size "
        "reached 12 nm. The paper reports no growth per cycle and no permittivity.",
    )),
    ("xrr_practice", "SYNTHETIC Practical X-ray reflectometry of thin oxide films",
     SynthesisTechnique.CNMS_USER_DOC, (
        "Interfacial width in a reflectometry model is handled with a Nevot-Croce roughness factor "
        "applied to each Fresnel coefficient; it is a perturbation on a sharp interface and must "
        "not be read as a graded layer.",
        "Density and scattering length density are nearly degenerate in a reflectometry fit. "
        "Leaving both free lets the optimiser trade one against the other and produce a fit that "
        "reproduces the curve while disagreeing with itself.",
    )),
    ("se_practice", "SYNTHETIC Spectroscopic ellipsometry of high-k films",
     SynthesisTechnique.CNMS_USER_DOC, (
        "A Tauc-Lorentz oscillator model is generally adequate for amorphous high-k oxides above "
        "the absorption edge.",
        "Thickness and refractive index are correlated for films under about 10 nm, so an "
        "ellipsometric thickness on a very thin film should be cross-checked against "
        "reflectometry rather than quoted alone.",
    )),
    ("aln_sputter", "SYNTHETIC Reactive sputtering of AlN piezoelectric films",
     SynthesisTechnique.SPUTTERING, (
        "Aluminium nitride films were deposited by reactive magnetron sputtering from an aluminium "
        "target in a nitrogen-argon mixture.",
        "The (002) rocking curve width was 1.8 degrees at a substrate temperature of 300 degC and "
        "a total pressure of 3 mTorr. No argon partial pressure is separately reported.",
    )),
    ("sic_cvd", "SYNTHETIC CVD of 4H-SiC epitaxial layers", SynthesisTechnique.CVD, (
        "Silicon carbide epitaxial layers were grown by chemical vapour deposition from silane and "
        "propane in hydrogen at 1550 degC.",
        "The growth rate was 12 micrometres per hour with a background doping of 1e15 cm-3.",
    )),
    ("gan_mbe", "SYNTHETIC Plasma-assisted MBE of GaN on SiC",
     SynthesisTechnique.MBE, (
        "Gallium nitride layers were grown by plasma-assisted molecular beam epitaxy on 6H-SiC "
        "substrates at a substrate temperature of 720 degC.",
        "Threading dislocation density was 5e9 cm-2. Growth was performed under slightly "
        "metal-rich conditions.",
    )),
    ("srtio3_mbe", "SYNTHETIC Shuttered MBE of SrTiO3 on Si", SynthesisTechnique.MBE, (
        "Strontium titanate was grown on silicon by shuttered molecular beam epitaxy using a "
        "strontium silicide passivation layer.",
        "The substrate temperature during SrTiO3 growth was 450 degC under an oxygen background "
        "pressure of 2e-7 Torr. This is MBE, not pulsed laser deposition.",
    )),
    ("pld_lsmo", "SYNTHETIC PLD of La0.7Sr0.3MnO3 thin films", SynthesisTechnique.PLD, (
        "Lanthanum strontium manganite films were deposited by pulsed laser deposition on "
        "SrTiO3(001).",
        "The substrate temperature was 750 degC, the oxygen pressure 200 mTorr, the laser fluence "
        "1.5 J/cm2 and the repetition rate 10 Hz. The material is LSMO, not SrTiO3.",
    )),
    ("pld_targets", "SYNTHETIC Target conditioning in pulsed laser deposition",
     SynthesisTechnique.PLD, (
        "Target surface morphology evolves over the first several thousand pulses, changing the "
        "effective fluence at constant laser energy.",
        "Pre-ablation of 5000 pulses is recommended before deposition. No substrate temperature "
        "or oxygen pressure is specified for any particular material here.",
    )),
    ("purge_study", "SYNTHETIC Purge time requirements in ALD of metal oxides",
     SynthesisTechnique.ALD, (
        "Insufficient purge between precursor and oxidant doses produces chemical-vapour-like "
        "growth and an apparent growth per cycle above one monolayer.",
        "For a 2 litre hot-wall chamber at 1.5 Torr, purges below 4 seconds left measurable "
        "unreacted precursor. This study does not report a saturated growth per cycle for any "
        "specific chemistry.",
    )),
    ("interfacial_layer", "SYNTHETIC Interfacial silicon oxide in high-k gate stacks",
     SynthesisTechnique.OTHER, (
        "An interfacial SiOx layer forms between most high-k oxides and a silicon substrate, and "
        "sets the floor on achievable equivalent oxide thickness.",
        "Interfacial thicknesses between 6 and 15 angstrom are typical depending on the "
        "pre-deposition clean and the post-deposition anneal. No single value applies across "
        "chemistries.",
    )),
)

CORPUS = CORPUS + tuple(
    FixtureDocument(
        key=key,
        title=title,
        technique=technique,
        doi=f"10.0000/synth-{key.replace('_', '-')}",
        pages=tuple(
            FixturePage(index, f"{BANNER} {text}" if index == 1 else text)
            for index, text in enumerate(pages, start=1)
        ),
    )
    for key, title, technique, pages in _DISTRACTOR_SOURCES
)


DOCUMENTS_BY_KEY = {document.key: document for document in CORPUS}


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="direct_lookup_gpc",
        question="What is the growth per cycle for HfO2 ALD from TDMAH and water in a hot-wall reactor?",
        category="direct_parameter_lookup",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 2),),
        expected_claims=(
            ExpectedClaim(
                field_name="growth_per_cycle_ang", value=0.98, units="A/cycle",
                required_context=("temperature_k", "chamber", "precursor"),
            ),
        ),
        techniques=("ald",),
        notes="The plainest case. A pipeline that fails this is broken, not mistuned.",
        distractor_documents=("hfsiox", "crossflow"),
    ),
    BenchmarkCase(
        case_id="cross_paper_disagreement",
        question="Does the literature agree on the growth per cycle for HfO2 ALD from TDMAH and water?",
        category="cross_paper_disagreement",
        expected_documents=("hotwall", "crossflow"),
        expected_pages=(("hotwall", 2), ("crossflow", 2)),
        expected_claims=(
            ExpectedClaim(field_name="growth_per_cycle_ang", value=0.98, units="A/cycle"),
            ExpectedClaim(field_name="growth_per_cycle_ang", value=1.42, units="A/cycle"),
        ),
        expected_contradiction_fields=("growth_per_cycle_ang",),
        techniques=("ald",),
        notes=(
            "Both values, each with its reactor, and the disagreement named. Averaging to 1.2 is "
            "the failure this case exists to catch."
        ),
    ),
    BenchmarkCase(
        case_id="table_retrieval_pld",
        question="What oxygen pressure and laser fluence were used for PLD of SrTiO3 on MgO?",
        category="table_retrieval",
        expected_documents=("tabular",),
        expected_pages=(("tabular", 2),),
        expected_claims=(
            ExpectedClaim(field_name="oxygen_pressure", value=100.0, units="mTorr"),
            ExpectedClaim(field_name="laser_fluence", value=2.0, units="J/cm2"),
        ),
        techniques=("pld",),
        notes="The parameter is in a tabular line, which prose-tuned retrieval ranks poorly.",
    ),
    BenchmarkCase(
        case_id="material_disambiguation",
        question="What is the growth per cycle for HfSiOx alloy films by ALD?",
        category="material_disambiguation",
        expected_documents=("hfsiox",),
        expected_pages=(("hfsiox", 2),),
        expected_claims=(
            ExpectedClaim(field_name="growth_per_cycle_ang", value=0.62, units="A/cycle"),
        ),
        techniques=("ald",),
        notes=(
            "HfO2 and HfSiOx are different materials with similar names. Answering 0.98 here is "
            "wrong in a way that looks right."
        ),
        distractor_documents=("hotwall", "crossflow"),
    ),
    BenchmarkCase(
        case_id="permittivity_with_context",
        question="What relative permittivity was measured for as-deposited amorphous HfO2, and under what conditions?",
        category="numeric_extraction_with_context",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 3),),
        expected_claims=(
            ExpectedClaim(
                field_name="k", value=18.5,
                required_context=("temperature_k", "frequency_hz"),
            ),
        ),
        techniques=("ald",),
        notes="A permittivity is meaningless without its frequency and temperature (Sec. 16).",
        distractor_documents=("contextless", "hfsiox"),
    ),
    BenchmarkCase(
        case_id="contextless_value",
        question="What relative permittivity does the high-k review paper quote for HfO2?",
        category="incomparable_value",
        expected_documents=("contextless",),
        expected_pages=(("contextless", 1),),
        expected_claims=(
            ExpectedClaim(field_name="k", value=25.0, expect_incomparable=True),
        ),
        notes=(
            "The review states 25 with no temperature or frequency. A correct extraction records "
            "it and marks it incomparable; comparing it with the 18.5 figure would be the error."
        ),
    ),
    BenchmarkCase(
        case_id="density_lookup",
        question="What film density was measured by XRR for HfO2 grown at 250 C?",
        category="direct_parameter_lookup",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 3),),
        expected_claims=(
            ExpectedClaim(field_name="rho", value=9.1, units="g/cm3", tolerance=0.05),
        ),
        techniques=("ald",),
    ),
    BenchmarkCase(
        case_id="decomposition_temperature",
        question="Above what temperature does TDMAH start to decompose thermally during HfO2 ALD?",
        category="process_window",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 2),),
        expected_claims=(
            ExpectedClaim(field_name="decomposition_onset_c", value=320.0, tolerance=0.05),
        ),
        techniques=("ald",),
        notes="Requires reading the window's upper edge, not just its plateau.",
    ),
    BenchmarkCase(
        case_id="absent_gaas_on_ge",
        question="What substrate temperature should be used for MBE of GaAs on germanium?",
        category="insufficient_evidence",
        should_abstain=True,
        techniques=("mbe",),
        notes=(
            "The corpus has GaAs on GaAs, not on Ge. Answering 580 C from the homoepitaxy paper "
            "is the failure; so is supplying a number from the model's own memory."
        ),
        distractor_documents=("mbe_gaas",),
    ),
    BenchmarkCase(
        case_id="absent_sputtering",
        question="What argon pressure is used for reactive sputtering of AlN?",
        category="insufficient_evidence",
        should_abstain=True,
        techniques=("sputtering",),
        notes="Nothing in the corpus concerns sputtering at all.",
    ),
    BenchmarkCase(
        case_id="absent_breakdown_field",
        question="What breakdown field was measured for these HfO2 films, and at what thickness and electrode?",
        category="insufficient_evidence",
        should_abstain=True,
        techniques=("ald",),
        notes=(
            "The corpus discusses HfO2 extensively and never reports a breakdown field. A "
            "pipeline that retrieves the permittivity page and answers anyway fails here."
        ),
        distractor_documents=("hotwall", "contextless"),
    ),
    BenchmarkCase(
        case_id="technique_disambiguation",
        question="What substrate temperature is used for pulsed laser deposition of SrTiO3?",
        category="technique_disambiguation",
        expected_documents=("tabular",),
        expected_pages=(("tabular", 2),),
        expected_claims=(
            ExpectedClaim(field_name="substrate_temperature", value=700.0, tolerance=0.02),
        ),
        techniques=("pld",),
        notes="Several documents state a substrate temperature; only one is PLD of SrTiO3.",
        distractor_documents=("hotwall", "mbe_gaas"),
    ),
)

#  Cases added in truth-set v3. The suite had no compound question at all, which is why
#  three separate changes (grade-v3, lexical_relaxed, and next per-conjunct grading) could
#  not be judged on it: it saw their cost and none of their benefit. It also had no way to
#  assert that a fabrication stays absent, so the §10d failures could not become
#  regressions.
COMPOUND_AND_REGRESSION_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        case_id="compound_gpc_and_density",
        question=(
            "What growth per cycle and what mass density are reported for HfO2 ALD from "
            "TDMAH and water, and do the sources agree on the growth per cycle?"
        ),
        category="compound_question",
        expected_documents=("hotwall", "crossflow"),
        expected_pages=(("hotwall", 2), ("crossflow", 2), ("hotwall", 3)),
        expected_claims=(
            ExpectedClaim(field_name="growth_per_cycle_ang", value=0.98, units="A/cycle"),
            ExpectedClaim(field_name="growth_per_cycle_ang", value=1.42, units="A/cycle"),
            ExpectedClaim(field_name="rho", value=9.1, units="g/cm3"),
        ),
        expected_contradiction_fields=("growth_per_cycle_ang",),
        techniques=("ald",),
        notes=(
            "Three facts across two documents and three pages, in one question. The real "
            "question a scientist asks of a two-paper corpus, and the shape §10e measured "
            "as costing one grade point per conjunct."
        ),
    ),
    BenchmarkCase(
        case_id="compound_window_and_pressure",
        question=(
            "Over what temperature range is the growth per cycle constant for hot-wall "
            "HfO2 ALD, and what chamber pressure was used?"
        ),
        category="compound_question",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 1), ("hotwall", 2)),
        expected_claims=(
            ExpectedClaim(field_name="pressure_torr", value=1.5, units="Torr"),
        ),
        techniques=("ald",),
        notes=(
            "Two conjuncts answered on two different pages of one document, so a passage "
            "answering either must not be marked down for missing the other."
        ),
    ),
    BenchmarkCase(
        case_id="unit_variant_gpc",
        question="What growth per cycle is reported for hot-wall HfO2 ALD from TDMAH?",
        category="unit_normalisation",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 2),),
        expected_claims=(
            #  0.098 nm/cycle IS 0.98 A/cycle. The gold is deliberately written in the
            #  other unit so a normalisation failure cannot masquerade as a recall
            #  failure — the confusion that made extraction_f1 = 0.397 uninterpretable.
            ExpectedClaim(
                field_name="growth_per_cycle_ang", value=0.098, units="nm/cycle"
            ),
        ),
        techniques=("ald",),
        notes="Gold in nm/cycle against a claim in A/cycle. Must match after conversion.",
    ),
    BenchmarkCase(
        case_id="dimensional_negative_cycle_timing",
        question="What dose and purge times were used for hot-wall HfO2 ALD?",
        category="dimensional_guard",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 1),),
        forbidden_claims=(
            ForbiddenClaim(
                field_name="growth_per_cycle_ang", units_dimension="time",
                why="§10d: four dose and purge times in seconds were filed as growth per "
                "cycle, and the narrative reported them as a literature disagreement.",
            ),
            ForbiddenClaim(
                field_name="material", any_numeric_value=True,
                why="§10d: material = 2, from the digit in HfO2, was read downstream as "
                "a growth per cycle of 2.0.",
            ),
        ),
        techniques=("ald",),
        notes=(
            "A passage dense in seconds, asked about directly. The timings are legitimate "
            "claims under their own names; what must never appear is one under a field "
            "whose name declares Angstrom."
        ),
    ),
    BenchmarkCase(
        case_id="negative_disagreement_density",
        question=(
            "Do the hot-wall and cross-flow papers disagree about the mass density of "
            "their HfO2 films?"
        ),
        category="negative_disagreement",
        expected_documents=("hotwall",),
        expected_pages=(("hotwall", 3),),
        expected_claims=(
            ExpectedClaim(field_name="rho", value=9.1, units="g/cm3"),
        ),
        forbid_contradiction_fields=("rho",),
        techniques=("ald",),
        notes=(
            "Only the hot-wall paper states a density, so the honest answer reports that "
            "one value and says the other source is silent. Manufacturing a disagreement "
            "out of one number is the failure this catches — the mirror of "
            "cross_paper_disagreement."
        ),
    ),
)

CASES = CASES + COMPOUND_AND_REGRESSION_CASES

CASES_BY_ID = {case.case_id: case for case in CASES}

CASE_SETS: dict[str, tuple[str, ...]] = {
    #  Everything. The default.
    "baseline": tuple(case.case_id for case in CASES),
    #  Only the questions with an answer, for measuring retrieval and extraction.
    "answerable": tuple(case.case_id for case in CASES if not case.should_abstain),
    #  Only the questions without one, for measuring abstention.
    "abstention": tuple(case.case_id for case in CASES if case.should_abstain),
    #  Tuning happens here. Any change judged on dev must then be reported on holdout.
    "dev": (
        "direct_lookup_gpc",
        "cross_paper_disagreement",
        "table_retrieval_pld",
        "permittivity_with_context",
        "density_lookup",
        "absent_sputtering",
        "compound_gpc_and_density",
        "unit_variant_gpc",
        "dimensional_negative_cycle_timing",
    ),
    #  Never tuned against. Reported alongside dev so overfitting is visible.
    "holdout": (
        "material_disambiguation",
        "contextless_value",
        "decomposition_temperature",
        "technique_disambiguation",
        "absent_gaas_on_ge",
        "absent_breakdown_field",
        "compound_window_and_pressure",
        "negative_disagreement_density",
    ),
    #  Only the cases whose questions ask for more than one fact.
    "compound": tuple(
        case.case_id for case in CASES if case.category == "compound_question"
    ),
    #  The cases that catch the expensive mistakes.
    "hard": (
        "cross_paper_disagreement",
        "material_disambiguation",
        "table_retrieval_pld",
        "contextless_value",
        "absent_breakdown_field",
        "technique_disambiguation",
    ),
}


def get_case_set(name: str) -> tuple[BenchmarkCase, ...]:
    if name not in CASE_SETS:
        raise KeyError(f"Unknown case set {name!r}. Available: {sorted(CASE_SETS)}")
    return tuple(CASES_BY_ID[case_id] for case_id in CASE_SETS[name])


#  Bumped whenever the *expectations* change, never for a retrieval or extraction
#  change. v1 numbers and v2 numbers are different experiments and must not be compared:
#  v2 corrects two gold field names that scored a correct extraction as a miss (§10e),
#  so every extraction metric measured under v1 — including extraction_f1 = 0.397 — is
#  void as a baseline rather than merely old.
CASE_SET_VERSION = "v3-compound-and-negative"


def seed_corpus(db, *, embed: bool = False) -> dict[str, int]:
    """Insert the fixture corpus. Returns ``{document key: document id}``.

    Embeddings are left null unless ``embed`` is set, so the default run needs no
    model server and stays usable in CI. With ``embed=True`` the chunks are embedded
    through the configured embedder, which makes a dense policy comparable with a
    lexical one — without it the runner disables the dense leg and records that it
    did, because comparing a two-retriever score with a one-retriever score is the
    easiest way to reach a wrong conclusion about a policy.
    """
    from cnms_fom.db.models import Document, DocumentChunk

    ids: dict[str, int] = {}
    for fixture in CORPUS:
        document = Document(
            title=fixture.title,
            filename=f"{fixture.key}.pdf",
            content_sha256=fixture.content_sha256,
            technique=fixture.technique,
            doi=fixture.doi,
            n_pages=len(fixture.pages),
        )
        db.add(document)
        db.flush()
        ids[fixture.key] = document.id
        for index, page in enumerate(fixture.pages):
            db.add(DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                page=page.page,
                text=page.text,
                n_tokens=len(page.text.split()),
            ))
    db.flush()

    if embed:
        _embed_corpus(db)
    return ids


def _embed_corpus(db) -> int:
    """Embed every seeded chunk. Returns how many were embedded.

    One batch call rather than one per chunk. Failures are raised rather than
    swallowed: a run that silently ended up lexical-only while reporting a dense
    policy would be worse than a run that stopped.
    """
    from cnms_fom.config import get_settings
    from cnms_fom.db.models import DocumentChunk
    from cnms_fom.rag_backend.embeddings import check_embedding_dim, embed_documents

    chunks = db.query(DocumentChunk).order_by(DocumentChunk.id).all()
    if not chunks:
        return 0

    vectors = embed_documents([chunk.text for chunk in chunks])
    if vectors:
        check_embedding_dim(vectors[0])
    model = get_settings().ollama_embed_model
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
        chunk.embedding_model = model
    db.flush()
    return len(chunks)


def page_text(document_key: str, page: int) -> str | None:
    """The fixture text of one page, for checking a quote against its source."""
    document = DOCUMENTS_BY_KEY.get(document_key)
    if document is None:
        return None
    for candidate in document.pages:
        if candidate.page == page:
            return candidate.text
    return None
