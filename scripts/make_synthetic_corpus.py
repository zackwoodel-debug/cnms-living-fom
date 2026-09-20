#!/usr/bin/env python3
"""Generate a small synthetic corpus for exercising ingestion and retrieval.

Why this exists: verifying the retrieval pipeline end to end needs real PDFs, and
the interesting behaviour is what happens when two sources *disagree*. So the two
documents written here report different growth-per-cycle values for the same
chemistry — 0.98 A/cycle in a hot-wall reactor, 1.42 in a cross-flow one — each
with the context that explains the difference. A pipeline that averages them to
1.2, or quotes one without its reactor, has failed in a way a single-document test
cannot detect.

Every page is stamped SYNTHETIC. The numbers are plausible and invented, the
files land in a gitignored directory, and nothing here is citable. Requires
``reportlab`` (not a project dependency — install it just for this).

    python scripts/make_synthetic_corpus.py
    cnms-fom ingest data/pdfs --technique ald
"""

from __future__ import annotations

import sys
from pathlib import Path

BANNER = (
    "<b>SYNTHETIC TEST DOCUMENT.</b> Generated to exercise an ingestion and retrieval "
    "pipeline. The numbers below are plausible but invented. This is not a real "
    "publication and must not be cited."
)

HOT_WALL = [
    ("h1", "Atomic Layer Deposition of HfO2 on Si(100) using TDMAH and H2O"),
    ("warn", BANNER),
    ("h2", "2. Experimental"),
    ("body", "Films were deposited in a Beneq TFS-200 hot-wall reactor on 100 mm Si(100) "
             "substrates. Prior to loading, substrates received a standard HF-last clean "
             "(10:1 HF, 60 s) followed by a deionised water rinse and N2 blow-dry. The "
             "hafnium precursor was tetrakis(dimethylamido)hafnium (TDMAH), held at 75 degC "
             "in a stainless steel bubbler, with deionised water vapour as the oxidant. "
             "Nitrogen at 300 sccm served as both carrier and purge gas, and the chamber "
             "was maintained at 1.5 Torr throughout."),
    ("body", "A single ALD cycle comprised a 0.2 s TDMAH dose, a 6 s N2 purge, a 0.1 s H2O "
             "dose, and a further 6 s N2 purge. Purge durations were established by "
             "extending each until the growth per cycle no longer changed; at 4 s purge the "
             "growth per cycle was still elevated, indicating incomplete removal of "
             "unreacted precursor."),
    ("h2", "3.1 Growth per cycle and the ALD window"),
    ("body", "Growth per cycle was measured by spectroscopic ellipsometry on films of 200 "
             "cycles. Between 200 and 300 degC the growth per cycle was constant at 0.98 "
             "angstrom per cycle, which we take as the ALD window for this precursor "
             "combination. Below 200 degC the growth per cycle rose to 1.3 angstrom per "
             "cycle, consistent with precursor condensation. Above 320 degC it fell to 0.71 "
             "angstrom per cycle, consistent with the onset of TDMAH thermal decomposition."),
    ("body", "We emphasise that the 0.98 angstrom per cycle figure applies only to the "
             "TDMAH/H2O chemistry in this chamber at 1.5 Torr. Measurements in a cross-flow "
             "reactor at 0.3 Torr, reported elsewhere, are not directly comparable."),
    ("h2", "3.2 Film density and structure"),
    ("body", "X-ray reflectometry at Cu K-alpha gave a mass density of 9.1 g/cm3 for films "
             "grown at 250 degC, below the 9.68 g/cm3 of bulk monoclinic HfO2, which we "
             "attribute to residual porosity and carbon incorporation from the amido "
             "ligands. Grazing-incidence diffraction showed the films to be amorphous as "
             "deposited, crystallising to the monoclinic phase after a 600 degC anneal in N2."),
    ("h2", "3.3 Interfacial layer"),
    ("body", "All films exhibited an interfacial SiOx layer between the HfO2 and the "
             "substrate. Reflectometry fitting gave an interfacial thickness of 8.6 angstrom "
             "for a 250 degC growth, increasing to 14 angstrom after the 600 degC anneal. "
             "The interfacial layer is the principal obstacle to reducing the equivalent "
             "oxide thickness below about 10 angstrom in this system."),
    ("h2", "3.4 Electrical characterisation"),
    ("body", "Capacitance-voltage measurements on Pt/HfO2/Si capacitors at 10 kHz gave a "
             "relative permittivity of 18.5 for the as-deposited amorphous films, rising to "
             "24 after crystallisation. Leakage at 1 V was 3e-7 A/cm2 for a 10 nm film."),
]

CROSS_FLOW = [
    ("h1", "Temperature dependence of HfO2 ALD in a cross-flow reactor: a re-examination"),
    ("warn", BANNER),
    ("h2", "1. Introduction"),
    ("body", "Reported growth per cycle for the TDMAH/H2O atomic layer deposition of HfO2 "
             "varies considerably across the literature. We revisit the temperature "
             "dependence in a cross-flow geometry and find a materially different "
             "saturation value from that commonly quoted for hot-wall systems."),
    ("h2", "2. Experimental"),
    ("body", "Depositions were performed in a custom cross-flow reactor at a base pressure "
             "of 0.3 Torr on Si(100) with the native oxide left intact. TDMAH was held at "
             "80 degC; the oxidant was deionised water vapour. Cycle timing was 0.5 s TDMAH, "
             "10 s purge, 0.5 s H2O, 10 s purge, with purge times deliberately long to "
             "guarantee self-limiting behaviour."),
    ("h2", "3. Results"),
    ("body", "Over the range 200 to 300 degC the growth per cycle saturated at 1.42 angstrom "
             "per cycle, substantially above the value of approximately 1 angstrom per cycle "
             "widely reported for hot-wall reactors. We attribute the difference to the "
             "reactor geometry and to the native oxide left on our substrates, which presents "
             "a higher initial hydroxyl density than an HF-last surface."),
    ("body", "We stress that a growth per cycle of 1.42 angstrom exceeds a single monolayer "
             "of monoclinic HfO2, which is approximately 3 angstrom across two formula units. "
             "Our value therefore sits close to, though below, the self-limiting ceiling, and "
             "we caution that a higher figure would indicate chemical vapour deposition rather "
             "than true atomic layer deposition."),
    ("body", "Film density from X-ray reflectometry was 8.7 g/cm3, and the interfacial oxide "
             "measured 12 angstrom, thicker than for HF-last substrates as expected from the "
             "retained native layer."),
    ("h2", "4. Discussion"),
    ("body", "The discrepancy between our saturation value and hot-wall reports cannot be "
             "resolved by appealing to measurement error: both are established by "
             "thickness-versus-cycle-count linearity over hundreds of cycles. We suggest the "
             "substrate preparation is the dominant variable and recommend that any quoted "
             "growth per cycle be accompanied by the surface pretreatment, the reactor "
             "geometry, and the operating pressure."),
]


def main() -> int:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        print(
            "This script needs reportlab, which is not a project dependency:\n"
            "    pip install reportlab",
            file=sys.stderr,
        )
        return 1

    sheet = getSampleStyleSheet()
    styles = {
        "body": ParagraphStyle("body", parent=sheet["BodyText"], fontSize=10, leading=14, spaceAfter=8),
        "h1": ParagraphStyle("h1", parent=sheet["Heading1"], fontSize=13, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=sheet["Heading2"], fontSize=11, spaceAfter=6),
    }
    styles["warn"] = ParagraphStyle(
        "warn", parent=styles["body"], textColor="#a00000", fontSize=9
    )

    out_dir = Path("data/pdfs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, blocks in (
        ("synthetic_ald_hfo2_hotwall.pdf", HOT_WALL),
        ("synthetic_ald_hfo2_crossflow.pdf", CROSS_FLOW),
    ):
        path = out_dir / name
        doc = SimpleDocTemplate(
            str(path), pagesize=LETTER,
            leftMargin=inch, rightMargin=inch, topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        )
        flow = []
        for style, text in blocks:
            flow.append(Paragraph(text, styles[style]))
            flow.append(Spacer(1, 2))
        doc.build(flow)
        print(f"wrote {path}")

    print(
        "\nThe two documents disagree on growth per cycle (0.98 vs 1.42 A/cycle) on purpose.\n"
        "Ingest and ask about it:\n"
        "    cnms-fom ingest data/pdfs --technique ald\n"
        '    cnms-fom ask "What is the growth per cycle for HfO2 ALD from TDMAH and water, '
        'and does the literature agree?" --technique ald\n'
        "A correct answer reports both values with their reactor and pressure, and says they "
        "disagree. Averaging them to 1.2 is the failure this corpus is built to catch."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
