"""API tests for /cards and the PDF upload path.

The upload tests deliberately do not need a working embedder: ingestion fails
without the rag extra, and what matters is that the *file is kept* and the error
says so, rather than the upload being silently lost.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base, get_db
from cnms_fom.db.models import KnowledgeCard  # noqa: F401 - registers mappers
from cnms_fom.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    from cnms_fom.config import get_settings

    get_settings.cache_clear()

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.db'}", future=True)
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()
    get_settings.cache_clear()


# --- upload ----------------------------------------------------------------


def test_a_non_pdf_is_rejected_with_the_reason(client):
    response = client.post(
        "/rag/ingest/upload",
        files={"files": ("notes.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 200
    entry = response.json()["ingested"][0]
    #  The reason matters: the chunker records a page per passage, and a format
    #  without pages cannot produce a citation.
    assert "Only PDFs" in entry["error"]
    assert "citation" in entry["error"]


def test_an_uploaded_pdf_is_kept_even_when_indexing_fails(client, tmp_path):
    """Without the rag extra the embed step fails; the file must not be lost."""
    response = client.post(
        "/rag/ingest/upload",
        files={"files": ("paper.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")},
    )
    assert response.status_code in (200, 501)
    saved = list((tmp_path / "corpus").glob("*.pdf"))
    assert [p.name for p in saved] == ["paper.pdf"]

    if response.status_code == 200:
        entry = response.json()["ingested"][0]
        assert entry["error"]
        assert entry["saved_to"].endswith("paper.pdf")
        assert "OCR" in entry["hint"]


def test_a_second_upload_of_the_same_filename_does_not_overwrite(client, tmp_path):
    """Two different papers can share a filename; the loser would vanish."""
    for _ in range(2):
        client.post(
            "/rag/ingest/upload",
            files={"files": ("paper.pdf", b"%PDF-1.4 x", "application/pdf")},
        )
    names = sorted(p.name for p in (tmp_path / "corpus").glob("*.pdf"))
    assert names == ["paper.pdf", "paper__2.pdf"]


def test_an_oversized_upload_is_refused_and_cleaned_up(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    from cnms_fom.config import get_settings

    get_settings.cache_clear()

    response = client.post(
        "/rag/ingest/upload",
        files={"files": ("big.pdf", b"%PDF-1.4" + b"x" * 2048, "application/pdf")},
    )
    entry = response.json()["ingested"][0]
    assert "limit" in entry["error"]
    #  The partial write is removed, not left as a truncated PDF.
    assert list((tmp_path / "corpus").glob("*.pdf")) == []
    get_settings.cache_clear()


# --- cards -----------------------------------------------------------------


CARD = {
    "slug": "concepts/ald-window-hfo2",
    "title": "ALD window for HfO2",
    "body": "GPC saturates at 0.98 A/cycle between 200 and 300 C [1].",
    "card_type": "concept",
    "sources": [{"kind": "document", "document_id": 1, "page": 7}],
    "tags": ["ald", "hfo2"],
}


def test_a_written_card_is_proposed_and_not_citable(client):
    body = client.post("/cards", json=CARD).json()
    assert body["status"] == "proposed"
    assert body["citable"] is False
    assert "not been reviewed" in body["caution"]


def test_a_bad_slug_is_422(client):
    response = client.post("/cards", json={**CARD, "slug": "Concepts/ALD Window"})
    assert response.status_code == 422
    assert "usable slug" in response.json()["detail"]


def test_review_requires_a_resolved_source_and_then_makes_it_citable(client):
    client.post("/cards", json={**CARD, "slug": "concepts/freetext", "sources": []})
    refused = client.post(
        "/cards/concepts/freetext/review", json={"reviewed_by": "Z. Woodel"}
    )
    assert refused.status_code == 409
    assert "no resolved source" in refused.json()["detail"]

    client.post("/cards", json=CARD)
    ok = client.post(
        "/cards/concepts/ald-window-hfo2/review", json={"reviewed_by": "Z. Woodel"}
    ).json()
    assert ok["status"] == "reviewed"
    assert ok["citable"] is True
    assert ok["caution"] is None


def test_editing_a_reviewed_card_makes_it_uncitable_again(client):
    client.post("/cards", json=CARD)
    client.post("/cards/concepts/ald-window-hfo2/review", json={"reviewed_by": "Z. Woodel"})

    edited = client.post("/cards", json={**CARD, "body": "Revised: 1.4 A/cycle."}).json()
    assert edited["status"] == "reviewed"
    assert edited["review_is_stale"] is True
    assert edited["citable"] is False


def test_a_contradicts_link_needs_a_note_and_shows_on_both_cards(client):
    client.post("/cards", json=CARD)
    client.post("/cards", json={**CARD, "slug": "concepts/other", "title": "Other"})

    refused = client.post(
        "/cards/links",
        json={
            "from_slug": "concepts/ald-window-hfo2",
            "to_slug": "concepts/other",
            "relation": "contradicts",
        },
    )
    assert refused.status_code == 422
    assert "needs a note" in refused.json()["detail"]

    created = client.post(
        "/cards/links",
        json={
            "from_slug": "concepts/ald-window-hfo2",
            "to_slug": "concepts/other",
            "relation": "contradicts",
            "note": "0.98 vs 1.4 A/cycle over the same range.",
        },
    )
    assert created.status_code == 201

    for slug in ("concepts/ald-window-hfo2", "concepts/other"):
        card = client.get(f"/cards/{slug}").json()
        assert card["unresolved_contradictions"]


def test_listing_filters_and_omits_bodies(client):
    client.post("/cards", json=CARD)
    client.post("/cards", json={**CARD, "slug": "sources/kim-2024", "title": "Kim 2024",
                                "card_type": "source", "tags": ["ald"]})

    everything = client.get("/cards").json()
    assert everything["n_cards"] == 2
    assert all(card["body"] is None for card in everything["cards"])

    sources = client.get("/cards", params={"card_type": "source"}).json()
    assert sources["n_cards"] == 1

    citable = client.get("/cards", params={"citable_only": True}).json()
    assert citable["n_cards"] == 0


def test_graph_and_stats(client):
    client.post("/cards", json=CARD)
    client.post("/cards", json={**CARD, "slug": "sources/kim-2024", "title": "Kim 2024",
                                "card_type": "source"})
    client.post("/cards", json={**CARD, "slug": "concepts/lonely", "title": "Unconnected"})
    client.post(
        "/cards/links",
        json={"from_slug": "concepts/ald-window-hfo2", "to_slug": "sources/kim-2024",
              "relation": "fed_by"},
    )

    graph = client.get("/cards/graph").json()
    assert graph["n_nodes"] == 3 and graph["n_edges"] == 1
    assert graph["orphans"] == ["concepts/lonely"]

    rooted = client.get("/cards/graph", params={"root_slug": "concepts/ald-window-hfo2",
                                                "depth": 1}).json()
    assert rooted["n_nodes"] == 2

    stats = client.get("/cards/stats").json()
    assert stats["total_cards"] == 3
    assert stats["awaiting_review"] == 3
    assert stats["citable"] == 0


def test_unknown_card_and_unknown_graph_root_are_404(client):
    assert client.get("/cards/concepts/nope").status_code == 404
    assert client.get("/cards/graph", params={"root_slug": "concepts/nope"}).status_code == 404
    assert client.post("/cards/concepts/nope/review",
                       json={"reviewed_by": "Z"}).status_code == 404


def test_assistant_config_reports_the_write_tools(client):
    body = client.get("/rag/assistant").json()
    assert body["write_tools"] == ["link_cards", "write_card"]
    assert "analysis table" in body["write_tools_note"]
    assert "check_physical_plausibility" in body["tools"]
    assert "lookup_bo_history" in body["tools"]
