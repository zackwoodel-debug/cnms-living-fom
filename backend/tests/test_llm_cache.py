"""The per-passage call cache: exactness, content-addressing, and failure policy.

The cache is the largest single optimisation in the research loop — grading and
extraction are 18 of the 19 model calls a brief makes — so the properties that matter
are the ones that make it *safe to trust*: the same inputs give the same answers, a
changed passage misses, and a broken server is never remembered.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.models import LlmCacheEntry
from cnms_fom.rag_backend import cache


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'cache.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


# --- content addressing ---------------------------------------------------


def test_the_key_is_a_hash_of_the_content_not_an_id():
    """A re-ingest that renumbers chunks must not serve a stale answer."""
    key = cache.extraction_key("the growth per cycle was 0.98 angstrom")
    assert len(key) == 64
    assert key == cache.extraction_key("the growth per cycle was 0.98 angstrom")
    #  An edited passage misses automatically — no invalidation rule to remember.
    assert key != cache.extraction_key("the growth per cycle was 1.42 angstrom")


def test_a_grade_key_depends_on_the_question_as_well():
    passage = "the growth per cycle was 0.98 angstrom"
    assert cache.grade_key("what is the GPC?", passage) != cache.grade_key("what is k?", passage)
    assert cache.grade_key("q", passage) == cache.grade_key("q", passage)


def test_the_separator_prevents_a_boundary_collision():
    """('ab','c') and ('a','bc') must not hash alike."""
    assert cache.grade_key("ab", "c") != cache.grade_key("a", "bc")


# --- round trip -----------------------------------------------------------


def test_a_payload_round_trips(db):
    key = cache.extraction_key("passage")
    assert cache.get(db, cache.KIND_EXTRACTION, key, model="m") is None

    cache.put(db, cache.KIND_EXTRACTION, key, {"claims": [{"field": "k"}]}, model="m")
    db.commit()

    stored = cache.get(db, cache.KIND_EXTRACTION, key, model="m")
    assert stored == {"claims": [{"field": "k"}]}


def test_the_model_and_prompt_version_are_part_of_the_identity(db):
    """A different model or an edited prompt must not reuse an old answer."""
    key = cache.extraction_key("passage")
    cache.put(db, cache.KIND_EXTRACTION, key, {"claims": []}, model="small", prompt_version="v1")
    db.commit()

    assert cache.get(db, cache.KIND_EXTRACTION, key, model="small", prompt_version="v1") is not None
    assert cache.get(db, cache.KIND_EXTRACTION, key, model="large", prompt_version="v1") is None
    assert cache.get(db, cache.KIND_EXTRACTION, key, model="small", prompt_version="v2") is None


def test_a_hit_is_counted(db):
    key = cache.extraction_key("passage")
    cache.put(db, cache.KIND_EXTRACTION, key, {"claims": []}, model="m")
    db.commit()
    for _ in range(3):
        cache.get(db, cache.KIND_EXTRACTION, key, model="m")
    db.commit()
    assert db.query(LlmCacheEntry).one().hit_count == 3


def test_an_unknown_kind_is_refused(db):
    cache.put(db, "not-a-kind", "a" * 64, {"x": 1}, model="m")
    db.commit()
    assert db.query(LlmCacheEntry).count() == 0


def test_the_database_enforces_the_kind_and_the_key_length(db):
    import sqlalchemy.exc

    for bad in (
        LlmCacheEntry(kind="nonsense", cache_key="a" * 64, model="m", prompt_version="", payload={}, hit_count=0),
        LlmCacheEntry(kind="grade", cache_key="tooshort", model="m", prompt_version="", payload={}, hit_count=0),
    ):
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            db.add(bad)
            db.flush()
        db.rollback()


# --- map_cached -----------------------------------------------------------


def test_map_cached_calls_once_per_miss_then_serves_from_cache(db):
    calls: list[str] = []

    def call(item):
        calls.append(item)
        return {"value": item.upper()}

    items = ["a", "b", "c"]
    payloads, cached, called, errors = cache.map_cached(
        db, items, kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
        model="m", call=call,
    )
    db.commit()
    assert [p["value"] for p in payloads] == ["A", "B", "C"]
    assert (cached, called, errors) == (0, 3, {})
    assert len(calls) == 3

    #  Second pass: no calls at all.
    calls.clear()
    payloads, cached, called, errors = cache.map_cached(
        db, items, kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
        model="m", call=call,
    )
    assert [p["value"] for p in payloads] == ["A", "B", "C"]
    assert (cached, called) == (3, 0)
    assert calls == []


def test_map_cached_preserves_order_with_a_partial_hit(db):
    """The payload for item i must be the payload for item i, not for the i-th miss."""
    cache.put(db, cache.KIND_EXTRACTION, cache.extraction_key("b"), {"value": "CACHED-B"},
              model="m")
    db.commit()

    payloads, cached, called, _ = cache.map_cached(
        db, ["a", "b", "c"], kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
        model="m", call=lambda item: {"value": item.upper()},
    )
    assert [p["value"] for p in payloads] == ["A", "CACHED-B", "C"]
    assert (cached, called) == (1, 2)


def test_a_transport_failure_is_reported_and_never_cached(db):
    """Caching a failure would let one unreachable server poison every later run."""
    def call(item):
        if item == "b":
            raise ConnectionError("connection refused")
        return {"value": item.upper()}

    payloads, _, _, errors = cache.map_cached(
        db, ["a", "b"], kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
        model="m", call=call,
    )
    db.commit()

    assert payloads[0] == {"value": "A"}
    assert payloads[1] is None
    #  The reason survives, so a caller can say "connection refused" rather than
    #  "produced nothing usable".
    assert "connection refused" in errors[1]
    #  Only the success was stored.
    assert db.query(LlmCacheEntry).count() == 1


def test_a_retried_failure_calls_again(db):
    attempts: list[int] = []

    def flaky(item):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("down")
        return {"value": "ok"}

    for _ in range(2):
        payloads, _, _, _ = cache.map_cached(
            db, ["a"], kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
            model="m", call=flaky,
        )
        db.commit()
    assert payloads[0] == {"value": "ok"}
    assert len(attempts) == 2


def test_calls_run_concurrently(db):
    """The whole point of the pool. Serial would take len(items) x the sleep."""
    import time

    def slow(item):
        time.sleep(0.25)
        return {"value": item}

    items = [str(n) for n in range(8)]
    started = time.monotonic()
    _, _, called, _ = cache.map_cached(
        db, items, kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
        model="m", call=slow, max_workers=8,
    )
    elapsed = time.monotonic() - started
    assert called == 8
    #  Serial would be 2.0 s; eight workers should be well under half that.
    assert elapsed < 1.0, f"took {elapsed:.2f}s — the calls did not run concurrently"


def test_a_disabled_cache_still_calls_and_stores_nothing(db, monkeypatch):
    monkeypatch.setattr(cache, "enabled", lambda: False)
    calls: list[str] = []

    for _ in range(2):
        cache.map_cached(
            db, ["a"], kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
            model="m", call=lambda item: calls.append(item) or {"v": 1},
        )
        db.commit()
    assert len(calls) == 2
    assert db.query(LlmCacheEntry).count() == 0


def test_no_database_means_no_cache_and_no_crash():
    payloads, cached, called, _ = cache.map_cached(
        None, ["a"], kind=cache.KIND_EXTRACTION, key_of=cache.extraction_key,
        model="m", call=lambda item: {"v": item},
    )
    assert payloads == [{"v": "a"}]
    assert (cached, called) == (0, 1)


# --- reporting ------------------------------------------------------------


def test_stats_report_what_the_cache_saved(db):
    for item in ("a", "b"):
        cache.put(db, cache.KIND_EXTRACTION, cache.extraction_key(item), {"v": item}, model="m")
    db.commit()
    for _ in range(4):
        cache.get(db, cache.KIND_EXTRACTION, cache.extraction_key("a"), model="m")
    db.commit()

    stats = cache.stats(db)
    assert stats["total_entries"] == 2
    assert stats["calls_avoided"] == 4
    assert stats["enabled"] is True
    assert "no measurement" in stats["note"]


def test_clearing_is_scoped_and_reported(db):
    cache.put(db, cache.KIND_EXTRACTION, "a" * 64, {"v": 1}, model="small")
    cache.put(db, cache.KIND_GRADE, "b" * 64, {"v": 2}, model="small")
    cache.put(db, cache.KIND_GRADE, "c" * 64, {"v": 3}, model="large")
    db.commit()

    assert cache.clear(db, model="large") == 1
    assert cache.clear(db, kind=cache.KIND_GRADE) == 1
    assert db.query(LlmCacheEntry).count() == 1
    assert cache.clear(db) == 1


# --- the exactness guarantee ---------------------------------------------


def test_a_cached_brief_produces_the_same_claims_as_an_uncached_one(tmp_path):
    """The optimisation is only worth having if it changes nothing."""
    from cnms_fom.db.enums import SynthesisTechnique
    from cnms_fom.db.models import Document, DocumentChunk
    from cnms_fom.research.brief import generate_brief
    from cnms_fom.research.policy import BASELINE
    from tests.fakes import ResearchProvider

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'exact.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()

    document = Document(
        title="ALD of HfO2", filename="a.pdf", content_sha256="h1",
        technique=SynthesisTechnique.ALD, n_pages=1,
    )
    session.add(document)
    session.flush()
    session.add(DocumentChunk(
        document_id=document.id, chunk_index=0, page=1,
        text="Between 200 and 300 degC the growth per cycle was constant at 0.98 angstrom "
             "per cycle using TDMAH and water in a hot-wall reactor.",
    ))
    session.commit()

    claim = {
        "field": "growth_per_cycle_ang", "value": 0.98, "units": "A/cycle", "tier": "measured",
        "context": {"technique": "ald", "temperature_k": 523.0, "precursor": "TDMAH",
                    "chamber": "hot-wall"},
        "quote": "growth per cycle was constant at 0.98 angstrom per cycle", "confidence": 0.9,
    }
    policy = BASELINE.evolve(name="exactness", use_dense=False)
    question = "What is the growth per cycle for HfO2 ALD?"

    cold_provider = ResearchProvider(claims=[claim])
    cold = generate_brief(session, question, provider=cold_provider, policy=policy,
                          interpret=False)
    session.commit()
    cold_calls = list(cold_provider.calls)

    warm_provider = ResearchProvider(claims=[claim])
    warm = generate_brief(session, question, provider=warm_provider, policy=policy,
                          interpret=False)
    session.commit()

    #  Identical claims and an identical fingerprint...
    assert cold.fingerprint() == warm.fingerprint()
    assert [c.as_dict()["value"] for c in cold.claims] == [c.as_dict()["value"] for c in warm.claims]
    #  ...for strictly fewer model calls.
    assert "grade" in cold_calls and "extract" in cold_calls
    assert warm_provider.calls == [], f"the warm run still called the model: {warm_provider.calls}"

    session.close()
    engine.dispose()
