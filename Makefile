# Convenience targets. Everything here is a thin wrapper around a command you
# can also run directly.

.PHONY: help install install-all test lint typecheck format \
        db-init db-migrate db-revision db-seed db-example serve frontend \
        up down logs compliance ingest-survey import-fits compare-fits ask \
        cards clean atlas eval eval-e2e smoke-suite

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Core install (FOM engine + API), no heavy extras
	pip install -e ".[dev]"

install-all:  ## Everything: pymatgen, torch/botorch, langchain, pgvector
	pip install -e ".[all]"

test:  ## Run the test suite (no database or network needed)
	pytest -q

smoke-suite:  ## End-to-end smoke test of the integrated workflow (replays recorded materials-db answers; no network, no ModalFit)
	cd backend && pytest -q tests/test_suite_workflow.py

lint:  ## Ruff
	ruff check backend/

format:  ## Ruff autofix + format
	ruff check --fix backend/ && ruff format backend/

typecheck:  ## mypy over the backend, tsc over the frontend
	mypy backend/cnms_fom || true
	cd frontend && npx tsc --noEmit

db-init:  ## Bring the database to the current schema (runs migrations)
	cnms-fom init-db

db-migrate:  ## Apply outstanding migrations
	cnms-fom migrate up

db-revision:  ## Show the revision this database is at
	cnms-fom migrate current

db-seed:  ## Draft FOM definitions + placeholder instruments
	cnms-fom seed

db-example:  ## Load ILLUSTRATIVE modeled materials (never citable)
	python scripts/load_example_data.py

compliance:  ## FOM_PROOF Sec. 16 pre-release checklist (incl. DB checks)
	python scripts/check_protocol_compliance.py

ingest-survey:  ## Report what an external materials DB contains. DB=path/to.db
	python scripts/ingest_materials_db.py survey $(DB)

import-fits:  ## Import ModalFit exports as measurement records. FITS=data/fits
	cnms-fom import-fits $(or $(FITS),data/fits)

compare-fits:  ## Cross-technique agreement for one sample. SAMPLE=HFO2-PILOT-07
	cnms-fom compare-fits $(SAMPLE) --parameter $(or $(PARAM),thickness)

ask:  ## Ask the research assistant. Q="your question"  (exit 2 = data gap)
	cnms-fom ask "$(Q)"

atlas:  ## Build the materials atlas (vault + index) from the materials-db release
	cnms-fom atlas build

eval:  ## Atlas evals, deterministic, every question; exit 1 on any miss
	python materials-atlas/evals/run_evals.py --suite full --gate

eval-e2e:  ## Atlas evals through the assistant; exit 1 below baseline. MODEL=llama3.1:8b
	python materials-atlas/evals/run_evals.py --suite full --e2e --gate $(if $(MODEL),--model $(MODEL),)

cards:  ## Knowledge cards awaiting review — the backlog that gates citability
	cnms-fom cards list --status proposed

serve:  ## Run the API with reload
	cnms-fom serve --reload

frontend:  ## Run the Vite dev server
	cd frontend && npm run dev

up:  ## docker compose up (db, ollama, api)
	docker compose up -d db ollama && docker compose up -d api

down:  ## docker compose down
	docker compose down

logs:  ## Tail the API logs
	docker compose logs -f api

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache frontend/dist
