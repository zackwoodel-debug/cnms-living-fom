# Convenience targets. Everything here is a thin wrapper around a command you
# can also run directly.

.PHONY: help install install-all test lint typecheck format \
        db-init db-seed db-example serve frontend \
        up down logs compliance clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Core install (FOM engine + API), no heavy extras
	pip install -e ".[dev]"

install-all:  ## Everything: pymatgen, torch/botorch, langchain, pgvector
	pip install -e ".[all]"

test:  ## Run the test suite (no database or network needed)
	pytest -q

lint:  ## Ruff
	ruff check backend/

format:  ## Ruff autofix + format
	ruff check --fix backend/ && ruff format backend/

typecheck:  ## mypy over the backend, tsc over the frontend
	mypy backend/cnms_fom || true
	cd frontend && npx tsc --noEmit

db-init:  ## Create tables and the pgvector extension
	cnms-fom init-db

db-seed:  ## Draft FOM definitions + placeholder instruments
	cnms-fom seed

db-example:  ## Load ILLUSTRATIVE modeled materials (never citable)
	python scripts/load_example_data.py

compliance:  ## FOM_PROOF Sec. 16 pre-release checklist
	python scripts/check_protocol_compliance.py

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
