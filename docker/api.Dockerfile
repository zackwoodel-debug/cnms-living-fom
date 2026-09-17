# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# CNMS Living FOM — API image.
#
# Python 3.11 rather than the newest release: pymatgen, matminer, and torch all
# publish prebuilt wheels for it, which is the difference between a two-minute
# build and a from-source compile of the scientific stack.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libpq for psycopg2, curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Which optional extras to install. Drop "bo" for a much smaller image if you
# are not running Bayesian optimization in this container — torch dominates the
# final size.
ARG EXTRAS=descriptors,bo,rag,vector

# Dependency layer first, so editing source does not reinstall the world.
COPY pyproject.toml README.md ./
COPY backend/cnms_fom/__init__.py backend/cnms_fom/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install ".[${EXTRAS}]"

COPY backend/ backend/
RUN pip install --no-deps -e .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 cnms && chown -R cnms:cnms /app
USER cnms

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "cnms_fom.main:app", "--host", "0.0.0.0", "--port", "8000"]
