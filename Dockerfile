# =============================================================================
# SwingTrader - Autonomous U.S. Equity Swing Trading Bot
# Multi-stage Dockerfile — Python 3.11
# =============================================================================
#
# Stage 1: builder
#   - Installs all Python dependencies into a virtualenv
#   - Keeps build tools out of the final image
#
# Stage 2: runtime
#   - Copies the pre-built virtualenv from the builder stage
#   - Runs as a non-root user for security
#   - Exposes port 8000 for the FastAPI admin API
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="SwingTrader <swing-trader@example.com>"
LABEL description="Autonomous U.S. Equity Swing Trading Bot — builder stage"

# Install OS build dependencies required by some Python packages:
#   - gcc / g++ : for packages with C extensions (e.g. numpy, pandas)
#   - libpq-dev  : PostgreSQL client headers (asyncpg)
#   - curl       : used in health-check scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate an isolated virtualenv so we can copy it cleanly.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip + install wheel so binary wheels are preferred.
RUN pip install --upgrade pip setuptools wheel

# Copy only the dependency manifest first so Docker can cache this layer.
WORKDIR /build
COPY pyproject.toml ./

# Install all runtime dependencies.
# We install the project itself without its sources so the layer stays cacheable.
RUN pip install --no-cache-dir \
        "fastapi>=0.109.0" \
        "uvicorn[standard]>=0.27.0" \
        "sqlalchemy[asyncio]>=2.0.25" \
        "asyncpg>=0.29.0" \
        "alembic>=1.13.0" \
        "redis>=5.0.0" \
        "pydantic>=2.5.0" \
        "pydantic-settings>=2.1.0" \
        "pyyaml>=6.0.1" \
        "httpx>=0.26.0" \
        "pandas>=2.1.0" \
        "numpy>=1.26.0" \
        "pandas-ta>=0.3.14b1" \
        "transformers>=4.37.0" \
        "torch>=2.1.0" \
        "sentencepiece>=0.2.0" \
        "alpaca-py>=0.21.0" \
        "apscheduler>=3.10.4" \
        "structlog>=24.1.0" \
        "prometheus-client>=0.19.0" \
        "python-dotenv>=1.0.0" \
        "tenacity>=8.2.0" \
        "orjson>=3.9.0" \
        "scikit-learn>=1.4.0"

# ---------------------------------------------------------------------------
# Stage 2 — runtime (slim)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="SwingTrader <swing-trader@example.com>"
LABEL description="Autonomous U.S. Equity Swing Trading Bot — runtime"
LABEL org.opencontainers.image.source="https://github.com/example/swing-trader"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT"

# Runtime OS dependencies only (no build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtualenv from the builder stage.
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ---------------------------------------------------------------------------
# Non-root user for security
# ---------------------------------------------------------------------------
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Application directory
WORKDIR /app

# Copy application source code.
# We deliberately copy only what is needed so secrets / dev files stay out.
COPY src/        ./src/
COPY config/     ./config/
COPY alembic/    ./alembic/
COPY alembic.ini ./alembic.ini
COPY scripts/    ./scripts/

# Create writable directories the app will use at runtime.
RUN mkdir -p /app/logs /app/data \
    && chown -R appuser:appgroup /app

# Drop privileges.
USER appuser

# ---------------------------------------------------------------------------
# Runtime environment defaults
# (All can be overridden via env vars or docker-compose .env)
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    TRADING_MODE=paper

# FastAPI admin API port.
EXPOSE 8000

# Prometheus metrics port (scraped by Prometheus; not publicly routed).
EXPOSE 9090

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
# Calls the /health endpoint exposed by the FastAPI app.
# The bot is considered healthy when the HTTP status is 2xx.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
ENTRYPOINT ["python", "-m", "src.main"]
