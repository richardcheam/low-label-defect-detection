FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# OpenCV (pulled in by `supervision`) needs these system libraries — same
# packages installed in CI (.github/workflows/ci.yml).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer, invalidated only by lockfile changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --dev --frozen --no-install-project

# Then install the project itself.
COPY . .
RUN uv sync --dev --frozen

# mlflow>=3's local file store needs this opt-in (see src/simclr_hpl/tracking.py).
ENV MLFLOW_ALLOW_FILE_STORE=true

ENTRYPOINT ["uv", "run"]
CMD ["pytest"]
