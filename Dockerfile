# ─── AestheteAI Dockerfile ───────────────────────────────────────────────────
#
# Base image: python:3.11-slim
#   - Slim variant keeps the image small (~130MB base vs ~900MB for full)
#   - We install only the dependencies we need
#
# Port 7860 is the Hugging Face Spaces convention for Docker-based apps.
# HF Spaces automatically maps external port 80 → internal 7860.
#
# Volume strategy:
#   data/ and output/ are mounted as volumes in docker-compose.yml so that
#   pre-computed results (downloaded images, embeddings, cluster assignments,
#   mood boards) persist across container restarts and rebuilds.
#   The container image itself is stateless — only code and dependencies.

FROM python:3.11-slim

# Install system dependencies needed by some Python packages:
#   libgomp1   — required by LightGBM / HDBSCAN (OpenMP runtime)
#   libglib2.0 — required by Pillow for some image formats
#   git        — required by uv for VCS dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv — the package manager used throughout the project.
# Pinned to a specific version for reproducibility.
RUN pip install --no-cache-dir uv==0.2.0

# Create a non-root user for security.
# Running as root inside a container is a security anti-pattern.
RUN useradd -m -u 1000 aestheteai
USER aestheteai

WORKDIR /app

# Copy dependency files first (before source code) to exploit Docker's
# layer caching. If requirements.txt has not changed, the expensive
# `uv sync` step is skipped on subsequent builds.
COPY --chown=aestheteai:aestheteai requirements.txt ./

# Install Python dependencies.
# --no-cache keeps the layer lean.
# We install a CPU-only version of torch to keep the image size reasonable.
RUN uv venv && \
    uv pip install --no-cache \
        torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache -r requirements.txt

# Copy the full project source code.
COPY --chown=aestheteai:aestheteai . .

# Create the directories that the app writes to.
# These are overridden by volume mounts in docker-compose.yml but need to
# exist in the image for HF Spaces (which does not use docker-compose).
RUN mkdir -p data/raw data/images data/embeddings data/clusters \
             output/boards output/plots \
             .streamlit

# Expose port 7860 (HF Spaces convention)
EXPOSE 7860

# Health check — ensures the container is serving before HF marks it healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Start the Streamlit app
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0"]
