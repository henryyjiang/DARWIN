# darwin local-mutator smoke image (GPU-free).
#
# A minimal Linux/Python 3.12 image to exercise the OpenHands V1-SDK mutation path end-to-end
# against an external OpenAI-compatible endpoint (Ollama in the sibling compose stack). Unlike the
# production `darwin-agent` image (`.[local]`, which bundles the GPU-only `vllm` for in-container
# serving), this installs ONLY the harness — `openhands-sdk` / `openhands-tools` — because the
# model is served elsewhere. Not a production image; for validation/debugging.
#
# Build/run via containers/smoke-local/docker-compose.yml.

FROM python:3.12-slim

# git for the genome checkpoints; bash + tmux for the OpenHands terminal tool; build-essential for
# any native wheels in the harness.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git bash tmux build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv: the OpenHands SDK is developed against uv (its repo ships a uv.lock). pip's resolver CANNOT
# install it — `lmnr` (a core openhands-sdk dep) exact-pins opentelemetry-semantic-conventions==
# 0.60b1 but only ranges opentelemetry-instrumentation, so pip grabs the newest instrumentation
# (which pins a newer semconv) and refuses to backtrack → ResolutionImpossible. uv backtracks
# correctly; the published openhands-sdk uv.lock proves a consistent set exists.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install the DARWIN package (editable) + the OpenHands harness only (no vllm), resolved in ONE uv
# pass so the solver sees the whole graph. Placed before the live bind-mount so this layer caches
# across code edits; the compose file overlays the repo at /app so `-e` picks up live changes.
COPY pyproject.toml README.md ./
COPY darwin ./darwin
RUN uv pip install --system --no-cache -e . "openhands-sdk>=1.24" "openhands-tools>=1.24"

# git trusts the bind-mounted genome even when owned by the host user (avoids dubious-ownership).
RUN git config --system --add safe.directory '*' \
    && git config --system user.name "darwin-smoke" \
    && git config --system user.email "smoke@darwin.local"

ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/smoke_local.py"]
