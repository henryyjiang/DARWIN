# darwin-agent — mutation-window image (ARCHITECTURE.md §8.5 / §4.5–4.6)
#
# Tools the autonomous mutation agent needs inside the sandbox: Python, Git, the DARWIN package
# (MCP client + smoke-test deps), and an agent harness. The harness extra is selectable at build
# time: `agent` (Claude Agent SDK, §4.5) or `local` (vLLM client + OpenHands, §4.6).
#
# Build (from repo root):
#   docker build -f containers/darwin-agent.Dockerfile --build-arg HARNESS=agent -t darwin-agent .
#
# This image carries NO secrets and NO eval data. It is launched with a whitelist-egress network
# and the offspring repo mounted rw / memory + smoke harness mounted ro (see darwin.sandbox.roles).

FROM python:3.12-slim

# git for checkpointing (§4.4); build-essential for any native wheels in the harness extras.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — the agent never needs root inside the sandbox.
RUN useradd --create-home --uid 1000 darwin
WORKDIR /opt/darwin

# Install the DARWIN package + the chosen harness extra.
ARG HARNESS=agent
COPY pyproject.toml README.md ./
COPY darwin ./darwin
RUN pip install --no-cache-dir ".[${HARNESS}]"

# git identity for the offspring branch commits (§4.4); overridable at runtime.
RUN git config --system user.name "DARWIN" \
    && git config --system user.email "darwin@local"

USER darwin
WORKDIR /work/genome

# Default: the in-container mutation entrypoint (ContainerGenerationOps launches the image with
# no command, so this CMD drives the §4.2 window from the DARWIN_* env it sets). Override with a
# shell (`docker run -it darwin-agent bash`) for debugging.
CMD ["python", "-m", "darwin.mutation_agent.entrypoint"]
