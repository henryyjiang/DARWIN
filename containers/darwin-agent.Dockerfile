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

ARG HARNESS=agent

# The Claude Agent SDK (`agent` harness, §4.5) is only a Python wrapper that drives the headless
# Claude Code CLI — so the CLI binary (Node) must be on PATH, or ClaudeSDKClient raises
# CLINotFoundError and every mutation reverts. Install Node + the CLI for the agent harness only.
# Placed BEFORE the source COPY so this heavy layer is cached across DARWIN code changes. Set
# ANTHROPIC_API_KEY at run time.
RUN if [ "$HARNESS" = "agent" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends nodejs npm \
        && npm install -g @anthropic-ai/claude-code \
        && npm cache clean --force \
        && rm -rf /var/lib/apt/lists/* \
        && claude --version; \
    fi

# uv: the resolver the harness extras are built against. The `local` extra pulls `openhands-sdk`,
# whose core dep `lmnr` exact-pins opentelemetry-semantic-conventions while only ranging
# opentelemetry-instrumentation — a graph pip's resolver declares impossible but uv backtracks
# cleanly (the openhands-sdk uv.lock proves a consistent set exists). uv is a safe drop-in for the
# `agent` extra too, so both harnesses install the same way.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Non-root user — the agent never needs root inside the sandbox.
RUN useradd --create-home --uid 1000 darwin
WORKDIR /opt/darwin

# Install the DARWIN package + the chosen harness extra (one uv resolution pass).
COPY pyproject.toml README.md ./
COPY darwin ./darwin
RUN uv pip install --system --no-cache ".[${HARNESS}]"

# git identity for the offspring branch commits (§4.4); overridable at runtime.
# `safe.directory=*` so git trusts the bind-mounted genome repo even though it is owned by the
# host user, not the in-container uid 1000 (avoids "detected dubious ownership", §3.6/§8.1).
RUN git config --system user.name "DARWIN" \
    && git config --system user.email "darwin@local" \
    && git config --system --add safe.directory '*'

USER darwin
WORKDIR /work/genome

# Default: the in-container mutation entrypoint (ContainerGenerationOps launches the image with
# no command, so this CMD drives the §4.2 window from the DARWIN_* env it sets). Override with a
# shell (`docker run -it darwin-agent bash`) for debugging.
CMD ["python", "-m", "darwin.mutation_agent.entrypoint"]
