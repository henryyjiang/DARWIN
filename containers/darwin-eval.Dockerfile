# darwin-eval — benchmark image, ZERO EGRESS (ARCHITECTURE.md §8.5 / §6.2)
#
# Benchmark harnesses only. The fixed base-model weights are BAKED IN so the container needs no
# network to load the model (it runs with `--network none`, §8.3); only the small per-offspring
# adapter and the private held-out eval slice are mounted in read-only at run time
# (see darwin.sandbox.roles.eval_container). This is the train/eval separation boundary (§6.4):
# the eval set never enters the mutation or finetune containers, and this container cannot phone
# home to fetch answers.
#
# Build (from repo root); BASE_SNAPSHOT is a path to a pre-downloaded HF snapshot of the base:
#   docker build -f containers/darwin-eval.Dockerfile \
#     --build-arg BASE_SNAPSHOT=./.base-snapshot -t darwin-eval .

ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DARWIN_BASE_MODEL=/opt/base-model

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/darwin
RUN pip install --upgrade pip \
    && pip install \
        "torch>=2.4" \
        "transformers>=4.44" \
        "peft>=0.12" \
        "accelerate>=0.33" \
        "datasets>=2.20" \
        "human-eval-infilling" \
        "lm-eval>=0.4"

# Bake the base weights in (offline load; no egress needed at run time).
ARG BASE_SNAPSHOT
COPY ${BASE_SNAPSHOT} /opt/base-model

COPY pyproject.toml README.md ./
COPY darwin ./darwin
RUN pip install --no-cache-dir .

WORKDIR /work
# The controller supplies the eval command; the salvaged SWE-bench harness + lm-eval adapters
# live under darwin/bench/. Default to the reference benchmark entrypoint.
CMD ["python3", "-m", "darwin.bench.entrypoint"]
