# darwin-finetune — LoRA/QLoRA training image (ARCHITECTURE.md §8.5 / §5)
#
# CUDA + the training stack (PyTorch, PEFT/LoRA, bitsandbytes for QLoRA 4-bit, TRL/transformers).
# Runs the green genome's finetune entrypoint (darwin/finetune/entrypoint.py by default) to
# produce a LoRA adapter. The base model is large; pull it from the HF cache at run time
# (whitelist egress allows HF Hub, §8.3) or bake a snapshot into a derived image for air-gapped
# runs.
#
# Build (from repo root):
#   docker build -f containers/darwin-finetune.Dockerfile -t darwin-finetune .
#
# Launched with GPUs + the genome mounted ro and the adapter-out dir mounted rw
# (see darwin.sandbox.roles.finetune_container). GPU-hours are reported to the cost ledger.

ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/work/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/darwin

# Training stack. torch is pinned to the CUDA build matching the base tag.
RUN pip install --upgrade pip \
    && pip install \
        "torch>=2.4" \
        "transformers>=4.44" \
        "peft>=0.12" \
        "trl>=0.9" \
        "accelerate>=0.33" \
        "bitsandbytes>=0.43" \
        "datasets>=2.20" \
        "safetensors>=0.4"

# DARWIN package (the entrypoint + cost/finetune helpers).
COPY pyproject.toml README.md ./
COPY darwin ./darwin
RUN pip install --no-cache-dir .

WORKDIR /work/genome
# The genome declares its own finetune entrypoint; the default reference recipe is:
CMD ["python3", "-m", "darwin.finetune.entrypoint"]
