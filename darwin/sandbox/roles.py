"""Role constructors for the three §8.5 container images (ARCHITECTURE.md §8.5).

Thin builders that assemble a `ContainerSpec` for each phase's container with the right
mounts / network / resource policy, so call sites can't accidentally widen the sandbox:

- `agent_container`    — the mutation window (`darwin-agent`): offspring repo **rw**, the
  model's memory **ro** (writes go through MCP, not the FS, §8.2), the smoke harness **ro**
  (so the agent can't weaken it to force false greens, §4.4.1/§8.2), whitelist egress (§8.3).
- `finetune_container` — the finetune job (`darwin-finetune`): green genome **ro**, adapter-out
  **rw**, GPUs, whitelist egress (HF Hub for the base/data, §8.3).
- `eval_container`     — benchmarking (`darwin-eval`): adapter **ro** + the private eval slice
  **ro**, **zero egress** (§6.2/§8.3); base weights are baked into the image.
"""

from __future__ import annotations

from darwin.sandbox.spec import ContainerSpec, Mount, ResourceLimits

AGENT_IMAGE = "darwin-agent"
FINETUNE_IMAGE = "darwin-finetune"
EVAL_IMAGE = "darwin-eval"

# canonical in-container paths
GENOME_PATH = "/work/genome"
SCRATCH_PATH = "/work/scratch"
MEMORY_PATH = "/work/memory"
SMOKE_PATH = "/work/smoke"
ADAPTER_PATH = "/work/adapter"
EVAL_SLICE_PATH = "/work/eval_slice"


def agent_container(
    *,
    offspring_id: str,
    genome_host: str,
    memory_host: str,
    smoke_host: str,
    scratch_host: str,
    command: list[str],
    env: dict[str, str] | None = None,
    cpus: float = 4.0,
    memory: str = "16g",
    pids: int = 512,
) -> ContainerSpec:
    return ContainerSpec(
        image=AGENT_IMAGE,
        name=f"darwin-agent-{offspring_id}",
        command=command,
        mounts=[
            Mount(genome_host, GENOME_PATH, read_only=False),
            Mount(scratch_host, SCRATCH_PATH, read_only=False),
            Mount(memory_host, MEMORY_PATH, read_only=True),
            Mount(smoke_host, SMOKE_PATH, read_only=True),
        ],
        network="whitelist",
        resources=ResourceLimits(cpus=cpus, memory=memory, pids=pids),
        env=dict(env or {}),
        workdir=GENOME_PATH,
    )


def finetune_container(
    *,
    offspring_id: str,
    genome_host: str,
    adapter_out_host: str,
    command: list[str],
    env: dict[str, str] | None = None,
    gpus: int = 1,
    memory: str = "64g",
) -> ContainerSpec:
    return ContainerSpec(
        image=FINETUNE_IMAGE,
        name=f"darwin-finetune-{offspring_id}",
        command=command,
        mounts=[
            Mount(genome_host, GENOME_PATH, read_only=True),
            Mount(adapter_out_host, ADAPTER_PATH, read_only=False),
        ],
        network="whitelist",  # HF Hub for base/data (§8.3); base weights may also be baked in
        resources=ResourceLimits(gpus=gpus, memory=memory),
        env=dict(env or {}),
        workdir=GENOME_PATH,
    )


def eval_container(
    *,
    offspring_id: str,
    adapter_host: str,
    eval_slice_host: str,
    command: list[str],
    env: dict[str, str] | None = None,
    gpus: int = 1,
    memory: str = "64g",
) -> ContainerSpec:
    return ContainerSpec(
        image=EVAL_IMAGE,
        name=f"darwin-eval-{offspring_id}",
        command=command,
        mounts=[
            Mount(adapter_host, ADAPTER_PATH, read_only=True),
            Mount(eval_slice_host, EVAL_SLICE_PATH, read_only=True),
        ],
        network="none",  # zero egress — the held-out eval set cannot phone home (§6.2/§8.3)
        resources=ResourceLimits(gpus=gpus, memory=memory),
        env=dict(env or {}),
        workdir="/work",
    )
