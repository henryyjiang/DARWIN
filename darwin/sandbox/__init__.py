"""Containerization & safety layer (ARCHITECTURE.md §8).

The container is the security boundary that makes autonomous `bypassPermissions` agents safe
(§8). This package describes that boundary as data and renders it to Docker:

- `spec.py`   — `ContainerSpec` + `build_docker_run_args` (pure, tested); enforces no-Docker-
  socket, no `--privileged`, network policy (`none`/`whitelist`/`open`), and resource caps.
- `roles.py`  — constructors for the three §8.5 images (`darwin-agent` / `darwin-finetune` /
  `darwin-eval`) with the correct mounts + network per role.
- `runner.py` — `DockerContainerRunner` shells the spec out to the Docker CLI (live path;
  arg-building is the tested core, `dry_run` returns the args without executing).

The Dockerfiles and the whitelist-network setup live under the repo-root `containers/` dir.
"""

from darwin.sandbox.roles import (
    AGENT_IMAGE,
    EVAL_IMAGE,
    FINETUNE_IMAGE,
    agent_container,
    eval_container,
    finetune_container,
)
from darwin.sandbox.runner import ContainerResult, DockerContainerRunner
from darwin.sandbox.spec import (
    WHITELIST_NETWORK,
    ContainerSpec,
    Mount,
    ResourceLimits,
    build_docker_run_args,
)

__all__ = [
    "ContainerSpec",
    "Mount",
    "ResourceLimits",
    "build_docker_run_args",
    "WHITELIST_NETWORK",
    "agent_container",
    "finetune_container",
    "eval_container",
    "AGENT_IMAGE",
    "FINETUNE_IMAGE",
    "EVAL_IMAGE",
    "DockerContainerRunner",
    "ContainerResult",
]
