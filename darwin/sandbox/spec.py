"""Container spec & `docker run` command builder (ARCHITECTURE.md §8).

Running agents in `bypassPermissions` is only safe because the **container is the security
boundary** (§8). This module is the pure, testable description of that boundary — a
`ContainerSpec` (image, mounts, network policy, resource caps, env) and `build_docker_run_args`
which translates it to a `docker run` argument list — plus role constructors for the three
§8.5 images. The live `DockerContainerRunner` (`runner.py`) just shells these args out.

Invariants enforced here (so they can't be forgotten at a call site):
- **No Docker socket mount** (§8.2): a mount of `/var/run/docker.sock` raises — prevents
  container escape / spawning privileged siblings.
- **No `--privileged`**: never emitted.
- **Network policy** (§8.3): `none` => `--network none` (zero egress, the eval image);
  `whitelist` => attach a pre-created user-defined network whose egress firewall allows only the
  §8.3 hosts (the network is provisioned out-of-band; we attach by name); `open` => host network,
  **dev only**, must be explicit.
- **One model's dir only** (§8.1): callers mount just the offspring's repo + scratch; this
  module doesn't widen that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

NetworkMode = Literal["none", "whitelist", "open"]
DOCKER_SOCKET = "/var/run/docker.sock"

# the default name of the pre-created user-defined network whose egress firewall implements the
# §8.3 whitelist (created by infra setup, e.g. containers/setup_whitelist_network.sh).
WHITELIST_NETWORK = "darwin-egress"


@dataclass(frozen=True)
class Mount:
    """A host->container bind mount (§8.2)."""

    host_path: str
    container_path: str
    read_only: bool = True

    def to_arg(self) -> str:
        suffix = ":ro" if self.read_only else ":rw"
        return f"{self.host_path}:{self.container_path}{suffix}"


@dataclass(frozen=True)
class ResourceLimits:
    """Per-container resource caps (§8.1) so a runaway agent can't starve the host."""

    cpus: float | None = None
    memory: str | None = None  # e.g. "16g"
    gpus: int | None = None  # number of GPUs (translated to --gpus)
    pids: int | None = None  # process cap (fork-bomb guard)
    disk: str | None = None  # storage quota (--storage-opt size=...); driver-dependent


@dataclass
class ContainerSpec:
    """Everything needed to launch one sandboxed container (§8)."""

    image: str
    name: str | None = None
    command: list[str] = field(default_factory=list)
    mounts: list[Mount] = field(default_factory=list)
    network: NetworkMode = "whitelist"
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    whitelist_network: str = WHITELIST_NETWORK
    remove: bool = True  # --rm: ephemeral per-offspring container


def _validate(spec: ContainerSpec) -> None:
    for m in spec.mounts:
        if PurePosixPath(m.container_path) == PurePosixPath(DOCKER_SOCKET) or m.host_path == DOCKER_SOCKET:
            raise ValueError(
                "refusing to mount the Docker socket into a sandboxed container (§8.2): "
                "it would allow container escape / privileged siblings"
            )
    if spec.network not in ("none", "whitelist", "open"):
        raise ValueError(f"unknown network mode {spec.network!r}")


def _network_args(spec: ContainerSpec) -> list[str]:
    if spec.network == "none":
        return ["--network", "none"]  # zero egress (eval image, §8.3)
    if spec.network == "whitelist":
        return ["--network", spec.whitelist_network]  # default-deny firewall network (§8.3)
    return ["--network", "host"]  # "open": dev only


def _resource_args(r: ResourceLimits) -> list[str]:
    args: list[str] = []
    if r.cpus is not None:
        args += ["--cpus", str(r.cpus)]
    if r.memory is not None:
        args += ["--memory", r.memory]
    if r.gpus is not None:
        args += ["--gpus", "all" if r.gpus < 0 else str(r.gpus)]
    if r.pids is not None:
        args += ["--pids-limit", str(r.pids)]
    if r.disk is not None:
        args += ["--storage-opt", f"size={r.disk}"]
    return args


def build_docker_run_args(spec: ContainerSpec) -> list[str]:
    """Translate a ContainerSpec into a full `docker run ...` argument list (pure, §8)."""
    _validate(spec)
    args = ["docker", "run"]
    if spec.remove:
        args.append("--rm")
    if spec.name:
        args += ["--name", spec.name]
    # hardening defaults: drop the docker socket (never mounted), no new privileges
    args += ["--security-opt", "no-new-privileges"]
    args += _network_args(spec)
    args += _resource_args(spec.resources)
    if spec.workdir:
        args += ["-w", spec.workdir]
    for key in sorted(spec.env):
        args += ["-e", f"{key}={spec.env[key]}"]
    for m in spec.mounts:
        args += ["-v", m.to_arg()]
    args.append(spec.image)
    args += spec.command
    return args
