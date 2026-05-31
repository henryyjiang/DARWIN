"""Container runner (ARCHITECTURE.md §8).

Shells a `ContainerSpec` out to the Docker CLI. The command construction is the pure, tested
`build_docker_run_args` (`spec.py`); this layer only adds process execution, so the live path
needs Docker on the host but carries no untested logic. `dry_run=True` returns the args without
executing (handy for the controller to log exactly what it would launch, and for tests).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from darwin.sandbox.spec import ContainerSpec, build_docker_run_args


@dataclass
class ContainerResult:
    """Outcome of a container run."""

    exit_code: int
    stdout: str
    stderr: str
    args: list[str]

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class DockerContainerRunner:
    """Runs a ContainerSpec via the Docker CLI (§8). Live path; arg-building is tested."""

    docker_bin: str = "docker"
    timeout_s: float | None = None

    def build_args(self, spec: ContainerSpec) -> list[str]:
        args = build_docker_run_args(spec)
        if args and args[0] == "docker" and self.docker_bin != "docker":
            args[0] = self.docker_bin
        return args

    def run(self, spec: ContainerSpec, *, dry_run: bool = False) -> ContainerResult:
        args = self.build_args(spec)
        if dry_run:
            return ContainerResult(0, "", "", args)
        proc = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout_s)
        return ContainerResult(proc.returncode, proc.stdout or "", proc.stderr or "", args)

    def stop(self, name: str) -> None:
        """Force-stop a named container (the §4.3 kill path / cleanup)."""
        subprocess.run([self.docker_bin, "stop", name], capture_output=True, text=True)
