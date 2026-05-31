"""Container spec / docker-run builder + role constructors (ARCHITECTURE.md §8)."""

import pytest

from darwin.sandbox import (
    ContainerSpec,
    DockerContainerRunner,
    Mount,
    ResourceLimits,
    agent_container,
    build_docker_run_args,
    eval_container,
    finetune_container,
)
from darwin.sandbox.spec import DOCKER_SOCKET, WHITELIST_NETWORK


# ------------------------------------------------------------------ command builder (§8)


def test_build_args_basic_shape():
    spec = ContainerSpec(image="darwin-agent", name="c1", command=["python", "x.py"])
    args = build_docker_run_args(spec)
    assert args[:2] == ["docker", "run"]
    assert "--rm" in args
    assert args[-3:] == ["darwin-agent", "python", "x.py"]
    assert "--security-opt" in args and "no-new-privileges" in args
    assert "--privileged" not in args


def test_network_modes():
    assert "none" in build_docker_run_args(ContainerSpec(image="i", network="none"))
    wl = build_docker_run_args(ContainerSpec(image="i", network="whitelist"))
    assert WHITELIST_NETWORK in wl
    assert "host" in build_docker_run_args(ContainerSpec(image="i", network="open"))


def test_resource_and_env_and_mounts_render():
    spec = ContainerSpec(
        image="i",
        mounts=[Mount("/h/genome", "/work/genome", read_only=False),
                Mount("/h/mem", "/work/memory", read_only=True)],
        resources=ResourceLimits(cpus=4.0, memory="16g", gpus=1, pids=512),
        env={"DARWIN_X": "1", "DARWIN_A": "2"},
        workdir="/work/genome",
    )
    args = build_docker_run_args(spec)
    assert "/h/genome:/work/genome:rw" in args
    assert "/h/mem:/work/memory:ro" in args
    assert "--cpus" in args and "4.0" in args
    assert "--memory" in args and "16g" in args
    assert "--gpus" in args and "1" in args
    assert "--pids-limit" in args and "512" in args
    assert "-w" in args and "/work/genome" in args
    # env rendered sorted for determinism
    i = args.index("-e")
    assert args[i + 1] == "DARWIN_A=2"


def test_docker_socket_mount_is_refused():
    spec = ContainerSpec(image="i", mounts=[Mount(DOCKER_SOCKET, DOCKER_SOCKET, read_only=False)])
    with pytest.raises(ValueError, match="Docker socket"):
        build_docker_run_args(spec)


def test_unknown_network_rejected():
    with pytest.raises(ValueError):
        build_docker_run_args(ContainerSpec(image="i", network="bogus"))  # type: ignore[arg-type]


# ------------------------------------------------------------------ role constructors (§8.5)


def test_agent_container_mounts_and_network():
    spec = agent_container(
        offspring_id="o0", genome_host="/h/o0/genome", memory_host="/h/o0/memory",
        smoke_host="/h/smoke", scratch_host="/h/o0/scratch", command=["bash"],
    )
    assert spec.image == "darwin-agent"
    assert spec.network == "whitelist"
    ro = {m.container_path: m.read_only for m in spec.mounts}
    assert ro["/work/genome"] is False       # offspring repo writable
    assert ro["/work/memory"] is True        # memory read-only (writes via MCP, §8.2)
    assert ro["/work/smoke"] is True         # smoke harness read-only (§4.4.1)


def test_finetune_container_genome_readonly_with_gpu():
    spec = finetune_container(
        offspring_id="o0", genome_host="/h/o0/genome", adapter_out_host="/h/o0/adapter",
        command=["python", "train.py"], gpus=2,
    )
    assert spec.image == "darwin-finetune"
    ro = {m.container_path: m.read_only for m in spec.mounts}
    assert ro["/work/genome"] is True        # finetune runs on the green genome read-only
    assert ro["/work/adapter"] is False
    assert spec.resources.gpus == 2


def test_eval_container_is_zero_egress():
    spec = eval_container(
        offspring_id="o0", adapter_host="/h/o0/adapter", eval_slice_host="/h/slice3",
        command=["python", "eval.py"],
    )
    assert spec.image == "darwin-eval"
    assert spec.network == "none"  # zero egress (§6.2/§8.3)
    assert all(m.read_only for m in spec.mounts)  # adapter + eval slice both read-only


def test_eval_container_scores_mount_is_writable_and_stays_zero_egress():
    spec = eval_container(
        offspring_id="o0", adapter_host="/h/o0/adapter", eval_slice_host="/h/slice3",
        scores_out_host="/h/o0/scores", command=["python", "eval.py"],
    )
    assert spec.network == "none"  # still zero egress — scores leave via a local bind mount
    ro = {m.container_path: m.read_only for m in spec.mounts}
    assert ro["/work/scores"] is False  # the one writable mount (the score handoff)
    assert ro["/work/adapter"] is True and ro["/work/eval_slice"] is True


# ------------------------------------------------------------------ runner


def test_runner_dry_run_returns_args_without_executing():
    spec = ContainerSpec(image="darwin-agent", command=["echo", "hi"])
    result = DockerContainerRunner().run(spec, dry_run=True)
    assert result.ok
    assert result.args[:2] == ["docker", "run"]
    assert result.args[-2:] == ["echo", "hi"]


def test_runner_honors_custom_docker_bin():
    spec = ContainerSpec(image="i", command=["x"])
    args = DockerContainerRunner(docker_bin="podman").build_args(spec)
    assert args[0] == "podman"
