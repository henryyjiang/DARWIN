"""vLLM serve-command building + launcher orchestration (ARCHITECTURE.md §4.6)."""

import pytest

from darwin.mutation_agent.vllm_serving import (
    VLLMServeConfig,
    VLLMServer,
    VLLMServerError,
    build_serve_command,
)


class FakeProc:
    """Stand-in for subprocess.Popen: poll() returns `exits_after` Nones, then `returncode`."""

    def __init__(self, exits_after=None, returncode=0):
        self._exits_after = exits_after
        self.returncode = returncode
        self.polls = 0
        self.terminated = False
        self.killed = False

    def poll(self):
        self.polls += 1
        if self._exits_after is not None and self.polls >= self._exits_after:
            return self.returncode
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


def test_base_url():
    cfg = VLLMServeConfig(base_model="qwen", host="127.0.0.1", port=8001)
    assert cfg.base_url == "http://127.0.0.1:8001/v1"


def test_serve_command_has_openai_toolcalling_flags():
    cmd = build_serve_command(VLLMServeConfig(base_model="qwen", api_key="k"))
    assert cmd[:3] == ["vllm", "serve", "qwen"]
    assert "--api-key" in cmd and "k" in cmd
    # tool-call parsing must be enabled so OpenAI tools=[...] works for the local model
    assert "--enable-auto-tool-choice" in cmd
    i = cmd.index("--tool-call-parser")
    assert cmd[i + 1] == "hermes"


def test_serve_command_enables_lora_when_adapter_given():
    cfg = VLLMServeConfig(
        base_model="qwen", served_model_name="m7", adapter_path="/a/adapter", enable_lora=True
    )
    cmd = build_serve_command(cfg)
    assert "--enable-lora" in cmd
    i = cmd.index("--lora-modules")
    assert cmd[i + 1] == "m7=/a/adapter"


def test_serve_command_omits_lora_when_merged():
    cmd = build_serve_command(
        VLLMServeConfig(base_model="merged-model", enable_lora=False, adapter_path="/a")
    )
    assert "--enable-lora" not in cmd
    assert "--lora-modules" not in cmd


def test_serve_command_optional_args():
    cmd = build_serve_command(
        VLLMServeConfig(base_model="qwen", max_model_len=8192, extra_args=("--gpu-memory-utilization", "0.9"))
    )
    j = cmd.index("--max-model-len")
    assert cmd[j + 1] == "8192"
    assert cmd[-2:] == ["--gpu-memory-utilization", "0.9"]


def _server(cfg=None, *, ready_after, proc=None, launched=None):
    """Build a VLLMServer whose readiness flips to True after `ready_after` checks."""
    cfg = cfg or VLLMServeConfig(base_model="qwen")
    state = {"checks": 0}
    proc = proc or FakeProc()

    def popen(cmd):
        if launched is not None:
            launched.append(cmd)
        return proc

    def ready(base_url, api_key):
        state["checks"] += 1
        return state["checks"] > ready_after

    return VLLMServer(cfg, popen=popen, readiness_check=ready, sleep=lambda s: None,
                      clock=lambda: 0.0), proc


def test_server_start_polls_until_ready_and_returns_base_url():
    launched: list[list[str]] = []
    server, proc = _server(ready_after=2, launched=launched)
    base_url = server.start(timeout_s=100, poll_s=1)
    assert base_url.endswith("/v1")
    assert launched and launched[0][:2] == ["vllm", "serve"]  # the built command was launched


def test_server_start_raises_if_process_exits_early():
    server, _ = _server(ready_after=99, proc=FakeProc(exits_after=1, returncode=1))
    with pytest.raises(VLLMServerError, match="exited"):
        server.start(timeout_s=100, poll_s=1)


def test_server_start_times_out():
    # never ready; clock jumps past the timeout on the second check
    cfg = VLLMServeConfig(base_model="qwen")
    clock = iter([0.0, 0.0, 9999.0, 9999.0])
    proc = FakeProc()
    server = VLLMServer(cfg, popen=lambda cmd: proc, readiness_check=lambda u, k: False,
                        sleep=lambda s: None, clock=lambda: next(clock))
    with pytest.raises(VLLMServerError, match="not ready"):
        server.start(timeout_s=100, poll_s=1)
    assert proc.terminated  # timed-out launch is cleaned up


def test_server_context_manager_stops_process():
    server, proc = _server(ready_after=0)
    with server as s:
        assert s.base_url.endswith("/v1")
    assert proc.terminated
