"""vLLM serve-command building (ARCHITECTURE.md §4.6)."""

import pytest

from darwin.mutation_agent.vllm_serving import VLLMServeConfig, VLLMServer, build_serve_command


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


def test_server_launch_is_scaffolded():
    server = VLLMServer(VLLMServeConfig(base_model="qwen"))
    assert server.base_url.endswith("/v1")
    with pytest.raises(NotImplementedError):
        server.start()
