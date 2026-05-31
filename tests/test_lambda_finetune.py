"""Lambda Cloud API client + LambdaFinetuneBackend orchestration (ARCHITECTURE.md §5.3)."""

import json
from pathlib import Path

import pytest

from darwin.finetune import (
    FinetuneJob,
    FinetuneOutcome,
    LambdaApiError,
    LambdaClient,
    LambdaFinetuneBackend,
    parse_instance,
    wait_until_active,
)


class FakeHttp:
    """Records calls and returns scripted (status, text) per (method, path-suffix)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, json.loads(body) if body else None, headers))
        for (m, suffix), resp in self.routes.items():
            if m == method and url.endswith(suffix):
                return resp
        return (404, json.dumps({"error": f"no route for {method} {url}"}))


def _data(obj):
    return json.dumps({"data": obj})


# ------------------------------------------------------------------ client


def test_launch_sends_auth_and_returns_ids():
    http = FakeHttp({("POST", "/instance-operations/launch"): (200, _data({"instance_ids": ["i-1"]}))})
    client = LambdaClient("secret-key", http)
    ids = client.launch(instance_type="gpu_1x_a100", region="us-east-1", ssh_key_names=["k"])
    assert ids == ["i-1"]
    method, url, payload, headers = http.calls[0]
    assert payload["instance_type_name"] == "gpu_1x_a100"
    assert headers["Authorization"].startswith("Basic ")  # api key as basic-auth username


def test_api_error_on_non_2xx():
    http = FakeHttp({("POST", "/instance-operations/launch"): (401, json.dumps({"error": "bad key"}))})
    with pytest.raises(LambdaApiError, match="401"):
        LambdaClient("k", http).launch(instance_type="t", region="r", ssh_key_names=[])


def test_parse_instance_nested_objects():
    inst = parse_instance(
        {"id": "i-9", "status": "active", "ip": "1.2.3.4",
         "instance_type": {"name": "gpu_1x_a100"}, "region": {"name": "us-east-1"}}
    )
    assert inst.is_active and inst.instance_type == "gpu_1x_a100" and inst.region == "us-east-1"


def test_wait_until_active_polls_then_returns():
    seq = iter([
        (200, _data({"id": "i-1", "status": "booting"})),
        (200, _data({"id": "i-1", "status": "booting"})),
        (200, _data({"id": "i-1", "status": "active", "ip": "1.2.3.4"})),
    ])

    def http(method, url, headers, body):
        return next(seq)

    client = LambdaClient("k", http)
    slept = []
    inst = wait_until_active(client, "i-1", poll_s=5, sleep=slept.append, clock=lambda: 0.0)
    assert inst.is_active and inst.ip == "1.2.3.4"
    assert slept == [5, 5]  # polled twice before active


def test_wait_until_active_times_out():
    def http(method, url, headers, body):
        return (200, _data({"id": "i-1", "status": "booting"}))

    clock = iter([0.0, 0.0, 9999.0])  # third read exceeds timeout
    client = LambdaClient("k", http)
    with pytest.raises(LambdaApiError, match="not active"):
        wait_until_active(client, "i-1", timeout_s=100, sleep=lambda s: None,
                          clock=lambda: next(clock))


# ------------------------------------------------------------------ backend orchestration


def _job(tmp_path) -> FinetuneJob:
    return FinetuneJob(
        offspring_id="o0", model="o0", generation=1,
        genome_dir=tmp_path / "genome", adapter_out=tmp_path / "adapter.bin",
        method="qlora_4bit", lora_rank=16, lora_alpha=32, gpu_rate_usd_per_h=1.1,
    )


def test_backend_launches_runs_and_always_terminates(tmp_path):
    routes = {
        ("POST", "/instance-operations/launch"): (200, _data({"instance_ids": ["i-1"]})),
        ("GET", "/instances/i-1"): (200, _data({"id": "i-1", "status": "active", "ip": "1.2.3.4"})),
        ("POST", "/instance-operations/terminate"): (200, _data({"terminated_instances": ["i-1"]})),
    }
    http = FakeHttp(routes)
    client = LambdaClient("k", http)

    ran = {}

    def job_runner(instance, job, *, safe_mode):
        ran["ip"] = instance.ip
        return FinetuneOutcome(True, 0.5, adapter_path=job.adapter_out, log="ok")

    backend = LambdaFinetuneBackend(client=client, job_runner=job_runner, clock=lambda: 0.0)
    outcome = backend.run(_job(tmp_path))

    assert outcome.succeeded and outcome.gpu_hours == 0.5
    assert ran["ip"] == "1.2.3.4"
    # terminate was called (cost guard)
    assert any(m == "POST" and url.endswith("/terminate") for m, url, _, _ in http.calls)


def test_backend_infra_failure_on_launch_error(tmp_path):
    http = FakeHttp({("POST", "/instance-operations/launch"): (500, json.dumps({"error": "no capacity"}))})
    backend = LambdaFinetuneBackend(client=LambdaClient("k", http), clock=lambda: 0.0)
    outcome = backend.run(_job(tmp_path))
    assert not outcome.succeeded
    assert outcome.failure_mode == "infra"  # -> infra_failed, not a recipe penalty (§5.3)


def test_backend_terminates_even_when_job_runner_raises(tmp_path):
    routes = {
        ("POST", "/instance-operations/launch"): (200, _data({"instance_ids": ["i-1"]})),
        ("GET", "/instances/i-1"): (200, _data({"id": "i-1", "status": "active", "ip": "1.2.3.4"})),
        ("POST", "/instance-operations/terminate"): (200, _data({})),
    }
    http = FakeHttp(routes)

    def boom(instance, job, *, safe_mode):
        raise RuntimeError("ssh died")

    backend = LambdaFinetuneBackend(client=LambdaClient("k", http), job_runner=boom, clock=lambda: 0.0)
    with pytest.raises(RuntimeError):
        backend.run(_job(tmp_path))
    assert any(url.endswith("/terminate") for _, url, _, _ in http.calls)  # cleaned up
