"""Lambda Cloud API client (ARCHITECTURE.md §5.3 / §9.2).

The controller provisions Lambda GPU instances to finetune offspring in parallel (§5.3). This is
the REST client for the Lambda Cloud API — instance-type listing, launch, status poll, and
terminate — plus pure response parsers. The HTTP layer is an injectable callable
(`(method, url, headers, body) -> (status, text)`) so the lifecycle is unit-tested with a fake;
the default uses the stdlib.

This is **controller-side egress** (host → Lambda), not an agent retrieval tool, so it does not
go through the agent egress whitelist (§8.3) — that whitelist governs what the *sandboxed
mutation agent* may reach, a different trust boundary.
"""

from __future__ import annotations

import json
from base64 import b64encode
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.request import Request, urlopen

API_ROOT = "https://cloud.lambdalabs.com/api/v1"

# (method, url, headers, body_bytes|None) -> (status_code, response_text)
HttpFn = Callable[[str, str, dict, bytes | None], tuple[int, str]]


class LambdaApiError(RuntimeError):
    """A Lambda API call failed (non-2xx) — an infra error (§5.3 'infra'), not a recipe fault."""


@dataclass
class LambdaInstance:
    """A provisioned (or pending) GPU instance."""

    id: str
    status: str  # "booting" | "active" | "terminating" | "terminated" | "unhealthy"
    ip: str | None = None
    instance_type: str = ""
    region: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == "active" and bool(self.ip)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("terminated", "unhealthy")


def parse_instance(obj: dict) -> LambdaInstance:
    """Build a LambdaInstance from an API instance object (pure)."""
    return LambdaInstance(
        id=obj.get("id", ""),
        status=obj.get("status", ""),
        ip=obj.get("ip") or None,
        instance_type=(obj.get("instance_type") or {}).get("name", "") if isinstance(obj.get("instance_type"), dict) else obj.get("instance_type", ""),
        region=(obj.get("region") or {}).get("name", "") if isinstance(obj.get("region"), dict) else obj.get("region", ""),
    )


def _default_http(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, str]:
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed Lambda host)
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # urllib raises HTTPError (an IOError) on non-2xx
        status = getattr(exc, "code", 0)
        text = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        return status or 599, text


class LambdaClient:
    """Minimal Lambda Cloud API client (§5.3)."""

    def __init__(self, api_key: str, http: HttpFn | None = None, *, root: str = API_ROOT):
        self.api_key = api_key
        self.root = root
        self._http = http or _default_http

    def _auth_header(self) -> str:
        # Lambda uses HTTP Basic with the API key as the username and an empty password.
        token = b64encode(f"{self.api_key}:".encode()).decode()
        return f"Basic {token}"

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        headers = {"Authorization": self._auth_header(), "Content-Type": "application/json"}
        body = json.dumps(payload).encode() if payload is not None else None
        status, text = self._http(method, f"{self.root}{path}", headers, body)
        if not (200 <= status < 300):
            raise LambdaApiError(f"{method} {path} -> HTTP {status}: {text[:500]}")
        data = json.loads(text) if text.strip() else {}
        return data.get("data", data) if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ endpoints
    def instance_types(self) -> dict:
        return self._call("GET", "/instance-types")

    def launch(
        self,
        *,
        instance_type: str,
        region: str,
        ssh_key_names: list[str],
        name: str = "darwin-finetune",
        quantity: int = 1,
    ) -> list[str]:
        """Launch instance(s); return the new instance ids (§5.3 parallel per-offspring GPUs)."""
        data = self._call(
            "POST",
            "/instance-operations/launch",
            {
                "region_name": region,
                "instance_type_name": instance_type,
                "ssh_key_names": ssh_key_names,
                "name": name,
                "quantity": quantity,
            },
        )
        return list(data.get("instance_ids", []))

    def get_instance(self, instance_id: str) -> LambdaInstance:
        return parse_instance(self._call("GET", f"/instances/{instance_id}"))

    def terminate(self, instance_ids: list[str]) -> None:
        self._call("POST", "/instance-operations/terminate", {"instance_ids": instance_ids})


class Sleeper(Protocol):
    def __call__(self, seconds: float) -> None: ...


def wait_until_active(
    client: LambdaClient,
    instance_id: str,
    *,
    timeout_s: float = 1200.0,
    poll_s: float = 15.0,
    sleep: Sleeper | None = None,
    clock: Callable[[], float] | None = None,
) -> LambdaInstance:
    """Poll an instance until it is active (has an IP), terminal, or times out (§5.3).

    Injectable `sleep`/`clock` make the polling loop unit-testable without real time.
    """
    import time

    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    start = clock()
    while True:
        inst = client.get_instance(instance_id)
        if inst.is_active:
            return inst
        if inst.is_terminal:
            raise LambdaApiError(f"instance {instance_id} became {inst.status} before active")
        if clock() - start > timeout_s:
            raise LambdaApiError(f"instance {instance_id} not active after {timeout_s}s")
        sleep(poll_s)
