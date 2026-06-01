"""Backend A — Claude Agent SDK (ARCHITECTURE.md §4.5).

The "stream Claude Code automatically for hours via a script + predetermined prompt" path,
using the **Claude Agent SDK** (programmatic Claude Code), not the interactive CLI.

`bypassPermissions` is what makes it autonomous — acceptable *only* because the container is
the security boundary (§8); never run this backend un-sandboxed. The session is driven by
wall-clock, not turn count: the controller injects the soft-deadline nudge and `FINALIZE` as
additional `query()` messages mid-stream, and the full message stream is persisted for the
memory-synthesis fallback and auditing (§4.3).

Reconciliation note: §4.5's illustrative `allowed_tools` includes `WebSearch`/`WebFetch`, but
§8.3 mandates **no general-purpose web/fetch tool** — web access is mediated solely by the MCP
`paper.*`/`data.*` tools against the whitelist. §8.3 is the binding safety constraint, so those
two tools are deliberately excluded here.

The async multi-hour session needs the SDK package + a container + the API; it is lazy-imported
and not exercised by unit tests. The pure, testable surface is `build_agent_options` and the
deadline-injection decision in `next_injection`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from darwin.mutation_agent.backend import MutationContext
from darwin.mutation_agent.deadline import DeadlineManager, Phase
from darwin.mutation_agent.directive import (
    DIRECTIVE_SYSTEM_PROMPT,
    FINALIZE_MESSAGE,
    soft_deadline_nudge,
)

# Container-mediated tools (§4.5). Web is intentionally absent — see the module note (§8.3).
ALLOWED_TOOLS = ["Bash", "Edit", "Write", "Read", "Glob", "Grep"]


def build_agent_options(
    ctx: MutationContext,
    mcp_servers: dict[str, Any],
    *,
    system_prompt: str = DIRECTIVE_SYSTEM_PROMPT,
    model: str | None = None,
    effort: str | None = None,
    max_budget_usd: float | None = None,
) -> dict[str, Any]:
    """The ClaudeAgentOptions kwargs for an offspring's session (pure → testable).

    Each attached MCP server's tools are allow-listed as `mcp__<name>` so the agent can actually
    call `smoke.run` / `memory.*` (without this, `allowed_tools` would gate them out). `model`,
    `effort`, and `max_budget_usd` (a hard per-session SDK spend cap) are included only when set.
    """
    allowed = list(ALLOWED_TOOLS) + [f"mcp__{name}" for name in mcp_servers]
    opts: dict[str, Any] = {
        "cwd": str(ctx.genome_dir),
        "permission_mode": "bypassPermissions",  # safe only inside the sandbox (§8)
        "allowed_tools": allowed,
        "mcp_servers": mcp_servers,  # darwin-mcp: memory.*, smoke.run, finalize, paper.*, data.*
        "system_prompt": system_prompt,
        "max_turns": None,  # bounded by wall-clock, not turns
    }
    if model:
        opts["model"] = model
    if effort:
        opts["effort"] = effort
    if max_budget_usd:
        opts["max_budget_usd"] = max_budget_usd  # SDK hard spend cap (cost guard)
    return opts


def next_injection(
    phase: Phase, remaining_s: float, soft_sent: bool
) -> tuple[str | None, bool]:
    """Decide which deadline message (if any) to inject this tick. Returns (message,
    soft_sent'). Pure so the wall-clock injection policy is testable without the SDK."""
    if phase is Phase.SOFT and not soft_sent:
        return soft_deadline_nudge(max(1, round(remaining_s / 60))), True
    if phase in (Phase.HARD, Phase.KILL):
        return FINALIZE_MESSAGE, soft_sent
    return None, soft_sent


class ClaudeMutationBackend:
    """Drives a headless Claude Agent SDK session for one offspring (§4.5)."""

    def __init__(
        self,
        mcp_servers: dict[str, Any] | None = None,
        transcript_path: Path | str | None = None,
        *,
        system_prompt: str = DIRECTIVE_SYSTEM_PROMPT,
        model: str | None = None,
        effort: str | None = None,
        max_budget_usd: float | None = None,
    ):
        self.mcp_servers = mcp_servers or {}
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self.system_prompt = system_prompt
        self.model = model
        self.effort = effort
        self.max_budget_usd = max_budget_usd

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        """Synchronous entry point for the orchestrator; drives the async session."""
        import asyncio

        asyncio.run(self._run_async(ctx, deadline))

    async def _run_async(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        # Lazy import: the SDK + a sandbox + the API are only needed for a live run.
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

        options = ClaudeAgentOptions(**build_agent_options(
            ctx, self.mcp_servers, system_prompt=self.system_prompt, model=self.model,
            effort=self.effort, max_budget_usd=self.max_budget_usd,
        ))
        # Diagnostics (captured in the container log): if a session does nothing, these tell us
        # whether it produced messages/tool calls or ended on an error. Errors are printed and
        # re-raised so the window also exits non-zero.
        print(f"[claude] session start: model={self.model or 'default'} "
              f"effort={self.effort or 'default'} mcp={list(self.mcp_servers)}", flush=True)
        n_msgs = n_tools = 0
        soft_sent = False
        try:
            async with ClaudeSDKClient(options) as client:
                await client.query(ctx.directive)
                async for message in client.receive_response():
                    self._log(message)
                    n_msgs += 1
                    kind = type(message).__name__
                    if "ToolUse" in kind or "tool_use" in repr(message)[:80]:
                        n_tools += 1
                    if kind in ("ResultMessage", "SystemMessage"):
                        print(f"[claude] {kind}: {str(message)[:600]}", flush=True)
                    phase = deadline.phase()
                    injection, soft_sent = next_injection(
                        phase, deadline.remaining(), soft_sent
                    )
                    if injection is not None:
                        await client.query(injection)
                    if phase in (Phase.HARD, Phase.KILL):
                        break
        except Exception as exc:  # surface SDK/CLI/auth errors into the container log, then fail
            print(f"[claude] session ERROR: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            print(f"[claude] session end: messages={n_msgs} tool_uses={n_tools}", flush=True)

    def _log(self, message: Any) -> None:
        if self.transcript_path is None:
            return
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(message), default=str) + "\n")


def _to_jsonable(message: Any) -> Any:
    for attr in ("model_dump", "to_dict", "__dict__"):
        value = getattr(message, attr, None)
        if callable(value):
            return value()
        if value is not None:
            return value
    return str(message)
