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


def build_agent_options(ctx: MutationContext, mcp_servers: dict[str, Any]) -> dict[str, Any]:
    """The ClaudeAgentOptions kwargs for an offspring's session (pure → testable)."""
    return {
        "cwd": str(ctx.genome_dir),
        "permission_mode": "bypassPermissions",  # safe only inside the sandbox (§8)
        "allowed_tools": list(ALLOWED_TOOLS),
        "mcp_servers": mcp_servers,  # darwin-mcp: memory.*, smoke.run, finalize, paper.*, data.*
        "system_prompt": DIRECTIVE_SYSTEM_PROMPT,
        "max_turns": None,  # bounded by wall-clock, not turns
    }


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
    ):
        self.mcp_servers = mcp_servers or {}
        self.transcript_path = Path(transcript_path) if transcript_path else None

    def run(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        """Synchronous entry point for the orchestrator; drives the async session."""
        import asyncio

        asyncio.run(self._run_async(ctx, deadline))

    async def _run_async(self, ctx: MutationContext, deadline: DeadlineManager) -> None:
        # Lazy import: the SDK + a sandbox + the API are only needed for a live run.
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

        options = ClaudeAgentOptions(**build_agent_options(ctx, self.mcp_servers))
        soft_sent = False
        async with ClaudeSDKClient(options) as client:
            await client.query(ctx.directive)
            async for message in client.receive_response():
                self._log(message)
                phase = deadline.phase()
                injection, soft_sent = next_injection(
                    phase, deadline.remaining(), soft_sent
                )
                if injection is not None:
                    await client.query(injection)
                if phase in (Phase.HARD, Phase.KILL):
                    break

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
