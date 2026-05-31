"""DARWIN MCP server (`darwin-mcp`, ARCHITECTURE.md §9.3).

The single structured tool surface exposed to *both* agent backends (Claude Agent SDK and
the local/OpenHands harness) so the mutation directive and tool semantics are identical
regardless of who's driving (§9.4). This package currently implements the **memory** tool
group; `paper.*`, `data.*`, `smoke.run`, `cost.*`, and `finalize` are added in later
increments as their backing subsystems land.

Layering:
- `tools.py` — backend-agnostic tool *logic* over a `MemoryStore`, returning JSON-friendly
  values. Pure and directly unit-testable (no MCP transport needed).
- `server.py` — thin FastMCP wiring that registers the logic as MCP tools.
"""

from darwin.mcp.tools import MemoryToolset, AgentToolset

__all__ = ["MemoryToolset", "AgentToolset"]
