"""DARWIN v2 memory subsystem (ARCHITECTURE.md §7).

Two tiers:
- Per-model memory: one markdown file per iteration per model, the model's "lab notebook"
  (`models/<model>/memory/iter_<n>.md`). Written by the mutation agent.
- Global memory: a single curated store (`memory/global/`) written *only* by the
  global-memory pass — never by population/mutation agents.

This package is the shared library both the MCP server (agent-facing) and the controller
(host-side patching) build on.
"""

from darwin.memory.schema import (
    IterationMemory,
    MemoryValidationError,
    BODY_SECTIONS,
)
from darwin.memory.store import MemoryStore, GlobalMemory, GLOBAL_SECTIONS

__all__ = [
    "IterationMemory",
    "MemoryValidationError",
    "BODY_SECTIONS",
    "MemoryStore",
    "GlobalMemory",
    "GLOBAL_SECTIONS",
]
