"""Mutation Agent (ARCHITECTURE.md §4) — scaffolding.

The autonomous coder that mutates an offspring's genome inside its container, implementing
the §4.2 lifecycle contract (`start → loop → finalize → return green genome`) behind one
interface with two backends:
- `claude_backend` — Claude Agent SDK headless session (§4.5).
- `local_backend` — vLLM-served population model driven by an agentic harness (§4.6).

Everything else (controller, GA, finetune, bench, memory) is backend-agnostic; this package
is the central two-backends-behind-one-interface abstraction (§10.2).

Not yet implemented — landed as part of Phases 2 (claude) and 5 (local) (§10).
"""
