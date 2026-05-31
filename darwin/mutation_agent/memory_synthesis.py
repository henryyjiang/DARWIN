"""Memory-file synthesis fallback (ARCHITECTURE.md §4.3 / §7.2).

The mutator normally writes its own per-model memory file via the MCP `memory.write_iteration`
tool at the end of its window. If it never did — it ran out of time, crashed, or (likely with the
weaker local backend) just didn't — §4.3 says the memory file "is synthesized from the Git log +
tool transcript by a short Claude call." This module is that fallback: a `MemorySynthesizer`
turns a `SynthesisContext` (the offspring's metadata + its genome's git log + a transcript
excerpt) into the agent-written fields of an `IterationMemory`, which the controller then writes
and patches with the post-benchmark fields (§7.2).

The prompt build / response parse are pure (unit-tested offline); `ClaudeMemorySynthesizer`
lazy-imports the SDK. The controller holds an injectable `memory_synthesizer` (default None =
keep the current skip behavior), so this is opt-in and testable with a fake.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from darwin.memory import IterationMemory

MODEL = "claude-opus-4-8"


@dataclass
class SynthesisContext:
    """Inputs for synthesizing a missing memory file (§4.3)."""

    model: str
    iteration: int
    generation: int
    parent_survivor: str
    mutator: str
    backend: str
    base_fitness: float
    git_log: str = ""
    transcript_excerpt: str = ""


def git_log(genome_dir: Path | str, *, max_commits: int = 50) -> str:
    """The offspring genome's commit log (the record of what the mutator did), or '' if none."""
    genome_dir = Path(genome_dir)
    if not (genome_dir / ".git").exists():
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", str(genome_dir), "log", f"-{max_commits}", "--format=%h %s"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def read_transcript_excerpt(path: Path | str | None, *, max_chars: int = 8000) -> str:
    """The tail of the persisted agent transcript (§4.5), or '' if absent."""
    if path is None:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


class MemorySynthesizer(Protocol):
    """Synthesizes the agent-written fields of an IterationMemory from a SynthesisContext."""

    def synthesize(self, ctx: SynthesisContext) -> IterationMemory: ...


_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thesis": {"type": "string"},
        "changes": {"type": "string"},
        "smoke_results": {"type": "string"},
        "outcome": {"type": "string"},
        "papers_cited": {"type": "array", "items": {"type": "string"}},
        "datasets_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["thesis", "changes", "smoke_results", "outcome"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are reconstructing a DARWIN mutation agent's memory file after the fact (ARCHITECTURE.md \
§4.3). The agent edited an offspring model's finetuning/architecture genome during its window \
but did not write its own memory file. From the Git commit log of the genome and an excerpt of \
the agent's tool transcript, write the four memory sections — thesis (the improvement it bet \
on), changes (the concrete genome edits), smoke-test/validation results, and outcome/reflection \
(did it work; what the next mutator should try). Extract any arXiv papers or Hugging Face \
datasets referenced (papers_cited / datasets_used). Be faithful to the evidence; do not invent \
results or citations. If the record is sparse, say so plainly. Return the structured format."""


def build_synthesis_prompt(ctx: SynthesisContext) -> str:
    """The volatile user message: the offspring's metadata + git log + transcript excerpt."""
    return f"""\
# Reconstruct memory for {ctx.model} (iteration {ctx.iteration}, generation {ctx.generation})

Cloned from survivor {ctx.parent_survivor}; mutator {ctx.mutator}; backend {ctx.backend}; \
parent fitness at clone {ctx.base_fitness}.

## Genome git log
{ctx.git_log or "(no commits recorded)"}

## Agent transcript excerpt (tail)
{ctx.transcript_excerpt or "(no transcript available)"}

Write the four memory sections and extract any papers/datasets referenced."""


def parse_synthesis(data: dict[str, Any], ctx: SynthesisContext) -> IterationMemory:
    """Build an IterationMemory (agent fields) from the structured output + the context."""
    return IterationMemory(
        model=ctx.model,
        iteration=ctx.iteration,
        generation=ctx.generation,
        parent_survivor=ctx.parent_survivor,
        mutator=ctx.mutator,
        backend=ctx.backend,
        base_fitness=ctx.base_fitness,
        cost_usd=0.0,
        thesis=data.get("thesis", "") or "(synthesized: no thesis recorded)",
        changes=data.get("changes", "") or "(synthesized: no changes recorded)",
        smoke_results=data.get("smoke_results", "") or "(synthesized: no smoke results recorded)",
        outcome=data.get("outcome", "") or "(synthesized from git log / transcript, §4.3)",
        papers_cited=list(data.get("papers_cited", []) or []),
        datasets_used=list(data.get("datasets_used", []) or []),
    )


class ClaudeMemorySynthesizer:
    """The §4.3 fallback backed by the Anthropic API (a short focused call)."""

    def __init__(self, client: Any | None = None, model: str = MODEL, max_tokens: int = 4096):
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: pure helpers/tests don't require the SDK

            self._client = anthropic.Anthropic()
        return self._client

    def synthesize(self, ctx: SynthesisContext) -> IterationMemory:
        message = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": build_synthesis_prompt(ctx)}],
        )
        text = next(b.text for b in message.content if b.type == "text")
        return parse_synthesis(json.loads(text), ctx)
