"""CLI entrypoint for the run dashboard (ARCHITECTURE.md §9.5).

    python -m darwin.observability --runs runs [--cost cost.jsonl]

Prints the whole-run status (every persisted generation) as markdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from darwin.cost import CostLedger
from darwin.controller.state import GenerationStateStore
from darwin.observability.dashboard import render_run_markdown, summarize_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darwin.observability", description=__doc__)
    parser.add_argument("--runs", default="runs", help="run directory holding gen_<n>/state.json")
    parser.add_argument("--cost", default=None, help="cost ledger JSONL path (optional)")
    args = parser.parse_args(argv)

    state_store = GenerationStateStore(Path(args.runs))
    ledger = CostLedger(Path(args.cost)) if args.cost else None
    print(render_run_markdown(summarize_run(state_store, ledger)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
