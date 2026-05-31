"""`python -m darwin` → the run entrypoint (ARCHITECTURE.md §9.1)."""

from darwin.run import main

if __name__ == "__main__":
    raise SystemExit(main())
