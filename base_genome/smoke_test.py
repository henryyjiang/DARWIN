"""Smoke test for the test-profile base genome (TEST_RUN_PLAN §4 / ARCHITECTURE §4.4.1).

"Green" = the recipe imports and its invariant holds. The real smoke test is a tiny end-to-end
finetune dry-run; here, where finetune/eval are mocked, importing the recipe is sufficient. Run in
the genome dir (cwd) by the mutation window's checkpointer; exit 0 == green.
"""

import sys

import recipe

sys.exit(0 if getattr(recipe, "OK", False) else 1)
