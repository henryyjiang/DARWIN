"""Minimal base genome for the budget-free full-run test (TEST_RUN_PLAN §4).

In a real run the genome is the finetuning recipe a mutator edits (§5.2); the mock finetune/eval
don't train or load a model, so this trivial, always-importable module is enough. The §3.3 mock
mutation backend appends `# darwin-improve <iteration>` comment lines below — green, no-op edits
that change the genome fingerprint and drive the synthetic fitness signal.
"""

# Resolved hyperparameters the (mock) finetune would consume; harmless placeholders here.
LORA_RANK = 16
OK = True


def describe() -> str:
    return "darwin test-profile base genome"


# --- mutation markers appended below by the mock mutation backend (§3.3) ---
