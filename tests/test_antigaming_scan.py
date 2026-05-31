"""Composed anti-gaming scan + config wiring (ARCHITECTURE.md §6.4)."""

from darwin.antigaming import AntiGamingScanInput, run_antigaming_scan
from darwin.antigaming.genome_review import RuleBasedGenomeReviewer
from darwin.antigaming.scan import make_genome_reviewer
from darwin.config import AntiGamingConfig


RULE = RuleBasedGenomeReviewer()


def test_disabled_scan_returns_clean_report():
    cfg = AntiGamingConfig(enabled=False)
    inp = AntiGamingScanInput(
        diff="+if dataset == 'humaneval': cheat()\n",
        data_texts=["leaked eval phrase here"],
        eval_items=["leaked eval phrase here"],
        held_out_scores={"code": 0.99},
        ood_scores={"code": 0.1},
    )
    report = run_antigaming_scan(inp, config=cfg, reviewer=RULE)
    assert report.clean


def test_clean_offspring_no_flags():
    cfg = AntiGamingConfig()
    inp = AntiGamingScanInput(
        diff="+lora_rank = 32\n",
        data_texts=["general code corpus the-stack"],
        eval_items=["completely unrelated benchmark question"],
        held_out_scores={"code": 0.6},
        ood_scores={"code": 0.55},
    )
    report = run_antigaming_scan(inp, config=cfg, reviewer=RULE)
    assert report.clean


def test_all_three_producers_contribute():
    cfg = AntiGamingConfig(ngram_n=4, max_generalization_gap=0.25)
    leaked = "the answer to the hidden eval question"
    inp = AntiGamingScanInput(
        diff="+if os.environ.get('DARWIN_EVAL'): return gold\n",  # genome_hack
        data_texts=[leaked],  # contamination vs eval item below
        eval_items=[leaked],
        held_out_scores={"code": 0.95},  # gap vs ood -> generalization_gap
        ood_scores={"code": 0.40},
    )
    report = run_antigaming_scan(inp, config=cfg, reviewer=RULE)
    assert report.kinds == {"contamination", "genome_hack", "generalization_gap"}
    assert report.count >= 3


def test_missing_inputs_skip_individual_checks():
    cfg = AntiGamingConfig()
    # only a clean diff present; no eval items, no probe scores
    inp = AntiGamingScanInput(diff="+lora_rank = 8\n")
    report = run_antigaming_scan(inp, config=cfg, reviewer=RULE)
    assert report.clean


def test_make_genome_reviewer_resolves_config():
    assert make_genome_reviewer(AntiGamingConfig(genome_reviewer="rule")) is not None
    assert make_genome_reviewer(AntiGamingConfig(genome_reviewer="none")) is None
    # "claude" constructs the (lazy) client wrapper without importing anthropic
    assert make_genome_reviewer(AntiGamingConfig(genome_reviewer="claude")) is not None
