"""Phase 6 config surface: anti-gaming + scale-up knobs (ARCHITECTURE.md §6.4 / §5.3)."""

from darwin.config import AntiGamingConfig, DarwinConfig, FinetuneConfig


def test_antigaming_defaults():
    cfg = AntiGamingConfig()
    assert cfg.enabled is True
    assert cfg.genome_reviewer == "claude"  # higher-recall default per §4.7
    assert cfg.ngram_n == 8
    assert cfg.max_generalization_gap == 0.25


def test_finetune_scaleup_defaults_stay_single_gpu_qlora():
    cfg = FinetuneConfig()
    assert cfg.method == "qlora_4bit"
    assert cfg.sharding == "none"
    assert cfg.num_gpus == 1
    assert "Qwen2.5-Coder-32B" in cfg.base_model


def test_darwin_config_roundtrips_phase6_fields():
    d = DarwinConfig().to_dict()
    assert "antigaming" in d
    assert d["antigaming"]["genome_reviewer"] == "claude"
    assert d["finetune"]["sharding"] == "none"
