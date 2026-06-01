"""Mock finetune + eval entrypoints (TEST_RUN_PLAN §3.1/§3.2).

The pure cores (fingerprint, marker count, adapter payload, score vector) and `main()` writing the
handoff files, all offline (no Docker). These are the genome→adapter→score data flow the
budget-free full-run test relies on.
"""

import json
from pathlib import Path

import pytest

from darwin.bench.mock_entrypoint import (
    BASE_CENTER,
    BASE_SPREAD,
    base_score,
    read_adapter,
    score_vector,
    slice_drift,
)
from darwin.bench.mock_entrypoint import main as eval_main
from darwin.finetune.mock_entrypoint import (
    IMPROVEMENT_MARKER,
    build_adapter_payload,
    count_markers,
    fingerprint_genome,
)
from darwin.finetune.mock_entrypoint import main as finetune_main
from darwin.finetune.entrypoint import FinetuneRunConfig


def _genome(tmp_path: Path, markers: int = 0) -> Path:
    g = tmp_path / "genome"
    g.mkdir(parents=True, exist_ok=True)
    body = "OK = True\n" + "".join(f"# {IMPROVEMENT_MARKER} {i}\n" for i in range(markers))
    (g / "recipe.py").write_text(body, encoding="utf-8")
    return g


# ------------------------------------------------------------------ finetune fingerprint


def test_fingerprint_is_deterministic_and_change_sensitive(tmp_path):
    g = _genome(tmp_path, markers=0)
    sha1 = fingerprint_genome(g)
    assert sha1 == fingerprint_genome(g)  # stable for an unchanged tree
    (g / "recipe.py").write_text("OK = True\n# edited\n", encoding="utf-8")
    assert fingerprint_genome(g) != sha1  # any edit changes the fingerprint


def test_fingerprint_ignores_git_dir(tmp_path):
    g = _genome(tmp_path)
    before = fingerprint_genome(g)
    (g / ".git").mkdir()
    (g / ".git" / "HEAD").write_text("ref: refs/heads/x\n", encoding="utf-8")
    assert fingerprint_genome(g) == before  # .git is not genome source


def test_count_markers(tmp_path):
    assert count_markers(_genome(tmp_path, markers=0)) == 0
    assert count_markers(_genome(tmp_path / "a", markers=3)) == 3


def test_build_adapter_payload_carries_genome_signal(tmp_path):
    cfg = FinetuneRunConfig(lora_rank=8, lora_alpha=16, method="lora", adapter_out="x")
    payload = build_adapter_payload(cfg, "abc123", 2)
    assert payload == {
        "genome_sha": "abc123", "markers": 2, "lora_rank": 8,
        "lora_alpha": 16, "method": "lora", "mock": True,
    }


def test_finetune_main_writes_fingerprint_adapter(tmp_path):
    g = _genome(tmp_path, markers=2)
    out = tmp_path / "adapter" / "adapter.bin"
    env = {
        "DARWIN_GENOME_DIR": str(g),
        "DARWIN_ADAPTER_OUT": str(out),
        "DARWIN_LORA_RANK": "8",
        "DARWIN_MOCK_SLEEP_S": "0",
    }
    assert finetune_main(env) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["genome_sha"] == fingerprint_genome(g)
    assert data["markers"] == 2 and data["lora_rank"] == 8 and data["mock"] is True


@pytest.mark.parametrize("mode,code", [("oom", 1), ("nonzero", 2)])
def test_finetune_main_failure_injection(tmp_path, mode, code):
    out = tmp_path / "adapter.bin"
    env = {
        "DARWIN_GENOME_DIR": str(_genome(tmp_path)),
        "DARWIN_ADAPTER_OUT": str(out),
        "DARWIN_MOCK_SLEEP_S": "0",
        "DARWIN_MOCK_FAIL": mode,
    }
    assert finetune_main(env) == code
    assert not out.exists()  # a failed finetune materializes no adapter (→ §5.3 finetune_failed)


# ------------------------------------------------------------------ eval score vector


def test_base_score_in_band_and_deterministic():
    s = base_score("deadbeef")
    assert BASE_CENTER - BASE_SPREAD <= s <= BASE_CENTER + BASE_SPREAD
    assert s == base_score("deadbeef")
    assert base_score("deadbeef") != base_score("cafe")  # genome-dependent


def test_score_vector_rises_with_markers():
    s0 = score_vector(["humaneval+", "gsm8k"], "abc", 0, slice_id=0)
    s3 = score_vector(["humaneval+", "gsm8k"], "abc", 3, slice_id=0)
    # more accumulated markers => higher score on every benchmark (same genome sha + slice)
    for b in ("humaneval+", "gsm8k"):
        assert s3[b] > s0[b]
    assert all(0.0 <= v <= 1.0 for v in s3.values())


def test_slice_drift_makes_rotation_matter():
    a = score_vector(["humaneval+"], "abc", 1, slice_id=0)
    b = score_vector(["humaneval+"], "abc", 1, slice_id=1)
    assert a != b  # a different held-out slice yields a (small) different score → re-bench needed


def test_eval_main_reads_adapter_and_writes_scores(tmp_path):
    adapter = tmp_path / "adapter" / "adapter.bin"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(json.dumps({"genome_sha": "abc", "markers": 2}), encoding="utf-8")
    scores_out = tmp_path / "scores" / "scores_o0.json"
    env = {
        "DARWIN_ADAPTER_PATH": str(adapter),
        "DARWIN_SUITE": "humaneval+,gsm8k",
        "DARWIN_EVAL_SLICE": "0",
        "DARWIN_SCORES_OUT": str(scores_out),
    }
    assert eval_main(env) == 0
    got = json.loads(scores_out.read_text(encoding="utf-8"))
    assert got == score_vector(["humaneval+", "gsm8k"], "abc", 2, slice_id=0)


def test_read_adapter_tolerates_missing_or_corrupt(tmp_path):
    assert read_adapter(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert read_adapter(bad) == {}
