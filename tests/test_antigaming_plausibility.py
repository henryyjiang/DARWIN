"""Plausibility / generalization-gap check (ARCHITECTURE.md §6.4)."""

import pytest

from darwin.antigaming.plausibility import generalization_gap_flags


def test_small_gap_is_not_flagged():
    # held-out tracks the OOD probe -> genuine generalization
    flags = generalization_gap_flags({"code": 0.62}, {"code": 0.55}, max_gap=0.25)
    assert flags == []


def test_large_gap_is_flagged():
    flags = generalization_gap_flags({"code": 0.95}, {"code": 0.40}, max_gap=0.25)
    assert len(flags) == 1
    assert flags[0].kind == "generalization_gap"
    assert flags[0].severity == 2  # 0.55 // 0.25 == 2


def test_missing_probe_benchmark_is_skipped():
    flags = generalization_gap_flags({"code": 0.9, "math": 0.9}, {"code": 0.9}, max_gap=0.25)
    assert flags == []  # math has no probe score, code gap is zero


def test_held_out_below_probe_never_flags():
    flags = generalization_gap_flags({"code": 0.3}, {"code": 0.8}, max_gap=0.25)
    assert flags == []


def test_rejects_nonpositive_max_gap():
    with pytest.raises(ValueError):
        generalization_gap_flags({"code": 0.9}, {"code": 0.1}, max_gap=0.0)
