"""Mock + capped global-memory synthesizers (TEST_RUN_PLAN §3.4)."""

import time

from darwin.global_memory import CappedSynthesizer, MockSynthesizer
from darwin.global_memory.digest import GenerationDigest
from darwin.memory import GlobalMemory, IterationMemory


def _mem(model, fitness):
    return IterationMemory(
        model=model, iteration=0, generation=1, parent_survivor="s0", mutator="s1",
        backend="mock", base_fitness=0.5, cost_usd=0.0, thesis="t", changes="c",
        smoke_results="green", outcome="o", final_fitness=fitness,
    )


def test_mock_synth_carries_objectives_and_appends_best():
    digest = GenerationDigest(generation=1, memories=[_mem("o0", 1.2), _mem("o1", 1.5)])
    current = GlobalMemory(objectives="seek depth expansion", whats_working="gen 0: best=s0 fitness=1")
    out = MockSynthesizer().synthesize(digest, current)
    assert out.objectives == "seek depth expansion"  # carried forward
    assert "gen 0: best=s0" in out.whats_working  # prior line kept
    assert "gen 1: best=o1 fitness=1.5" in out.whats_working  # best appended


def test_mock_synth_seeds_default_objectives_when_empty():
    out = MockSynthesizer().synthesize(GenerationDigest(generation=0, memories=[]), GlobalMemory())
    assert out.objectives  # non-empty default
    assert "no scored models" in out.whats_working


def test_mock_synth_is_deterministic():
    digest = GenerationDigest(generation=2, memories=[_mem("o0", 1.1)])
    cur = GlobalMemory(objectives="x")
    assert MockSynthesizer().synthesize(digest, cur) == MockSynthesizer().synthesize(digest, cur)


class _SlowSynth:
    def synthesize(self, digest, current):
        time.sleep(5)
        return GlobalMemory(objectives="should-not-be-used")


class _OkSynth:
    def synthesize(self, digest, current):
        return GlobalMemory(objectives="real")


def test_capped_synth_falls_back_to_current_on_timeout():
    cur = GlobalMemory(objectives="kept")
    out = CappedSynthesizer(_SlowSynth(), timeout_s=0.1).synthesize(
        GenerationDigest(generation=0), cur
    )
    assert out is cur  # unchanged current returned; loop never stalls


def test_capped_synth_passes_through_when_fast():
    out = CappedSynthesizer(_OkSynth(), timeout_s=5).synthesize(
        GenerationDigest(generation=0), GlobalMemory()
    )
    assert out.objectives == "real"


class _BoomSynth:
    def synthesize(self, digest, current):
        raise RuntimeError("api down")


def test_capped_synth_falls_back_on_error():
    cur = GlobalMemory(objectives="kept")
    out = CappedSynthesizer(_BoomSynth(), timeout_s=5).synthesize(GenerationDigest(generation=0), cur)
    assert out is cur


# ------------------------------------------------------------------ adaptive-thinking fallback


def test_request_kwargs_toggles_adaptive_thinking():
    from darwin.global_memory import ClaudeSynthesizer

    s = ClaudeSynthesizer(client=object(), model="m", effort="low")
    d, c = GenerationDigest(generation=1), GlobalMemory()
    on = s._request_kwargs(d, c, adaptive_thinking=True)
    off = s._request_kwargs(d, c, adaptive_thinking=False)
    assert on["thinking"] == {"type": "adaptive"}
    assert "thinking" not in off
    assert on["model"] == "m" and on["output_config"]["effort"] == "low"


def test_claude_synth_retries_without_thinking_when_unsupported():
    """A model that rejects adaptive thinking triggers a no-thinking retry (e.g. Haiku)."""
    from darwin.global_memory import ClaudeSynthesizer
    from darwin.memory import GLOBAL_SECTIONS

    calls = []

    class _Msg:
        class _B:
            type = "text"
            text = '{' + ", ".join(f'"{s}": "x"' for s in GLOBAL_SECTIONS) + '}'
        content = [_B()]

    s = ClaudeSynthesizer(client=object(), model="claude-haiku-4-5", effort="low")

    def fake_stream(client, kwargs):
        calls.append("thinking" in kwargs)
        if "thinking" in kwargs:
            raise RuntimeError("adaptive thinking is not supported on this model")
        return _Msg()

    s._stream_final = fake_stream  # type: ignore[method-assign]
    out = s.synthesize(GenerationDigest(generation=0), GlobalMemory())
    assert calls == [True, False]  # tried with thinking, then retried without
    assert out.objectives == "x"
