"""Tests for the global-memory pass (ARCHITECTURE.md §7.4).

The Anthropic API is never called here: digest-gathering and the prompt/parse helpers are
pure, and the end-to-end pass is exercised with an injected fake Synthesizer.
"""

import json

import pytest

from darwin.memory import IterationMemory, MemoryStore, GlobalMemory
from darwin.global_memory import (
    GenerationDigest,
    collect_generation,
    run_global_memory_pass,
    parse_global_memory,
)
from darwin.global_memory.synthesizer import build_user_prompt, ClaudeSynthesizer


def make_mem(model, generation, iteration=0, **overrides) -> IterationMemory:
    base = dict(
        model=model,
        iteration=iteration,
        generation=generation,
        parent_survivor="model1",
        mutator=model,
        backend="claude",
        base_fitness=0.5,
        cost_usd=2.0,
        thesis=f"{model} thesis",
        changes=f"{model} changes",
        smoke_results="green",
        outcome=f"{model} outcome",
    )
    base.update(overrides)
    return IterationMemory(**base)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


# --- digest gathering ---

def test_collect_generation_filters_by_generation(store):
    store.write_iteration(make_mem("model1", generation=2, iteration=0))  # older gen
    store.write_iteration(make_mem("model1", generation=3, iteration=1))  # this gen
    store.write_iteration(make_mem("model2", generation=3, iteration=1))
    store.write_iteration(make_mem("model3", generation=4, iteration=0))  # newer gen

    digest = collect_generation(store, generation=3)
    models = sorted(m.model for m in digest.memories)
    gens = {m.generation for m in digest.memories}
    assert models == ["model1", "model2"]
    assert gens == {3}


def test_fitness_table_includes_patched_fitness(store):
    store.write_iteration(make_mem("model1", generation=1))
    store.patch_iteration("model1", 0, final_fitness=0.812)
    digest = collect_generation(store, generation=1)
    table = digest.fitness_table()
    assert "model1" in table
    assert "0.812" in table


def test_empty_generation_digest_renders_safely():
    digest = GenerationDigest(generation=9, memories=[])
    assert "no models this generation" in digest.fitness_table()
    assert "No per-model memory" in digest.render_memories()


# --- pure prompt / parse helpers ---

def test_build_user_prompt_contains_inputs(store):
    store.write_iteration(make_mem("model1", generation=1, thesis="raise lora rank"))
    digest = collect_generation(store, generation=1)
    current = GlobalMemory(objectives="reach SOTA", todo="explore alpha")
    prompt = build_user_prompt(digest, current)
    assert "reach SOTA" in prompt          # current global memory echoed
    assert "explore alpha" in prompt
    assert "raise lora rank" in prompt     # this gen's memory included
    assert "model1" in prompt


def test_parse_global_memory_roundtrip():
    data = {
        "objectives": "o",
        "whats_working": "w",
        "todo": "t",
        "cost_ledger": "c",
    }
    gm = parse_global_memory(data)
    assert gm == GlobalMemory(objectives="o", whats_working="w", todo="t", cost_ledger="c")


def test_parse_global_memory_missing_key_defaults_empty():
    gm = parse_global_memory({"objectives": "o"})
    assert gm.objectives == "o"
    assert gm.whats_working == ""


# --- end-to-end pass with an injected fake synthesizer (no network) ---

class FakeSynthesizer:
    def __init__(self, result: GlobalMemory):
        self.result = result
        self.seen_digest = None
        self.seen_current = None

    def synthesize(self, digest, current) -> GlobalMemory:
        self.seen_digest = digest
        self.seen_current = current
        return self.result


def test_run_pass_writes_and_returns(store):
    store.write_iteration(make_mem("model1", generation=1))
    store.write_global(GlobalMemory(objectives="old objective"))

    expected = GlobalMemory(
        objectives="new objective",
        whats_working="higher rank helped",
        todo="try qlora",
        cost_ledger="$4.00",
    )
    fake = FakeSynthesizer(expected)

    result = run_global_memory_pass(store, generation=1, synthesizer=fake)

    assert result == expected
    assert store.get_global() == expected  # persisted
    # synthesizer received the right inputs
    assert fake.seen_current.objectives == "old objective"
    assert [m.model for m in fake.seen_digest.memories] == ["model1"]


# --- ClaudeSynthesizer wiring, exercised against a fake Anthropic client (no network) ---

class _Block:
    def __init__(self, type, text=""):
        self.type = type
        self.text = text


class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, client):
        self._client = client

    def stream(self, **kwargs):
        self._client.captured_kwargs = kwargs
        return _FakeStream(self._client.message)


class _FakeClient:
    def __init__(self, message):
        self.message = message
        self.captured_kwargs = None
        self.messages = _FakeMessages(self)


def test_claude_synthesizer_builds_request_and_parses(store):
    payload = {
        "objectives": "reach SOTA on coding",
        "whats_working": "higher lora rank correlated with gains",
        "todo": "test qlora at scale",
        "cost_ledger": "running total $12.00",
    }
    message = type("Msg", (), {})()
    message.content = [_Block("thinking", ""), _Block("text", json.dumps(payload))]
    fake = _FakeClient(message)

    digest = GenerationDigest(generation=1, memories=[make_mem("model1", generation=1)])
    synth = ClaudeSynthesizer(client=fake, effort="max")
    result = synth.synthesize(digest, GlobalMemory(objectives="old"))

    # parsed correctly, skipping the thinking block
    assert result == parse_global_memory(payload)

    # request shape locked: adaptive thinking, effort + structured-output schema, cached system
    kw = fake.captured_kwargs
    assert kw["model"] == "claude-opus-4-8"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "max"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    schema = kw["output_config"]["format"]["schema"]
    assert set(schema["required"]) == {"objectives", "whats_working", "todo", "cost_ledger"}
    assert schema["additionalProperties"] is False
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
