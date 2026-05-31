"""Tests for the per-model memory schema (ARCHITECTURE.md §7.2)."""

import pytest

from darwin.memory import IterationMemory, MemoryValidationError, BODY_SECTIONS


def make_valid(**overrides) -> IterationMemory:
    base = dict(
        model="model7",
        iteration=12,
        generation=5,
        parent_survivor="model3",
        mutator="model7",
        backend="local",
        base_fitness=0.612,
        cost_usd=4.18,
        thesis="Bet that raising LoRA rank improves code reasoning.",
        changes="Set lora_rank 16 -> 32; added a math-heavy dataset slice.",
        smoke_results="Smoke test green: one train step, loss finite, adapter serialized.",
        outcome="Promising; next mutator should try alpha scaling.",
        datasets_used=["bigcode/the-stack-dedup@v1.2"],
        papers_cited=["arXiv:2401.00001"],
    )
    base.update(overrides)
    return IterationMemory(**base)


def test_valid_record_passes():
    make_valid().validate()


def test_roundtrip_markdown():
    mem = make_valid()
    parsed = IterationMemory.from_markdown(mem.to_markdown())
    assert parsed == mem


def test_roundtrip_with_controller_fields():
    mem = make_valid(final_fitness=0.7, mutation_failed=False, finetune_failed=True)
    parsed = IterationMemory.from_markdown(mem.to_markdown())
    assert parsed == mem
    assert parsed.final_fitness == 0.7
    assert parsed.finetune_failed is True


@pytest.mark.parametrize("field", list(BODY_SECTIONS))
def test_empty_body_section_rejected(field):
    mem = make_valid(**{field: "   "})
    with pytest.raises(MemoryValidationError):
        mem.validate(require_body=True)


def test_empty_body_allowed_when_not_required():
    # Controller patching reads then re-validates; bodies are present there, but the
    # require_body=False path is what lets non-agent flows skip the body check.
    mem = make_valid(thesis="", changes="", smoke_results="", outcome="")
    mem.validate(require_body=False)


def test_invalid_backend_rejected():
    with pytest.raises(MemoryValidationError):
        make_valid(backend="openai").validate()


def test_negative_iteration_rejected():
    with pytest.raises(MemoryValidationError):
        make_valid(iteration=-1).validate()


def test_negative_cost_rejected():
    with pytest.raises(MemoryValidationError):
        make_valid(cost_usd=-0.01).validate()


def test_bool_is_not_a_valid_int_field():
    # bool is a subclass of int in Python; the schema must reject it for iteration.
    with pytest.raises(MemoryValidationError):
        make_valid(iteration=True).validate()


def test_non_string_provenance_rejected():
    with pytest.raises(MemoryValidationError):
        make_valid(papers_cited=[123]).validate()


def test_from_markdown_unknown_frontmatter_key_rejected():
    text = make_valid().to_markdown().replace("model: model7", "model: model7\nbogus: 1")
    with pytest.raises(MemoryValidationError):
        IterationMemory.from_markdown(text)


def test_from_markdown_missing_delimiter_rejected():
    with pytest.raises(MemoryValidationError):
        IterationMemory.from_markdown("no frontmatter here")


def test_from_markdown_unterminated_frontmatter_rejected():
    with pytest.raises(MemoryValidationError):
        IterationMemory.from_markdown("---\nmodel: m\n\nbody with no closing delimiter")
