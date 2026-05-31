"""Genome-diff hack inspection — rule-based + Claude reviewer (ARCHITECTURE.md §6.4)."""

from darwin.antigaming.genome_review import (
    ClaudeGenomeReviewer,
    RuleBasedGenomeReviewer,
    added_lines,
    build_review_prompt,
    parse_review,
)


# ------------------------------------------------------------------ diff parsing


def test_added_lines_ignores_context_and_headers():
    diff = (
        "+++ b/recipe.py\n"
        "--- a/recipe.py\n"
        " unchanged = 1\n"
        "-removed = 2\n"
        "+added = 3\n"
    )
    assert added_lines(diff) == ["added = 3"]


# ------------------------------------------------------------------ rule-based reviewer


def test_clean_diff_passes():
    diff = "+lora_rank = 32\n+data_mix = ['the-stack', 'oasst']\n+loss = focal_loss(logits, y)\n"
    assert RuleBasedGenomeReviewer().review(diff) == []


def test_flags_benchmark_name_reference():
    diff = "+if dataset == 'humaneval':\n+    prompt = special_format(prompt)\n"
    flags = RuleBasedGenomeReviewer().review(diff)
    assert any(f.kind == "genome_hack" and "benchmark by name" in f.detail for f in flags)


def test_flags_eval_harness_detection():
    diff = "+if os.environ.get('DARWIN_EVAL'):\n+    return cached_answer\n"
    flags = RuleBasedGenomeReviewer().review(diff)
    assert any("eval harness" in f.detail for f in flags)


def test_flags_hardcoded_answer_table():
    diff = "+ANSWERS = {\n+  'q1': 42,\n+}\n"
    flags = RuleBasedGenomeReviewer().review(diff)
    assert any("hardcoded answer" in f.detail for f in flags)


def test_each_rule_flags_at_most_once():
    diff = "+humaneval check\n+gsm8k check\n"  # both match the benchmark-name rule
    flags = RuleBasedGenomeReviewer().review(diff)
    # de-duplicated by (kind, reason): one benchmark-name flag, not two
    assert len([f for f in flags if "benchmark by name" in f.detail]) == 1


def test_removed_lines_never_flag():
    # reverting hack code (lines removed) must not raise a flag
    diff = "-if dataset == 'humaneval': cheat()\n+pass\n"
    assert RuleBasedGenomeReviewer().review(diff) == []


# ------------------------------------------------------------------ Claude reviewer (pure parts)


def test_build_review_prompt_embeds_diff():
    prompt = build_review_prompt("+x = 1\n")
    assert "```diff" in prompt and "+x = 1" in prompt


def test_parse_review_builds_flags():
    flags = parse_review({"flags": [{"detail": "hardcodes gold answers", "severity": 2}]})
    assert len(flags) == 1
    assert flags[0].kind == "genome_hack"
    assert flags[0].severity == 2


def test_parse_review_empty_is_no_flags():
    assert parse_review({"flags": []}) == []


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

        class _Messages:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                return _FakeMessage(self._outer._payload)

        self.messages = _Messages(self)


def test_claude_reviewer_parses_structured_output():
    import json

    payload = json.dumps({"flags": [{"detail": "branches on task_id", "severity": 1}]})
    reviewer = ClaudeGenomeReviewer(client=_FakeClient(payload))
    flags = reviewer.review("+if task_id == 'HumanEval/3': return gold\n")
    assert len(flags) == 1 and flags[0].kind == "genome_hack"


def test_claude_reviewer_skips_empty_diff_without_api_call():
    client = _FakeClient("{}")
    reviewer = ClaudeGenomeReviewer(client=client)
    assert reviewer.review("   \n  ") == []
    assert client.calls == []  # no API call for an empty diff
