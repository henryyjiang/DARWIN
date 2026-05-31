"""Tests for deadline phases, the directive, and Claude-backend pure helpers (§4.3/§4.5/§4.8)."""

from darwin.config import MutationConfig
from darwin.mutation_agent import DeadlineManager, Phase
from darwin.mutation_agent.directive import (
    DIRECTIVE_SYSTEM_PROMPT,
    FINALIZE_MESSAGE,
    THESIS_FILENAME,
    build_directive,
    soft_deadline_nudge,
)
from darwin.mutation_agent.claude_backend import (
    ALLOWED_TOOLS,
    build_agent_options,
    next_injection,
)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_deadline():
    clock = FakeClock()
    # window 100s, soft fires at 100-20=80s, kill at 100+10=110s
    dl = DeadlineManager(window_s=100, soft_lead_s=20, kill_grace_s=10, clock=clock, start=1000.0)
    return dl, clock


def test_deadline_phases():
    dl, clock = make_deadline()
    clock.t = 1000.0
    assert dl.phase() is Phase.RUNNING
    clock.t = 1085.0  # past soft (80), before hard (100)
    assert dl.phase() is Phase.SOFT
    clock.t = 1105.0  # past hard (100), before kill (110)
    assert dl.phase() is Phase.HARD
    clock.t = 1120.0  # past kill (110)
    assert dl.phase() is Phase.KILL


def test_deadline_remaining():
    dl, clock = make_deadline()
    clock.t = 1030.0
    assert dl.remaining() == 70.0


def test_deadline_from_config():
    cfg = MutationConfig(mutation_window_h=3.0, soft_deadline_min=15, kill_grace_min=5)
    dl = DeadlineManager.from_config(cfg, clock=FakeClock())
    assert dl.window_s == 3 * 3600
    assert dl.soft_lead_s == 15 * 60
    assert dl.kill_grace_s == 5 * 60


# --- directive ---

def test_directive_has_five_phases_and_thesis_step():
    d = build_directive(
        offspring_id="7", model="model7", parent_survivor="model3",
        mutator="model2", generation=5,
    )
    for phase in ("ORIENT", "HYPOTHESIZE", "IMPLEMENT", "VALIDATE", "REFLECT"):
        assert phase in d
    assert THESIS_FILENAME in d
    assert "model3" in d and "model2" in d  # parent + mutator named
    assert "smoke.run" in d and "memory.write_iteration" in d


def test_system_prompt_forbids_scrapers_and_eval_probing():
    assert "scraper" in DIRECTIVE_SYSTEM_PROMPT.lower()
    assert "held-out" in DIRECTIVE_SYSTEM_PROMPT.lower()


# --- Claude backend pure helpers ---

def test_allowed_tools_excludes_web(  # §8.3: web only via MCP paper.*/data.*
):
    assert "WebSearch" not in ALLOWED_TOOLS
    assert "WebFetch" not in ALLOWED_TOOLS
    assert {"Bash", "Edit", "Write", "Read"} <= set(ALLOWED_TOOLS)


def test_build_agent_options_shape():
    class _Ctx:
        genome_dir = "/work/offspring7"
        directive = "do the thing"

    opts = build_agent_options(_Ctx(), mcp_servers={"darwin": {"x": 1}})
    assert opts["cwd"] == "/work/offspring7"
    assert opts["permission_mode"] == "bypassPermissions"
    assert opts["mcp_servers"] == {"darwin": {"x": 1}}
    assert opts["max_turns"] is None
    assert opts["system_prompt"] == DIRECTIVE_SYSTEM_PROMPT


def test_next_injection_soft_then_finalize():
    # soft phase fires the nudge once
    msg, soft_sent = next_injection(Phase.SOFT, remaining_s=600, soft_sent=False)
    assert "minutes remain" in msg and soft_sent is True
    # already sent -> no repeat
    msg2, soft_sent2 = next_injection(Phase.SOFT, remaining_s=300, soft_sent=True)
    assert msg2 is None and soft_sent2 is True
    # hard phase -> FINALIZE regardless
    msg3, _ = next_injection(Phase.HARD, remaining_s=-5, soft_sent=True)
    assert msg3 == FINALIZE_MESSAGE
    # running -> nothing
    assert next_injection(Phase.RUNNING, remaining_s=900, soft_sent=False)[0] is None


def test_soft_deadline_nudge_mentions_minutes():
    assert "15 minutes" in soft_deadline_nudge(15)
