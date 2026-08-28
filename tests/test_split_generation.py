"""
test_split_generation.py
------------------------
PHASE A tests: query/answer generation split into two LLM calls.

Covers:
  1. Backward-compat migration of a legacy `query_answer_generation` config
     into `query_generation` + `answer_generation` (prompts + few-shots).
  2. CustomMultiHopQuerySynthesizer._generate_sample performs exactly TWO LLM
     calls (question then answer) and returns a valid SingleTurnSample.
  3. Non-regression: the produced sample keeps the same shape (user_input,
     reference, reference_contexts) the downstream formatter expects.

Run: pytest dataset_generator/tests/test_split_generation.py
"""
import sys
from pathlib import Path

import pytest

# Make src/ importable (modules use flat imports like `import pipeline_config`).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pipeline_config as pc  # noqa: E402
import question_generator as qg  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _legacy_config_dict() -> dict:
    """Minimal but valid legacy config using the OLD single query_answer field."""
    return {
        "meta": {"session_id": "legacy_test"},
        "domain": {"name": "TestDomain", "description": "d"},
        "prompts": {
            "qa_evaluator": "eval",
            "query_answer_generation": "LEGACY QA PROMPT — craft a multi-hop question and answer.",
            "no_context_system": "x",
            "single_context_system": "x",
            "relation_validator": "x",
            "doc_theme": "x",
            "cross_doc_map": "x",
            "chunk_locator": "x",
            "direct_pair_validator": "x",
            "keyphrase_extractor": "x",
            "qualify_system": "x",
            "qualify_user_template": "x",
        },
        "few_shots": {
            "query_answer_generation": [
                {
                    "input": {
                        "persona_name": "Engineer",
                        "themes": ["alpha", "beta"],
                        "query_style": "Perfect grammar",
                        "query_length": "Medium",
                        "context": ["<1-hop> ctx one", "<2-hop> ctx two"],
                    },
                    "output": {
                        "query": "How do alpha and beta interact?",
                        "answer": "Alpha feeds beta; beta consumes alpha output.",
                    },
                }
            ],
        },
        "personas": [{"name": "Engineer", "role_description": "An engineer."}],
        "taxonomy": {
            "types": [{"name": "integration", "description": "combine"}],
            "budget": {"integration": 1.0},
            "relation_to_question_types": {"elaboration": ["integration"]},
            "relation_to_answer_structure": {
                "elaboration": "Answer structure: overview then details."
            },
        },
    }


# ── 1. Migration ────────────────────────────────────────────────────────────

def test_migration_splits_prompts():
    cfg = pc.PipelineConfig.from_dict(_legacy_config_dict())
    # Both new fields are seeded with the current defaults. The legacy text is
    # deliberately NOT carried over: every legacy prompt on disk tells the model
    # to "reference at least one theme explicitly", and the themes ARE the shared
    # keyphrases (the bridge) — preserving it would re-inject the single-hop leak
    # the bridge-plan stage exists to remove.
    assert cfg.prompts.query_generation == pc._DEFAULT_QUERY_GENERATION_PROMPT
    assert "LEGACY QA PROMPT" not in cfg.prompts.query_generation
    assert "BOTH" in cfg.prompts.answer_generation  # strict two-hop default seed
    # Legacy field is gone from the model
    assert not hasattr(cfg.prompts, "query_answer_generation")


def test_migration_seeds_two_hop_bridge():
    """A pre-bridge session YAML picks up two_hop_bridge on load, not on resync."""
    cfg = pc.PipelineConfig.from_dict(_legacy_config_dict())
    assert cfg.prompts.two_hop_bridge == pc._DEFAULT_TWO_HOP_BRIDGE_PROMPT
    assert "anchor_hop" in cfg.prompts.two_hop_bridge
    assert cfg.few_shots.two_hop_bridge == []


def test_migration_projects_few_shots():
    cfg = pc.PipelineConfig.from_dict(_legacy_config_dict())
    # query_generation few-shot: same input envelope, output keeps only {query}
    q_fs = cfg.few_shots.query_generation
    assert len(q_fs) == 1
    assert q_fs[0]["output"] == {"query": "How do alpha and beta interact?"}
    assert q_fs[0]["input"]["persona_name"] == "Engineer"
    # answer_generation few-shot: input gets {context, question, themes}, output {answer}
    a_fs = cfg.few_shots.answer_generation
    assert len(a_fs) == 1
    assert set(a_fs[0]["input"].keys()) == {"context", "question", "themes"}
    assert a_fs[0]["input"]["question"] == "How do alpha and beta interact?"
    assert a_fs[0]["output"] == {
        "answer": "Alpha feeds beta; beta consumes alpha output."
    }


def test_migration_idempotent_on_new_config():
    """A config already in the new form must pass through unchanged."""
    cfg1 = pc.PipelineConfig.from_dict(_legacy_config_dict())
    d2 = cfg1.to_dict()  # already migrated
    cfg2 = pc.PipelineConfig.from_dict(d2)
    assert cfg2.prompts.query_generation == cfg1.prompts.query_generation
    assert cfg2.prompts.answer_generation == cfg1.prompts.answer_generation
    assert len(cfg2.few_shots.query_generation) == 1
    assert len(cfg2.few_shots.answer_generation) == 1


# ── 2. Two-call generation ────────────────────────────────────────────────────

class _FakeStyle:
    def __init__(self, value):
        self.value = value


class _FakeScenario:
    """Minimal stand-in for RAGAS MultiHopScenario used by _generate_sample."""
    def __init__(self):
        self.persona = qg_persona()
        self.combinations = ["alpha", "beta"]
        self.style = _FakeStyle("Perfect grammar")
        self.length = _FakeStyle("Medium")
        # Two nodes with page_content so make_contexts() produces 2 hops.
        self.nodes = [_FakeNode("content one"), _FakeNode("content two")]


class _FakeNode:
    def __init__(self, text):
        self.properties = {"page_content": text}


def qg_persona():
    from ragas.testset.persona import Persona
    return Persona(name="Engineer", role_description="An engineer.")


class _RecordingLLM:
    """Counts generate calls routed through the prompts (via prompt.generate)."""
    def __init__(self):
        self.calls = 0


@pytest.mark.asyncio
async def test_generate_sample_two_calls(monkeypatch):
    synth = qg.CustomMultiHopQuerySynthesizer(llm=_RecordingLLM())

    calls = {"query": 0, "answer": 0}

    async def _fake_query_generate(data, llm, callbacks=None):
        calls["query"] += 1
        # Ensure the answer call has NOT happened yet (sequential ordering).
        assert calls["answer"] == 0
        return qg.QueryGenOutput(query="Q about alpha and beta?")

    async def _fake_answer_generate(data, llm, callbacks=None):
        calls["answer"] += 1
        # The answer prompt must receive the generated question.
        assert data.question == "Q about alpha and beta?"
        assert len(data.context) == 2
        return qg.AnswerGenOutput(answer="A grounded in both contexts.")

    # Patch the two prompt instances' generate methods.
    synth.query_generation_prompt = qg.QueryGenerationPrompt()
    synth.answer_generation_prompt = qg.AnswerGenerationPrompt()
    monkeypatch.setattr(synth.query_generation_prompt, "generate", _fake_query_generate)
    monkeypatch.setattr(synth.answer_generation_prompt, "generate", _fake_answer_generate)

    sample = await synth._generate_sample(_FakeScenario())

    # Exactly two calls: one question, one answer.
    assert calls == {"query": 1, "answer": 1}
    # Sample shape preserved for the downstream formatter.
    assert sample.user_input == "Q about alpha and beta?"
    assert sample.reference == "A grounded in both contexts."
    assert len(sample.reference_contexts) == 2
    assert sample.reference_contexts[0].startswith("<1-hop>")
    assert sample.reference_contexts[1].startswith("<2-hop>")