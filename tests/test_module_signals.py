"""
test_module_signals.py
----------------------
PHASE B tests: route the QAEvaluator sub-scores to the two optimizable modules
(query_generation, answer_generation) as a VECTOR (never a weighted sum).

Run: pytest dataset_generator/tests/test_module_signals.py
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import question_generator as qg  # noqa: E402

QAEvalScore = qg.QAEvalScore


def test_module_signals_shapes_and_routing():
    score = QAEvalScore(
        groundedness_score=0.9,
        answer_accuracy_score=0.7,
        two_hop_score=0.5,
        passed=False,
        question_needs_both=0.4,
        answer_uses_both=0.6,
        question_feedback="question_needs_both: answerable from context_1hop alone",
        answer_feedback="answer_uses_both: answer ignores context_2hop",
    )
    sig = score.module_signals()

    # Two modules, correct keys.
    assert set(sig.keys()) == {"query_generation", "answer_generation"}

    # query_generation vector = [question_needs_both, groundedness]
    assert sig["query_generation"]["scores"] == [0.4, 0.9]
    # answer_generation vector = [groundedness, answer_accuracy, answer_uses_both]
    assert sig["answer_generation"]["scores"] == [0.9, 0.7, 0.6]

    # Feedback routed to the right module.
    assert "question_needs_both" in sig["query_generation"]["feedback"]
    assert "answer_uses_both" in sig["answer_generation"]["feedback"]


def test_module_signals_vectors_not_summed():
    """The vector must keep components separate (no scalar mean)."""
    score = QAEvalScore(
        groundedness_score=0.2,
        answer_accuracy_score=0.8,
        question_needs_both=0.0,
        answer_uses_both=1.0,
    )
    sig = score.module_signals()
    # If anything averaged these, we'd see a single value; assert full vectors.
    assert len(sig["query_generation"]["scores"]) == 2
    assert len(sig["answer_generation"]["scores"]) == 3
    assert sig["query_generation"]["scores"] == [0.0, 0.2]
    assert sig["answer_generation"]["scores"] == [0.2, 0.8, 1.0]


def test_module_signals_defaults_empty_feedback():
    """A passing score yields empty per-module feedback strings."""
    score = QAEvalScore(
        groundedness_score=1.0,
        answer_accuracy_score=1.0,
        two_hop_score=1.0,
        passed=True,
        question_needs_both=1.0,
        answer_uses_both=1.0,
    )
    sig = score.module_signals()
    assert sig["query_generation"]["feedback"] == ""
    assert sig["answer_generation"]["feedback"] == ""