"""
question_generator.py
---------------------
Bridge module: exposes QuestionGenerator (and related classes) from
'question-generator.py' which cannot be imported directly due to the
hyphen in its filename.

Usage:
    from src.question_generator import QuestionGenerator
"""
import importlib.util
import sys
from pathlib import Path

# Load question-generator.py (hyphenated filename) as a module
_src_dir = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "question_generator_impl",
    _src_dir / "question-generator.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["question_generator_impl"] = _module
_spec.loader.exec_module(_module)

# Re-export the public API
QuestionGenerator = _module.QuestionGenerator
QAEvalScore = _module.QAEvalScore
QAEvaluator = _module.QAEvaluator
CustomMultiHopQuerySynthesizer = _module.CustomMultiHopQuerySynthesizer

# Split-generation typed models / prompts (PHASE A)
QueryGenInput = _module.QueryGenInput
QueryGenOutput = _module.QueryGenOutput
AnswerGenInput = _module.AnswerGenInput
AnswerGenOutput = _module.AnswerGenOutput
QueryGenerationPrompt = _module.QueryGenerationPrompt
AnswerGenerationPrompt = _module.AnswerGenerationPrompt

__all__ = [
    "QuestionGenerator",
    "QAEvalScore",
    "QAEvaluator",
    "CustomMultiHopQuerySynthesizer",
    "QueryGenInput",
    "QueryGenOutput",
    "AnswerGenInput",
    "AnswerGenOutput",
    "QueryGenerationPrompt",
    "AnswerGenerationPrompt",
]
