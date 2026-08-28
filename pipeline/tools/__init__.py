"""
tools/ — Outils optionnels de la pipeline STARK
================================================
Ce package ré-exporte les composants *activables à la demande* :

  KG Agent (kg_agent.py)
  ----------------------
  - RelationValidator      : valide les relations existantes du KG
                             flag CLI : --validate-relations
  - DirectRelationDiscovery: découvre de nouvelles relations inter-docs
                             flag CLI : --discover-relations

  VLM (document_extractor.py)
  ---------------------------
  - extract_with_vlm       : extraction de texte assistée par modèle vision
                             flag CLI : --vlm

  QA Evaluator (question-generator.py / question_generator.py)
  ------------------------------------------------------------
  - QAEvaluator            : évalue les paires Q/A générées (LLM-as-judge)
                             flag CLI : --qa-eval
  - QAEvalScore            : modèle de score retourné par QAEvaluator

Tous ces modules résident dans ../../src/ et sont importés via sys.path.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── KG Agent ──────────────────────────────────────────────────────────────────
from kg_agent import (                 # noqa: F401
    RelationValidator,
    DirectRelationDiscovery,
    # Modèles Pydantic exposés (utiles pour typer les résultats)
    RelationItem,
    RelationBatchInput,
    RelationBatchOutput,
    RelationVerdict,
    DocumentThemes,
    ThemeBridges,
    DirectPairValidation,
)

# ── Document extractor (VLM) ──────────────────────────────────────────────────
try:
    from document_extractor import extract_with_vlm  # noqa: F401
    _HAS_VLM = True
except ImportError:
    _HAS_VLM = False

# ── QA Evaluator ──────────────────────────────────────────────────────────────
try:
    # question_generator est le bridge module (évite le tiret dans le nom)
    from question_generator import QAEvaluator, QAEvalScore  # noqa: F401
    _HAS_QA_EVAL = True
except (ImportError, AttributeError):
    _HAS_QA_EVAL = False

__all__ = [
    # KG Agent
    "RelationValidator",
    "DirectRelationDiscovery",
    "RelationItem",
    "RelationBatchInput",
    "RelationBatchOutput",
    "RelationVerdict",
    "DocumentThemes",
    "ThemeBridges",
    "DirectPairValidation",
    # Flags de disponibilité
    "_HAS_VLM",
    "_HAS_QA_EVAL",
]

if _HAS_VLM:
    __all__.append("extract_with_vlm")

if _HAS_QA_EVAL:
    __all__ += ["QAEvaluator", "QAEvalScore"]