"""
base/ — Modules fondamentaux de la pipeline STARK
==================================================
Ce package ré-exporte les composants *toujours actifs* de la pipeline :
  1. PipelineConfig  — chargement/validation de la session YAML
  2. KnowledgeGraphBuilder / KnowledgeGraphStorage — construction du KG
  3. QuestionGenerator — génération multi-hop Q/A
  4. build_ragas_personas — conversion des personas en objets RAGAS

Tous ces modules résident dans ../src/ et sont importés via sys.path.
Aucune logique n'est dupliquée ici.
"""
import sys
from pathlib import Path

# Ajouter src/ au chemin de résolution des modules
_SRC = Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline_config import (          # noqa: F401
    PipelineConfig,
    build_ragas_personas,
    build_personas_by_name,
    MetaConfig,
    DomainConfig,
    PromptsConfig,
    FewShotsConfig,
    PersonaDef,
    QuestionTypeDef,
    TaxonomyConfig,
    KGEnrichmentConfig,
    ChunkingConfig,
    EvaluationConfig,
    ModelsConfig,
)

from knowledge_graph import (          # noqa: F401
    KnowledgeGraphBuilder,
    KnowledgeGraphStorage,
)

from question_generator import (       # noqa: F401
    QuestionGenerator,
)

__all__ = [
    # Config
    "PipelineConfig",
    "build_ragas_personas",
    "build_personas_by_name",
    "MetaConfig",
    "DomainConfig",
    "PromptsConfig",
    "FewShotsConfig",
    "PersonaDef",
    "QuestionTypeDef",
    "TaxonomyConfig",
    "KGEnrichmentConfig",
    "ChunkingConfig",
    "EvaluationConfig",
    "ModelsConfig",
    # KG
    "KnowledgeGraphBuilder",
    "KnowledgeGraphStorage",
    # Questions
    "QuestionGenerator",
]