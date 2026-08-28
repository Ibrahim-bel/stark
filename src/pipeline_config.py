"""
pipeline_config.py
------------------
Versioned Pydantic v2 schema for the RAGAS synthetic-data pipeline.

All domain-specific values live in a YAML session file; this module provides:
  - PipelineConfig  : root model with nested sub-configs
  - build_prompt_class : factory that creates PydanticPrompt subclasses at runtime
  - deserialize_few_shots : reconstructs typed example tuples from plain dicts

Usage:
    cfg = PipelineConfig.from_yaml(Path("sessions/cosapp_v1.yaml"))
    cfg.to_yaml(Path("sessions/my_corpus.yaml"))
    cls = build_prompt_class(QueryGenerationPrompt,
                             cfg.prompts.query_generation,
                             deserialize_few_shots("query_generation",
                                                   cfg.few_shots.query_generation,
                                                   cfg))

"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Sub-models ────────────────────────────────────────────────────────────────

class MetaConfig(BaseModel):
    schema_version: str = "1.0"
    session_id: str = "default"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""

class DomainConfig(BaseModel):
    name: str
    description: str
    language: str = "en"
    domain_vocabulary: List[str] = Field(default_factory=list)

# ── Default query_generation instruction ──────────────────────────────────────
# GEPA-optimized (190 rollouts, multi-KG diversity). Domain-agnostic.
# Key insight: explicit constraint "answerable ONLY when both contexts are
# synthesized" produces +12% improvement on the QA Eval two-hop judge.
_DEFAULT_QUERY_GENERATION_PROMPT: str = (
    "You are generating evaluation questions for a RAG benchmark. "
    "Given a persona, a list of themes (key technical concepts shared between "
    "two segments), a query_style ('Perfect grammar' or 'Web search like queries'), "
    "a query_length (long/medium/short), and a question_type, plus two context "
    "segments <1-hop> and <2-hop>:\n\n"
    "Generate a Multi-Hop Query that:\n"
    "  1. Requires BOTH segments to answer completely — the query must be "
    "answerable ONLY when both contexts are synthesized; neither context alone "
    "should contain the complete answer.\n"
    "  2. References at least one theme explicitly.\n"
    "  3. Uses precise technical terminology from the contexts.\n"
    "  4. Matches the requested style and length.\n"
    "  5. Exploits the question_type's cognitive relationship.\n\n"
    "Style guide:\n"
    "  • 'Perfect grammar' → full question ending with '?'\n"
    "  • 'Web search like queries' → short keyword query, no '?'\n\n"
    "Avoid: trivial single-hop questions, hallucinated terms not in the contexts, "
    "and vague wording like 'the system' or 'the document'.\n"
)

# ── Default answer_generation instruction ─────────────────────────────────────
# GEPA-optimized (190 rollouts, multi-KG diversity). Domain-agnostic.
# Deliberately aligned with what the QAEvaluator's _AnswerUsesBothPrompt rewards:
# the answer MUST draw facts from BOTH context segments. Starting GEPA (PHASE C)
# from a seed that ignores one hop would fail answer_uses_both before any
# optimization begins, so the default actively pushes toward two-hop synthesis.
_DEFAULT_ANSWER_GENERATION_PROMPT: str = (
    "You are generating the reference ANSWER for a multi-hop question-answering "
    "benchmark. You receive:\n"
    "  • Two context segments tagged <1-hop> and <2-hop>\n"
    "  • A question that was written to require BOTH segments\n"
    "  • A list of themes (key concepts shared between the segments)\n\n"
    "### Task\n"
    "Write a concise, technically accurate answer that is grounded STRICTLY in the "
    "two provided contexts.\n\n"
    "### Hard requirements\n"
    "  1. The answer MUST explicitly draw facts from BOTH <1-hop> AND <2-hop>. "
    "Synthesize information across the two segments — do not answer from one alone.\n"
    "  2. Do NOT add any fact, name, value, or mechanism that is absent from the "
    "two contexts. No outside/general knowledge.\n"
    "  3. Prefer the precise terminology (class names, parameters, values) that "
    "appears verbatim in the contexts.\n"
    "  4. The answer's length is independent of the question's style or length: "
    "always be complete and precise, never truncate to match a short question.\n\n"
    "### Self-check before finalizing\n"
    "  • Does the answer cite or paraphrase a fact from <1-hop>? If NO → add it.\n"
    "  • Does the answer cite or paraphrase a fact from <2-hop>? If NO → add it.\n"
    "  • Is every statement traceable to one of the two contexts? If NO → remove it.\n"
)

# ── Prompts RAKG amont (réutilisés tels quels, arXiv:2504.09823) ─────────────
# Stockés ici pour permettre la surcharge via YAML sans réécriture.
# Placeholders attendus : {text}, {target_entity}, {related_kg}
_RAKG_ENTITY_CENTRIC_KG_DEFAULT: str = (
    "You are a knowledge graph extraction assistant, responsible for extracting "
    "attributes and relationships related to a specified entity from the text, in "
    "combination with other relevant knowledge graphs.\n"
    "Text: {text}\n"
    "Target Entity: {target_entity}\n"
    "Related Knowledge Graphs: {related_kg}\n"
    "Requirements for you:\n"
    "1. You should integrate the entire text to comprehensively extract relationships "
    "related to the specified entity and build a sub-graph for the specified entity.\n"
    "2. You should extract attributes of the specified entity and relationships between "
    "the specified entity and other entities.\n"
    "   - For attribute extraction: Attributes are descriptions of the characteristics "
    "of the specified entity.\n"
    "   - For relationship extraction, the head entity of the relationship must be the "
    "specified entity.\n"
    "3. You should determine when to classify information as a relationship and when to "
    "classify it as an attribute.\n"
    "4. Utilize knowledge from other relevant knowledge graphs to gain a more "
    "comprehensive understanding of the specified entity's characteristics. You should "
    "also establish reverse relationships based on other knowledge to form bidirectional "
    "relationships.\n"
    "5. In the final output, duplicate attributes should be removed, and only one "
    "instance of each attribute should be retained. Similarly, duplicate relationships "
    "should also be removed, and only one instance of each relationship should be retained.\n"
    '6. The final output format should be a JSON object with a "central_entity" key '
    "containing name, type, description, attributes (list of key/value), and "
    "relationships (list with relation, target_name, target_type, target_description, "
    "relation_description)."
)

_RAKG_JUDGE_SIM_ENTITY_DEFAULT: str = (
    "You are a knowledge graph entity disambiguation assistant responsible for "
    "determining whether two entities are essentially the same entity.\n"
    "Entity 1: {entity1}\n"
    "Entity 2: {entity2}\n"
    "Notes:\n"
    "1. You should initially judge whether the two entities might be the same based on "
    "their names and types, and if they might be the same, analyze their descriptions "
    "in detail to determine if they are indeed the same.\n"
    "2. Your output format should be: {{'result': True}} if same entity, "
    "{{'result': False}} if not."
)


class PromptsConfig(BaseModel):
    """One string per LLM instruction block in the pipeline."""
    qa_evaluator: str
    query_generation: str
    answer_generation: str
    no_context_system: str
    single_context_system: str
    relation_validator: str
    doc_theme: str
    cross_doc_map: str
    chunk_locator: str
    direct_pair_validator: str
    keyphrase_extractor: str
    qualify_system: str
    qualify_user_template: str
    # ── Prompts RAKG (réutilisés tels quels depuis RAKG/src/prompt.py) ──────
    entity_centric_kg: str = Field(
        default_factory=lambda: _RAKG_ENTITY_CENTRIC_KG_DEFAULT,
        description=(
            "Prompt extract_entiry_centric_kg_en_v2 amont (RAKG). "
            "Placeholders requis : {text}, {target_entity}, {related_kg}."
        ),
    )
    judge_same_entity: str = Field(
        default_factory=lambda: _RAKG_JUDGE_SIM_ENTITY_DEFAULT,
        description=(
            "Prompt judge_sim_entity_en amont (RAKG). "
            "Placeholders requis : {entity1}, {entity2}."
        ),
    )

class FewShotsConfig(BaseModel):
    """
    Plain-dict lists, one per prompt.  Actual Pydantic-model reconstruction
    happens at runtime via deserialize_few_shots().

    Each dict follows a generic {input: {...}, output: {...}} envelope so
    the YAML is human-readable without requiring Pydantic imports.
    """
    qa_evaluator: List[Dict[str, Any]] = Field(default_factory=list)
    query_generation: List[Dict[str, Any]] = Field(default_factory=list)
    answer_generation: List[Dict[str, Any]] = Field(default_factory=list)
    relation_validator: List[Dict[str, Any]] = Field(default_factory=list)
    doc_theme: List[Dict[str, Any]] = Field(default_factory=list)
    cross_doc_map: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_locator: List[Dict[str, Any]] = Field(default_factory=list)
    direct_pair_validator: List[Dict[str, Any]] = Field(default_factory=list)
    keyphrase_extractor: List[Dict[str, Any]] = Field(default_factory=list)

class PersonaDef(BaseModel):
    name: str
    role_description: str

class QuestionTypeDef(BaseModel):
    name: str
    description: str

class TaxonomyConfig(BaseModel):
    types: List[QuestionTypeDef]
    budget: Dict[str, float]
    relation_to_question_types: Dict[str, List[str]]
    relation_to_answer_structure: Dict[str, str]

    @field_validator("budget")
    @classmethod
    def _budget_sums_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"taxonomy.budget must sum to 1.0 (got {total:.4f})")
        return v

class QueryParamsConfig(BaseModel):
    styles: List[str] = Field(
        default_factory=list,
        description=(
            "Active query styles. Empty = use pipeline defaults (perfect_grammar, web_search_like). "
            "Valid values: perfect_grammar, web_search_like, misspelled, poor_grammar."
        ),
    )
    lengths: List[str] = Field(
        default_factory=list,
        description=(
            "Active query lengths. Empty = use pipeline defaults (long, medium, short). "
            "Valid values: long, medium, short."
        ),
    )

class KGEnrichmentConfig(BaseModel):
    blacklist_words: List[str] = Field(default_factory=list)
    blacklist_phrases: List[str] = Field(default_factory=list)
    domain_blacklist: List[str] = Field(default_factory=list)
    regex_blacklist_patterns: List[str] = Field(default_factory=list)
    idf_threshold: float = 0.693
    jaccard_kp_threshold: float = 0.6
    cosine_sim_min: float = 0.6
    cosine_sim_max: float = 0.9
    cosine_anti_dup_jaccard: float = 0.8
    overlap_score_threshold: float = 0.02
    overlap_distance_threshold: float = 0.9
    shared_keyphrase_min_count: int = 3
    shared_keyphrase_min_kps: int = 3
    max_keyphrases: int = 10
    semantic_relation_types: List[str] = Field(
        default_factory=lambda: [
            # Seuls 4 types de relations sémantiques sont conservés.
            "keyphrases_overlap",   # OverlapScoreBuilder (Ragas), 0 LLM
            "cosine_similarity",    # similarité vectorielle inter-chunks, 0 LLM
            "agent_discovered",     # KG Agent (DirectRelationDiscovery), LLM
            "llm_triplet",          # triplets validés LLM (kg_enrich_universal)
        ]
    )
    structural_relation_types: List[str] = Field(
        default_factory=lambda: ["child", "next"]
    )

class ChunkingConfig(BaseModel):
    max_tokens: int = 1024
    overlap_ratio: float = 0.1
    min_chunk_tokens: int = 50

class EvaluationConfig(BaseModel):
    max_context_chars: int = 12000
    qa_eval_threshold: float = 0.8
    max_retry: int = 2
    relation_validator_confidence_threshold: float = 0.5
    discovery_min_confidence: float = 0.65
    frame_bridge_min_confidence: float = 0.70
    max_content_chars_agents: int = 4000
    scenario_buffer_ratio: float = 0.50
    max_qualify_chars: int = 3000
    target_only_passed: bool = Field(
        default=True,
        description=(
            "Si True, la cible num_questions compte UNIQUEMENT les questions qui "
            "passent le QA Eval (qa_eval.passed = true). La génération continue "
            "donc tant que la cible de questions VALIDES n'est pas atteinte (ou "
            "que les scénarios sont épuisés). Les questions rejetées restent "
            "présentes dans le dataset (marquées passed=false) mais ne comptent "
            "pas dans la cible. Si False, on retrouve l'ancien comportement : la "
            "cible compte toutes les questions générées, rejetées incluses."
        ),
    )
    drop_failed_questions: bool = Field(
        default=False,
        description=(
            "Si True, les questions qui échouent définitivement le QA Eval après "
            "tous les retries sont exclues du dataset final au lieu d'y figurer "
            "avec passed=false. N'a d'effet que si target_only_passed=True."
        ),
    )


class ModelsConfig(BaseModel):
    fallback_haiku: str = "claude-haiku-4-5-20251001"
    fallback_scoring: str = "gpt-4.1-mini"
    fallback_qualify: str = "gpt-4o-mini"


class EnrichmentModulesConfig(BaseModel):
    """
    Flags on/off pour chaque module d'enrichissement du KG.
    Permet de choisir avant le lancement quels modules sont actifs.
    Tous les flags sont backward-compat (valeurs par défaut = comportement actuel).
    """
    # ── Enrichissement sémantique standard (KnowledgeGraphBuilder) ───────────
    keyphrases_overlap: bool = Field(
        default=True,
        description="Activer les relations keyphrases_overlap (extraction TF-IDF + cosine).",
    )
    cosine_similarity: bool = Field(
        default=False,
        description="Activer les relations cosine_similarity entre chunks.",
    )
    # ── KG Agent (RelationValidator + DirectRelationDiscovery + SemanticFrameBridgeDiscovery) ──
    kg_agent: bool = Field(
        default=False,
        description="Activer le KG Agent (agent_discovered, inter-documents). Coûteux en LLM.",
    )
    relation_validator: bool = Field(
        default=True,
        description="Activer la validation des relations existantes (RelationValidator). Actif seulement si kg_agent=True.",
    )
    frame_bridge: bool = Field(
        default=False,
        description=(
            "Activer SemanticFrameBridgeDiscovery (System 3) : analyse bottom-up des frames "
            "sémantiques par chunk pour découvrir des bridges conditionnels et comparatifs. "
            "Complémentaire à DirectRelationDiscovery (ne duplique pas les relations existantes). "
            "Actif seulement si kg_agent=True."
        ),
    )
    # ── Enrichissement rétrospectif centré-entité (RAKG §III-D) ─────────────
    retrospective: bool = Field(
        default=True,
        description="Activer l'enrichissement rétrospectif RAKG (retrospective_entity).",
    )
    # ── QA Eval (évaluation des questions générées) ──────────────────────────
    qa_eval: bool = Field(
        default=False,
        description="Activer le QA Eval (RAGAS + 2-Hop judge) lors de la génération.",
    )
    # ── Enrichissement universel (knowledge_graph/rahulnyk) ──────────────────
    # Méthodes domain-agnostic issues du repo knowledge_graph, adaptées à RAGAS.
    # Tous OFF par défaut → aucun changement de comportement existant.
    universal_proximity: bool = Field(
        default=False,
        description=(
            "Relations contextual_proximity entre chunks voisins (voisinage "
            "séquentiel intra-doc). 0 appel LLM. Source: knowledge_graph "
            "df_helpers.py::contextual_proximity()."
        ),
    )
    universal_metrics: bool = Field(
        default=False,
        description=(
            "Calcule PageRank + betweenness + degree centrality sur les nœuds "
            "(injectés dans node.properties pour prioriser les chunks centraux). "
            "0 appel LLM. Source: knowledge_graph compute_centrality_metrics()."
        ),
    )
    universal_communities: bool = Field(
        default=False,
        description=(
            "Détection de communautés Girvan-Newman + couleurs par cluster "
            "thématique (community_id / community_color). 0 appel LLM. "
            "Source: knowledge_graph extract_graph.ipynb."
        ),
    )
    universal_triplets: bool = Field(
        default=False,
        description=(
            "Extraction de triplets {node_1, node_2, edge} par chunk via LLM "
            "(graphPrompt). Coûteux en LLM. Source: knowledge_graph "
            "prompts.py::graphPrompt()."
        ),
    )
    universal_concepts: bool = Field(
        default=False,
        description=(
            "Extraction de concepts + score d'importance (1-5) par chunk via LLM. "
            "Coûteux en LLM. Source: knowledge_graph prompts.py::extractConcepts()."
        ),
    )
    universal_triplet_relations: bool = Field(
        default=False,
        description=(
            "Transforme les triplets LLM (graphPrompt) en VRAIES relations "
            "chunk↔chunk 'llm_triplet', chacune VALIDÉE par un appel LLM (approche "
            "macro : 1 appel par concept-pont, le LLM sélectionne les paires "
            "réellement reliées). Nécessite universal_triplets=True (les triplets "
            "doivent être extraits d'abord). Meilleur type de relation multi-hop "
            "(qualité 0.95 mesurée). Coûteux en LLM."
        ),
    )
    # ── Enrichissement Graphify (extraction sémantique LLM + AST code) ───────
    graphify: bool = Field(
        default=False,
        description=(
            "Activer l'enrichissement Graphify : extraction sémantique LLM + "
            "relations structurelles code (AST) sur les chunks, en mémoire, puis "
            "injection de relations graphify_semantic (multi-hop) et graphify_code "
            "(calls/imports/inherits/references) dans le KG. Coûteux en LLM."
        ),
    )


class RetrospectiveConfig(BaseModel):
    """
    Configuration pour l'enrichissement rétrospectif centré-entité (RAKG §III-D).

    Tous les paramètres ont des valeurs par défaut identiques à l'amont RAKG,
    feature disabled par défaut (enabled=False) → zéro impact sur les pipelines
    existants.
    """
    enabled: bool = True
    entity_sim_threshold: float = Field(
        default=0.60,
        description="Seuil cosine pour les paires d'entités candidates (= défaut amont).",
    )
    chunk_top_k: int = Field(
        default=5,
        description="top_k pour get_retriever_context (= défaut amont).",
    )
    max_extra_chunks: int = Field(
        default=5,
        description="Nombre max de chunks similaires additionnels.",
    )
    use_graph_retrieval: bool = Field(
        default=False,
        description=(
            "Écart #1 : activer le Graph Structure Retrieval pour remplir {related_kg}. "
            "OFF par défaut = comportement amont (related_kg='none')."
        ),
    )
    judge_threshold: float = Field(
        default=0.0,
        description=(
            "Écart #2 : seuil de score pour le filtre de triplets. "
            "0.0 = juge DÉSACTIVÉ (court-circuit, aucun appel LLM). Les audits "
            "montrent que 100% des paires sont réellement liées, donc le filtre "
            "dégrade plus qu'il n'aide."
        ),
    )
    enable_disambiguation: bool = Field(
        default=False,
        description=(
            "Désambiguïsation d'entités (fusion de doublons via LLM). "
            "OFF par défaut : dans STARK 1 chunk = 1 entité, deux chunks ne sont "
            "jamais des doublons → les appels LLM de désambiguïsation sont inutiles "
            "(observé : 393 appels pour 0 fusion)."
        ),
    )
    use_logprobs: bool = Field(
        default=True,
        description=(
            "Utiliser le score JSON pour le jugement de triplets. "
            "False = fallback binaire yes/no."
        ),
    )
    link_to_existing_chunks: bool = Field(
        default=True,
        description=(
            "Résoudre la cible de chaque relation vers un chunk existant par "
            "similarité d'embedding (crée des relations chunk↔chunk exploitables "
            "pour le multi-hop). False = toujours créer un nœud synthétique (parité amont)."
        ),
    )
    chunk_link_threshold: float = Field(
        default=0.75,
        description=(
            "Seuil cosine minimum pour relier la cible d'une relation à un chunk "
            "existant. En-dessous, on crée un nœud synthétique."
        ),
    )
    # ── Nouveau flux NER (vraie méthode RAKG) ────────────────────────────────
    use_ner_extractor: bool = Field(
        default=True,
        description=(
            "Nouveau flux : extraire de vraies entités nommées (RAGAS NERExtractor) "
            "et relier les chunks qui PARTAGENT une même entité (vraie méthode RAKG), "
            "au lieu de traiter chaque chunk comme une entité. "
            "False = ancien flux (chunk=entité + résolution cosine)."
        ),
    )
    ner_max_entities: int = Field(
        default=15,
        description="Nombre max d'entités nommées extraites par chunk (RAGAS NERExtractor).",
    )
    max_parallel_entities: int = Field(
        default=8,
        description=(
            "Parallélisme des extractions centré-entité (appels LLM indépendants). "
            "Accélère fortement la Voie B (défaut : 8)."
        ),
    )
    max_relations_per_chunk: int = Field(
        default=3,
        description=(
            "Anti-hub : nombre max de relations retrospective_entity par chunk. "
            "On garde les entités-pont les plus spécifiques (IDF élevé = entité rare)."
        ),
    )
    require_cross_document: bool = Field(
        default=True,
        description=(
            "N'établir un lien entité-partagée que si les deux chunks proviennent "
            "de documents (fichiers) différents. True = vrai multi-hop inter-doc."
        ),
    )
    enable_complementarity_judge: bool = Field(
        default=True,
        description=(
            "Juge LLM de complémentarité : ne garder une paire que si les deux chunks "
            "apportent des informations DIFFÉRENTES et complémentaires sur l'entité "
            "partagée (évite les paires redondantes/paraphrasées)."
        ),
    )
    complementarity_threshold: float = Field(
        default=0.5,
        description="Score minimum de complémentarité (0-1) pour garder une paire entité-partagée.",
    )
    min_entity_chars: int = Field(
        default=3,
        description="Longueur minimale d'un nom d'entité (filtre le bruit type 'a', 'is').",
    )
    max_entity_document_frequency: float = Field(
        default=0.5,
        description=(
            "Filtre anti-entité-générique : ignorer les entités présentes dans plus de "
            "cette fraction des chunks (ex. 0.5 = présente dans >50% des chunks → trop générique)."
        ),
    )


# ── Root model ────────────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    meta: MetaConfig = Field(default_factory=MetaConfig)
    domain: DomainConfig
    prompts: PromptsConfig
    few_shots: FewShotsConfig = Field(default_factory=FewShotsConfig)
    personas: List[PersonaDef]
    taxonomy: TaxonomyConfig
    query_params: QueryParamsConfig = Field(default_factory=QueryParamsConfig)
    kg_enrichment: KGEnrichmentConfig = Field(default_factory=KGEnrichmentConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrospective: RetrospectiveConfig = Field(
        default_factory=RetrospectiveConfig,
        description="Configuration RAKG rétrospectif (disabled par défaut).",
    )
    enrich_modules: EnrichmentModulesConfig = Field(
        default_factory=EnrichmentModulesConfig,
        description="Flags on/off pour chaque module d'enrichissement du KG.",
    )
    graphify_reuse_path: Optional[str] = Field(
        default=None,
        description=(
            "Chemin absolu vers un graphify_graph.json (format NetworkX node-link) "
            "à réutiliser pour l'étape 3.7/4. None = reconstruction LLM."
        ),
    )

    # ── Backward-compat migration ─────────────────────────────────────────────

    @model_validator(mode="before")
    @classmethod
    def _migrate_query_answer_split(cls, data: Any) -> Any:
        """
        Migrate legacy configs that store a single `query_answer_generation`
        prompt/few-shot into the new two-module form (`query_generation` +
        `answer_generation`).

        Runs before field validation so old YAML/dicts load unchanged. A config
        already using the new fields is left untouched (idempotent).
        """
        if not isinstance(data, dict):
            return data
        return _migrate_query_answer_split(data)

    # ── Loaders / serializers ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls.model_validate(d)

    def to_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    def to_dict(self) -> dict:
        return self.model_dump()

    # ── Convenience accessors ─────────────────────────────────────────────────

    def question_type_names(self) -> List[str]:
        return [t.name for t in self.taxonomy.types]

    def question_type_proportions(self) -> Dict[str, float]:
        return dict(self.taxonomy.budget)

    def personas_by_name(self) -> Dict[str, "PersonaDef"]:
        return {p.name: p for p in self.personas}

# ── Backward-compat migration helper ──────────────────────────────────────────

def _migrate_query_answer_split(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split a legacy `query_answer_generation` prompt + few-shots into the new
    `query_generation` / `answer_generation` pair, in place on a shallow copy.

    Rules (idempotent):
      • prompts:
          - If `query_generation` is absent but legacy `query_answer_generation`
            exists, set `query_generation` ← legacy text (it already knows how to
            craft a multi-hop question).
          - If `answer_generation` is absent, seed it with the strict two-hop
            default template (_DEFAULT_ANSWER_GENERATION_PROMPT).
          - Drop the legacy `query_answer_generation` key.
      • few_shots: project each legacy example {input, output:{query, answer}}:
          - query_generation  ← {input: <legacy input>, output: {query}}
          - answer_generation ← {input: {context, question:<query>, themes},
                                  output: {answer}}
        The legacy `query_answer_generation` few-shots key is dropped.

    A config already using the new fields is returned unchanged.
    """
    data = dict(data)  # shallow copy; nested dicts copied below as needed

    # ── prompts ───────────────────────────────────────────────────────────────
    prompts = data.get("prompts")
    if isinstance(prompts, dict) and "query_answer_generation" in prompts:
        prompts = dict(prompts)
        prompts.pop("query_answer_generation")
        prompts.setdefault("query_generation", _DEFAULT_QUERY_GENERATION_PROMPT)
        prompts.setdefault("answer_generation", _DEFAULT_ANSWER_GENERATION_PROMPT)
        data["prompts"] = prompts

    # ── few_shots ───────────────────────────────────────────────────────────────
    few_shots = data.get("few_shots")
    if isinstance(few_shots, dict) and "query_answer_generation" in few_shots:
        few_shots = dict(few_shots)
        legacy_examples = few_shots.pop("query_answer_generation") or []
        query_examples: List[Dict[str, Any]] = []
        answer_examples: List[Dict[str, Any]] = []
        for ex in legacy_examples:
            if not isinstance(ex, dict):
                continue
            inp = dict(ex.get("input", {}) or {})
            out = dict(ex.get("output", {}) or {})
            query_text = out.get("query", "")
            answer_text = out.get("answer", "")
            # query_generation: same input envelope → output {query}
            query_examples.append({"input": inp, "output": {"query": query_text}})
            # answer_generation: input {context, question, themes} → output {answer}
            ctx = inp.get("context", [])
            if not ctx:
                ctx = [c for c in [inp.get("context_1", ""), inp.get("context_2", "")] if c]
            answer_examples.append({
                "input": {
                    "context": ctx,
                    "question": query_text,
                    "themes": inp.get("themes", []),
                },
                "output": {"answer": answer_text},
            })
        few_shots.setdefault("query_generation", query_examples)
        few_shots.setdefault("answer_generation", answer_examples)
        data["few_shots"] = few_shots

    return data

# ── Prompt class factory ──────────────────────────────────────────────────────

def build_prompt_class(base_cls: Type, instruction: str, examples: list) -> Type:
    """
    Return a new PydanticPrompt subclass with the given instruction and examples.

    input_model / output_model are inherited from base_cls unchanged.
    The returned *class* (not instance) can be instantiated normally:

        MyPrompt = build_prompt_class(QueryAnswerGenerationPrompt, instr, exs)
        instance = MyPrompt()

    This replaces all the static domain-specific PydanticPrompt subclasses
    (e.g. _CoSAppQueryAnswerGenerationPrompt) with runtime-built equivalents.
    """
    return type(
        f"Domain{base_cls.__name__}",
        (base_cls,),
        {"instruction": instruction, "examples": examples},
    )

# ── Few-shot deserializer ─────────────────────────────────────────────────────
#
# Each prompt type stores examples as plain dicts:
#   { "input": {...}, "output": {...} }
#
# At runtime this function reconstructs the typed (InputModel, OutputModel)
# tuples that PydanticPrompt expects.
#
# Persona resolution for query_answer_generation uses a name-lookup into
# the RAGAS Persona objects built from cfg.personas.

def deserialize_few_shots(
    prompt_type: str,
    examples: List[Dict[str, Any]],
    cfg: "PipelineConfig",
) -> list:
    """
    Reconstruct typed (InputModel, OutputModel) tuples for the given prompt type.

    Args:
        prompt_type : key matching FewShotsConfig field names.
        examples    : list of {input: {...}, output: {...}} dicts from YAML.
        cfg         : full PipelineConfig (needed for persona resolution).

    Returns:
        List of (InputModel, OutputModel) tuples ready for PydanticPrompt.examples.
        Returns [] on import errors (graceful degradation when RAGAS not installed).
    """
    if not examples:
        return []

    # Look the deserializer up FIRST: doing it inside the try below would make
    # any internal KeyError (e.g. a few-shot dict missing its "input" key) be
    # misreported as "unknown prompt_type", hiding the real cause.
    deserializer = _DESERIALIZERS.get(prompt_type)
    if deserializer is None:
        logging.warning(
            "deserialize_few_shots: unknown prompt_type %r (known: %s) — returning []",
            prompt_type, ", ".join(sorted(_DESERIALIZERS)),
        )
        return []

    try:
        return deserializer(examples, cfg)
    except KeyError as exc:
        logging.warning(
            "deserialize_few_shots: malformed few-shot for prompt_type %r — missing "
            "key %s. Each example must be shaped {input: {...}, output: {...}}. "
            "Returning [] (prompt will run WITHOUT few-shots).",
            prompt_type, exc,
        )
        return []
    except Exception as exc:

        logging.warning(
            "deserialize_few_shots: failed for prompt_type %r (%s) — returning []",
            prompt_type, exc,
        )
        return []

# ── Per-prompt deserializer functions ─────────────────────────────────────────

def _deser_qa_evaluator(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    """DEPRECATED stub: qa_evaluator few-shots are no longer applied anywhere.

    The GEval internal judge that consumed them was removed, so no prompt reads
    these examples. The field is kept in FewShotsConfig only so that existing
    session YAML files keep loading. We warn loudly, because silently dropping
    non-empty examples looks like a working feature from the UI.
    """
    if examples:
        logging.warning(
            "deserialize_few_shots: 'qa_evaluator' few-shots are DEPRECATED and "
            "IGNORED (%d example(s) dropped) — the GEval internal judge that used "
            "them was removed. Remove them from the session config; editing them "
            "has no effect on generation.",
            len(examples),
        )
    return []


def _resolve_persona(inp_d: Dict, cfg: "PipelineConfig"):
    """Resolve a RAGAS Persona from a few-shot input dict (by persona_name)."""
    from ragas.testset.persona import Persona as RagasPersona
    persona_map = {
        p.name: RagasPersona(name=p.name, role_description=p.role_description)
        for p in cfg.personas
    }
    persona_name = inp_d.get("persona_name", "")
    persona = persona_map.get(persona_name)
    if persona is None:
        persona = next(iter(persona_map.values())) if persona_map else RagasPersona(
            name=persona_name, role_description=""
        )
    return persona


def _coerce_context(inp_d: Dict) -> List[str]:
    """Extract a context list from a few-shot input, supporting legacy keys."""
    context_raw = inp_d.get("context", [])
    if not context_raw:
        ctx1 = inp_d.get("context_1", "")
        ctx2 = inp_d.get("context_2", "")
        context_raw = [c for c in [ctx1, ctx2] if c]
    return context_raw


def _deser_query_generation(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    """Reconstruct (QueryGenInput, QueryGenOutput) tuples.

    The typed models live in the question generator module (loaded through the
    `question_generator` bridge); import lazily so pipeline_config stays
    importable without RAGAS/heavy deps.
    """
    try:
        from question_generator import QueryGenInput, QueryGenOutput  # type: ignore
    except Exception:
        return []

    result = []
    for ex in examples:
        inp_d = ex["input"]
        inp = QueryGenInput(
            persona=_resolve_persona(inp_d, cfg),
            themes=inp_d.get("themes", []),
            query_style=inp_d.get("query_style", "Perfect grammar"),
            query_length=inp_d.get("query_length", "Medium"),
            context=_coerce_context(inp_d),
        )
        out = QueryGenOutput.model_validate(ex["output"])
        result.append((inp, out))
    return result


def _deser_answer_generation(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    """Reconstruct (AnswerGenInput, AnswerGenOutput) tuples."""
    try:
        from question_generator import AnswerGenInput, AnswerGenOutput  # type: ignore
    except Exception:
        return []

    result = []
    for ex in examples:
        inp_d = ex["input"]
        inp = AnswerGenInput(
            context=_coerce_context(inp_d),
            question=inp_d.get("question", ""),
            themes=inp_d.get("themes", []),
        )
        out = AnswerGenOutput.model_validate(ex["output"])
        result.append((inp, out))
    return result

def _deser_relation_validator(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    from kg_agent import RelationBatchInput, RelationBatchOutput
    result = []
    for ex in examples:
        inp = RelationBatchInput.model_validate(ex["input"])
        out = RelationBatchOutput.model_validate(ex["output"])
        result.append((inp, out))
    return result

def _deser_doc_theme(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    from kg_agent import DocumentInput, DocumentThemes
    result = []
    for ex in examples:
        inp = DocumentInput.model_validate(ex["input"])
        out = DocumentThemes.model_validate(ex["output"])
        result.append((inp, out))
    return result

def _deser_cross_doc_map(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    from kg_agent import AllThemeEntries, ThemeBridges
    result = []
    for ex in examples:
        inp = AllThemeEntries.model_validate(ex["input"])
        out = ThemeBridges.model_validate(ex["output"])
        result.append((inp, out))
    return result

def _deser_chunk_locator(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    from kg_agent import ChunkLocationRequest, LocatedChunks
    result = []
    for ex in examples:
        inp = ChunkLocationRequest.model_validate(ex["input"])
        out = LocatedChunks.model_validate(ex["output"])
        result.append((inp, out))
    return result

def _deser_direct_pair_validator(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    from kg_agent import DirectPairInput, DirectPairValidation
    result = []
    for ex in examples:
        inp = DirectPairInput.model_validate(ex["input"])
        out = DirectPairValidation.model_validate(ex["output"])
        result.append((inp, out))
    return result

def _deser_keyphrase_extractor(
    examples: List[Dict], cfg: "PipelineConfig"
) -> list:
    from ragas.testset.transforms.extractors.llm_based import (
        Keyphrases, TextWithExtractionLimit,
    )
    result = []
    for ex in examples:
        inp = TextWithExtractionLimit.model_validate(ex["input"])
        out = Keyphrases.model_validate(ex["output"])
        result.append((inp, out))
    return result

_DESERIALIZERS = {
    "qa_evaluator": _deser_qa_evaluator,
    "query_generation": _deser_query_generation,
    "answer_generation": _deser_answer_generation,
    "relation_validator": _deser_relation_validator,
    "doc_theme": _deser_doc_theme,
    "cross_doc_map": _deser_cross_doc_map,
    "chunk_locator": _deser_chunk_locator,
    "direct_pair_validator": _deser_direct_pair_validator,
    "keyphrase_extractor": _deser_keyphrase_extractor,
}

# ── RAGAS Persona builder ─────────────────────────────────────────────────────

def build_ragas_personas(cfg: "PipelineConfig") -> list:
    """
    Convert cfg.personas (List[PersonaDef]) into RAGAS Persona objects.
    Avoids importing RAGAS at module level.
    """
    from ragas.testset.persona import Persona as RagasPersona
    return [
        RagasPersona(name=p.name, role_description=p.role_description)
        for p in cfg.personas
    ]

def build_personas_by_name(cfg: "PipelineConfig") -> dict:
    """Return {name: RagasPersona} lookup from config."""
    from ragas.testset.persona import Persona as RagasPersona
    return {
        p.name: RagasPersona(name=p.name, role_description=p.role_description)
        for p in cfg.personas
    }

