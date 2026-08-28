"""
config_agent.py — v2 (high-quality prompt generation)
------------------------------------------------------
Same public API as v1:
    from config_agent import run_config_agent, run_config_agent_sync, validate_config
    cfg, issues = await run_config_agent(llm, description="...", samples=[...])

Key improvements vs v1:
  - FullConfigBuilderAgent is replaced by 3 specialised prompt-builder sub-agents,
    each carrying a GOLD-STANDARD CoSApp example → the LLM imitates the structure
    and adapts it to the new domain rather than hallucinating from scratch.
  - Stages 4-5-6 (the three builders) run in PARALLEL (asyncio.gather) → ~2-3x faster.
  - qualify_user_template is automatically repaired if the LLM drops a placeholder.
  - validate_config tests that qualify_user_template is a valid .format() string.

Pipeline (7 stages):
    1. CorpusAnalyzerAgent    → CorpusProfile
    2. TaxonomyDesignerAgent  → TaxonomyProposal
    3. PersonaGeneratorAgent  → PersonaProposal
    4. GenEvalPromptsBuilder  → 4 prompts (generation + evaluation)  ┐
    5. KGPromptsBuilder       → 6 prompts (knowledge-graph)          ├─ PARALLEL
    6. QualifyPromptsBuilder  → 2 prompts (qualification)            ┘
    7. ValidationLayer        → List[ConfigIssue]  (rule-based, no LLM)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt

from pipeline_config import (
    ChunkingConfig,
    DomainConfig,
    EvaluationConfig,
    FewShotsConfig,
    KGEnrichmentConfig,
    MetaConfig,
    ModelsConfig,
    PersonaDef,
    PipelineConfig,
    PromptsConfig,
    QueryParamsConfig,
    QuestionTypeDef,
    TaxonomyConfig,
    _migrate_query_answer_split,
)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED MODELS
# ══════════════════════════════════════════════════════════════════════════════

class CorpusProfile(BaseModel):
    domain_name: str
    domain_description: str
    technical_vocabulary: List[str]
    document_style: Literal["api_reference", "tutorial", "conceptual", "mixed"]
    chunk_examples: List[str] = Field(
        description="2-3 representative text excerpts (≤ 400 chars each)"
    )
    suggested_language: str = "en"


class QuestionTypeProposal(BaseModel):
    name: str
    description: str
    budget_fraction: float = Field(ge=0.0, le=1.0)


class TaxonomyProposal(BaseModel):
    question_types: List[QuestionTypeProposal]
    relation_to_question_types: Dict[str, List[str]]
    relation_to_answer_structure: Dict[str, str]


class PersonaProposal(BaseModel):
    personas: List[PersonaDef]


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 1 : CorpusAnalyzerAgent
# ══════════════════════════════════════════════════════════════════════════════

class CorpusAnalyzerInput(BaseModel):
    description: str = ""
    samples: List[str] = Field(default_factory=list)


_CORPUS_ANALYZER_INSTRUCTION = (
    "You are a technical documentation analyst.\n\n"
    "You receive a free-text description of a corpus, sample document excerpts, "
    "or both. Analyse them and return a CorpusProfile.\n\n"
    "### Fields\n"
    "domain_name         : Short name of the technology/framework (1-5 words).\n"
    "domain_description  : One paragraph describing what the corpus covers.\n"
    "technical_vocabulary: 12-20 domain-specific terms, class names, method names, "
    "or concepts that appear frequently AND discriminate this corpus from a generic one.\n"
    "document_style      : One of api_reference / tutorial / conceptual / mixed.\n"
    "chunk_examples      : 2-3 representative short excerpts (≤ 400 chars each).\n"
    "suggested_language  : ISO-639-1 language code (default 'en').\n\n"
    "Rules:\n"
    "  - technical_vocabulary MUST be specific to the domain, never generic "
    "(forbidden: 'class', 'method', 'function', 'parameter', 'value', 'object').\n"
    "  - Prefer real identifiers extracted verbatim from samples (CamelCase class "
    "names, snake_case methods, multi-word technical concepts).\n"
    "  - If both description and samples are provided, trust the samples more.\n"
    "  - Extract chunk_examples verbatim from samples when possible.\n"
)

_CORPUS_ANALYZER_EXAMPLES = [
    (
        CorpusAnalyzerInput(
            description=(
                "Technical documentation for FEniCS, a Python FEM library. "
                "Covers function spaces, variational forms, mesh refinement, "
                "PDE solvers, and boundary condition APIs."
            ),
            samples=[
                "FunctionSpace defines the finite element approximation space on a mesh. "
                "Usage: V = FunctionSpace(mesh, 'Lagrange', 1). The degree parameter "
                "controls polynomial order.",
                "DirichletBC applies essential boundary conditions by constraining DOF values. "
                "bc = DirichletBC(V, Constant(0.0), boundary). Call bc.apply(A, b) after assembly.",
            ],
        ),
        CorpusProfile(
            domain_name="FEniCS FEM",
            domain_description=(
                "FEniCS is a Python library for solving PDEs using the finite element method. "
                "Its documentation covers FunctionSpace definitions, variational form assembly, "
                "mesh handling, DirichletBC boundary conditions, and linear/nonlinear solvers."
            ),
            technical_vocabulary=[
                "FunctionSpace", "DirichletBC", "TrialFunction", "TestFunction",
                "variational form", "mesh refinement", "Lagrange element",
                "boundary condition", "DOF", "assembly", "FEniCS", "UFL",
                "Constant", "Function", "solve",
            ],
            document_style="api_reference",
            chunk_examples=[
                "FunctionSpace defines the finite element approximation space on a mesh. "
                "Usage: V = FunctionSpace(mesh, 'Lagrange', 1).",
                "DirichletBC applies essential boundary conditions by constraining DOF values. "
                "bc = DirichletBC(V, Constant(0.0), boundary).",
            ],
            suggested_language="en",
        ),
    ),
]


class CorpusAnalyzerPrompt(PydanticPrompt[CorpusAnalyzerInput, CorpusProfile]):
    instruction: str = _CORPUS_ANALYZER_INSTRUCTION
    input_model = CorpusAnalyzerInput
    output_model = CorpusProfile
    examples = _CORPUS_ANALYZER_EXAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 2 : TaxonomyDesignerAgent
# ══════════════════════════════════════════════════════════════════════════════

_TAXONOMY_DESIGNER_INSTRUCTION = (
    "You are designing a question taxonomy for a RAG evaluation benchmark.\n\n"
    "Given a CorpusProfile, propose:\n"
    "  1. 4-6 question types relevant to the corpus (name + description + budget_fraction).\n"
    "  2. relation_to_question_types: mapping each of the 5 RAGAS relation names "
    "(prerequisite, contrast, elaboration, example_of, shared_concept) to 1-3 "
    "compatible question types from your list.\n"
    "  3. relation_to_answer_structure: one answer-structure sentence per relation.\n\n"
    "### Rules\n"
    "  - budget_fraction values MUST sum to exactly 1.0.\n"
    "  - Question type names: lowercase, underscore_separated, no spaces.\n"
    "  - Each description must (a) name the cognitive operation (HOW/WHY/contrast/list/"
    "precise-value) and (b) reference the domain explicitly using domain_name.\n"
    "  - Types must reflect what the corpus actually supports: api_reference → "
    "'implementation' + 'factual'; tutorial → 'step_by_step' + 'integration'; "
    "conceptual → 'design_rationale' + 'comparison'.\n"
    "  - All 5 relation names must appear in relation_to_question_types, each mapping "
    "to ≥1 question type FROM YOUR LIST (never invent a type in the mapping).\n"
    "  - relation_to_answer_structure values are concrete 1-sentence guidance strings "
    "using '(1)…; (2)…' structure or explicit markers like 'whereas'/'unlike'.\n"
)

_TAXONOMY_DESIGNER_EXAMPLES = [
    (
        CorpusProfile(
            domain_name="FEniCS FEM",
            domain_description=(
                "FEniCS is a Python FEM library. Docs cover FunctionSpace, "
                "variational forms, DirichletBC, assembly, and solvers."
            ),
            technical_vocabulary=["FunctionSpace", "DirichletBC", "variational form", "mesh"],
            document_style="api_reference",
            chunk_examples=["FunctionSpace defines the approximation space on a mesh."],
            suggested_language="en",
        ),
        TaxonomyProposal(
            question_types=[
                QuestionTypeProposal(
                    name="implementation",
                    description="Asks HOW to implement or configure a FEniCS API by combining both segments.",
                    budget_fraction=0.30,
                ),
                QuestionTypeProposal(
                    name="integration",
                    description="Requires combining two FEniCS components that exchange data or wiring.",
                    budget_fraction=0.25,
                ),
                QuestionTypeProposal(
                    name="comparison",
                    description="Contrasts two FEniCS classes, boundary-condition types, or solver strategies.",
                    budget_fraction=0.20,
                ),
                QuestionTypeProposal(
                    name="design_rationale",
                    description="Asks WHY a FEniCS design decision was made (one segment explains the other).",
                    budget_fraction=0.15,
                ),
                QuestionTypeProposal(
                    name="factual",
                    description="Expects a precise FEniCS parameter name, default value, or method signature.",
                    budget_fraction=0.10,
                ),
            ],
            relation_to_question_types={
                "prerequisite":   ["implementation", "design_rationale"],
                "contrast":       ["comparison", "integration"],
                "elaboration":    ["integration", "implementation"],
                "example_of":     ["factual", "implementation"],
                "shared_concept": ["integration", "comparison"],
            },
            relation_to_answer_structure={
                "prerequisite":   "Answer: (1) what the first chunk establishes; (2) how the second depends on it.",
                "contrast":       "Answer: point-by-point contrast using 'whereas' or 'unlike' markers.",
                "elaboration":    "Answer: (1) high-level overview; (2) detailed implementation from the second chunk.",
                "example_of":     "Answer: (1) the general concept; (2) the concrete FEniCS example.",
                "shared_concept": "Answer: how each chunk covers a different facet of the same concept.",
            },
        ),
    ),
]


class TaxonomyDesignerPrompt(PydanticPrompt[CorpusProfile, TaxonomyProposal]):
    instruction: str = _TAXONOMY_DESIGNER_INSTRUCTION
    input_model = CorpusProfile
    output_model = TaxonomyProposal
    examples = _TAXONOMY_DESIGNER_EXAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 3 : PersonaGeneratorAgent
# ══════════════════════════════════════════════════════════════════════════════

_PERSONA_GENERATOR_INSTRUCTION = (
    "You are designing user personas for a RAG evaluation benchmark.\n\n"
    "Given a CorpusProfile, propose 3 to 5 user personas who would realistically "
    "query this documentation.\n\n"
    "Each persona has:\n"
    "  name             : A specific job title or role (2-5 words), NOT generic.\n"
    "  role_description : A rich profile that MUST contain, in order:\n"
    "    (a) 2-3 sentences on background, expertise level, and goals;\n"
    "    (b) the specific sub-systems / APIs / concepts this persona works with, "
    "using actual terms from technical_vocabulary;\n"
    "    (c) 2-3 concrete example questions this persona would ask, each referencing "
    "real domain identifiers (class names, method names, parameters).\n\n"
    "### Rules\n"
    "  - Personas MUST be diverse: cover different expertise levels and concerns "
    "(e.g. one API user, one architect/designer, one optimizer/analyst).\n"
    "  - Every example question must name at least one specific identifier from "
    "technical_vocabulary — never vague ('the system', 'the function').\n"
    "  - FORBIDDEN generic personas: 'developer', 'user', 'engineer' alone.\n"
    "  - Persona names must be specific to the domain (prefix or qualify with "
    "domain_name where natural).\n"
)

_PERSONA_GENERATOR_EXAMPLES = [
    (
        CorpusProfile(
            domain_name="FEniCS FEM",
            domain_description="FEniCS Python FEM library for solving PDEs.",
            technical_vocabulary=["FunctionSpace", "DirichletBC", "variational form", "mesh", "TrialFunction"],
            document_style="api_reference",
            chunk_examples=["FunctionSpace defines the approximation space on a mesh."],
            suggested_language="en",
        ),
        PersonaProposal(personas=[
            PersonaDef(
                name="FEniCS PDE Solver",
                role_description=(
                    "A computational scientist who writes FEniCS code to solve PDEs numerically. "
                    "Comfortable with the variational formulation but still learning the assembly internals. "
                    "Works daily with FunctionSpace, TrialFunction/TestFunction, variational form definition, "
                    "and DirichletBC application. "
                    "Example questions: 'How do I apply a DirichletBC to a mixed FunctionSpace defined with "
                    "MixedElement?' "
                    "'What is the difference between assemble() and assemble_system() when imposing "
                    "boundary conditions?' "
                    "'Why does my TestFunction need to live in the same FunctionSpace as my TrialFunction?'"
                ),
            ),
            PersonaDef(
                name="FEM Mesh Engineer",
                role_description=(
                    "A computational engineer focused on mesh generation and adaptive refinement for FEniCS. "
                    "Expert in geometry handling, weaker on solver theory. "
                    "Works with mesh refinement, MeshFunction markers, SubDomain tagging, and parallel partitioning. "
                    "Example questions: 'How does refine() interact with MeshFunction markers across refinement levels?' "
                    "'What boundary-marking strategy combines SubDomain and MeshFunction correctly?' "
                    "'How are DOF re-numbered after a mesh refinement step?'"
                ),
            ),
            PersonaDef(
                name="PDE Optimization Researcher",
                role_description=(
                    "A researcher using FEniCS for PDE-constrained optimization and adjoint methods. "
                    "Strong in math, treats FEniCS as a tool. "
                    "Works with dolfin-adjoint, Functional definitions, Control objects, and the variational form "
                    "as the forward problem. "
                    "Example questions: 'How do I define a Functional over a volume integral using the dx measure?' "
                    "'What is the workflow to compute sensitivities via the adjoint approach in FEniCS?' "
                    "'How does DirichletBC affect the adjoint equation derivation?'"
                ),
            ),
        ]),
    ),
]


class PersonaGeneratorPrompt(PydanticPrompt[CorpusProfile, PersonaProposal]):
    instruction: str = _PERSONA_GENERATOR_INSTRUCTION
    input_model = CorpusProfile
    output_model = PersonaProposal
    examples = _PERSONA_GENERATOR_EXAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# SHARED INPUT FOR PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

class PromptBuilderInput(BaseModel):
    domain_name: str
    domain_description: str
    technical_vocabulary: List[str]
    document_style: str
    chunk_examples: List[str] = Field(default_factory=list)
    question_type_names: List[str] = Field(default_factory=list)
    persona_names: List[str] = Field(default_factory=list)
    question_type_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps question type name → its cognitive definition (from TaxonomyDesignerAgent). "
        "Use these verbatim — do NOT reinvent definitions.",
    )
    persona_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps persona name → full role_description (from PersonaGeneratorAgent). "
        "Use to tailor examples to each persona's real concerns and domain vocabulary.",
    )
    sample_chunk_pairs: List[Dict[str, str]] = Field(
        default_factory=list,
        description="0-3 real {chunk_a, chunk_b} pairs from the corpus. "
        "When present, use them as context_1hop/context_2hop in few-shot examples "
        "instead of hallucinating content.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 4 : GenEvalPromptsBuilder (query/answer generation + evaluation)
# ══════════════════════════════════════════════════════════════════════════════

class GenEvalPromptsDraft(BaseModel):
    query_answer_generation: str = Field(
        description="Instruction for the multi-hop query/answer generator. Must list the "
        "question types, demand domain-specific terminology, give a style guide "
        "(perfect grammar vs web-search), and a 'what to avoid' section."
    )
    qa_evaluator: str = Field(
        description="GEval-style judge instruction with 3 criteria: multi_hop_necessity, "
        "grounding, specificity. The specificity criterion must reference the new domain's "
        "identifiers."
    )
    no_context_system: str = Field(
        description="System prompt for answering WITHOUT context: expert-on-domain persona, "
        "concise, admits uncertainty."
    )
    single_context_system: str = Field(
        description="System prompt for answering with ONE context only: answer strictly from "
        "context, state what is missing."
    )


_GEN_EVAL_INSTRUCTION = (
    "You are writing domain-adapted LLM instruction strings for a RAG benchmark pipeline.\n\n"
    "Below is a GOLD-STANDARD set of instructions written for the CoSApp / CoSApp-Turbo "
    "domain. Study its structure, tone, and level of detail. Then produce the EQUIVALENT "
    "instructions for the NEW domain described in the input — same structure and rigor, "
    "but every CoSApp-specific term replaced by terms from the new domain's "
    "technical_vocabulary, and every question type replaced by the new question_type_names.\n\n"
    "### CRITICAL REQUIREMENTS\n"
    "  - Replace EVERY mention of CoSApp / CoSApp-Turbo with the new domain_name.\n"
    "  - Replace CoSApp examples (velocity triangle, extend_rows_3d, ChannelAeroSolverMeridian, "
    "ports fl_in/fl_out) with concrete terms from technical_vocabulary.\n"
    "  - In query_answer_generation, list exactly the provided question_type_names. "
    "Use the definition from question_type_descriptions for each — do NOT invent new ones.\n"
    "  - In qa_evaluator, keep the 3-criteria structure (multi_hop_necessity, grounding, "
    "specificity) and make criterion 3 check for the NEW domain's identifiers.\n"
    "  - Keep length/style guidance (long ≥ 20 words, medium 10-19, short ≤ 9; "
    "perfect grammar ends with '?', web-search-like has no '?').\n"
    "  - Do NOT copy CoSApp text verbatim. Adapt fully.\n\n"
    "### Question type definitions (authoritative — use verbatim)\n"
    "The input field question_type_descriptions maps each question type name to its exact "
    "cognitive definition as designed for this domain. In query_answer_generation, reproduce "
    "each definition AS-IS — do NOT paraphrase or reinvent. This is the single source of truth "
    "for what each question type means in the generated prompt.\n\n"
    "════════ GOLD STANDARD (CoSApp) — query_answer_generation ════════\n"
    "You are generating evaluation questions for a RAG benchmark focused on the CoSApp Python "
    "simulation framework and its turbomachinery extension CoSApp-Turbo. Given a persona, a list "
    "of themes (key technical concepts shared between two segments), a query_style "
    "('Perfect grammar' or 'Web search like queries'), a query_length (long/medium/short), and a "
    "question_type (integration / comparison / design_rationale / implementation / enumeration / "
    "factual), plus two context segments <1-hop> and <2-hop>:\n"
    "1. Generate a Multi-Hop Query that requires BOTH segments, references ≥1 theme explicitly, "
    "uses precise CoSApp terminology (class/method/port names), matches style+length, and exploits "
    "the question_type's cognitive relationship.\n"
    "2. Generate a concise reference answer grounded strictly in the context.\n"
    "Style guide: 'Perfect grammar' → full question ending with '?'; 'Web search like queries' → "
    "short keyword query, no '?'. Avoid trivial single-hop questions, hallucinated identifiers, and "
    "vague wording like 'the system'.\n\n"
    "════════ GOLD STANDARD (CoSApp) — qa_evaluator ════════\n"
    "You are a quality reviewer for a CoSApp RAG benchmark. Score a (question, answer) pair against "
    "three independent criteria: (1) multi_hop_necessity — can it be answered from one context alone? "
    "score ≥0.7 only if BOTH are needed; (2) grounding — does the answer add claims absent from the "
    "contexts? ≥0.7 if fully grounded; (3) specificity — does the question name precise CoSApp "
    "identifiers rather than 'the system'/'the method'? Set passed=True only if all three ≥ threshold. "
    "On failure give a ≤30-word feedback citing the criterion that failed.\n\n"
    "════════ GOLD STANDARD (CoSApp) — no_context_system ════════\n"
    "You are an expert on CoSApp (a Python multidisciplinary system-simulation framework) and its "
    "turbomachinery extension CoSApp-Turbo. Answer concisely and accurately from training knowledge; "
    "if unsure, say so explicitly.\n\n"
    "════════ GOLD STANDARD (CoSApp) — single_context_system ════════\n"
    "You are an expert on CoSApp and CoSApp-Turbo. Answer using ONLY the provided context. Be concise "
    "and technically accurate. If the context lacks enough information, say explicitly what is missing.\n"
)


class GenEvalPromptsBuilder(PydanticPrompt[PromptBuilderInput, GenEvalPromptsDraft]):
    instruction: str = _GEN_EVAL_INSTRUCTION
    input_model = PromptBuilderInput
    output_model = GenEvalPromptsDraft
    examples = []


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 5 : KGPromptsBuilder (knowledge-graph stage)
# ══════════════════════════════════════════════════════════════════════════════

class KGPromptsDraft(BaseModel):
    relation_validator: str = Field(
        description="Instruction for judging whether an A→B relation is reliable using BOTH "
        "shared-keyphrase quality AND content agreement (strict AND)."
    )
    doc_theme: str = Field(
        description="Instruction for extracting 3-5 specific technical themes from a document preview."
    )
    cross_doc_map: str = Field(
        description="Instruction for merging theme names and finding bridge themes spanning ≥2 docs."
    )
    chunk_locator: str = Field(
        description="Instruction for selecting the 1-3 chunks that best cover a theme within a document."
    )
    direct_pair_validator: str = Field(
        description="Instruction for validating an inter-document chunk pair and classifying the "
        "relation (elaboration/contrast/prerequisite/example_of/shared_concept)."
    )
    keyphrase_extractor: str = Field(
        description="Instruction for extracting the most technically discriminant keyphrases, with a "
        "priority order and an explicit ban-list of generic words."
    )


_KG_INSTRUCTION = (
    "You are writing domain-adapted LLM instruction strings for the knowledge-graph stage of a RAG "
    "benchmark pipeline. Below are GOLD-STANDARD instructions written for CoSApp / CoSApp-Turbo. "
    "Reproduce their structure and rigor for the NEW domain, replacing all CoSApp terms with terms "
    "from technical_vocabulary and the new domain_name.\n\n"
    "### CRITICAL REQUIREMENTS\n"
    "  - Replace CoSApp/CoSApp-Turbo with domain_name everywhere.\n"
    "  - In keyphrase_extractor, set the 'what to extract' priority and the 'reliable vs unreliable' "
    "examples using REAL terms from technical_vocabulary — do NOT invent identifiers.\n"
    "  - question_type_descriptions is available: ensure the themes extracted by keyphrase_extractor "
    "and relation_validator are specific enough to support those question types.\n"
    "  - In direct_pair_validator, keep exactly the ten relation labels: elaboration, contrast, "
    "prerequisite, example_of, shared_concept, conditional_behavior, operation_comparison, "
    "generalization_pattern, convergent_goal, complementary_aspect.\n"
    "  - In relation_validator, keep the strict-AND rule over the two signals (keyphrase quality + "
    "content agreement).\n"
    "  - Adapt fully; do not copy CoSApp text verbatim.\n\n"
    "════════ GOLD STANDARD (CoSApp) — keyphrase_extractor ════════\n"
    "Extract the most technically discriminant keyphrases. Priority: (1) class names (CamelCase, e.g. "
    "ChannelAeroSolverMeridian); (2) method/function names (snake_case, e.g. extend_rows_3d); "
    "(3) composite domain terms (e.g. velocity triangle, meridian framework); (4) enum/constant values. "
    "BAN generic words: setup, method, function, class, parameter, value, default, object, system, type, "
    "name, data, code, input, output, example. Each keyphrase must appear verbatim; prefer specific over "
    "general; keep original casing; return at most N, rarest first; empty list if no technical content.\n\n"
    "════════ GOLD STANDARD (CoSApp) — relation_validator ════════\n"
    "You are a KG quality reviewer for CoSApp. For each proposed semantic relation between two chunks, "
    "use BOTH signals and require they AGREE: Signal 1 (keyphrases) — are shared keyphrases domain-specific "
    "class/method names rather than generic words? Signal 2 (content) — do both chunks genuinely discuss the "
    "same concept or a real dependency? is_reliable=True only if BOTH are reliable; False if EITHER fails. "
    "Give confidence ∈ [0,1] and a ≤25-word reason. Default to reliable when confidence < 0.6.\n\n"
    "════════ GOLD STANDARD (CoSApp) — doc_theme ════════\n"
    "You are analysing a CoSApp/CoSApp-Turbo documentation page given as breadcrumbs + excerpts. Extract "
    "3-5 SPECIFIC technical themes. Each theme is domain-specific (e.g. 'meridian aerodynamic solver'), "
    "discriminative, and grounded in the excerpt — never generic ('Python class', 'configuration'). For "
    "each theme write one sentence on what aspect the doc covers. Return doc_id unchanged.\n\n"
    "════════ GOLD STANDARD (CoSApp) — cross_doc_map ════════\n"
    "You receive a compact theme registry (theme name → doc_ids). Identify BRIDGE THEMES appearing in ≥2 "
    "distinct documents. Merge entries expressing the same concept with different wording, accumulate their "
    "doc_ids, reject generic themes, and prefer fewer high-quality bridges. One-sentence description per "
    "bridge. Empty list if no genuine cross-doc bridge exists.\n\n"
    "════════ GOLD STANDARD (CoSApp) — chunk_locator ════════\n"
    "You receive chunks from one CoSApp document and a theme. Identify the 1-3 chunks that BEST and "
    "substantively cover the theme (not incidental keyword mentions). Return 0-based indices ordered by "
    "relevance, at most 3, within range; empty list only if none is genuinely relevant.\n\n"
    "════════ GOLD STANDARD (CoSApp) — direct_pair_validator ════════\n"
    "You receive two chunks A and B from DIFFERENT documents and a shared theme. A direct relation is valid "
    "only if both substantively discuss the theme and reading both together adds real value. If valid, "
    "classify as ONE of: elaboration (B details A), contrast (B is an alternative to A), prerequisite (A "
    "needed before B), example_of (B is a concrete example of A), shared_concept (complementary angles on the "
    "same concept), conditional_behavior (same mechanism behaves differently under different conditions), "
    "operation_comparison (same object processed by different operations), generalization_pattern (general "
    "principle vs specific instantiation), convergent_goal (different mechanisms achieve the same result), "
    "complementary_aspect (distinct orthogonal aspects of the same entity). Prefer the most specific type; "
    "use shared_concept only as fallback. Output is_valid, confidence ∈ [0,1], relation_type (empty if "
    "invalid), one-sentence rationale.\n"
)


class KGPromptsBuilder(PydanticPrompt[PromptBuilderInput, KGPromptsDraft]):
    instruction: str = _KG_INSTRUCTION
    input_model = PromptBuilderInput
    output_model = KGPromptsDraft
    examples = []


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 6 : QualifyPromptsBuilder (question-type qualification)
# ══════════════════════════════════════════════════════════════════════════════

class QualifyPromptsDraft(BaseModel):
    qualify_system: str = Field(
        description="Short system prompt forcing JSON-only output for the question-type "
        "qualification step."
    )
    qualify_user_template: str = Field(
        description="A Python format string with {context_1} and {context_2} placeholders that asks, "
        "per question type, whether a meaningful question of that type can be generated from the two "
        "segments, and demands a JSON {\"compatible_types\": [...]} reply."
    )


_QUALIFY_INSTRUCTION = (
    "You are writing the question-type QUALIFICATION prompts for a RAG benchmark in a NEW domain. "
    "Below is the CoSApp gold standard. Reproduce it for the new domain.\n\n"
    "### CRITICAL REQUIREMENTS\n"
    "  - qualify_user_template MUST be a valid Python .format() string containing EXACTLY the "
    "placeholders {context_1} and {context_2} (and no other single-brace tokens; escape any literal "
    "JSON braces as double braces {{ }}).\n"
    "  - List exactly the provided question_type_names, each with a one-line yes/no test phrased for "
    "the new domain (using domain_name and technical_vocabulary).\n"
    "  - End the template by demanding ONLY a JSON object: {{\"compatible_types\": [...]}} with no "
    "markdown, no prose.\n"
    "  - qualify_system: one or two sentences, expert-on-domain, JSON-only, no code fence.\n\n"
    "════════ GOLD STANDARD (CoSApp) — qualify_system ════════\n"
    "You are an expert on CoSApp framework documentation analysis. Respond ONLY with a valid JSON "
    "object — no markdown, no explanation, no code fence.\n\n"
    "════════ GOLD STANDARD (CoSApp) — qualify_user_template ════════\n"
    "Given these two CoSApp documentation segments:\n\n"
    "<1-hop> {context_1}\n\n"
    "<2-hop> {context_2}\n\n"
    "For each question type below, decide if it is compatible with these two segments (a meaningful "
    "question of that type CAN be generated from them together):\n"
    "- integration      : do the two segments describe components that connect or exchange data?\n"
    "- comparison       : do the two segments describe two elements that can be contrasted?\n"
    "- design_rationale : does one segment explain WHY something in the other was designed that way?\n"
    "- implementation   : do both segments together explain HOW to use or configure something?\n"
    "- enumeration      : do both segments together yield a list of steps/ports/parameters?\n"
    "- factual          : do both segments together yield a precise value/type/parameter?\n\n"
    "Return ONLY this JSON (no markdown, no extra keys):\n"
    "{{\"compatible_types\": [\"integration\", \"comparison\"]}}\n"
)


class QualifyPromptsBuilder(PydanticPrompt[PromptBuilderInput, QualifyPromptsDraft]):
    instruction: str = _QUALIFY_INSTRUCTION
    input_model = PromptBuilderInput
    output_model = QualifyPromptsDraft
    examples = []


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 7 : FewShotBuilderAgent (query_answer_generation few-shots)
# ══════════════════════════════════════════════════════════════════════════════

class FewShotExample(BaseModel):
    persona_name: str = Field(description="Exact name of one of the personas")
    themes: List[str] = Field(description="1-2 technical terms shared by both contexts")
    query_style: str = Field(description="'Perfect grammar' or 'Web search like queries'")
    query_length: str = Field(description="'long', 'medium', or 'short'")
    context_1hop: str = Field(description="First documentation chunk (≥ 100 chars), prefixed '<1-hop>'")
    context_2hop: str = Field(description="Second documentation chunk (≥ 100 chars), prefixed '<2-hop>'")
    question_type: str = Field(description="One of the question type names from the taxonomy")
    query: str = Field(description="The multi-hop question requiring BOTH contexts")
    answer: str = Field(description="Reference answer grounded strictly in both contexts")


class FewShotsDraft(BaseModel):
    examples: List[FewShotExample] = Field(
        description="2 to 3 diverse few-shot examples for query_answer_generation"
    )


_FEWSHOT_INSTRUCTION = (
    "You are generating high-quality few-shot examples for a multi-hop QA generation prompt "
    "in a RAG benchmark pipeline.\n\n"
    "Below are SIX GOLD-STANDARD examples written for the CoSApp / CoSApp-Turbo domain. "
    "Study their structure carefully: realistic documentation contexts with specific class/method "
    "names, diverse styles and lengths, queries that require BOTH contexts, and grounded answers. "
    "Then produce 2-3 examples of the SAME quality for the NEW domain — same structure and "
    "rigour, every CoSApp identifier replaced by terms from the new domain's technical_vocabulary, "
    "question types replaced by the provided question_type_names.\n\n"
    "### CRITICAL: do NOT copy CoSApp text verbatim — adapt fully to the new domain.\n\n"
    "════════ GOLD STANDARD 1 — Perfect grammar / Short ════════\n"
    "persona_name: CoSApp-Turbo Aerodynamics Engineer\n"
    "themes: [\"turbine\"]\n"
    "query_style: Perfect grammar\n"
    "query_length: short\n"
    "question_type: implementation\n"
    "context_1hop: <1-hop> The output JSON file is overwritten if it already exists. "
    "The caller must ensure that the ChannelSet object is fully populated with valid geometric "
    "and aerodynamic data before invoking this function, as no internal validation beyond the "
    "empty-list check is performed. json_dump_trb_ax serializes a turbomachinery channel set "
    "into a CAD export JSON file by extracting the channel list from a ChannelSet.\n"
    "context_2hop: <2-hop> In the project, build_cad_dict is called in the test test_imp_Vm1, "
    "where a Compressor instance (cmp) is passed after running its drivers. The returned "
    "dictionary is then consumed by channelCadBuild.json_cad_to_gbs to reconstruct the "
    "geometric B-spline representation of the machine, confirming that build_cad_dict produces "
    "a fully valid and structured output compatible with downstream CAD tooling. Only "
    "FanWithBPR, Turbine, and Compressor instances are supported.\n"
    "query: How does the build_cad_dict function handle turbine systems, and what structured "
    "output is produced for CAD export when a turbine's channel set is processed?\n"
    "answer: build_cad_dict handles turbine systems by dispatching to build_trb_cad_dict, "
    "passing the turbine's channel set (system.ch_set) as an argument. The function produces "
    "a structured JSON output containing two top-level sections: 'meridian_curves' (hub, mean "
    "line, and shroud B-spline data with constraints and parametrization modes) and "
    "'channel_sets' (blade geometry, aerodynamic section parameters, axial positions z, and "
    "blade types). This output is compatible with channelCadBuild.json_cad_to_gbs for "
    "B-spline reconstruction.\n\n"
    "════════ GOLD STANDARD 2 — Perfect grammar / Medium ════════\n"
    "persona_name: CoSApp-Turbo CAD and Geometry Engineer\n"
    "themes: [\"unstructured3dmeshport\"]\n"
    "query_style: Perfect grammar\n"
    "query_length: medium\n"
    "question_type: integration\n"
    "context_1hop: <1-hop> BladeSectionDeformation is instantiated within "
    "BladeMechanicalPost2's setup method as a child system named 'deformation'. The ports cad, "
    "mesh_deformed, channel_set_cad, and the parameters nu and nv are pulled up to the parent "
    "system level, making them directly accessible. The leading edge radius (r_le) is computed "
    "as the Euclidean distance using the hypot function applied to the x-coordinates of pt_le.\n"
    "context_2hop: <2-hop> The compute method of BladeSectionDeformation accesses specific "
    "attributes of Unstructured3dMeshPort: it retrieves mesh_deformed.vertices to extract 3D "
    "vertex positions for the centerline patch of each blade section by indexing using indices "
    "in mesh_deformed.patches['cl']. It also accesses mesh_deformed.fields to retrieve "
    "parametric UV coordinates. The method relies on the Unstructured3dMeshPort interface "
    "through its vertices, patches, and fields attributes, and uses gbs for B-spline curve "
    "interpolation.\n"
    "query: How does the BladeMechanicalPost2 system integrate the Unstructured3dMeshPort "
    "data from the deformed mesh to compute blade section deformations, and what specific mesh "
    "attributes are accessed during the deformation computation process?\n"
    "answer: BladeMechanicalPost2 pulls up the mesh_deformed input port from its child "
    "BladeSectionDeformation to the parent level. During computation, BladeSectionDeformation "
    "accesses mesh_deformed.vertices (3D vertex positions for the centerline patch via "
    "mesh_deformed.patches['cl'] indices) and mesh_deformed.fields (parametric UV coordinates "
    "for spanwise sections). The extracted 3D points and parametric coordinates are passed to "
    "gbs.interpolate_cn to construct a smooth B-spline curve representing the deformed blade "
    "section centerline.\n\n"
    "════════ GOLD STANDARD 3 — Perfect grammar / Long ════════\n"
    "persona_name: CoSApp Developer\n"
    "themes: [\"blade\"]\n"
    "query_style: Perfect grammar\n"
    "query_length: long\n"
    "question_type: design_rationale\n"
    "context_1hop: <1-hop> extend_rows_3d is a method of FluidMachineryBase that iterates "
    "over all bladed rows registered in the assembly and delegates 3D blade generation to the "
    "extend_3d method, automatically determining the correct extension strategy based on the "
    "aerodynamic nature of each row. The method iterates over self.aero.rows_info, which yields "
    "pairs of (row_name, is_rotating) for each bladed row. For each row, it evaluates the "
    "is_rotating flag to select the appropriate Extend2dMode.\n"
    "context_2hop: <2-hop> Valid aerodynamic state data must be available in self.aero before "
    "calling this method, as the velocity triangle calculations depend on current values of "
    "fl_in.Vm, fl_out.Vm, psi, rpm, and fl_in.Vu. The extend_3d operation completely rebuilds "
    "the sections of the target blade, discarding any previously defined section configurations. "
    "The spans list should include 0.0 and 1.0 to define hub and shroud sections respectively.\n"
    "query: How does the extend_rows_3d method automatically determine the correct blade "
    "extension strategy for each bladed row in the assembly, and what aerodynamic state data "
    "must be available before the extend_3d method can properly generate 3D blade geometry "
    "using the ConstantPsi mode for rotating blade rows?\n"
    "answer: extend_rows_3d iterates over self.aero.rows_info, yielding (row_name, is_rotating) "
    "pairs. For each row it assigns Extend2dMode.ConstantPsi when is_rotating is True (rotor) "
    "or Extend2dMode.Extrude when False (stator), then calls extend_3d with the row name, mode, "
    "and shared spans list. Before extend_3d can generate 3D geometry in ConstantPsi mode, "
    "valid aerodynamic state data must be present in self.aero: fl_in.Vm, fl_out.Vm, psi, rpm, "
    "and fl_in.Vu. The spans list must include 0.0 and 1.0 for hub and shroud; the operation "
    "discards any prior section configurations.\n\n"
    "════════ GOLD STANDARD 4 — Web search / Short ════════\n"
    "persona_name: CoSApp-Turbo Aerodynamics Engineer\n"
    "themes: [\"channelaerosolvermeridian\"]\n"
    "query_style: Web search like queries\n"
    "query_length: short\n"
    "question_type: factual\n"
    "context_1hop: <1-hop> ChannelSetAeroMeridianMultiRotor is a system class that sets up "
    "and manages aerodynamic channel analysis for multi-rotor configurations in a meridian "
    "framework, supporting multiple independent shafts with distinct rotational speeds. rows is "
    "a sequence of tuples (blade_row_name, shaft_index); a negative shaft index means the blade "
    "row belongs to no shaft. rpm_{shaft_index} inward variables (default 0.0, units rpm) are "
    "dynamically added for each unique non-negative shaft index.\n"
    "context_2hop: <2-hop> A ChannelAeroSolverMeridian child named 'solver' is added with "
    "ports 'fl_in', 'fl_out', and 'solver_case' pulled up to the parent level. The mesh output "
    "port of the preprocessor (self.pre.mesh) is connected to the solver's mesh input "
    "(self.solver.mesh) via a PlainConnector. For each (row_name, shaft_index) pair, the method "
    "connects the parent's rpm_{shaft_index} inward to the solver's {row_name}_rpm inward. "
    "Blade rows with negative shaft indices are skipped via a try/except AttributeError block.\n"
    "query: ChannelAeroSolverMeridian multi-rotor rpm variables blade rows connection\n"
    "answer: ChannelSetAeroMeridianMultiRotor connects the preprocessor mesh to the solver via "
    "a PlainConnector. For each unique non-negative shaft index a dynamic inward port "
    "rpm_{shaft_index} is registered. The setup method then connects each parent "
    "rpm_{shaft_index} inward to the solver's {row_name}_rpm inward. Blade rows with negative "
    "shaft indices are silently skipped via try/except AttributeError, designating them as "
    "stationary.\n\n"
    "════════ GOLD STANDARD 5 — Web search / Medium ════════\n"
    "persona_name: Simulation and Optimization Engineer\n"
    "themes: [\"thicknessmodifierbase\"]\n"
    "query_style: Web search like queries\n"
    "query_length: medium\n"
    "question_type: comparison\n"
    "context_1hop: <1-hop> compute is a method of ThicknessModifierBase that executes the "
    "thickness modification pipeline by transforming the input curve f_in into the output curve "
    "f_out through the compute_f_out method. self holds the input curve f_in and stores the "
    "result in f_out.\n"
    "context_2hop: <2-hop> ThicknessModifierBase is a base class that defines the interface "
    "and default pass-through behavior for thickness law modifiers applied to 2D blade section "
    "geometry. f_in is an inward port accepting a BSCfunction representing the input thickness "
    "law, initialized with a default linear B-spline defined by poles [0.05, 0.05]. In the base "
    "class, compute_f_out acts as a pass-through returning the input curve unchanged; subclasses "
    "override compute_f_out to introduce specific thickness modification behavior.\n"
    "query: ThicknessModifierBase compute method compute_f_out relationship template method "
    "pattern blade section\n"
    "answer: ThicknessModifierBase uses a template method pattern: compute() calls "
    "compute_f_out(f_in) and stores the result in f_out. In the base class, compute_f_out is a "
    "pass-through that returns f_in unchanged. Subclasses override compute_f_out to apply custom "
    "thickness modifications while keeping compute() unchanged, ensuring a consistent and "
    "predictable pipeline.\n\n"
    "════════ GOLD STANDARD 6 — Web search / Long ════════\n"
    "persona_name: CoSApp-Turbo Aerodynamics Engineer\n"
    "themes: [\"section_line\"]\n"
    "query_style: Web search like queries\n"
    "query_length: long\n"
    "question_type: enumeration\n"
    "context_1hop: <1-hop> ChannelPlotData is a data container class that stores geometric and "
    "aerodynamic information required to plot axisymmetric turbomachinery channels. r_hub, z_hub, "
    "r_shr, z_shr store radial and axial coordinates for hub and shroud points along the channel "
    "path. le_lines and te_lines accumulate leading and trailing edge line representations and "
    "are only populated for blade-type channels.\n"
    "context_2hop: <2-hop> AxisymPoint is a named tuple representing a point in an axisymmetric "
    "coordinate system with axial (z) and radial (r) coordinates. Index 0 is z (axial) and "
    "index 1 is r (radial). The section_line static method constructs [[r_hub, r_shr], "
    "[z_hub, z_shr]] from hub and shroud AxisymPoint instances via index access "
    "(hub[0], hub[1], shr[0], shr[1]), suitable for 2D plotting routines.\n"
    "query: ChannelPlotData AxisymPoint section_line leading trailing edge axisymmetric blade "
    "channel hub shroud coordinates visualization\n"
    "answer: ChannelPlotData uses the section_line static method to build leading and trailing "
    "edge line representations from hub and shroud AxisymPoint instances. section_line extracts "
    "z (index 0) and r (index 1) from each AxisymPoint and returns [[r_hub, r_shr], "
    "[z_hub, z_shr]] for 2D plotting. add_le_line and add_te_line call section_line for "
    "blade-type channels only; the resulting segments are accumulated in le_lines and te_lines.\n\n"
    "════════ END OF GOLD STANDARDS ════════\n\n"
    "Now produce 2-3 diverse few-shot examples for the NEW domain following the same structure.\n\n"
    "### Persona profiles (use persona_descriptions — do NOT flatten to generic roles)\n"
    "The input field persona_descriptions maps each persona name to their full role_description "
    "(expertise level, sub-systems they use, concrete example questions they would ask). "
    "Each generated few-shot example MUST reflect its persona's actual domain concerns: "
    "reference the specific APIs, classes, and parameters listed in their profile. "
    "A persona's question should feel like it came from someone with that exact background.\n\n"
    "### Real corpus contexts (MANDATORY when sample_chunk_pairs is provided)\n"
    "If the input field sample_chunk_pairs is non-empty, you MUST use them AS-IS as "
    "context_1hop and context_2hop. Copy the text verbatim (trim to ≥100 chars if needed). "
    "Do NOT paraphrase, summarize, or invent new contexts when real ones are available. "
    "This is CRITICAL: few-shot examples must be anchored in real corpus data so the "
    "generation prompt matches the actual distribution of chunks the pipeline will encounter. "
    "If sample_chunk_pairs is empty, write realistic synthetic contexts following the gold "
    "standards above — never generic placeholder text.\n\n"
    "### Question type definitions (use question_type_descriptions — do NOT reinvent)\n"
    "The input field question_type_descriptions maps each question type name to its exact "
    "cognitive definition. Set question_type to one of question_type_names and ensure the "
    "query and answer actually exercise the cognitive operation described in its definition.\n\n"
    "### Each example MUST\n"
    "  1. Use a persona_name EXACTLY as given in persona_names.\n"
    "  2. Use themes that are real technical terms from technical_vocabulary.\n"
    "  3. Have context_1hop and context_2hop that are REALISTIC documentation snippets "
    "for the domain (≥ 100 chars each), mimicking the density and specificity of the gold "
    "standards above. Prefix them with '<1-hop>' and '<2-hop>'.\n"
    "  4. Set question_type to one of question_type_names.\n"
    "  5. Have a query that requires BOTH contexts to answer — NOT answerable from one alone.\n"
    "  6. Have an answer that is concise, technically accurate, and grounded in both contexts.\n\n"
    "### Diversity rules\n"
    "  - Use different question_type values across examples.\n"
    "  - Mix query_style: at least one 'Perfect grammar' and one 'Web search like queries'.\n"
    "  - Mix query_length: at least one 'long' (≥20 words) and one 'short' (≤9 words) or "
    "'medium' (10-19 words).\n"
    "  - Use different personas across examples.\n\n"
    "### Quality bar\n"
    "  - Contexts must contain specific class names, method names, or parameter names "
    "from technical_vocabulary — never generic text.\n"
    "  - The query must name at least one identifier from technical_vocabulary.\n"
    "  - 'Perfect grammar' → ends with '?'. 'Web search like queries' → no '?'.\n"
    "  - Do NOT copy CoSApp text verbatim — adapt fully to the new domain.\n"
)


class FewShotBuilderAgent(PydanticPrompt[PromptBuilderInput, FewShotsDraft]):
    instruction: str = _FEWSHOT_INSTRUCTION
    input_model = PromptBuilderInput
    output_model = FewShotsDraft
    examples = []


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 8 : KGFewShotBuilderAgent (relation_validator + direct_pair_validator)
# ══════════════════════════════════════════════════════════════════════════════

class KGRelationExample(BaseModel):
    relation_type: str = Field(description="e.g. 'keyphrases_overlap' or 'cosine_similarity'")
    source_breadcrumb: str = Field(description="Section path of source chunk")
    target_breadcrumb: str = Field(description="Section path of target chunk")
    source_content: str = Field(description="Source chunk content (100-300 chars)")
    target_content: str = Field(description="Target chunk content (100-300 chars)")
    shared_keyphrases: List[str] = Field(description="2-4 shared keyphrases")
    is_reliable: bool = Field(description="True if relation is genuine, False if spurious")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="≤25 words explaining why reliable or not")


class KGPairExample(BaseModel):
    chunk_a_breadcrumb: str
    chunk_a_doc_id: str
    chunk_a_content: str = Field(description="Chunk A content (100-300 chars)")
    chunk_b_breadcrumb: str
    chunk_b_doc_id: str
    chunk_b_content: str = Field(description="Chunk B content (100-300 chars)")
    shared_theme: str
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    relation_type: str = Field(description="elaboration/contrast/prerequisite/example_of/shared_concept or empty")
    rationale: str = Field(description="One sentence explaining the decision")


class KGFewShotsDraft(BaseModel):
    relation_validator_reliable: KGRelationExample = Field(
        description="One example of a RELIABLE relation (is_reliable=True) — domain-specific "
        "keyphrases AND genuine content overlap."
    )
    relation_validator_unreliable: KGRelationExample = Field(
        description="One example of an UNRELIABLE relation (is_reliable=False) — generic "
        "keyphrases OR no real content dependency."
    )
    direct_pair_valid: KGPairExample = Field(
        description="One example of a VALID direct pair (is_valid=True) — both chunks "
        "substantively discuss the shared theme."
    )
    direct_pair_invalid: KGPairExample = Field(
        description="One example of an INVALID direct pair (is_valid=False) — theme is "
        "only incidentally mentioned or chunks are disconnected."
    )


_KG_FEWSHOT_INSTRUCTION = (
    "You are generating calibration examples for the knowledge-graph validation agents "
    "of a RAG benchmark pipeline.\n\n"
    "These examples teach the LLM judges what 'reliable' vs 'unreliable' LOOKS LIKE "
    "for this specific domain. Without them, the judges have no reference point for "
    "scoring confidence.\n\n"
    "### What to generate\n"
    "1. **relation_validator_reliable**: A relation where shared_keyphrases are "
    "domain-specific identifiers from technical_vocabulary AND both chunks genuinely "
    "discuss the same technical concept. Set is_reliable=True, confidence ≥0.85.\n\n"
    "2. **relation_validator_unreliable**: A relation where shared_keyphrases are generic "
    "words (e.g. 'method', 'configuration', 'process') OR the chunks discuss unrelated "
    "topics despite sharing keywords. Set is_reliable=False, confidence ≥0.80.\n\n"
    "3. **direct_pair_valid**: Two chunks from DIFFERENT documents that both substantively "
    "discuss a shared theme from technical_vocabulary. Reading both adds real value. "
    "Set is_valid=True, pick the best relation_type, confidence ≥0.85.\n\n"
    "4. **direct_pair_invalid**: Two chunks that only incidentally mention the shared "
    "theme, or discuss it from entirely disconnected angles. Set is_valid=False, "
    "relation_type='', confidence ≥0.80.\n\n"
    "### CRITICAL RULES\n"
    "  - Use ONLY terms from technical_vocabulary as shared_keyphrases / shared_theme.\n"
    "  - If sample_chunk_pairs is provided, use those real excerpts for chunk content. "
    "Otherwise write realistic synthetic content (100-300 chars) that sounds like actual "
    "documentation for this domain — never generic placeholder text.\n"
    "  - The 'reason' and 'rationale' fields must be ≤25 words and cite the specific "
    "signal that passed or failed.\n"
    "  - source_breadcrumb / chunk_a_breadcrumb should use realistic section paths "
    "for the domain (e.g. 'Domain > SubTopic > ClassName').\n"
    "  - For the unreliable/invalid examples, make it CLEAR why they fail — the LLM "
    "judge must learn to distinguish borderline cases.\n"
)


class KGFewShotBuilderAgent(PydanticPrompt[PromptBuilderInput, KGFewShotsDraft]):
    instruction: str = _KG_FEWSHOT_INSTRUCTION
    input_model = PromptBuilderInput
    output_model = KGFewShotsDraft
    examples = []


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION LAYER (rule-based, no LLM)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigIssue:
    severity: Literal["error", "warning"]
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


def validate_config(cfg: PipelineConfig) -> List[ConfigIssue]:
    """Rule-based consistency checks. Returns [] if config is valid."""
    issues: List[ConfigIssue] = []
    type_names = cfg.question_type_names()

    # ── Budget ────────────────────────────────────────────────────────────────
    total = sum(cfg.taxonomy.budget.values())
    if not (0.98 <= total <= 1.02):
        issues.append(ConfigIssue("error", "taxonomy.budget",
            f"Proportions sum to {total:.4f}, expected 1.0 ±0.02"))
    for t, frac in cfg.taxonomy.budget.items():
        if t not in type_names:
            issues.append(ConfigIssue("error", f"taxonomy.budget.{t}",
                f"Budget key '{t}' not found in taxonomy.types"))
        if not (0.0 <= frac <= 1.0):
            issues.append(ConfigIssue("error", f"taxonomy.budget.{t}",
                f"Fraction {frac} outside [0, 1]"))
    for t in type_names:
        if t not in cfg.taxonomy.budget:
            issues.append(ConfigIssue("warning", "taxonomy.budget",
                f"Type '{t}' defined in taxonomy.types but missing from budget"))

    # ── relation_to_question_types ────────────────────────────────────────────
    valid_relations = {"prerequisite", "contrast", "elaboration", "example_of", "shared_concept"}
    for rel, qtypes in cfg.taxonomy.relation_to_question_types.items():
        if rel not in valid_relations:
            issues.append(ConfigIssue("warning", f"taxonomy.relation_to_question_types.{rel}",
                f"Unknown relation name '{rel}' (expected one of {sorted(valid_relations)})"))
        for qt in qtypes:
            if qt not in type_names:
                issues.append(ConfigIssue("error", f"taxonomy.relation_to_question_types.{rel}",
                    f"Question type '{qt}' referenced but not in taxonomy.types"))
    for rel in valid_relations:
        if rel not in cfg.taxonomy.relation_to_question_types:
            issues.append(ConfigIssue("warning", "taxonomy.relation_to_question_types",
                f"RAGAS relation '{rel}' has no mapping"))

    # ── Personas ──────────────────────────────────────────────────────────────
    if not cfg.personas:
        issues.append(ConfigIssue("error", "personas", "At least one persona must be defined"))
    persona_names = {p.name for p in cfg.personas}
    for ex in cfg.few_shots.query_generation:
        pname = (ex.get("input") or {}).get("persona_name", "")
        if pname and pname not in persona_names:
            issues.append(ConfigIssue("warning", "few_shots.query_generation",
                f"Few-shot example references persona '{pname}' not in personas list"))

    # ── Types minimum ─────────────────────────────────────────────────────────
    if len(type_names) < 2:
        issues.append(ConfigIssue("error", "taxonomy.types",
            "At least 2 question types must be defined"))

    # ── Numeric thresholds ────────────────────────────────────────────────────
    ev = cfg.evaluation
    kg = cfg.kg_enrichment
    _check_range(issues, "evaluation.qa_eval_threshold", ev.qa_eval_threshold, 0.0, 1.0)
    _check_range(issues, "evaluation.relation_validator_confidence_threshold",
                 ev.relation_validator_confidence_threshold, 0.0, 1.0)
    _check_range(issues, "evaluation.discovery_min_confidence", ev.discovery_min_confidence, 0.0, 1.0)
    _check_range(issues, "evaluation.scenario_buffer_ratio", ev.scenario_buffer_ratio, 0.0, 2.0)
    _check_range(issues, "kg_enrichment.cosine_sim_min", kg.cosine_sim_min, 0.0, 1.0)
    _check_range(issues, "kg_enrichment.cosine_sim_max", kg.cosine_sim_max, 0.0, 1.0)
    if kg.cosine_sim_min >= kg.cosine_sim_max:
        issues.append(ConfigIssue("error", "kg_enrichment",
            f"cosine_sim_min ({kg.cosine_sim_min}) must be < cosine_sim_max ({kg.cosine_sim_max})"))
    _check_range(issues, "kg_enrichment.jaccard_kp_threshold", kg.jaccard_kp_threshold, 0.0, 1.0)
    _check_range(issues, "kg_enrichment.idf_threshold", kg.idf_threshold, 0.0, 20.0)
    if ev.max_retry < 0:
        issues.append(ConfigIssue("error", "evaluation.max_retry", "max_retry must be ≥ 0"))
    if ev.max_context_chars < 1000:
        issues.append(ConfigIssue("warning", "evaluation.max_context_chars",
            f"max_context_chars={ev.max_context_chars} is very small (< 1000)"))

    # ── Query params ──────────────────────────────────────────────────────────
    valid_styles = {"perfect_grammar", "web_search_like", "misspelled", "poor_grammar"}
    for s in cfg.query_params.styles:
        if s not in valid_styles:
            issues.append(ConfigIssue("warning", "query_params.styles",
                f"Style '{s}' is not a known RAGAS QueryStyle value"))
    valid_lengths = {"long", "medium", "short"}
    for length in cfg.query_params.lengths:
        if length not in valid_lengths:
            issues.append(ConfigIssue("warning", "query_params.lengths",
                f"Length '{length}' is not a known RAGAS QueryLength value"))

    # ── qualify_user_template placeholder + .format() safety ─────────────────
    tmpl = cfg.prompts.qualify_user_template
    if "{context_1}" not in tmpl or "{context_2}" not in tmpl:
        issues.append(ConfigIssue("error", "prompts.qualify_user_template",
            "Template must contain {context_1} and {context_2} placeholders"))
    try:
        tmpl.format(context_1="X", context_2="Y")
    except (KeyError, IndexError, ValueError) as exc:
        issues.append(ConfigIssue("error", "prompts.qualify_user_template",
            f"Template is not a valid .format() string ({exc}); escape literal braces as {{{{ }}}}"))

    return issues


def _check_range(issues: List[ConfigIssue], field: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        issues.append(ConfigIssue("error", field,
            f"Value {value} outside expected range [{lo}, {hi}]"))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# The 10 UNIVERSAL prompts (domain-agnostic, GEPA-optimized / frozen).
# They are the single source of truth: config generation injects them at creation
# time AND resync_universal_prompts() re-applies them to old sessions whose stored
# copies have drifted from the current code.
_UNIVERSAL_PROMPT_KEYS = (
    "query_generation", "answer_generation", "qa_evaluator",
    "no_context_system", "single_context_system", "relation_validator",
    "doc_theme", "cross_doc_map", "chunk_locator", "direct_pair_validator",
)

# The 3 SESSION-SPECIFIC prompts (generated per-domain by the LLM). Never touched
# by the resync — they are the only editable prompts in the UI.
_SESSION_PROMPT_KEYS = (
    "keyphrase_extractor", "qualify_system", "qualify_user_template",
)


def universal_prompts() -> Dict[str, str]:
    """Return the 10 UNIVERSAL prompts (hardcoded, GEPA-optimized / frozen).

    Single source of truth reused by both config generation and the resync of
    existing sessions, so a session's universal prompts always match the code.
    query_generation / answer_generation come from pipeline_config defaults.
    """
    from pipeline_config import (
        _DEFAULT_QUERY_GENERATION_PROMPT,
        _DEFAULT_ANSWER_GENERATION_PROMPT,
    )
    return {
        "query_generation":  _DEFAULT_QUERY_GENERATION_PROMPT,
        "answer_generation": _DEFAULT_ANSWER_GENERATION_PROMPT,
        "qa_evaluator": (
            "You are a quality reviewer for a RAG benchmark. "
            "Score a (question, answer) pair against three independent criteria: "
            "(1) multi_hop_necessity — does answering the question genuinely require "
            "information from BOTH context segments? (2) groundedness — is every "
            "claim in the answer traceable to the provided contexts? "
            "(3) answer_accuracy — is the answer factually correct and complete "
            "given the contexts? Score each criterion 0.0-1.0. A question passes "
            "only if ALL three scores exceed the threshold."
        ),
        "no_context_system": (
            "You are a knowledgeable expert. Answer the following question using "
            "only your general knowledge. Be concise and technically precise. "
            "If you are unsure, say so rather than hallucinating."
        ),
        "single_context_system": (
            "You are an expert assistant. Answer using ONLY the provided context. "
            "Be concise and technically precise. Do not add information beyond "
            "what is stated in the context."
        ),
        "relation_validator": (
            "You are a KG quality reviewer. For each proposed semantic relation "
            "between two chunks, assess whether the relation is GENUINE (both chunks "
            "substantively discuss related topics, not mere keyword overlap). "
            "Use BOTH structural signals (shared keyphrases, overlap score) AND "
            "semantic coherence. Output a verdict: is_reliable (bool), confidence "
            "(0.0-1.0), and a brief reason."
        ),
        "doc_theme": (
            "You are analysing a documentation page given as breadcrumbs + excerpts. "
            "Extract 3-5 SPECIFIC technical themes. Each theme must be a precise "
            "concept, not a generic category. Return theme names and brief descriptions."
        ),
        "cross_doc_map": (
            "You receive a compact theme registry (theme name → doc_ids). "
            "Identify BRIDGE THEMES appearing in ≥2 distinct documents. "
            "Merge entries expressing the same concept differently. "
            "Return the consolidated bridge themes with their document sets."
        ),
        "chunk_locator": (
            "You receive chunks from one document and a theme. "
            "Identify the 1-3 chunks that BEST and substantively cover the theme "
            "(not incidental keyword matches). Return chunk indices and confidence."
        ),
        "direct_pair_validator": (
            "You receive two chunks A and B from DIFFERENT documents and a shared theme. "
            "A direct relation is valid only if both substantively discuss the same "
            "concept and reading them together reveals complementary information "
            "(definition/use, cause/effect, process/sub-process). "
            "Return: is_valid, confidence (0.0-1.0), relation_type, rationale."
        ),
    }


def resync_universal_prompts(cfg_dict: dict) -> tuple[dict, list[str]]:
    """Re-apply the current UNIVERSAL prompts to an (old) session config dict.

    Old sessions embed a snapshot of the 10 universal prompts taken at creation
    time; when the hardcoded prompts evolve in the code, those snapshots drift.
    This overwrites the 10 universal prompts with the current code values while
    PRESERVING the 3 session-specific prompts (keyphrase_extractor, qualify_system,
    qualify_user_template) and everything else in the config.

    Also migrates the legacy `query_answer_generation` field to the split
    `query_generation` + `answer_generation` form so nothing stays inconsistent.

    Returns (updated_cfg_dict, changed_keys).
    """
    import copy
    cfg_dict = copy.deepcopy(cfg_dict)
    prompts = dict(cfg_dict.get("prompts") or {})

    # Drop the legacy combined field — universal_prompts() provides the split form.
    prompts.pop("query_answer_generation", None)

    current = universal_prompts()
    changed: list[str] = []
    for key, value in current.items():
        if prompts.get(key) != value:
            changed.append(key)
        prompts[key] = value

    cfg_dict["prompts"] = prompts
    return cfg_dict, changed


def _build_fallback_qualify_template(qtype_names: List[str]) -> str:
    """Deterministic, always-valid qualify_user_template if the LLM's output is broken."""
    lines = [
        "Given these two documentation segments:",
        "",
        "<1-hop> {context_1}",
        "",
        "<2-hop> {context_2}",
        "",
        "For each question type below, decide if a meaningful question of that type "
        "CAN be generated from the two segments together:",
    ]
    for name in qtype_names:
        lines.append(f"- {name}")
    lines += [
        "",
        "Return ONLY this JSON (no markdown, no extra keys):",
        '{{"compatible_types": ["' + (qtype_names[0] if qtype_names else "integration") + '"]}}',
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def run_config_agent(
    llm: Any,
    description: str = "",
    samples: Optional[List[str]] = None,
    chunk_pairs: Optional[List[Dict[str, str]]] = None,
    base_config: Optional[PipelineConfig] = None,
) -> Tuple[PipelineConfig, List[ConfigIssue]]:
    """
    Run the 7-stage config agent and return (PipelineConfig, issues).

    Stages 1-3 are sequential (each feeds the next).
    Stages 4-5-6 run in PARALLEL (asyncio.gather) for speed.
    Stage 7 is rule-based validation (no LLM).
    """
    if not description and not samples:
        raise ValueError("Provide at least one of: description, samples")

    samples = samples or []

    # ── Stage 1: CorpusAnalyzerAgent ─────────────────────────────────────────
    logging.info("ConfigAgent [1/7] CorpusAnalyzerAgent...")
    try:
        corpus_profile: CorpusProfile = await CorpusAnalyzerPrompt().generate(
            llm=llm,
            data=CorpusAnalyzerInput(description=description, samples=samples[:10]),
        )
    except Exception as exc:
        raise RuntimeError(f"CorpusAnalyzerAgent failed: {exc}") from exc
    logging.info("  domain='%s' style=%s", corpus_profile.domain_name, corpus_profile.document_style)

    # ── Stage 2: TaxonomyDesignerAgent ───────────────────────────────────────
    logging.info("ConfigAgent [2/7] TaxonomyDesignerAgent...")
    try:
        taxonomy_proposal: TaxonomyProposal = await TaxonomyDesignerPrompt().generate(
            llm=llm, data=corpus_profile
        )
    except Exception as exc:
        raise RuntimeError(f"TaxonomyDesignerAgent failed: {exc}") from exc
    logging.info("  %d question types", len(taxonomy_proposal.question_types))

    # ── Stage 3: PersonaGeneratorAgent ───────────────────────────────────────
    logging.info("ConfigAgent [3/7] PersonaGeneratorAgent...")
    try:
        persona_proposal: PersonaProposal = await PersonaGeneratorPrompt().generate(
            llm=llm, data=corpus_profile
        )
    except Exception as exc:
        raise RuntimeError(f"PersonaGeneratorAgent failed: {exc}") from exc
    logging.info("  %d personas", len(persona_proposal.personas))

    # ── Shared builder input (used by stages 4-5-6) ──────────────────────────
    qtype_names = [t.name for t in taxonomy_proposal.question_types]
    builder_input = PromptBuilderInput(
        domain_name=corpus_profile.domain_name,
        domain_description=corpus_profile.domain_description,
        technical_vocabulary=corpus_profile.technical_vocabulary,
        document_style=corpus_profile.document_style,
        chunk_examples=corpus_profile.chunk_examples,
        question_type_names=qtype_names,
        persona_names=[p.name for p in persona_proposal.personas],
        # Enriched context — eliminates LLM reinvention of definitions/personas
        question_type_descriptions={
            t.name: t.description for t in taxonomy_proposal.question_types
        },
        persona_descriptions={
            p.name: p.role_description for p in persona_proposal.personas
        },
        sample_chunk_pairs=(chunk_pairs or [])[:3],
    )

    # ── Stages 4-8 : PARALLEL prompt builders + few-shot generators ──────────
    logging.info("ConfigAgent [4-8/8] GenEval + KG + Qualify + FewShots + KGFewShots (parallel)...")
    try:
        gen_eval, kg_prompts, qualify, fewshots_draft, kg_fewshots_draft = await asyncio.gather(
            GenEvalPromptsBuilder().generate(llm=llm, data=builder_input),
            KGPromptsBuilder().generate(llm=llm, data=builder_input),
            QualifyPromptsBuilder().generate(llm=llm, data=builder_input),
            FewShotBuilderAgent().generate(llm=llm, data=builder_input),
            KGFewShotBuilderAgent().generate(llm=llm, data=builder_input),
        )
    except Exception as exc:
        raise RuntimeError(f"Prompt builders failed: {exc}") from exc

    # ── Repair qualify_user_template if the LLM dropped a placeholder ─────────
    qtmpl = qualify.qualify_user_template
    if "{context_1}" not in qtmpl or "{context_2}" not in qtmpl:
        logging.warning("qualify_user_template missing placeholders — using safe fallback")
        qtmpl = _build_fallback_qualify_template(qtype_names)
    else:
        try:
            qtmpl.format(context_1="X", context_2="Y")
        except (KeyError, IndexError, ValueError):
            logging.warning("qualify_user_template not .format()-safe — using safe fallback")
            qtmpl = _build_fallback_qualify_template(qtype_names)

    # ── Convert FewShotsDraft → FewShotsConfig-compatible dict lists ─────────
    _qa_few_shots: List[Dict] = []
    for ex in fewshots_draft.examples:
        try:
            _qa_few_shots.append({
                "input": {
                    "persona_name": ex.persona_name,
                    "themes":       ex.themes,
                    "query_style":  ex.query_style,
                    "query_length": ex.query_length,
                    "context": [ex.context_1hop, ex.context_2hop],
                },
                "output": {
                    "query":  ex.query,
                    "answer": ex.answer,
                },
            })
        except Exception as _fe:
            logging.warning("FewShotBuilderAgent: skipping malformed example (%s)", _fe)
    logging.info("ConfigAgent: %d few-shot example(s) generated", len(_qa_few_shots))

    # ── Convert KGFewShotsDraft → FewShotsConfig-compatible dict lists ────────
    _rv_few_shots: List[Dict] = []
    _dpv_few_shots: List[Dict] = []
    try:
        # relation_validator: 2 examples (reliable + unreliable)
        for ex in [kg_fewshots_draft.relation_validator_reliable,
                   kg_fewshots_draft.relation_validator_unreliable]:
            _rv_few_shots.append({
                "input": {
                    "relations": [{
                        "relation_id": 0,
                        "relation_type": ex.relation_type,
                        "source_breadcrumb": ex.source_breadcrumb,
                        "target_breadcrumb": ex.target_breadcrumb,
                        "source_content": ex.source_content,
                        "target_content": ex.target_content,
                        "shared_keyphrases": ex.shared_keyphrases,
                    }],
                },
                "output": {
                    "verdicts": [{
                        "relation_id": 0,
                        "is_reliable": ex.is_reliable,
                        "confidence": ex.confidence,
                        "reason": ex.reason,
                    }],
                },
            })
        # direct_pair_validator: 2 examples (valid + invalid)
        for ex in [kg_fewshots_draft.direct_pair_valid,
                   kg_fewshots_draft.direct_pair_invalid]:
            _dpv_few_shots.append({
                "input": {
                    "chunk_a": {
                        "breadcrumb": ex.chunk_a_breadcrumb,
                        "doc_id": ex.chunk_a_doc_id,
                        "content": ex.chunk_a_content,
                    },
                    "chunk_b": {
                        "breadcrumb": ex.chunk_b_breadcrumb,
                        "doc_id": ex.chunk_b_doc_id,
                        "content": ex.chunk_b_content,
                    },
                    "shared_theme": ex.shared_theme,
                    "theme_description": "",
                },
                "output": {
                    "is_valid": ex.is_valid,
                    "confidence": ex.confidence,
                    "relation_type": ex.relation_type,
                    "rationale": ex.rationale,
                },
            })
        logging.info(
            "ConfigAgent: KG few-shots generated — %d relation_validator, %d direct_pair_validator",
            len(_rv_few_shots), len(_dpv_few_shots),
        )
    except Exception as _kfe:
        logging.warning("KGFewShotBuilderAgent: conversion failed (%s) — KG few-shots will be empty", _kfe)

    # ── Assemble PromptsConfig ────────────────────────────────────────────────
    # 10 prompts are UNIVERSAL (domain-agnostic, hardcoded) — they never change
    # between sessions. Only 3 prompts are session-specific (generated by LLM):
    #   - keyphrase_extractor  (domain vocabulary)
    #   - qualify_system       (domain question types)
    #   - qualify_user_template (domain qualification logic)
    prompts_kwargs = {
        # ── 10 UNIVERSAL prompts (hardcoded, GEPA-optimized / frozen) ────────
        **universal_prompts(),
        # ── 3 SESSION-SPECIFIC prompts (generated by LLM, domain-adapted) ────
        "keyphrase_extractor":     kg_prompts.keyphrase_extractor,
        "qualify_system":          qualify.qualify_system,
        "qualify_user_template":   qtmpl,
    }

    # ── Normalise budget to exactly 1.0 ──────────────────────────────────────
    raw_budget = {t.name: t.budget_fraction for t in taxonomy_proposal.question_types}
    total = sum(raw_budget.values()) or 1.0
    budget = {k: round(v / total, 6) for k, v in raw_budget.items()}
    diff = 1.0 - sum(budget.values())
    if budget:
        first_key = next(iter(budget))
        budget[first_key] = round(budget[first_key] + diff, 6)

    # ── Apply the same legacy→split migration used for YAML loading ──────────
    # Build the raw prompts + few_shots dicts (with the legacy single field),
    # then run the shared migration helper so `query_answer_generation` becomes
    # `query_generation` + `answer_generation` consistently.
    _few_shots_raw = {
        "query_answer_generation": _qa_few_shots,
        "relation_validator": _rv_few_shots,
        "direct_pair_validator": _dpv_few_shots,
    }
    _migrate_pc = _migrate_query_answer_split({
        "prompts": prompts_kwargs,
        "few_shots": _few_shots_raw,
    })

    # ── Base config defaults ──────────────────────────────────────────────────
    if base_config is not None:
        base_query_params  = base_config.query_params
        base_kg_enrichment = base_config.kg_enrichment
        base_chunking      = base_config.chunking
        base_evaluation    = base_config.evaluation
        base_models        = base_config.models
    else:
        base_query_params  = QueryParamsConfig()
        base_kg_enrichment = KGEnrichmentConfig()
        base_chunking      = ChunkingConfig()
        base_evaluation    = EvaluationConfig()
        base_models        = ModelsConfig()

    # ── Assemble PipelineConfig ───────────────────────────────────────────────
    cfg = PipelineConfig(
        meta=MetaConfig(
            session_id=f"{corpus_profile.domain_name.lower().replace(' ', '_')}_draft",
            description=f"Auto-generated config for: {corpus_profile.domain_name}",
        ),
        domain=DomainConfig(
            name=corpus_profile.domain_name,
            description=corpus_profile.domain_description,
            language=corpus_profile.suggested_language,
            domain_vocabulary=corpus_profile.technical_vocabulary,
        ),
        prompts=PromptsConfig(**_migrate_pc["prompts"]),
        few_shots=FewShotsConfig(**_migrate_pc["few_shots"]),
        personas=persona_proposal.personas,
        taxonomy=TaxonomyConfig(
            types=[
                QuestionTypeDef(name=t.name, description=t.description)
                for t in taxonomy_proposal.question_types
            ],
            budget=budget,
            relation_to_question_types=taxonomy_proposal.relation_to_question_types,
            relation_to_answer_structure=taxonomy_proposal.relation_to_answer_structure,
        ),
        query_params=base_query_params,
        kg_enrichment=KGEnrichmentConfig(
            blacklist_words=base_kg_enrichment.blacklist_words,
            blacklist_phrases=base_kg_enrichment.blacklist_phrases,
            domain_blacklist=base_kg_enrichment.domain_blacklist,
            regex_blacklist_patterns=base_kg_enrichment.regex_blacklist_patterns,
            idf_threshold=base_kg_enrichment.idf_threshold,
            jaccard_kp_threshold=base_kg_enrichment.jaccard_kp_threshold,
            cosine_sim_min=base_kg_enrichment.cosine_sim_min,
            cosine_sim_max=base_kg_enrichment.cosine_sim_max,
            cosine_anti_dup_jaccard=base_kg_enrichment.cosine_anti_dup_jaccard,
            overlap_score_threshold=base_kg_enrichment.overlap_score_threshold,
            overlap_distance_threshold=base_kg_enrichment.overlap_distance_threshold,
            shared_keyphrase_min_count=base_kg_enrichment.shared_keyphrase_min_count,
            shared_keyphrase_min_kps=base_kg_enrichment.shared_keyphrase_min_kps,
            max_keyphrases=base_kg_enrichment.max_keyphrases,
            semantic_relation_types=base_kg_enrichment.semantic_relation_types,
            structural_relation_types=base_kg_enrichment.structural_relation_types,
        ),
        chunking=base_chunking,
        evaluation=base_evaluation,
        models=base_models,
    )

    issues = validate_config(cfg)
    error_count = sum(1 for i in issues if i.severity == "error")
    logging.info(
        "ConfigAgent complete. Validation: %d error(s), %d warning(s)",
        error_count,
        sum(1 for i in issues if i.severity == "warning"),
    )

    return cfg, issues


def run_config_agent_sync(
    llm: Any,
    description: str = "",
    samples: Optional[List[str]] = None,
    chunk_pairs: Optional[List[Dict[str, str]]] = None,
    base_config: Optional[PipelineConfig] = None,
) -> Tuple[PipelineConfig, List[ConfigIssue]]:
    """Synchronous wrapper around run_config_agent."""
    import asyncio as _asyncio
    # Use DefaultEventLoopPolicy to avoid uvloop.Loop incompatibility
    _asyncio.set_event_loop_policy(_asyncio.DefaultEventLoopPolicy())
    loop = _asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            run_config_agent(llm, description, samples, chunk_pairs, base_config)
        )
    finally:
        loop.close()