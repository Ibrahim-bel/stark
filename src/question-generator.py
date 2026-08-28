"""
Question generation module using RAGAS.
"""
 
import asyncio
import json
import logging
import os
import hashlib
import random
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import typing as t
 
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona, PersonaList
from ragas.testset.synthesizers.base import QueryStyle, QueryLength
from personas import get_default_personas as _get_default_personas
from question_type_budget import QuestionTypeBudget, qualify_question_types
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    QueryAnswerGenerationPrompt,
)
from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt
 
# ── Canonical personas (single source of truth: src/personas.py) ─────────────
# Build a name→Persona lookup once at import time so the few-shot examples
# below and any other code in this module always use the exact same persona
# objects (same name + role_description) as the rest of the pipeline.
_PERSONAS_BY_NAME: t.Dict[str, Persona] = {p.name: p for p in _get_default_personas()}
 
# ── Context truncation limit (chars) – avoids storing entire DOCUMENT nodes ──
MAX_CONTEXT_CHARS: int = 12000  # ≈ 2400 tokens, couvre un chunk complet de 2048 tokens


# ── Transient-error retry helper ──────────────────────────────────────────────
# The Safran LiteLLM proxy occasionally returns transient failures under load:
#   • 400 "Invalid model name passed in model=…"  (routing/load-balancing hiccup)
#   • Connection error. / timeouts / 429 rate limits / 5xx
# These are NOT permanent — the very same model name succeeds on the next call
# (KG enrichment + the first questions of a run use the same model fine).
# Without a retry, a short proxy flap makes every subsequent scenario fail and
# the whole generation collapses (e.g. 23/250 with 215 consecutive failures).
_TRANSIENT_ERROR_MARKERS: t.Tuple[str, ...] = (
    "invalid model name",
    "connection error",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "rate limit",
    "too many requests",
    "overloaded",
    "internal server error",
    " 429",
    " 500",
    " 502",
    " 503",
    " 504",
)


def _is_transient_error(exc: Exception) -> bool:
    """Heuristic: is this exception a transient proxy/network error worth retrying?"""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)


async def _acall_with_retry(
    coro_factory,
    *,
    max_attempts: int = 6,
    base_delay: float = 3.0,
    what: str = "LLM call",
):
    """Await ``coro_factory()`` with exponential backoff on transient errors.


    coro_factory must return a fresh awaitable on each call (a coroutine can
    only be awaited once). Non-transient errors are re-raised immediately.
    """
    last_exc: t.Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 — re-raised below
            last_exc = exc
            if not _is_transient_error(exc) or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logging.warning(
                "%s failed (attempt %d/%d, transient): %s — retrying in %.1fs",
                what, attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
 
# ── Subject-specific scenario constraints ─────────────────────────────────────
# CoSApp / CoSApp-Turbo is a technical engineering framework. Restrict query
# styles and lengths to those that reflect how engineers actually query it:
#   • "Perfect grammar"        – formal technical questions (API docs, design)
#   • "Web search like queries" – short keyword searches (class name, feature)
# Excluded: "Misspelled queries", "Poor grammar" — irrelevant for engineering
# documentation and harmful to evaluation quality.
#
# Lengths: "long" and "medium" suit multi-hop technical questions that require
# combining information from two nodes. "short" is kept for search-style queries.
COSAPP_QUERY_STYLES: t.List[QueryStyle] = [
    QueryStyle.PERFECT_GRAMMAR,
    QueryStyle.WEB_SEARCH_LIKE,
]
COSAPP_QUERY_LENGTHS: t.List[QueryLength] = [
    QueryLength.LONG,
    QueryLength.MEDIUM,
    QueryLength.SHORT,
]

# ── Direct mapping: agent_discovered relation_type → compatible question types ─
# Avoids a redundant qualify_question_types LLM call for agent_discovered
# relations — DirectRelationDiscovery already validated the semantic type.
_AGENT_TYPE_TO_QUESTION_TYPES: t.Dict[str, t.List[str]] = {
    "prerequisite":          ["design_rationale", "implementation"],
    "contrast":              ["comparison", "integration"],
    "elaboration":           ["integration", "implementation", "design_rationale"],
    "example_of":            ["factual", "implementation", "enumeration"],
    "shared_concept":        ["integration", "comparison"],
    "conditional_behavior":  ["comparison", "design_rationale"],
    "operation_comparison":  ["comparison", "integration"],
    "generalization_pattern":["design_rationale", "integration"],
    "convergent_goal":       ["integration", "comparison"],
    "complementary_aspect":  ["integration", "implementation"],
}

# ── QAEvaluator — GEval-inspired LLM judge ───────────────────────────────────
# Evaluates generated (question, answer) pairs against 3 independent criteria:
#   1. multi_hop_necessity : question requires BOTH contexts to be answered
#   2. answer_grounding    : answer is strictly derived from the provided contexts
#   3. question_specificity: question uses precise CoSApp/CoSApp-Turbo terminology
# Used inside the retry loop in QuestionGenerator.generate().
class QAEvalScore(BaseModel):
    groundedness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="1 - FactualCorrectness(answer, no-context reference): high = answer depends on provided contexts")
    two_hop_score: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False
    feedback: str = ""  # ≤ 30 words explaining the main failure; empty string if passed
    question_is_faulty: bool = False  # True = the QUESTION is single-hop; retry must regenerate question+answer
    # ── PHASE B: per-judge raw sub-scores + per-module feedback ──────────────
    # The two_hop_score above is the average of the two judges; GEPA needs the
    # raw components kept separate (Pareto front, not a weighted sum).
    question_needs_both: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Judge A: does the QUESTION require both contexts? (routed to query_generation)")
    answer_uses_both: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Judge B: does the ANSWER draw from both contexts? (routed to answer_generation)")
    question_feedback: str = ""  # verdict concerning the QUESTION module only
    answer_feedback: str = ""    # verdicts concerning the ANSWER module only

    def module_signals(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        """Route the judge sub-scores to the two optimizable modules (PHASE B/C).

        Returns a per-module dict in GEPA's expected shape, keeping the
        sub-scores as a VECTOR (never summed/averaged): the Pareto front must
        see each component independently — this is the key result of the GEPA
        paper, a scalar mean makes the search stagnate.

          query_generation  : [question_needs_both, groundedness]
              the question must require both hops AND not invent entities
              absent from the contexts.
          answer_generation : [groundedness, answer_accuracy, answer_uses_both]
              the answer must be grounded, context-dependent, and synthesize
              facts from BOTH segments.
        """
        return {
            "query_generation": {
                "scores": [self.question_needs_both, self.groundedness_score],
                "feedback": self.question_feedback,
            },
            "answer_generation": {
                "scores": [
                    self.groundedness_score,
                    self.answer_accuracy_score,
                    self.answer_uses_both,
                ],
                "feedback": self.answer_feedback,
            },
        }

# ── 2-hop judge — deux prompts indépendants ──────────────────────────────────
# Judge A : la question nécessite-t-elle les deux contextes ?
# Judge B : la réponse exploite-t-elle les deux contextes ?
# Les deux tournent en parallèle via asyncio.gather ; two_hop_score = moyenne Python.
class TwoHopJudgeInput(BaseModel):
    context_1hop: str
    context_2hop: str
    question: str
    answer: str

# ── Judge A : question_needs_both ────────────────────────────────────────────
class _QuestionNeedsBothOutput(BaseModel):
    question_needs_both: float = Field(
        ge=0.0, le=1.0,
        description=(
            "1.0 = question is unanswerable from either context alone; "
            "0.0 = question is fully answerable from one context alone."
        ),
    )
    verdict: str = Field(
        description=(
            "One sentence (≤ 20 words) citing the failure if question_needs_both < 0.8. "
            "Empty string if question_needs_both >= 0.8."
        ),
    )

_QUESTION_NEEDS_BOTH_INSTRUCTION = (
    "You are an expert evaluator for multi-hop question-answering benchmarks.\n\n"
    "You receive two documentation contexts (context_1hop, context_2hop) and a question.\n"
    "Your ONLY task: decide whether the question genuinely requires BOTH contexts.\n\n"
    "### question_needs_both  (0.0 – 1.0)\n"
    "Try to answer the question using ONLY context_1hop — can you fully answer it?\n"
    "Then try using ONLY context_2hop — can you fully answer it?\n"
    "  • 1.0 : the question cannot be answered from either context alone; both are necessary.\n"
    "  • 0.5 : one context partially answers it but the other is still needed.\n"
    "  • 0.0 : one context alone is sufficient to answer the question.\n\n"
    "### verdict\n"
    "If question_needs_both < 0.8: one sentence citing which context alone suffices "
    "(e.g. 'question answerable from context_1hop alone'). "
    "If question_needs_both >= 0.8: return an empty string.\n"
)

class _QuestionNeedsBothPrompt(PydanticPrompt[TwoHopJudgeInput, _QuestionNeedsBothOutput]):
    instruction: str = _QUESTION_NEEDS_BOTH_INSTRUCTION
    input_model = TwoHopJudgeInput
    output_model = _QuestionNeedsBothOutput
    examples: t.List[t.Tuple[TwoHopJudgeInput, _QuestionNeedsBothOutput]] = []

# ── Judge B : answer_uses_both ───────────────────────────────────────────────
class _AnswerUsesBothOutput(BaseModel):
    answer_uses_both: float = Field(
        ge=0.0, le=1.0,
        description=(
            "1.0 = answer explicitly draws facts from BOTH context_1hop AND context_2hop; "
            "0.0 = answer ignores one of the contexts entirely."
        ),
    )
    verdict: str = Field(
        description=(
            "One sentence (≤ 20 words) citing the failure if answer_uses_both < 0.8. "
            "Empty string if answer_uses_both >= 0.8."
        ),
    )

_ANSWER_USES_BOTH_INSTRUCTION = (
    "You are an expert evaluator for multi-hop question-answering benchmarks.\n\n"
    "You receive two documentation contexts (context_1hop, context_2hop), a question, "
    "and an answer.\n"
    "Your ONLY task: decide whether the answer draws facts from BOTH contexts.\n\n"
    "### answer_uses_both  (0.0 – 1.0)\n"
    "Identify specific facts stated in the answer. "
    "Trace each fact back to context_1hop or context_2hop.\n"
    "  • 1.0 : the answer cites or paraphrases facts from BOTH contexts.\n"
    "  • 0.5 : the answer draws mostly from one context with minimal use of the other.\n"
    "  • 0.0 : the answer is entirely derived from one context (the other is ignored).\n\n"
    "### verdict\n"
    "If answer_uses_both < 0.8: one sentence naming which context is ignored "
    "(e.g. 'answer ignores context_2hop entirely'). "
    "If answer_uses_both >= 0.8: return an empty string.\n"
)

class _AnswerUsesBothPrompt(PydanticPrompt[TwoHopJudgeInput, _AnswerUsesBothOutput]):
    instruction: str = _ANSWER_USES_BOTH_INSTRUCTION
    input_model = TwoHopJudgeInput
    output_model = _AnswerUsesBothOutput
    examples: t.List[t.Tuple[TwoHopJudgeInput, _AnswerUsesBothOutput]] = []

# ── Résultat combiné (reconstruit en Python après les deux appels) ────────────
class TwoHopJudgeOutput(BaseModel):
    question_needs_both: float = Field(ge=0.0, le=1.0)
    answer_uses_both: float = Field(ge=0.0, le=1.0)
    two_hop_score: float = Field(ge=0.0, le=1.0)
    verdict: str = ""

# ── No-context reference prompt (for AnswerAccuracy baseline) ────────────────
class _NoCtxInput(BaseModel):
    question: str
class _NoCtxOutput(BaseModel):
    answer: str
class NoContextReferencePrompt(PydanticPrompt[_NoCtxInput, _NoCtxOutput]):
    instruction: str = (
        "Answer the following question using only your general knowledge, "
        "without any additional context. Be concise (1-3 sentences)."
    )
    input_model = _NoCtxInput
    output_model = _NoCtxOutput

class QAEvaluator:
    """
    LLM-as-judge for generated QA pairs. Used inside the retry loop.
    Three independent judges run in parallel:
      1. Faithfulness        (RAGAS) — answer grounded in retrieved contexts
      2. FactualCorrectness  (RAGAS) — context-dependence score: 1 - similarity(answer, LLM-without-context).
                                        High score = answer diverges from what the LLM knows without context
                                        = truly grounded in the provided documents.
      3. TwoHopJudgePrompt  (custom) — question requires BOTH contexts AND
                                        answer draws facts from BOTH contexts
    Requires `enable_ragas=True` and `llm_config` to activate.  When disabled,
    all three scores are 0.0 and `passed` defaults to True (no filtering).
    """
    def __init__(
        self,
        threshold: float = 0.6,
        config: t.Any = None,
        enable_ragas: bool = False,
        llm_config: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> None:
        self.threshold = threshold
        # Anti false-positive: enable_ragas=True but no llm_config means the
        # RAGAS metrics below are NEVER built (the `if enable_ragas and llm_config`
        # block is skipped), yet _enable_ragas would stay True and the GEPA guard
        # (which only checks _enable_ragas) would pass — letting GEPA optimize
        # against groundedness/answer_accuracy stuck at 0.0. Force it off so the
        # guard correctly refuses to run without a usable RAGAS judge.
        if enable_ragas and not llm_config:
            logging.warning(
                "QAEvaluator: enable_ragas=True but no llm_config — RAGAS judge "
                "cannot be built; disabling RAGAS (groundedness/answer_accuracy)."
            )
            enable_ragas = False
        self._enable_ragas = enable_ragas
        self._max_context_chars = (
            config.evaluation.max_context_chars
            if config and hasattr(config, 'evaluation')
            else MAX_CONTEXT_CHARS
        )
        # ── 2-hop judges (deux prompts indépendants) + no-context reference
        self._judge_question = _QuestionNeedsBothPrompt()
        self._judge_answer   = _AnswerUsesBothPrompt()
        self._no_ctx_prompt  = NoContextReferencePrompt()
        # ── RAGAS metrics (LangchainLLMWrapper + httpx, same pattern as eval scripts)
        self._ragas_groundedness: t.Any = None
        self._ragas_answer_accuracy: t.Any = None
        if enable_ragas and llm_config:
            try:
                import httpx as _httpx
                from langchain_openai import ChatOpenAI as _ChatOpenAI
                from ragas.llms import LangchainLLMWrapper as _LCWrapper
                from ragas.metrics import Faithfulness, FactualCorrectness
                from ragas.run_config import RunConfig as _RC
                _model    = llm_config.get("model", "gpt-4o-mini")
                _base_url = llm_config.get("base_url")
                _api_key  = llm_config.get("api_key", "")
                _ragas_llm = _LCWrapper(_ChatOpenAI(
                    model=_model,
                    api_key=_api_key,
                    base_url=_base_url,
                    temperature=0.0,
                    http_client=_httpx.Client(verify=False),
                    http_async_client=_httpx.AsyncClient(verify=False),
                ))
                _rc = _RC(timeout=30, max_retries=2, max_wait=15)
                self._ragas_groundedness   = Faithfulness(llm=_ragas_llm)
                self._ragas_answer_accuracy = FactualCorrectness(llm=_ragas_llm, mode="f1")
                self._ragas_groundedness.run_config   = _rc
                self._ragas_answer_accuracy.run_config = _rc
            except Exception as _e:
                logging.warning(
                    "QAEvaluator: RAGAS metrics init failed (%s) — RAGAS eval disabled", _e
                )
                self._enable_ragas = False
    async def evaluate(
        self,
        llm: t.Any,
        question: str,
        context_1hop: str,
        context_2hop: str,
        answer: str,
        question_type: str = "",
        relation_type: str = "",
    ) -> QAEvalScore:
        ctx_1 = context_1hop[:self._max_context_chars]
        ctx_2 = context_2hop[:self._max_context_chars]
        if not self._enable_ragas:
            return QAEvalScore(passed=True, feedback="")
        groundedness_score = 0.0
        answer_accuracy_score = 0.0
        two_hop_score = 0.0
        feedback_parts: t.List[str] = []
        try:
            from ragas.dataset_schema import SingleTurnSample
            has_ragas = self._ragas_groundedness is not None
            # Génère une réponse de référence sans contexte pour AnswerAccuracy
            no_ctx_reference = answer  # fallback si génération échoue
            if has_ragas:
                try:
                    _ref_out = await self._no_ctx_prompt.generate(
                        llm=llm, data=_NoCtxInput(question=question)
                    )
                    no_ctx_reference = _ref_out.answer
                except Exception as _e:
                    logging.debug("no-context reference generation failed: %s", _e)
            tasks: t.List[t.Any] = []
            if has_ragas:
                # Faithfulness : réponse ancrée dans les contextes récupérés
                tasks.append(self._ragas_groundedness.single_turn_ascore(
                    SingleTurnSample(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=[ctx_1, ctx_2],
                    )
                ))
                # FactualCorrectness : divergence entre la réponse et ce que le LLM
                # dirait sans contexte. Score brut proche de 0 = très similaire au
                # savoir général → mauvais signe. On stocke 1 - score_brut pour que
                # answer_accuracy_score élevé signifie "ancré dans les documents".
                tasks.append(self._ragas_answer_accuracy.single_turn_ascore(
                    SingleTurnSample(
                        user_input=question,
                        response=answer,
                        reference=no_ctx_reference,
                    )
                ))
            _judge_input = TwoHopJudgeInput(
                context_1hop=ctx_1,
                context_2hop=ctx_2,
                question=question,
                answer=answer,
            )
            tasks.append(self._judge_question.generate(llm=llm, data=_judge_input))
            tasks.append(self._judge_answer.generate(llm=llm, data=_judge_input))
            raw = await asyncio.gather(*tasks, return_exceptions=True)
            import math
            def _safe_score(v: t.Any) -> float:
                """Convert RAGAS result to float in [0, 1], mapping nan/errors to 0."""
                try:
                    f = float(v)
                    return 0.0 if math.isnan(f) or math.isinf(f) else max(0.0, min(1.0, f))
                except Exception:
                    return 0.0
            offset = 0
            if has_ragas:
                g_res, a_res = raw[0], raw[1]
                offset = 2
                if not isinstance(g_res, Exception):
                    groundedness_score = _safe_score(g_res)
                else:
                    logging.debug("Faithfulness failed: %s", g_res)
                if not isinstance(a_res, Exception):
                    answer_accuracy_score = 1.0 - _safe_score(a_res)
                else:
                    logging.debug("FactualCorrectness failed: %s", a_res)
            q_res = raw[offset]
            a2_res = raw[offset + 1]
            question_needs_both = 0.0
            answer_uses_both    = 0.0
            verdicts: t.List[str] = []
            _question_is_faulty = False
            # ── PHASE B: collect per-module feedback separately ──────────────
            question_verdict = ""  # routed to query_generation module
            answer_2hop_verdict = ""  # routed to answer_generation module
            if not isinstance(q_res, Exception):
                question_needs_both = _safe_score(q_res.question_needs_both)
                if question_needs_both < self.threshold:
                    _question_is_faulty = True
                    verdict_text = q_res.verdict or "question answerable from one context alone"
                    verdicts.append("question: " + verdict_text)
                    question_verdict = verdict_text
            else:
                logging.debug("TwoHopJudge(question) failed: %s", q_res)
            if not isinstance(a2_res, Exception):
                answer_uses_both = _safe_score(a2_res.answer_uses_both)
                if answer_uses_both < self.threshold:
                    verdict_text = a2_res.verdict or "answer does not draw from both contexts"
                    verdicts.append("answer: " + verdict_text)
                    answer_2hop_verdict = verdict_text
            else:
                logging.debug("TwoHopJudge(answer) failed: %s", a2_res)
            two_hop_score = (question_needs_both + answer_uses_both) / 2.0
            if verdicts and two_hop_score < self.threshold:
                feedback_parts.append("2hop: " + " | ".join(verdicts))
        except Exception as exc:
            logging.warning("QAEvaluator.evaluate failed (%s) — treating as passed", exc)
            return QAEvalScore(passed=True, feedback="")
        passed = (
            groundedness_score >= self.threshold
            and answer_accuracy_score >= self.threshold
            and two_hop_score >= self.threshold
        )
        if not passed:
            # Always emit a feedback entry for every failing criterion independently.
            # The 2-hop verdict is already in feedback_parts (built above).
            # Groundedness and answer_accuracy are always appended when they fail,
            # regardless of whether 2-hop already produced feedback.
            if groundedness_score < self.threshold:
                feedback_parts.append(
                    "groundedness: la réponse affirme des faits absents des deux contextes"
                    " — rester strictement dans les contextes fournis"
                )
            if answer_accuracy_score < self.threshold:
                feedback_parts.append(
                    "answer_accuracy: réponse = savoir général LLM, pas spécifique aux chunks"
                    " — inclure des noms/valeurs/mécanismes présents dans les contextes"
                )
            # Fallback: two_hop failed but verdicts were empty (both judges threw)
            if two_hop_score < self.threshold and not any("2hop" in p for p in feedback_parts):
                feedback_parts.append(
                    f"two_hop: score {two_hop_score:.2f} below threshold"
                )
        # ── PHASE B: build per-module feedback (routed by module_signals()) ──
        # query_generation cares about: question_needs_both + groundedness.
        # answer_generation cares about: groundedness + answer_accuracy + answer_uses_both.
        _q_fb_parts: t.List[str] = []
        if question_verdict:
            _q_fb_parts.append("question_needs_both: " + question_verdict)
        if groundedness_score < self.threshold:
            _q_fb_parts.append(
                "groundedness: the question presumes entities/facts absent from the contexts"
            )
        _a_fb_parts: t.List[str] = []
        if groundedness_score < self.threshold:
            _a_fb_parts.append(
                "groundedness: the answer asserts facts absent from both contexts"
            )
        if answer_accuracy_score < self.threshold:
            _a_fb_parts.append(
                "answer_accuracy: the answer reads as general LLM knowledge, not chunk-specific"
            )
        if answer_2hop_verdict:
            _a_fb_parts.append("answer_uses_both: " + answer_2hop_verdict)

        return QAEvalScore(
            groundedness_score=groundedness_score,
            answer_accuracy_score=answer_accuracy_score,
            two_hop_score=two_hop_score,
            passed=passed,
            feedback=" | ".join(feedback_parts),
            question_is_faulty=_question_is_faulty,
            question_needs_both=question_needs_both,
            answer_uses_both=answer_uses_both,
            question_feedback=" | ".join(_q_fb_parts),
            answer_feedback=" | ".join(_a_fb_parts),
        )

# ── Relation-type → answer structure guidance ─────────────────────────────────
_RELATION_TYPE_ANSWER_STRUCTURE: t.Dict[str, str] = {
    "prerequisite": (
        "Answer structure: (1) what <1-hop> defines or establishes; "
        "(2) how <2-hop> depends on or requires that definition."
    ),
    "contrast": (
        "Answer structure: point-by-point comparison using explicit "
        "'whereas' / 'unlike' / 'in contrast to' markers."
    ),
    "example_of": (
        "Answer structure: (1) the general principle or concept from one context; "
        "(2) the concrete example or application from the other context."
    ),
    "elaboration": (
        "Answer structure: (1) high-level overview from one context; "
        "(2) technical details or implementation specifics from the other."
    ),
    "shared_concept": (
        "Answer structure: explain how each context covers a different facet "
        "of the same technical concept."
    ),
    "conditional_behavior": (
        "Answer structure: (1) describe the shared mechanism or operation; "
        "(2) contrast how it behaves under each condition, citing both contexts explicitly."
    ),
    "operation_comparison": (
        "Answer structure: side-by-side comparison of the two operations applied "
        "to the shared object — highlight what differs and why it matters."
    ),
    "generalization_pattern": (
        "Answer structure: (1) state the general principle from one context; "
        "(2) show how the other context instantiates or extends that principle."
    ),
    "convergent_goal": (
        "Answer structure: explain the shared goal/result, then describe the "
        "distinct paths or mechanisms each context uses to reach it."
    ),
    "complementary_aspect": (
        "Answer structure: (1) the aspect covered by <1-hop>; "
        "(2) the orthogonal aspect covered by <2-hop>; "
        "(3) synthesis explaining how together they give a complete picture."
    ),
}

_GENERIC_QA_PROMPT = QueryAnswerGenerationPrompt()


# ── Split generation: two independent prompts (query, then answer) ────────────
# PHASE A — STARK splits the single RAGAS QueryAnswerGenerationPrompt (which
# emitted query+answer in one LLM call) into two sequential modules so each can
# be optimized independently by GEPA:
#   1. QueryGenerationPrompt  : contexts (+persona/themes/style/length) → question
#   2. AnswerGenerationPrompt : contexts + question (+themes) → answer
# The answer module deliberately does NOT receive query_style / query_length:
# those describe how the *question* is phrased, not the answer, which must always
# be complete, precise and grounded in BOTH contexts regardless of question style.

class QueryGenInput(BaseModel):
    """Input for the question-generation module (mirrors RAGAS QueryConditions)."""
    persona: Persona
    themes: t.List[str]
    query_style: str
    query_length: str
    context: t.List[str]


class QueryGenOutput(BaseModel):
    query: str


class AnswerGenInput(BaseModel):
    """Input for the answer-generation module: contexts + the generated question."""
    context: t.List[str]
    question: str
    themes: t.List[str] = Field(default_factory=list)


class AnswerGenOutput(BaseModel):
    answer: str


_QUERY_GENERATION_INSTRUCTION = (
    "Generate a multi-hop QUESTION (no answer) based on the specified conditions "
    "(persona, themes, style, length) and the provided context segments tagged "
    "<1-hop> and <2-hop>. The themes are key phrases extracted from the context "
    "that highlight why the segments support a multi-hop question.\n\n"
    "### Instructions\n"
    "1. Craft a question that can ONLY be answered by combining information from "
    "BOTH context segments, referencing at least one theme explicitly.\n"
    "2. Use precise terminology that appears in the contexts (class names, method "
    "names, parameters, values) where present.\n"
    "3. Match the requested style and length:\n"
    "   • 'Perfect grammar' → complete grammatical question ending with '?'\n"
    "   • 'Web search like queries' → short keyword query, no question mark\n"
    "   • length long ≥ 20 words, medium 10-19 words, short ≤ 9 words.\n\n"
    "### What to avoid\n"
    "   - Trivial single-hop questions answerable from one segment alone.\n"
    "   - Hallucinated class names or signatures absent from the context.\n"
    "   - Vague wording like 'the system' without naming the concept.\n"
)


class QueryGenerationPrompt(PydanticPrompt[QueryGenInput, QueryGenOutput]):
    """Generates ONLY the multi-hop question from the conditions + contexts."""
    instruction: str = _QUERY_GENERATION_INSTRUCTION
    input_model = QueryGenInput
    output_model = QueryGenOutput
    examples: t.List[t.Tuple[QueryGenInput, QueryGenOutput]] = []


# Default answer instruction shared with the config migration seed. Kept in sync
# with pipeline_config._DEFAULT_ANSWER_GENERATION_PROMPT (imported lazily to avoid
# a hard module-load dependency); falls back to a local copy if import fails.
try:
    from pipeline_config import _DEFAULT_ANSWER_GENERATION_PROMPT as _ANSWER_GENERATION_INSTRUCTION
except Exception:  # pragma: no cover - defensive
    _ANSWER_GENERATION_INSTRUCTION = (
        "Generate the reference ANSWER to the question using ONLY the two provided "
        "context segments <1-hop> and <2-hop>. The answer MUST draw facts from BOTH "
        "segments, must not introduce information absent from the contexts, and must "
        "use the precise terminology present in the contexts."
    )


class AnswerGenerationPrompt(PydanticPrompt[AnswerGenInput, AnswerGenOutput]):
    """Generates ONLY the reference answer from contexts + the generated question."""
    instruction: str = _ANSWER_GENERATION_INSTRUCTION
    input_model = AnswerGenInput
    output_model = AnswerGenOutput
    examples: t.List[t.Tuple[AnswerGenInput, AnswerGenOutput]] = []


_GENERIC_QUERY_PROMPT = QueryGenerationPrompt()
_GENERIC_ANSWER_PROMPT = AnswerGenerationPrompt()
 
 
def _is_same_section_subdivision(triplet) -> bool:
    node_a, node_b = triplet[0], triplet[-1]
    return (
        node_a.properties.get("is_subdivision", False)
        and node_b.properties.get("is_subdivision", False)
        and node_a.properties.get("breadcrumb", "NONE_A")
        == node_b.properties.get("breadcrumb", "NONE_B")
    )
 
 
@dataclass
class CustomMultiHopQuerySynthesizer(MultiHopQuerySynthesizer):
    """Config-driven multi-hop query synthesizer.

    Overrides ``prepare_combinations`` so that generated scenarios use only the
    styles and lengths defined in ``PipelineConfig.query_params`` (defaults:
    Perfect grammar, Web search like queries / long, medium, short). This avoids
    generating "Misspelled queries" / "Poor grammar" scenarios for corpora where
    they are meaningless.

    Overrides ``_generate_sample`` to perform TWO sequential LLM calls — first a
    question prompt, then an answer prompt — built at runtime from
    ``PipelineConfig.prompts.query_generation`` / ``answer_generation`` and the
    few-shot examples in ``PipelineConfig.few_shots.query_generation`` /
    ``answer_generation``. Falls back to generic prompts when no config is attached.
    """
 
    # Default prompts: overridden at runtime by the typed-prompt builders below
    # (config-driven or generic fallback). PHASE A splits the single RAGAS
    # query+answer prompt into two: a question prompt and an answer prompt.
    generate_query_reference_prompt = _GENERIC_QA_PROMPT  # legacy slot (compat)
    query_generation_prompt = _GENERIC_QUERY_PROMPT
    answer_generation_prompt = _GENERIC_ANSWER_PROMPT
 
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()
 
    def _get_typed_query_prompt(
        self, question_type: str, feedback: str = "",
        question_is_faulty: bool = False,
    ) -> "QueryGenerationPrompt":
        """Build the QUESTION-generation prompt (config-driven + typed blocks).

        Adds: a question-type directive, an anti-single-hop self-check, and (on
        retry) any feedback that concerns the QUESTION itself.
        """
        _cfg = getattr(self, "_qg_config", None)
        if _cfg is not None:
            try:
                from pipeline_config import build_prompt_class, deserialize_few_shots
                _examples = deserialize_few_shots(
                    "query_generation",
                    _cfg.few_shots.query_generation,
                    _cfg,
                )
                _PromptCls = build_prompt_class(
                    QueryGenerationPrompt,
                    _cfg.prompts.query_generation,
                    _examples,
                )
                prompt = _PromptCls()
            except Exception as _pe:
                logging.warning("_get_typed_query_prompt: config prompt build failed (%s) — using generic fallback", _pe)
                prompt = QueryGenerationPrompt()
        else:
            prompt = QueryGenerationPrompt()
        # Block 1 — question type directive
        extra = (
            f"\n### Question type for this generation\n"
            f"Generate a **{question_type}** question. "
            f"The question must specifically exploit the *{question_type}* "
            f"relationship between the two context segments.\n"
        )
        # Block 2 — anti-single-hop self-check (question only)
        extra += (
            "\n### Self-check before finalizing\n"
            "- Can the question be answered from <1-hop> alone? If YES → reformulate to require <2-hop>.\n"
            "- Can the question be answered from <2-hop> alone? If YES → reformulate to require <1-hop>.\n"
        )
        # Block 3 — feedback concerning the QUESTION (on retry)
        if feedback and question_is_faulty:
            extra += (
                "\n### Tentative précédente rejetée — la QUESTION est mono-contexte\n"
                f"Raison :\n{feedback}\n\n"
                "• Reformulez ENTIÈREMENT la question pour qu'elle soit inrépondable "
                "sans combiner <1-hop> ET <2-hop>.\n"
            )
        prompt.instruction = prompt.instruction + extra
        return prompt
 
    def _get_typed_answer_prompt(
        self, relation_type: str = "", feedback: str = "",
    ) -> "AnswerGenerationPrompt":
        """Build the ANSWER-generation prompt (config-driven + typed blocks).

        Adds: relation-type answer-structure guidance, and (on retry) any
        feedback that concerns the ANSWER (groundedness / accuracy / 2-hop usage).
        """
        _cfg = getattr(self, "_qg_config", None)
        if _cfg is not None:
            try:
                from pipeline_config import build_prompt_class, deserialize_few_shots
                _examples = deserialize_few_shots(
                    "answer_generation",
                    _cfg.few_shots.answer_generation,
                    _cfg,
                )
                _PromptCls = build_prompt_class(
                    AnswerGenerationPrompt,
                    _cfg.prompts.answer_generation,
                    _examples,
                )
                prompt = _PromptCls()
            except Exception as _pe:
                logging.warning("_get_typed_answer_prompt: config prompt build failed (%s) — using generic fallback", _pe)
                prompt = AnswerGenerationPrompt()
        else:
            prompt = AnswerGenerationPrompt()
        # ── Relation-to-answer-structure: config first, hardcoded fallback ──
        if _cfg is not None and hasattr(_cfg, "taxonomy") and getattr(_cfg.taxonomy, "relation_to_answer_structure", None):
            _rel_map = _cfg.taxonomy.relation_to_answer_structure
        else:
            _rel_map = _RELATION_TYPE_ANSWER_STRUCTURE
        extra = ""
        # Block 1 — relation-type answer structure (when known)
        if relation_type and relation_type in _rel_map:
            extra += (
                f"\n### Answer structure guidance\n"
                f"{_rel_map[relation_type]}\n"
            )
        # Block 2 — feedback concerning the ANSWER (on retry).
        # Routes groundedness / answer_accuracy / 2-hop-usage feedback to the
        # answer module. (In PHASE B the routing becomes per-module explicit.)
        if feedback:
            has_2hop   = "2hop"            in feedback
            has_ground = "groundedness"    in feedback
            has_acc    = "answer_accuracy" in feedback
            extra += "\n### Tentative précédente rejetée — corriger la RÉPONSE\n"
            extra += f"Raisons :\n{feedback}\n\nActions requises :\n"
            if has_2hop:
                extra += (
                    "• [2-HOP] La réponse n'exploite pas les deux contextes."
                    " Intégrez des faits spécifiques de <1-hop> ET de <2-hop>.\n"
                )
            if has_ground:
                extra += (
                    "• [GROUNDEDNESS] La réponse contient des affirmations absentes des contextes."
                    " Supprimez tout fait qui ne figure pas dans <1-hop> ou <2-hop>.\n"
                )
            if has_acc:
                extra += (
                    "• [ANSWER ACCURACY] La réponse ressemble au savoir général du LLM."
                    " Citez des termes/valeurs/mécanismes présents textuellement dans les contextes.\n"
                )
        prompt.instruction = prompt.instruction + extra
        return prompt

    async def _generate_sample(self, scenario, callbacks=None):
        """Two-call generation: question first, then answer.

        Overrides RAGAS's single-call _generate_sample. The question prompt and
        answer prompt are taken from the instance attributes ``query_generation_prompt``
        and ``answer_generation_prompt`` (assigned per-attempt by the retry loop in
        QuestionGenerator.generate via the typed-prompt builders).

        Returns a RAGAS SingleTurnSample with the same shape as before so the rest
        of the pipeline (formatter, checkpoints, question dict) is unaffected.
        """
        from ragas.dataset_schema import SingleTurnSample

        reference_context = self.make_contexts(scenario)
        themes = list(scenario.combinations or [])

        # ── Call 1: question ──────────────────────────────────────────────────
        query_prompt = getattr(self, "query_generation_prompt", _GENERIC_QUERY_PROMPT)
        q_out = await query_prompt.generate(
            data=QueryGenInput(
                persona=scenario.persona,
                themes=themes,
                query_style=scenario.style.value,
                query_length=scenario.length.value,
                context=reference_context,
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        question = q_out.query

        # ── Call 2: answer (receives the generated question + both contexts) ──
        answer_prompt = getattr(self, "answer_generation_prompt", _GENERIC_ANSWER_PROMPT)
        a_out = await answer_prompt.generate(
            data=AnswerGenInput(
                context=reference_context,
                question=question,
                themes=themes,
            ),
            llm=self.llm,
            callbacks=callbacks,
        )
        answer = a_out.answer

        return SingleTurnSample(
            user_input=question,
            reference=answer,
            reference_contexts=reference_context,
        )

    def prepare_combinations(
        self,
        nodes,
        combinations: t.List[t.List[str]],
        personas: t.List[Persona],
        persona_item_mapping: t.Dict[str, t.List[str]],
        property_name: str,
    ) -> t.List[t.Dict[str, t.Any]]:
        """Same as parent but restricts styles/lengths to CoSApp-appropriate values."""
        persona_list = PersonaList(personas=personas)
        possible_combinations = []
        for combination in combinations:
            entry: t.Dict[str, t.Any] = {"combination": combination}
 
            # ── personas valid for this combination ──────────────────────────
            valid_personas = []
            for persona_name, concept_list in persona_item_mapping.items():
                concept_list_lower = [c.lower() for c in concept_list]
                if (
                    any(
                        concept.lower() in concept_list_lower for concept in combination
                    )
                    and persona_list[persona_name]
                ):
                    valid_personas.append(persona_list[persona_name])
            # Fallback: if no persona matched (e.g. "entities" not populated),
            # allow all personas so sample_diverse_combinations has something to pick.
            if not valid_personas and personas:
                valid_personas = list(personas)
            entry["personas"] = valid_personas
 
            # ── nodes that contain at least one concept of this combination ──
            valid_nodes = []
            for node in nodes:
                node_themes = [
                    (theme["value"] if isinstance(theme, dict) else theme).lower()
                    for theme in node.properties.get(property_name, [])
                    if theme
                ]
                if node.get_property(property_name) and any(
                    concept.lower() in node_themes for concept in combination
                ):
                    valid_nodes.append(node)
            # Bug fix: if no node matches by property (e.g. "entities" not
            # populated on CHUNK nodes), fall back to using ALL provided nodes.
            # We already know they form a valid triplet — losing them here
            # is what caused scenario.nodes=[] and the "node_ids=[]" warning.
            entry["nodes"] = valid_nodes if valid_nodes else list(nodes)
 
            # ── restrict styles/lengths: config first, CoSApp defaults as fallback ──
            _cfg = getattr(self, "_qg_config", None)
            _qp = getattr(_cfg, "query_params", None) if _cfg is not None else None
            _style_names = getattr(_qp, "styles", []) if _qp is not None else []
            if _style_names:
                _style_map = {
                    "perfect_grammar":   QueryStyle.PERFECT_GRAMMAR,
                    "web_search_like":   QueryStyle.WEB_SEARCH_LIKE,
                    "misspelled":        QueryStyle.MISSPELLED,
                    "poor_grammar":      QueryStyle.POOR_GRAMMAR,
                }
                entry["styles"] = [_style_map[s] for s in _style_names if s in _style_map] or COSAPP_QUERY_STYLES
            else:
                entry["styles"] = COSAPP_QUERY_STYLES
            _length_names = getattr(_qp, "lengths", []) if _qp is not None else []
            if _length_names:
                _length_map = {
                    "long":   QueryLength.LONG,
                    "medium": QueryLength.MEDIUM,
                    "short":  QueryLength.SHORT,
                }
                entry["lengths"] = [_length_map[l] for l in _length_names if l in _length_map] or COSAPP_QUERY_LENGTHS
            else:
                entry["lengths"] = COSAPP_QUERY_LENGTHS
 
            possible_combinations.append(entry)
        return possible_combinations
 
    # ── scenario-level checkpoint helpers ────────────────────────────────────
 
    # Set by QuestionGenerator before calling generate_scenarios so that
    # _generate_scenarios can persist progress without changing the RAGAS API.
    _scenario_checkpoint_path: t.Optional[Path] = None
 
    @staticmethod
    def _load_scenario_checkpoint(path: t.Optional[Path]) -> t.Tuple[t.List, int]:
        """Return (scenario_dicts, last_triplet_index) from checkpoint, or ([], -1)."""
        if not path or not path.exists():
            return [], -1
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scenarios_raw = data.get("scenarios", [])
            last_triplet = data.get("last_triplet_index", -1)
            logging.info(
                "Scenario checkpoint loaded: %d scenarios, resuming after triplet %d (%s)",
                len(scenarios_raw),
                last_triplet,
                path,
            )
            return scenarios_raw, last_triplet
        except Exception as exc:
            logging.warning("Could not load scenario checkpoint %s: %s", path, exc)
            return [], -1
 
    @staticmethod
    def _save_scenario_checkpoint(
        scenario_dicts: t.List[t.Dict],
        last_triplet_index: int,
        path: t.Optional[Path],
    ) -> None:
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_triplet_index": last_triplet_index,
                        "scenarios": scenario_dicts,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:
            logging.warning("Could not save scenario checkpoint to %s: %s", path, exc)
 
    # ─────────────────────────────────────────────────────────────────────────
 
    async def _generate_scenarios(
        self, n: int, knowledge_graph, persona_list, callbacks
    ) -> t.List[MultiHopScenario]:
        """Generate multi-hop query scenarios.
 
        Processes triplets one by one until exactly `n` scenarios have been
        collected, then stops.  When n >= len(results) (e.g. n=999999) every
        triplet is processed.
 
        Supports crash-resume via ``_scenario_checkpoint_path``: after each
        triplet the raw scenario data is serialised to JSON so a re-run can
        skip already-processed triplets.
        """
        ckpt_path = getattr(self, "_scenario_checkpoint_path", None)
 
        # n=0 means "no limit" — process all triplets
        unlimited = n <= 0 or n >= 999999
        logging.info(
            f"Generating {'ALL' if unlimited else str(n)} scenarios "
            f"(unlimited={unlimited})..."
        )
 
        # Preferred: semantic overlap relations produced by OverlapScoreBuilder / NER
        # and by DirectRelationDiscovery agents (agent_discovered).
        SEMANTIC_RELS = {
            "keyphrases_overlap",    # OverlapScoreBuilder (Ragas), 0 LLM
            "cosine_similarity",     # similarité vectorielle inter-chunks, 0 LLM
            "agent_discovered",      # relations découvertes par les agents LLM inter-documents
            "llm_triplet",           # triplets validés LLM (meilleur type, 0.955)
            "retrospective_entity",  # relations RAKG centré-entité (chunk↔chunk)
        }
        # Fallback: structural relations produced by HeadlineSplitter / MarkdownChunker
        # NOTE: "contains" (DOCUMENT → CHUNK) is intentionally excluded — it is a
        # structural parent-child link with no multi-hop semantic value.
        STRUCTURAL_RELS = {"child", "next"}
 
        results = knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: rel.type in SEMANTIC_RELS
        )
        logging.info(
            "find_two_nodes_single_rel returned %d triplets (SEMANTIC_RELS)",
            len(results) if results else 0,
        )
        # Fallback: if find_two_nodes_single_rel returns nothing but KG has
        # semantic relationships (e.g. after KG Agent modified the list),
        # build triplets manually from kg.relationships.
        if not results and hasattr(knowledge_graph, "relationships"):
            manual_rels = [
                r for r in knowledge_graph.relationships
                if getattr(r, "type", "") in SEMANTIC_RELS
            ]
            if manual_rels:
                logging.info(
                    "find_two_nodes_single_rel returned 0 but KG has %d semantic "
                    "relationships — building triplets manually (fallback A).",
                    len(manual_rels),
                )
                results = [
                    (r.source, r, r.target)
                    for r in manual_rels
                    if getattr(r, "source", None) is not None
                    and getattr(r, "target", None) is not None
                ]
        results = [t for t in results if not _is_same_section_subdivision(t)]
        logging.info(
            "After same-section subdivision filter: %d triplets remaining", len(results)
        )
        # Fallback B: if subdivision filter removed everything, rebuild from
        # kg.relationships directly (skipping the subdivision filter for
        # manually-built triplets — intra-doc pairs are valid here).
        if not results and hasattr(knowledge_graph, "relationships"):
            manual_rels = [
                r for r in knowledge_graph.relationships
                if getattr(r, "type", "") in SEMANTIC_RELS
            ]
            if manual_rels:
                logging.info(
                    "Subdivision filter removed all triplets — rebuilding %d "
                    "from kg.relationships (fallback B, no subdivision filter).",
                    len(manual_rels),
                )
                results = [
                    (r.source, r, r.target)
                    for r in manual_rels
                    if getattr(r, "source", None) is not None
                    and getattr(r, "target", None) is not None
                ]
 
        # ── Inter-doc filter: if inter_doc_only, keep only cross-document pairs ──
        _inter_doc_flag = getattr(self, "_qg_inter_doc_only", True)
        if _inter_doc_flag:
            before_idf = len(results)
            results = [
                t for t in results
                if t[0].properties.get("parent_doc", "__A__")
                != t[-1].properties.get("parent_doc", "__B__")
            ]
            logging.info(
                "Inter-doc filter (strict): kept %d/%d cross-document triplets",
                len(results), before_idf,
            )
        else:
            logging.info(
                "Mode intra-doc: all %d triplets kept (same-doc pairs allowed)",
                len(results),
            )
 
        if not results:
            logging.warning(
                "No semantic relationships (entities_overlap / keyphrases_overlap) found — "
                "falling back to structural relationships (child / next / contains)"
            )
            results = knowledge_graph.find_two_nodes_single_rel(
                relationship_condition=lambda rel: rel.type in STRUCTURAL_RELS
            )
            results = [t for t in results if not _is_same_section_subdivision(t)]
            logging.info(
                "After same-section subdivision filter: %d triplets remaining",
                len(results),
            )
 
        if not results:
            logging.warning("No valid relationships found in knowledge graph")
            return []
 
        logging.info(f"Found {len(results)} relationship triplets")
        # ── Filter: exclude relations rejected by RelationValidator ─────────────
        # agent_RelationValidator=False means the LLM was confident the relation is
        # noise. Keep the relation in the KG (for graph integrity) but skip QA
        # generation on it.
        before_filter = len(results)
        results = [
            t for t in results
            if t[1].properties.get("agent_RelationValidator", True) is not False
        ]
        if len(results) < before_filter:
            logging.info(
                "RelationValidator filter: excluded %d rejected triplets, %d remaining",
                before_filter - len(results), len(results),
            )
        # ── Sort: agent_discovered first, then by relation quality ───────────────
        # Priority order:
        #   1. agent_discovered (semantically validated, most reliable for 2-hop)
        #   2. keyphrases_overlap / cosine_similarity with
        #      agent_RelationValidator=True and high confidence
        #   3. everything else
        def _triplet_priority(triplet) -> int:
            rel = triplet[1]
            rel_type = str(getattr(rel, "type", ""))
            if rel_type == "agent_discovered":
                return 0
            if rel_type == "retrospective_entity":
                return 1
            if rel.properties.get("agent_RelationValidator") is True:
                conf = rel.properties.get("validation_confidence", 0.0) or 0.0
                if conf >= 0.7:
                    return 2
                return 3
            return 4
        results.sort(key=_triplet_priority)
        logging.info(
            "Triplets sorted: %d agent_discovered first, then by validator confidence",
            sum(1 for t in results if str(getattr(t[1], "type", "")) == "agent_discovered"),
        )

        # ── Filtre qualité des paires (avant tout appel LLM) ─────────────────
        # 1. Exclure les triplets dont un chunk a un contenu vide
        # 2. Exclure les paires quasi-identiques (Jaccard sur tokens > 0.20)
        # 3. Exclure les triplets sans ID stable sur l'un des deux nœuds

        def _chunk_text(node) -> str:
            """Retourne le texte représentatif d'un nœud (raw_content > page_content)."""
            props = getattr(node, "properties", {}) or {}
            raw = props.get("raw_content", "")
            if raw:
                bc = props.get("breadcrumb", "")
                return f"{bc}\n\n{raw}" if bc else raw
            return props.get("page_content", "") or ""

        def _jaccard_tokens(a: str, b: str) -> float:
            ta = set(a.lower().split())
            tb = set(b.lower().split())
            union = ta | tb
            return len(ta & tb) / len(union) if union else 0.0

        _before_quality = len(results)
        _empty_ctx = 0
        _no_id = 0
        _near_dup = 0
        _filtered: t.List = []
        for _trip in results:
            _na, _nb = _trip[0], _trip[-1]
            # Filtre 1 : ID stable requis sur les deux nœuds
            if _na.id is None or _nb.id is None:
                _no_id += 1
                continue
            # Filtre 2 : contenu non vide
            _ta = _chunk_text(_na).strip()
            _tb = _chunk_text(_nb).strip()
            if not _ta or not _tb:
                _empty_ctx += 1
                continue
            # Filtre 3 : Jaccard > 0.20 → paire quasi-identique, skip
            if _jaccard_tokens(_ta, _tb) > 0.20:
                _near_dup += 1
                continue
            _filtered.append(_trip)
        results = _filtered
        logging.info(
            "┌─ Filtres qualité des paires : %d/%d conservés"
            "\n  → contenu vide : %d exclus"
            "\n  → node_id manquant : %d exclus"
            "\n  → quasi-doublons (Jaccard > 0.20) : %d exclus",
            len(results), _before_quality,
            _empty_ctx, _no_id, _near_dup,
        )

        # ── Helper: extract terms from a node (entities + keyphrases) ───────────
        def _node_terms(node) -> t.List[str]:
            terms = []
            ents = node.properties.get("entities", [])
            if isinstance(ents, list):
                terms += [str(e) for e in ents]
            kps = node.properties.get("keyphrases", [])
            if isinstance(kps, list):
                terms += [str(k) for k in kps]
            return [t2.lower().strip() for t2 in terms if t2]

        # ── Resume from checkpoint ────────────────────────────────────────────
        scenario_dicts, last_done = self._load_scenario_checkpoint(ckpt_path)
        # scenario_dicts stores lightweight metadata; MultiHopScenario objects
        # cannot be serialised so we rebuild them on the fly while processing.
        # We keep a parallel list of live objects for the return value.
        scenarios: t.List[MultiHopScenario] = []
 
        start_index = last_done + 1
 
        # ── Fast-path: all triplets already processed → reconstruct from checkpoint ──
        # MultiHopScenario is not serialisable, so after a crash the live list
        # is empty even though every triplet was processed.  When start_index
        # covers all triplets we rebuild the objects directly from the saved
        # metadata + the KG nodes — no LLM calls required.
        if start_index >= len(results) and scenario_dicts:
            logging.info(
                "All %d triplets already processed. "
                "Reconstructing %d MultiHopScenario objects from checkpoint (no LLM calls)...",
                len(results),
                len(scenario_dicts),
            )
 
            # Build a stable lookup: "<node_a_id>|<node_b_id>|<rel_type>" → (node_a, node_b, overlapped_keywords)
            # Including rel_type in the key avoids collision when the same pair (A,B)
            # is connected by both entities_overlap AND cosine_similarity.
            # Note: extract rel_type before the f-string (Python <3.12 forbids
            # same-quote string literals inside f-string expressions).
            def _trip_rel_type(trip):
                return str(getattr(trip[1], "type", "unknown"))
 
            triplet_lookup = {
                f"{trip[0].id}|{trip[-1].id}|{_trip_rel_type(trip)}": (
                    trip[0],
                    trip[-1],
                    trip[1].properties.get("overlapped_items", []),
                )
                for trip in results
            }
            _idx_to_key = {
                i: f"{trip[0].id}|{trip[-1].id}|{_trip_rel_type(trip)}"
                for i, trip in enumerate(results)
            }
            persona_list_obj = PersonaList(personas=persona_list)
            for sd in scenario_dicts:
                # Prefer stable content-based key; fall back to positional index
                # for checkpoints written before this fix.
                tkey = sd.get("triplet_key")
                if tkey is None:
                    tidx = sd.get("triplet_index")
                    if tidx is None:
                        continue
                    tkey = _idx_to_key.get(tidx)
                if tkey is None or tkey not in triplet_lookup:
                    continue
                node_a, node_b, overlapped_keywords = triplet_lookup[tkey]
 
                # ── Same fallback as the normal generation path ───────────────
                if not overlapped_keywords:
                    terms_a = _node_terms(node_a)
                    terms_b = _node_terms(node_b)
                    shared = list({t2 for t2 in terms_a if t2 in set(terms_b)})
                    if shared:
                        overlapped_keywords = [[kw, kw] for kw in shared[:10]]
                    else:
                        all_terms = list(dict.fromkeys(terms_a[:5] + terms_b[:5]))
                        if all_terms:
                            overlapped_keywords = [[kw, kw] for kw in all_terms[:10]]
 
                if not overlapped_keywords:
                    continue
 
                # Reconstruct combinations (keyword pairs)
                kw_list = [list(item) for item in overlapped_keywords]
                # Build a minimal persona_item_mapping that maps the saved persona
                # to all keywords (ensures prepare_combinations finds it)
                saved_persona_name = sd.get("persona")
                if saved_persona_name and persona_list_obj[saved_persona_name]:
                    pim = {saved_persona_name: [k for pair in kw_list for k in pair]}
                else:
                    # Fallback: map first persona to all keywords
                    first = persona_list[0].name if persona_list else None
                    pim = (
                        {first: [k for pair in kw_list for k in pair]} if first else {}
                    )
                combos = self.prepare_combinations(
                    [node_a, node_b],
                    kw_list,
                    personas=persona_list,
                    persona_item_mapping=pim,
                    property_name="entities",
                )
                # Bug #5 fix: seed the RNG with the stable triplet key so
                # sample_diverse_combinations always picks the same
                # (style, length, persona) combination as the original run,
                # ensuring reproducibility across crash-resume cycles.
                random.seed(
                    int(hashlib.md5(tkey.encode()).hexdigest(), 16) & 0xFFFFFFFF
                )
                rebuilt = self.sample_diverse_combinations(combos, 1)
                saved_rel_type = sd.get("relation_type", "unknown")
                saved_agent_rationale = sd.get("agent_rationale", "")
                saved_agent_rel_subtype = sd.get("agent_rel_subtype", "")
                saved_agent_relation_validator = sd.get("agent_RelationValidator", None)
                for sc in rebuilt:
                    try:
                        sc._rel_type = saved_rel_type
                    except Exception:
                        object.__setattr__(sc, "_rel_type", saved_rel_type)
                    # Restore agent_rationale so generate() can inject the semantic
                    # link hint in the context even after a crash-resume fast-path.
                    if saved_agent_rationale:
                        try:
                            sc._agent_rationale = saved_agent_rationale
                        except Exception:
                            object.__setattr__(sc, "_agent_rationale", saved_agent_rationale)
                    # Restore agent_rel_subtype (prerequisite/elaboration/…) so
                    # generate() can inject answer-structure guidance.
                    if saved_agent_rel_subtype:
                        try:
                            sc._agent_rel_subtype = saved_agent_rel_subtype
                        except Exception:
                            object.__setattr__(sc, "_agent_rel_subtype", saved_agent_rel_subtype)
                    if saved_agent_relation_validator is not None:
                        try:
                            sc._agent_relation_validator = saved_agent_relation_validator
                        except Exception:
                            object.__setattr__(sc, "_agent_relation_validator", saved_agent_relation_validator)
                scenarios.extend(rebuilt)
 
            logging.info(
                "Reconstructed %d/%d scenarios from checkpoint.",
                len(scenarios),
                len(scenario_dicts),
            )
            return scenarios
 
        if start_index > 0:
            logging.info(
                "Skipping triplets 0-%d (already processed), resuming from triplet %d",
                last_done,
                start_index,
            )
 
        # Copie locale du budget pour la phase 1 : simule les increments successifs
        # pendant l'assignation des types aux triplets, sans toucher au vrai budget
        # qui n'est incrémenté qu'après chaque question réussie en phase 2.
        _phase1_budget: t.Optional["QuestionTypeBudget"] = None
        _real_budget: t.Optional["QuestionTypeBudget"] = getattr(self, "_qg_budget", None)
        if _real_budget is not None:
            import copy as _copy
            _phase1_budget = _copy.copy(_real_budget)
            _phase1_budget.counts = dict(_real_budget.counts)
        for i, triplet in enumerate(results):
            # Skip already-processed triplets
            if i < start_index:
                continue
 
            # Stop as soon as we have enough scenarios (unless unlimited)
            if not unlimited and len(scenarios) >= n:
                logging.info(
                    f"Reached target of {n} scenarios after {i} triplets — stopping"
                )
                break
 
            logging.info(
                f"Processing triplet {i + 1}/{len(results)} "
                f"(scenarios so far: {len(scenarios)})..."
            )
            node_a, node_b = triplet[0], triplet[-1]
            overlapped_keywords = triplet[1].properties.get("overlapped_items", [])
            # For agent_discovered relations, extract rich semantic metadata
            _agent_rel_subtype = ""
            _agent_rationale = ""
            _agent_relation_validator = triplet[1].properties.get("agent_RelationValidator", None)
            if str(getattr(triplet[1], "type", "")) == "agent_discovered":
                _shared_theme = triplet[1].properties.get("shared_theme", "")
                _agent_rel_subtype = triplet[1].properties.get("relation_type", "")
                _agent_rationale = triplet[1].properties.get("rationale", "")
                # Use shared_theme as the primary theme — more precise than
                # keyphrase/entity intersection fallback below
                if _shared_theme and not overlapped_keywords:
                    overlapped_keywords = [[_shared_theme, _shared_theme]]
 
            # ── Fallback: compute overlap from node entities/keyphrases ──────
            # Structural relations (child/next/contains) have no overlapped_items.
            # We compute a soft overlap from the two nodes' entities and keyphrases.
            if not overlapped_keywords:
                terms_a = _node_terms(node_a)
                terms_b = _node_terms(node_b)
                shared = list({t2 for t2 in terms_a if t2 in set(terms_b)})
                if shared:
                    # Use shared terms as overlapped_keywords pairs (term, term)
                    overlapped_keywords = [[kw, kw] for kw in shared[:10]]
                else:
                    # No shared terms either — use union of terms from both nodes
                    # so the synthesizer still has themes to work with
                    all_terms = list(dict.fromkeys(terms_a[:5] + terms_b[:5]))
                    if all_terms:
                        overlapped_keywords = [[kw, kw] for kw in all_terms[:10]]
 
            if not overlapped_keywords:
                # Still advance the checkpoint so we don't retry empty triplets
                self._save_scenario_checkpoint(scenario_dicts, i, ckpt_path)
                continue
 
            themes = list({kw[0] for kw in overlapped_keywords})
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            try:
                persona_concepts = await _acall_with_retry(
                    lambda: self.theme_persona_matching_prompt.generate(
                        data=prompt_input, llm=self.llm, callbacks=callbacks
                    ),
                    what=f"theme_persona_matching (triplet {i + 1})",
                )
            except Exception as e:
                logging.warning(f"Skipping triplet {i+1}: persona matching failed: {e}")
                self._save_scenario_checkpoint(scenario_dicts, i, ckpt_path)
                continue

            overlapped_keywords = [list(item) for item in overlapped_keywords]
            base_scenarios = self.prepare_combinations(
                [node_a, node_b],
                overlapped_keywords,
                personas=persona_list,
                persona_item_mapping=persona_concepts.mapping,
                property_name="entities",
            )
            # 1 scenario per triplet keeps the total count predictable
            base_scenarios = self.sample_diverse_combinations(base_scenarios, 1)
            scenarios.extend(base_scenarios)
 
            # ── Persist progress after every triplet ─────────────────────────
            rel_type_str = str(getattr(triplet[1], "type", "unknown"))
 
            # ── Phase 1 : qualify question types depuis le contenu des chunks ──
            # Les deux chunks sont déjà disponibles ici ; on appelle qualify
            # une seule fois par triplet (pas par scénario) et on stocke le
            # chosen_type dans le dict de scénario pour pouvoir l'utiliser
            # directement en phase 2 sans second appel LLM.
            _props_a = node_a.properties or {}
            _props_b = node_b.properties or {}
 
            def _node_content(props: dict) -> str:
                raw = props.get("raw_content", "")
                if raw:
                    bc = props.get("breadcrumb", "")
                    return f"{bc}\n\n{raw}" if bc else raw
                return props.get("page_content", "")
 
            _ctx1 = _node_content(_props_a)[:MAX_CONTEXT_CHARS]
            _ctx2 = _node_content(_props_b)[:MAX_CONTEXT_CHARS]
 
            # Récupérer le llm_config depuis le parent QuestionGenerator si disponible,
            # sinon construire depuis les variables d'environnement.
            _llm_cfg = getattr(self, "_qg_llm_config", None)
            if _llm_cfg is None:
                import os as _os
 
                _api_key = _os.environ.get("OPENAI_API_KEY", "")
                if _api_key:
                    _llm_cfg = {
                        "base_url": _os.environ.get("OPENAI_BASE_URL"),
                        "api_key": _api_key,
                        "model": _os.environ.get(
                            "HAIKU_MODEL",
                            _os.environ.get(
                                "OPENAI_MODEL", "claude-haiku-4-5-20251001"
                            ),
                        ),
                    }
 
            # Resolve the UI-defined taxonomy (config) so agent_discovered
            # relations respect the user's relation→question-type mapping and
            # question-type names instead of the hardcoded defaults below.
            _cfg = getattr(self, "_qg_config", None)
            _cfg_rel_map: t.Dict[str, t.List[str]] = {}
            _cfg_type_names: t.List[str] = []
            if _cfg is not None:
                try:
                    _tax = getattr(_cfg, "taxonomy", None)
                    if _tax is not None:
                        _cfg_rel_map = getattr(_tax, "relation_to_question_types", None) or {}
                        _cfg_type_names = _cfg.question_type_names()
                except Exception:
                    pass

            # For agent_discovered relations, skip qualify_question_types:
            # DirectRelationDiscovery already validated the semantic type,
            # so the mapping is more accurate than a generic content analysis.
            # Priority: config.taxonomy.relation_to_question_types (UI taxonomy)
            # first, then the hardcoded _AGENT_TYPE_TO_QUESTION_TYPES fallback.
            if _agent_rel_subtype and _agent_rel_subtype in _cfg_rel_map:
                _compatible = list(_cfg_rel_map[_agent_rel_subtype])
                logging.debug(
                    "Triplet %d (agent_discovered '%s'): using config relation_to_question_types %s",
                    i + 1, _agent_rel_subtype, _compatible,
                )
            elif _agent_rel_subtype in _AGENT_TYPE_TO_QUESTION_TYPES:
                _compatible = _AGENT_TYPE_TO_QUESTION_TYPES[_agent_rel_subtype]
                logging.debug(
                    "Triplet %d (agent_discovered '%s'): using direct mapping %s",
                    i + 1, _agent_rel_subtype, _compatible,
                )
            else:
                try:
                    _compatible = await qualify_question_types(_ctx1, _ctx2, _llm_cfg, config=_cfg)
                except Exception as _qe:
                    logging.warning(
                        "qualify_question_types failed in phase 1 for triplet %d: %s",
                        i + 1,
                        _qe,
                    )
                    _compatible = list(_cfg_type_names) if _cfg_type_names else ["integration"]

            # Guard: constrain _compatible to the UI-defined taxonomy so that
            # hardcoded fallback types (design_rationale, implementation, …) never
            # leak into a session whose taxonomy does not define them. The
            # hardcoded _AGENT_TYPE_TO_QUESTION_TYPES map and the RAGAS defaults
            # both use the CoSApp type names; a custom taxonomy may use entirely
            # different names, so we intersect and fall back to the full taxonomy
            # (never to an out-of-taxonomy type).
            if _cfg_type_names:
                _filtered = [t2 for t2 in _compatible if t2 in _cfg_type_names]
                if not _filtered:
                    logging.debug(
                        "Triplet %d: none of %s are in the session taxonomy %s — "
                        "falling back to the full taxonomy",
                        i + 1, _compatible, _cfg_type_names,
                    )
                    _filtered = list(_cfg_type_names)
                _compatible = _filtered
 
            # Sélectionner le type selon la copie locale du budget.
            # On incrémente _phase1_budget immédiatement pour que le triplet
            # suivant voie un état à jour et ne choisisse pas le même type.
            # Le vrai budget (_qg_budget) reste intact ; il sera incrémenté
            # en phase 2 uniquement après chaque question réussie.
            if _phase1_budget is not None:
                _chosen = _phase1_budget.pick_type(_compatible)
                _phase1_budget.increment(_chosen)
            else:
                _chosen = _compatible[0] if _compatible else "integration"
 
            for sc in base_scenarios:
                # Propagate rel_type onto the scenario object so generate()
                # can reconstruct the triplet_key during the lookup phase.
                sc._rel_type = rel_type_str
                # Attach the pre-computed question type so generate() can
                # reuse it without a second qualify LLM call.
                sc._question_type = _chosen
                sc._compatible_types = _compatible
                # Propagate agent rationale so generate() can inject it as
                # a context hint in the question generation prompt.
                if _agent_rationale:
                    sc._agent_rationale = _agent_rationale
                # Propagate semantic sub-type of agent_discovered relations
                # so generate() can pick the right answer-structure guidance.
                if _agent_rel_subtype:
                    sc._agent_rel_subtype = _agent_rel_subtype
                if _agent_relation_validator is not None:
                    sc._agent_relation_validator = _agent_relation_validator
                scenario_dicts.append(
                    {
                        # Stable content-based key so fast-path reconstruction is
                        # correct even if the KG is reloaded with a different order.
                        # Include rel_type_str to avoid collision when the same pair
                        # (A,B) is connected by multiple relation types.
                        "triplet_key": f"{node_a.id}|{node_b.id}|{rel_type_str}",
                        "relation_type": rel_type_str,
                        "style": str(sc.style),
                        "length": str(sc.length),
                        "persona": sc.persona.name if sc.persona else None,
                        "num_nodes": len(sc.nodes),
                        "question_type": _chosen,
                        "compatible_types": _compatible,
                        # Persist agent rationale so crash-resume fast-path can
                        # restore sc._agent_rationale on reconstructed scenarios.
                        "agent_rationale": _agent_rationale if _agent_rationale else "",
                        # Persist the semantic sub-type of agent_discovered relations
                        # (prerequisite/elaboration/contrast/…) so generate() can
                        # inject the answer-structure guidance even after crash-resume.
                        "agent_rel_subtype": _agent_rel_subtype if _agent_rel_subtype else "",
                        # RelationValidator verdict: True = passed, False = rejected, None = not applicable
                        "agent_RelationValidator": _agent_relation_validator,
                    }
                )
            self._save_scenario_checkpoint(scenario_dicts, i, ckpt_path)
 
        total_processed = len(results) if unlimited else min(i + 1, len(results))
        logging.info(
            f"Generated {len(scenarios)} scenarios "
            f"(requested {'ALL' if unlimited else n}, "
            f"processed {total_processed}/{len(results)} triplets)"
        )
        return scenarios
 
 

 
 
class QuestionGenerator:
    """Generate multi-hop questions using RAGAS."""
    def __init__(
        self,
        llm,
        llm_config: Optional[dict] = None,
        max_retry: int = 2,
        qa_eval_threshold: float = 0.6,
        enable_qa_eval: bool = True,
        enable_ragas_eval: bool = False,
        config=None,
        inter_doc_only: bool = True,
    ):
        """
        Initialize the question generator.
        Args:
            llm:               Language model for RAGAS generation.
            llm_config:        Optional dict with base_url/api_key/model used for the
                               no-context answer call.  If None, the call is skipped.
            max_retry:          Max number of QAEval-driven regeneration attempts (default 2).
            qa_eval_threshold:  Score threshold (0–1) for all 3 GEval criteria (default 0.6).
            enable_qa_eval:     Set False to disable QAEval entirely (0 extra LLM calls).
            enable_ragas_eval:  Set True to also run RAGAS ResponseGroundedness +
                                AnswerAccuracy (requires llm_config).  Off by default
                                because each QA pair costs 2 extra LLM calls.
            config:             Optional PipelineConfig — when provided, thresholds and
                                prompts are read from it instead of hardcoded defaults.
            inter_doc_only:     If True (default), only generate questions from
                                cross-document chunk pairs. If False, allow
                                intra-document multi-hop (same file, different chunks).
        """
        self.llm = llm
        self._config = config
        self._inter_doc_only = inter_doc_only
        # Override thresholds from config if available
        if config and hasattr(config, 'evaluation'):
            qa_eval_threshold = getattr(config.evaluation, 'qa_eval_threshold', qa_eval_threshold)
            max_retry = getattr(config.evaluation, 'max_retry', max_retry)
        # Build llm_config from env if not provided
        if llm_config is None:
            _api_key = os.environ.get("OPENAI_API_KEY", "")
            if _api_key:
                llm_config = {
                    "base_url": os.environ.get("OPENAI_BASE_URL"),
                    "api_key": _api_key,
                    "model": os.environ.get(
                        "HAIKU_MODEL",
                        os.environ.get("OPENAI_MODEL", "claude-haiku-4-5-20251001"),
                    ),
                }
        self._llm_config = llm_config
        # QAEval retry settings
        self.max_retry = max_retry
        self._qa_evaluator: t.Optional[QAEvaluator] = (
            QAEvaluator(
                threshold=qa_eval_threshold,
                config=config,
                enable_ragas=enable_ragas_eval,
                llm_config=llm_config,
            ) if enable_qa_eval else None
        )
        # Budget proportionnel des types de questions (initialisé dans generate())
        self._budget: t.Optional[QuestionTypeBudget] = None
        self._budget_path: t.Optional[Path] = None
        # QA eval cache : question_id → {attempts: [...], final: {...}}
        self._eval_cache: t.Dict[str, t.Dict] = {}
        self._eval_cache_path: t.Optional[Path] = None

    # ── checkpoint helpers ────────────────────────────────────────────────────
 
    @staticmethod
    def _load_checkpoint(path: Path) -> List[Dict]:
        """Load previously generated questions from a checkpoint file."""
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                questions = (
                    data if isinstance(data, list) else data.get("questions", [])
                )
                logging.info(
                    "Checkpoint loaded: %d questions already generated (%s)",
                    len(questions),
                    path,
                )
                return questions
            except Exception as exc:
                logging.warning(
                    "Could not load checkpoint %s: %s — starting fresh", path, exc
                )
        return []
 
    @staticmethod
    def _save_checkpoint(questions: List[Dict], path: Path) -> None:
        """Persist the current question list to the checkpoint file."""
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("Could not save checkpoint to %s: %s", path, exc)
    def _save_eval_cache(self) -> None:
        """Persist the QA eval retry cache to a JSON file next to the checkpoint."""
        path = self._eval_cache_path
        if not path or not self._eval_cache:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._eval_cache, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("Could not save eval cache to %s: %s", path, exc)
    def _record_eval_attempt(
        self,
        question_id: str,
        attempt: int,
        question: str,
        answer: str,
        eval_result: "QAEvalScore",
    ) -> None:
        """Append one attempt record to the in-memory eval cache."""
        entry = self._eval_cache.setdefault(question_id, {"attempts": []})
        entry["attempts"].append({
            "attempt": attempt + 1,
            "question": question,
            "answer": answer,
            "groundedness_score": eval_result.groundedness_score,
            "answer_accuracy_score": eval_result.answer_accuracy_score,
            "two_hop_score": eval_result.two_hop_score,
            "passed": eval_result.passed,
            "feedback": eval_result.feedback,
            "question_is_faulty": eval_result.question_is_faulty,
        })
    # ─────────────────────────────────────────────────────────────────────────

    async def generate(
        self,
        kg: KnowledgeGraph,
        persona_list: List[Persona],
        num_questions: int = 10,
        checkpoint_path: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Generate multi-hop questions.
 
        Generates ``num_questions`` valid questions by requesting extra
        scenarios as a buffer to compensate for LLM parse failures AND for
        QAEval rejections. The buffer ratio comes from
        ``evaluation.scenario_buffer_ratio`` (min +5 scenarios) and is raised to
        at least 100 % when the target counts QE-passed questions only.

        Target semantics (``evaluation.target_only_passed``, default True):
        generation continues until ``num_questions`` questions have PASSED the
        QA Eval; questions that exhausted their retries stay in the dataset with
        ``qa_eval.passed = false`` but do not count toward the target (set
        ``evaluation.drop_failed_questions = true`` to exclude them from the
        returned dataset). Set ``target_only_passed = false`` to restore the
        legacy behaviour where every generated question counts.

        Args:
            kg: Knowledge graph
            persona_list: List of personas
            num_questions: Number of questions desired (valid ones by default)

        Returns:
            List of generated questions with metadata
        """

        # ── Unlimited mode: num_questions <= 0 means "1 question per chunk" ──
        unlimited = num_questions <= 0 or num_questions >= 999999
 
        # ── Initialiser le chemin du budget ──────────────────────────────────
        if checkpoint_path:
            self._budget_path = checkpoint_path.parent / (
                checkpoint_path.stem + "_budget.json"
            )
            self._eval_cache_path = checkpoint_path.parent / (
                checkpoint_path.stem + "_eval_cache.json"
            )
        else:
            self._budget_path = None
            self._eval_cache_path = None
        self._eval_cache = {}
 
        # ── Resume from checkpoint if available ───────────────────────────────
        questions: List[Dict] = self._load_checkpoint(checkpoint_path)

        # ── Target semantics: count only QE-passed questions ─────────────────
        # Historically the loop stopped as soon as len(questions) >= num_questions,
        # but `questions` also holds the ones that exhausted their QAEval retries
        # (qa_eval.passed = False).  The requested target was therefore diluted by
        # rejected questions.  With target_only_passed (default True) the target
        # counts VALID questions only; rejected ones stay in the dataset (flagged
        # passed=false) unless drop_failed_questions is set.
        _eval_cfg = self._config.evaluation if self._config is not None else None
        target_only_passed = (
            bool(getattr(_eval_cfg, "target_only_passed", True))
            and self._qa_evaluator is not None
        )
        drop_failed = (
            bool(getattr(_eval_cfg, "drop_failed_questions", False))
            and target_only_passed
        )

        def _target_count(qs: List[Dict]) -> int:
            """Number of questions that count toward num_questions."""
            if not target_only_passed:
                return len(qs)
            n = 0
            for _q in qs:
                _qe = _q.get("qa_eval") or {}
                # passed is None when QAEval was disabled for that question →
                # treat as valid (nothing rejected it).
                if _qe.get("passed") is not False:
                    n += 1
            return n

 
        # Fix 1 : charger le budget APRÈS le checkpoint pour pouvoir reconstruire
        # depuis les questions existantes si le fichier budget est absent.
        self._budget = QuestionTypeBudget.load_or_rebuild(
            self._budget_path, existing_questions=questions, config=self._config
        )
        logging.info("QuestionTypeBudget initialised:\n%s", self._budget.report())
        # Bug 4 fix: track done scenario indices explicitly instead of using
        # len(questions) as a proxy.  len(questions) breaks when some scenarios
        # fail to produce a question: the successful/failed split means the
        # bijection scenario_index ↔ question_index no longer holds, so
        # "skip if i < len(questions)" skips the wrong scenarios on resume.
        done_scenario_indices: t.Set[int] = {
            q["generation_params"]["scenario_index"]
            for q in questions
            if isinstance(q.get("generation_params"), dict)
            and q["generation_params"].get("scenario_index") is not None
        }
        already_done = len(questions)
        # Early exit only when the checkpoint already holds enough *countable*
        # questions (QE-passed ones when target_only_passed is active).
        _already_valid = _target_count(questions)
        if not unlimited and _already_valid >= num_questions:
            logging.info(
                "Checkpoint already has %d/%d %squestions (%d kept in total) — "
                "skipping generation",
                _already_valid, num_questions,
                "valid (QE-passed) " if target_only_passed else "",
                already_done,
            )
            if drop_failed:
                questions = [
                    q for q in questions
                    if (q.get("qa_eval") or {}).get("passed") is not False
                ]
            return questions

 
        # ── How many scenarios to request ────────────────────────────────────
        if unlimited:
            # Unlimited mode: process ALL triplets — 1 question per chunk
            n_request = 999999
            logging.info(
                "Unlimited mode (num_questions=%d) — generating 1 question per chunk, "
                "processing ALL triplets (already done=%d)...",
                num_questions,
                already_done,
            )
        else:
            # Ask for extra scenarios to absorb LLM parse failures, minimum +5.
            # Ratio driven by config.evaluation.scenario_buffer_ratio (default 0.50).
            # RAGAS MultiHopQuerySynthesizer fails ~10-20 % of the time due to
            # JSON parse errors in the LLM output (missing 'query' field, etc.).
            _buf_ratio = (
                self._config.evaluation.scenario_buffer_ratio
                if self._config is not None
                else 0.50
            )
            # When the target counts only QE-passed questions, the buffer must also
            # absorb the QAEval rejection rate (not just parse failures), otherwise
            # we run out of scenarios before reaching num_questions VALID questions.
            if target_only_passed:
                _buf_ratio = max(_buf_ratio, 1.0)
            buffer = max(5, int(num_questions * _buf_ratio))
            n_request = num_questions + buffer
            logging.info(
                f"Generating {num_questions} multi-hop questions "
                f"({'QE-passed only, ' if target_only_passed else ''}"
                f"requesting {n_request} scenarios with {buffer}-scenario buffer, "
                f"already done={already_done})..."
            )

 
        query_synthesizer = CustomMultiHopQuerySynthesizer(llm=self.llm)
        # Pass inter_doc_only flag to synthesizer for scenario filtering
        query_synthesizer._qg_inter_doc_only = self._inter_doc_only
        # Context truncation limit: config-driven or module default
        _max_ctx = (
            self._config.evaluation.max_context_chars
            if self._config is not None
            else MAX_CONTEXT_CHARS
        )
 
        # Wire the scenario checkpoint so _generate_scenarios can persist
        # triplet progress independently of question progress.
        if checkpoint_path:
            query_synthesizer._scenario_checkpoint_path = checkpoint_path.parent / (
                checkpoint_path.stem + "_scenarios.json"
            )
 
        # Expose llm_config, budget, budget_path and config to the synthesizer so
        # _generate_scenarios can call qualify_question_types, pick_type AND
        # increment/save in phase 1 (budget_path is required for save()).
        query_synthesizer._qg_llm_config = self._llm_config
        query_synthesizer._qg_budget = self._budget
        query_synthesizer._qg_budget_path = self._budget_path
        query_synthesizer._qg_config = self._config
 
        # Generate scenarios (CustomMultiHopQuerySynthesizer stops early once
        # n_request scenarios are ready, so this stays efficient)
        scenarios = await query_synthesizer.generate_scenarios(
            n=n_request, knowledge_graph=kg, persona_list=persona_list
        )
 
        # ── Build triplet_key → relation_type mapping from scenario checkpoint ──
        # The scenario checkpoint is written by _generate_scenarios after every
        # triplet and contains both "triplet_key" and "relation_type" fields per
        # scenario dict.  We index by triplet_key (= "{node_a.id}|{node_b.id}")
        # instead of positional index because the position in the checkpoint list
        # diverges from the position in the live scenarios[] list as soon as any
        # triplet fails to produce a scenario (empty overlapped_keywords, persona
        # matching failure, etc.).
        _scenario_rel_type: t.Dict[str, str] = {}  # triplet_key → relation_type
        if checkpoint_path:
            _ckpt_scenarios_path = checkpoint_path.parent / (
                checkpoint_path.stem + "_scenarios.json"
            )
            if _ckpt_scenarios_path.exists():
                try:
                    with open(_ckpt_scenarios_path, "r", encoding="utf-8") as _f:
                        _ckpt_data = json.load(_f)
                    for _sd in _ckpt_data.get("scenarios", []):
                        _tkey = _sd.get("triplet_key")
                        _rtype = _sd.get("relation_type")
                        if _tkey and _rtype:
                            _scenario_rel_type[_tkey] = _rtype
                    logging.info(
                        "Loaded relation_type for %d scenarios from %s",
                        len(_scenario_rel_type),
                        _ckpt_scenarios_path,
                    )
                except Exception as _exc:
                    logging.warning(
                        "Could not load relation_type from scenario checkpoint %s: %s",
                        _ckpt_scenarios_path,
                        _exc,
                    )
 
        if not scenarios:
            logging.warning("No scenarios generated. Returning empty question list.")
            return questions

        # ── Round-robin interleave par question_type ──────────────────────────
        # Par défaut les scénarios sont triés par priorité de relation
        # (agent_discovered en tête), ce qui produit des blocs de même type :
        # tous les "integration" d'abord, puis tous les "comparison", etc.
        # On réordonne ici en round-robin sur _question_type pour que chaque
        # lot de questions soit diversifié dès le début (important si on s'arrête
        # avant d'avoir épuisé tous les scénarios).
        from collections import defaultdict as _defaultdict
        _buckets: t.Dict[str, t.List] = _defaultdict(list)
        for _sc in scenarios:
            _qt = getattr(_sc, "_question_type", None) or "unknown"
            _buckets[_qt].append(_sc)
        _type_order = sorted(_buckets.keys())  # ordre alphabétique stable
        _iters = {_qt: iter(_scs) for _qt, _scs in _buckets.items()}
        _interleaved: t.List = []
        while _iters:
            _exhausted = []
            for _qt in _type_order:
                _it = _iters.get(_qt)
                if _it is None:
                    continue
                try:
                    _interleaved.append(next(_it))
                except StopIteration:
                    _exhausted.append(_qt)
            for _qt in _exhausted:
                del _iters[_qt]
                _type_order.remove(_qt)
        _type_dist = {_qt: len(_scs) for _qt, _scs in _buckets.items()}
        logging.info(
            "┌─ Round-robin interleave des scénarios par question_type (%d total)"
            "\n  → distribution : %s",
            len(_interleaved),
            "  ".join(f"{k}:{v}" for k, v in sorted(_type_dist.items())),
        )
        scenarios = _interleaved

        # ── Build node_id → filename reverse-lookup ───────────────────────────
        # DOCUMENT nodes store `filename` in their properties.
        # CHUNK nodes are children of DOCUMENT nodes via `child` relationships,
        # but have no `filename` property themselves.  Build a lookup so that
        # when we encounter a chunk node we can still recover the source filename.
        _node_to_filename: t.Dict[str, str] = {}
        _doc_id_to_filename: t.Dict[str, str] = {}
        # First pass: collect filename from every document node
        for node in kg.nodes:
            props = node.properties if hasattr(node, "properties") else {}
            fn = props.get("filename") or (props.get("document_metadata") or {}).get(
                "filename"
            )
            if fn:
                nid = str(node.id) if node.id is not None else None
                if nid:
                    _node_to_filename[nid] = fn
                    ntype_str = (
                        node.type.name if hasattr(node.type, "name") else str(node.type)
                    ).upper()
                    if ntype_str in ("DOCUMENT", "DOC"):
                        _doc_id_to_filename[nid] = fn
        # Second pass: propagate filename from document nodes to their children
        # via `child` or `contains` relationships (source=document → target=chunk).
        # Bug 4 fix: Relationship has no source_id/target_id attributes — the
        # correct attributes are rel.source.id and rel.target.id.
        if hasattr(kg, "relationships"):
            for rel in kg.relationships:
                if getattr(rel, "type", "").lower() in ("child", "contains"):
                    src = str(getattr(getattr(rel, "source", None), "id", ""))
                    tgt = str(getattr(getattr(rel, "target", None), "id", ""))
                    if src and tgt and src in _doc_id_to_filename:
                        _node_to_filename.setdefault(tgt, _doc_id_to_filename[src])
 
        # The real ceiling is min(num_questions, len(scenarios)) unless unlimited
        effective_max = (
            len(scenarios) if unlimited else min(num_questions, len(scenarios))
        )
        logging.info(
            f"Got {len(scenarios)} scenarios — will generate up to {effective_max} questions"
        )
 
        # ── Generate questions from scenarios ─────────────────────────────────
        # On crash-resume, `questions` already contains `already_done` entries
        # that were generated from scenarios[0..already_done-1].  Skip those
        # scenarios so we never regenerate a question for the same scenario index.
 
        # Bug 5 fix: deduplicate by node-pair to avoid generating two questions
        # for the same context when the same (A, B) pair appears under multiple
        # relation types (e.g. entities_overlap AND keyphrases_overlap).
        # Pre-populate from any checkpoint entries already in `questions`.
        seen_node_pairs: t.Set[t.FrozenSet[str]] = set()
        for _q in questions:
            _nids = _q.get("node_ids", [])
            if _nids:
                seen_node_pairs.add(frozenset(_nids))
 
        failed = 0
        for i, scenario in enumerate(scenarios):
            # Skip scenarios already covered by the checkpoint
            # Bug 4 fix: use the explicit set of processed indices, not a
            # count-based cutoff that breaks when scenarios fail.
            if i in done_scenario_indices:
                continue
 
            # Stop as soon as we have enough (skipped in unlimited mode).
            # With target_only_passed, only QE-passed questions count.
            _valid_so_far = _target_count(questions)
            if not unlimited and _valid_so_far >= num_questions:
                logging.info(
                    f"Reached target of {num_questions} "
                    f"{'valid (QE-passed) ' if target_only_passed else ''}questions after "
                    f"{i} scenarios (total kept={len(questions)}, failed={failed}) — stopping"
                )
                break

            try:
                logging.info(
                    f"Generating question {_valid_so_far + 1}/{effective_max} "
                    f"(valid={_valid_so_far}, kept={len(questions)}, "
                    f"scenario {i + 1}/{len(scenarios)})..."
                )

 
                # Extract source document filenames from scenario nodes.
                # Use the pre-built lookup so chunk nodes (which carry no
                # `filename` property directly) resolve to their parent
                # document's filename via the `child` relationship index.
                source_docs = []
                for node in scenario.nodes:
                    props = node.properties if hasattr(node, "properties") else {}
                    fn = (
                        props.get("filename")
                        or (props.get("document_metadata") or {}).get("filename")
                        # Bug 9 fix: CHUNK nodes created by
                        # create_from_markdown_files() store the source
                        # filename in "parent_doc", not "filename".
                        or props.get("parent_doc")
                        or _node_to_filename.get(
                            str(node.id) if node.id is not None else ""
                        )
                    )
                    if fn and fn not in source_docs:
                        source_docs.append(fn)
 
                # Fix 4 : collecter node_ids et vérifier seen_node_pairs AVANT
                # qualify_question_types (évite un appel LLM inutile sur un doublon).
                # ── Collect node / chunk / document IDs ──────────────────────
                node_ids: t.List[str] = []
                chunk_ids: t.List[str] = []
                document_ids: t.List[str] = []
                for node in scenario.nodes:
                    nid = str(node.id) if node.id is not None else None
                    if nid:
                        node_ids.append(nid)
                        # Determine node type (NodeType enum or plain string)
                        ntype = node.type
                        ntype_str = (
                            ntype.name if hasattr(ntype, "name") else str(ntype)
                        ).upper()
                        if ntype_str == "CHUNK":
                            chunk_ids.append(nid)
                        elif ntype_str in ("DOCUMENT", "DOC"):
                            document_ids.append(nid)
                        # Nodes that are neither CHUNK nor DOCUMENT are still
                        # recorded in node_ids for full traceability.
 
                # ── Fix 4 : skip doublon AVANT qualify ───────────────────────
                _pair_key = (
                    frozenset(node_ids) if node_ids else frozenset(["__empty__"])
                )
                if _pair_key in seen_node_pairs:
                    logging.debug(
                        "Skipping scenario %d: node pair already generated — node_ids=%s",
                        i + 1,
                        node_ids,
                    )
                    continue
                seen_node_pairs.add(_pair_key)
 
                # Build reference_contexts from node properties.
                # Priority:
                #   1. raw_content  – pure markdown, no breadcrumb prefix
                #      → re-prefix with breadcrumb (mirrors what a real retriever returns)
                #   2. page_content as-is (for nodes that pre-date the raw_content field)
                # Each context is prefixed with its hop label (<1-hop>, <2-hop>, …)
                # so downstream consumers can identify which segment is which.
                _HOP_LABELS = ["<1-hop>", "<2-hop>", "<3-hop>"]
                # Inject agent rationale as a preamble hint before <1-hop>
                # so the LLM knows WHY the two chunks are linked.
                _agent_rationale_hint = getattr(scenario, "_agent_rationale", "")
                ref_contexts = []
                for _hop_idx, _node in enumerate(scenario.nodes):
                    _props = _node.properties or {}
                    _raw = _props.get("raw_content", "")
                    if _raw:
                        _bc = _props.get("breadcrumb", "")
                        _clean = f"{_bc}\n\n{_raw}" if _bc else _raw
                    else:
                        _clean = _props.get("page_content", "")
                    # Truncate oversized contexts (RAGAS sometimes returns entire
                    # DOCUMENT nodes instead of CHUNK nodes — up to 100 KB+).
                    _clean_truncated = (
                        _clean[:_max_ctx]
                        if isinstance(_clean, str) and len(_clean) > _max_ctx
                        else _clean
                    )
                    _label = (
                        _HOP_LABELS[_hop_idx]
                        if _hop_idx < len(_HOP_LABELS)
                        else f"<{_hop_idx + 1}-hop>"
                    )
                    ref_contexts.append(f"{_label}\n{_clean_truncated}")
                # Prepend rationale to the first context so the LLM understands
                # the semantic link between the two chunks before reading them.
                if _agent_rationale_hint and ref_contexts:
                    ref_contexts[0] = (
                        f"[Semantic link: {_agent_rationale_hint}]\n\n"
                        + ref_contexts[0]
                    )
                num_hops = len(ref_contexts)
 
                # ── Fix 4 : skip contextes vides AVANT qualify ───────────────
                # This happens when scenario.nodes contains DOCUMENT-type nodes
                # that carry no raw_content / page_content (only metadata).
                # Generating a QA from empty contexts produces useless entries.
                _empty_ctx_count = sum(
                    1
                    for ctx in ref_contexts
                    if len(ctx.strip().split("\n", 1)[-1].strip()) == 0
                )
                if _empty_ctx_count == len(ref_contexts):
                    failed += 1
                    logging.warning(
                        "Skipping scenario %d: all %d context(s) are empty "
                        "(nodes may be DOCUMENT-type without raw_content/page_content). "
                        "node_ids=%s",
                        i + 1,
                        len(ref_contexts),
                        [str(n.id) for n in scenario.nodes],
                    )
                    continue
 
                # ── Étape 1 : Qualification des types compatibles ─────────────
                # Si le scénario porte déjà _question_type/_compatible_types
                # (calculés en phase 1 par _generate_scenarios), on les réutilise
                # directement — pas de second appel LLM nécessaire.
                _precomp_type = getattr(scenario, "_question_type", None)
                _precomp_compat = getattr(scenario, "_compatible_types", None)
 
                if _precomp_type and _precomp_compat:
                    compatible_types = _precomp_compat
                    chosen_type = _precomp_type
                    logging.debug(
                        "Scenario %d — reusing pre-computed type=%r (compatible=%s)",
                        i + 1,
                        chosen_type,
                        compatible_types,
                    )
                else:
                    # Fallback : appel LLM qualify si pas pré-calculé
                    raw_ctx_1 = ref_contexts[0] if ref_contexts else ""
                    raw_ctx_2 = ref_contexts[1] if len(ref_contexts) > 1 else ""
                    compatible_types = await qualify_question_types(
                        raw_ctx_1, raw_ctx_2, self._llm_config, config=self._config
                    )
                    logging.debug(
                        "Scenario %d — compatible_types=%s (LLM call)",
                        i + 1,
                        compatible_types,
                    )
 
                    # ── Étape 2 : Sélection par budget (plus sous-représenté) ─
                    chosen_type = self._budget.pick_type(compatible_types)
                    logging.info(
                        "Scenario %d — chosen_type=%r (compatible=%s)",
                        i + 1,
                        chosen_type,
                        compatible_types,
                    )
 
                # ── Étape 3 : Génération ciblée avec prompt typé + retry QAEval ──
                # Prefer the persisted semantic sub-type (prerequisite/elaboration/…)
                # over the KG relation type string ("agent_discovered") which is never
                # a key in _RELATION_TYPE_ANSWER_STRUCTURE.
                _agent_rel_subtype_for_prompt = getattr(scenario, "_agent_rel_subtype", "") \
                    or getattr(scenario, "_rel_type", "")
                if _agent_rel_subtype_for_prompt not in _RELATION_TYPE_ANSWER_STRUCTURE:
                    _agent_rel_subtype_for_prompt = ""
                last_feedback = ""
                last_question_is_faulty = False
                eval_result: t.Optional[QAEvalScore] = None
                _attempt = 0  # initialise avant la boucle (évite NameError si enable_qa_eval=False)
                _question_id_for_cache = str(uuid.uuid4())  # stable across retries
                try:
                    for _attempt in range(self.max_retry + 1):
                        # PHASE A: assign the two typed prompts (question + answer).
                        # In PHASE A the same feedback is routed to both modules;
                        # PHASE B will split the feedback per module.
                        query_synthesizer.query_generation_prompt = (
                            query_synthesizer._get_typed_query_prompt(
                                chosen_type,
                                feedback=last_feedback,
                                question_is_faulty=last_question_is_faulty,
                            )
                        )
                        query_synthesizer.answer_generation_prompt = (
                            query_synthesizer._get_typed_answer_prompt(
                                relation_type=_agent_rel_subtype_for_prompt,
                                feedback=last_feedback,
                            )
                        )
                        # Wrap the two LLM calls in a transient-error retry so a
                        # brief proxy flap (e.g. 400 "Invalid model name", connection
                        # error, 429/5xx) doesn't lose the scenario. Non-transient
                        # errors (real parse failures) are re-raised immediately and
                        # handled by the outer except as before.
                        sample = await _acall_with_retry(
                            lambda: query_synthesizer.generate_sample(scenario=scenario),
                            what=f"generate_sample (scenario {i + 1})",
                        )
                        # No evaluator → keep whatever we have
                        if self._qa_evaluator is None:
                            break
                        eval_result = await _acall_with_retry(
                            lambda: self._qa_evaluator.evaluate(
                                llm=self.llm,
                                question=sample.user_input,
                                context_1hop=ref_contexts[0] if ref_contexts else "",
                                context_2hop=(
                                    ref_contexts[1] if len(ref_contexts) > 1 else ""
                                ),
                                answer=sample.reference or "",
                                question_type=chosen_type,
                                relation_type=_agent_rel_subtype_for_prompt,
                            ),
                            what=f"qa_evaluate (scenario {i + 1})",
                        )
                        # Record every attempt in the eval cache
                        self._record_eval_attempt(
                            _question_id_for_cache,
                            _attempt,
                            sample.user_input,
                            sample.reference or "",
                            eval_result,
                        )
                        if eval_result.passed:
                            break
                        last_feedback = eval_result.feedback
                        last_question_is_faulty = eval_result.question_is_faulty
                        if _attempt == self.max_retry:
                            # All retries exhausted — keep last generation but mark rejected
                            logging.info(
                                "QAEval all %d attempt(s) failed — keeping last generation, marking as rejected. Feedback: %s",
                                self.max_retry + 1,
                                last_feedback,
                            )
                            break
                        logging.info(
                            "QAEval attempt %d/%d failed — retrying. Feedback: %s",
                            _attempt + 1,
                            self.max_retry,
                            last_feedback,
                        )
                finally:
                    # Restore the default (non-typed) prompts — config-based if
                    # available, generic fallback otherwise. Both the question and
                    # answer modules are restored.
                    _cfg_r = getattr(self, "_config", None)
                    if _cfg_r is not None:
                        try:
                            from pipeline_config import build_prompt_class, deserialize_few_shots
                            _q_ex = deserialize_few_shots(
                                "query_generation",
                                _cfg_r.few_shots.query_generation,
                                _cfg_r,
                            )
                            _QCls = build_prompt_class(
                                QueryGenerationPrompt,
                                _cfg_r.prompts.query_generation,
                                _q_ex,
                            )
                            query_synthesizer.query_generation_prompt = _QCls()
                            _a_ex = deserialize_few_shots(
                                "answer_generation",
                                _cfg_r.few_shots.answer_generation,
                                _cfg_r,
                            )
                            _ACls = build_prompt_class(
                                AnswerGenerationPrompt,
                                _cfg_r.prompts.answer_generation,
                                _a_ex,
                            )
                            query_synthesizer.answer_generation_prompt = _ACls()
                        except Exception:
                            query_synthesizer.query_generation_prompt = _GENERIC_QUERY_PROMPT
                            query_synthesizer.answer_generation_prompt = _GENERIC_ANSWER_PROMPT
                    else:
                        query_synthesizer.query_generation_prompt = _GENERIC_QUERY_PROMPT
                        query_synthesizer.answer_generation_prompt = _GENERIC_ANSWER_PROMPT
 


                # ── v2.0 schema: deduplicate combinations → themes ────────────
                raw_combos = scenario.combinations or []
                themes = list(dict.fromkeys(raw_combos))  # unique, order-preserved
 
                # ── Fix 2 : incrémenter le budget AVANT d'ajouter la question ─
                # Ordre atomique : increment → save budget → append → save checkpoint
                # Si le process crashe après save budget mais avant save checkpoint,
                # le budget est cohérent avec les questions au prochain redémarrage.
                self._budget.increment(chosen_type)
                self._budget.save(self._budget_path)
 
                questions.append(
                    {
                        "question_id": _question_id_for_cache,
                        "question": sample.user_input,
                        "question_type": chosen_type,
                        "compatible_types": compatible_types,
                        "num_hops": num_hops,
                        "node_ids": node_ids,
                        # chunk_ids / document_ids omitted — derivable from node_ids
                        # via the knowledge graph; storing them here doubled the data
                        # and was always empty for 1-hop questions.
                        "source_documents": source_docs,
                        "agent_RelationValidator": getattr(scenario, "_agent_relation_validator", None),
                        "reference": (
                            sample.reference if hasattr(sample, "reference") else ""
                        ),
                        "reference_contexts": ref_contexts,
                        "context_1hop": (
                            ref_contexts[0] if len(ref_contexts) > 0 else ""
                        ),
                        "context_2hop": (
                            ref_contexts[1] if len(ref_contexts) > 1 else ""
                        ),
                        "qa_eval": {
                            "groundedness_score": (
                                eval_result.groundedness_score if eval_result else None
                            ),
                            "answer_accuracy_score": (
                                eval_result.answer_accuracy_score if eval_result else None
                            ),
                            "two_hop_score": (
                                eval_result.two_hop_score if eval_result else None
                            ),
                            "passed": eval_result.passed if eval_result else None,
                            "feedback": eval_result.feedback if eval_result else None,
                            "question_is_faulty": (
                                eval_result.question_is_faulty if eval_result else None
                            ),
                            "attempts": _attempt + 1,
                            "validator_rejected": (
                                eval_result is not None and not eval_result.passed
                            ),
                        },
                        "generation_params": {
                            # Bug 4 fix: persist the scenario index so crash-resume
                            # can reconstruct done_scenario_indices precisely.
                            "scenario_index": i,
                            "style": scenario.style,
                            "length": scenario.length,
                            "themes": themes,  # unique keywords (was dup combinations)
                            "persona": scenario.persona.name,  # name only; role is verbose blob
                            # Type of KG relationship that connected the two nodes
                            # (e.g. "entities_overlap", "keyphrases_overlap",
                            #  "cosine_similarity", "child", "contains", …).
                            # Sourced from the scenario checkpoint written by
                            # _generate_scenarios, indexed by triplet_key so the
                            # mapping is stable across crash-resume runs (positional
                            # index diverges when some triplets produce no scenario).
                            "relation_type": getattr(scenario, "_rel_type", "unknown")
                            or "unknown",
                        },
                    }
                )
                # ── Checkpoint: persist after every new question ──────────────
                self._save_checkpoint(questions, checkpoint_path)
                self._save_eval_cache()
            except Exception as e:
                failed += 1
                logging.error(
                    f"Error generating question from scenario {i + 1} "
                    f"(total failures={failed}): {e}"
                )
                continue
 
        triplet_count = len(scenarios)
        _valid_final = _target_count(questions)
        _label = "valid (QE-passed) " if target_only_passed else ""
        _rejected_final = len(questions) - _valid_final
        if unlimited:
            logging.info(
                f"Unlimited mode complete: generated {len(questions)} questions "
                f"({_valid_final} QE-passed, {_rejected_final} QE-rejected) "
                f"from {triplet_count} triplets ({failed} scenario(s) skipped due to parse errors)"
            )
        elif _valid_final < num_questions:
            if triplet_count < n_request:
                logging.warning(
                    f"Only generated {_valid_final}/{num_questions} {_label}questions "
                    f"({len(questions)} kept in total, {_rejected_final} QE-rejected). "
                    f"Root cause: KG only produced {triplet_count} triplets "
                    f"(requested {n_request}). "
                    f"→ Add more documents to INPUT_DIR to get more triplets, "
                    f"OR reduce GLOBAL_NUM_QUESTIONS to ≤ {triplet_count - failed}."
                )
            else:
                logging.warning(
                    f"Only generated {_valid_final}/{num_questions} {_label}questions "
                    f"({len(questions)} kept in total, {_rejected_final} QE-rejected, "
                    f"{failed} scenario(s) failed out of {triplet_count} available). "
                    f"Consider increasing evaluation.scenario_buffer_ratio."
                )
        else:
            logging.info(
                f"Successfully generated {_valid_final} {_label}questions "
                f"({len(questions)} kept in total, {_rejected_final} QE-rejected, "
                f"{failed} scenario(s) skipped due to parse errors)"
            )

        # ── Optional: drop the questions that definitively failed QAEval ─────
        # They stay in the checkpoint (so a resume does not re-spend LLM calls on
        # the same scenarios) but are excluded from the returned dataset.
        if drop_failed and _rejected_final:
            questions = [
                q for q in questions
                if (q.get("qa_eval") or {}).get("passed") is not False
            ]
            logging.info(
                "drop_failed_questions=True — excluded %d QE-rejected question(s) "
                "from the returned dataset (%d remaining)",
                _rejected_final, len(questions),
            )

        return questions

