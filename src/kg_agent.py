"""
Agentic post-processing for Ragas KnowledgeGraph.

System 1 — RelationValidator
  Validates existing A→B relations using BOTH full chunk content AND keyphrases.
  A relation is reliable only when both signals agree.

System 2 — DirectRelationDiscovery (4 sub-agents, context-only, no O(n²), no cosine sim)
  Discovers new direct A→B relations between chunks from different documents.
  Pipeline:
    DocumentThemeAgent       — semantic themes per document      O(D) calls
    CrossDocMapAgent         — bridge themes spanning ≥2 docs    O(1) call
    ChunkLocatorAgent        — best chunks per (theme, doc)      O(T×D) calls
    DirectPairValidatorAgent — validate (A, B) inter-doc pairs   O(P) calls
  Output: agent_discovered relations (A→B) with relation_type and rationale.

Entry point:
    from kg_agent import run_agentic_enrichment
    kg, stats = run_agentic_enrichment(kg, llm)
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship


# ============================================================================
# MODULE-LEVEL HELPERS
# ============================================================================

def _node_id(node: Node) -> str:
    return str(getattr(node, "id", id(node)))


def _doc_id(node: Node) -> str:
    return (
        node.properties.get("parent_doc")
        or node.properties.get("filename")
        or _node_id(node)
    )


def _relation_key(r: Relationship) -> Tuple[str, str, str]:
    """Canonical dedup key for a relationship.

    For 'agent_discovered' relations the key includes the shared_theme, so the
    generated key is EXACTLY the same format as the candidate keys built by
    DirectRelationDiscovery / SemanticFrameBridgeDiscovery
    (``f"agent_discovered:{theme}"``). Without this, re-running the enrichment
    on an already-enriched KG silently re-added duplicate relations because the
    stored key ("agent_discovered") could never match the candidate key
    ("agent_discovered:<theme>").

    Two chunks may still be linked by several relations when they share several
    distinct themes — only strictly identical (pair + theme) relations are
    considered duplicates.
    """
    a = _node_id(r.source)
    b = _node_id(r.target)
    rtype = getattr(r, "type", "")
    if rtype == "agent_discovered":
        theme = (getattr(r, "properties", None) or {}).get("shared_theme", "")
        rtype = f"agent_discovered:{theme}"
    return (min(a, b), max(a, b), rtype)



def _existing_relation_keys(kg: KnowledgeGraph) -> set:
    return {_relation_key(r) for r in getattr(kg, "relationships", [])}


def _group_chunks_by_doc(kg: KnowledgeGraph) -> Dict[str, List[Node]]:
    """Build doc_id → list of CHUNK nodes mapping."""
    result: Dict[str, List[Node]] = defaultdict(list)
    for node in getattr(kg, "nodes", []):
        if getattr(node, "type", None) == NodeType.CHUNK:
            result[_doc_id(node)].append(node)
    return dict(result)


def _best_content(node: Node, max_chars: int) -> str:
    """Return the richest available text for a node.

    Priority:
      1. summary  — full-chunk summary produced by Ragas SummaryExtractor
                    (enrich_prechunked_official). No truncation needed: summaries
                    are compact by design and faithfully represent the whole chunk.
      2. raw_content / page_content truncated to max_chars — fallback when no
                    summary exists (enrich_prechunked_custom or lightweight mode).
    """
    summary = node.properties.get("summary", "")
    if summary:
        return summary
    raw = (
        node.properties.get("raw_content")
        or node.properties.get("page_content", "")
    )
    return raw[:max_chars]


# ============================================================================
# SYSTEM 1 — RELATION VALIDATOR
# Uses BOTH full chunk content AND keyphrases. Both signals must agree.
# ============================================================================

class RelationItem(BaseModel):
    relation_id: int
    relation_type: str
    source_breadcrumb: str
    target_breadcrumb: str
    source_content: str       # full chunk content, not a short snippet
    target_content: str
    shared_keyphrases: List[str]
    overlap_score: Optional[float] = None
    cosine_score: Optional[float] = None


class RelationBatchInput(BaseModel):
    relations: List[RelationItem]


class RelationVerdict(BaseModel):
    relation_id: int
    is_reliable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class RelationBatchOutput(BaseModel):
    verdicts: List[RelationVerdict]


_VALIDATOR_INSTRUCTION: str = (
    "You are a knowledge graph quality reviewer for a technical documentation corpus.\n\n"
    "You receive a batch of proposed SEMANTIC RELATIONS between documentation chunks. "
    "Each relation was created automatically based on shared keyphrases or embedding "
    "similarity. For EACH relation, decide whether it is GENUINELY RELIABLE.\n\n"
    "### Important constraints\n"
    "- Judge ONLY from information explicitly present in the two chunks. "
    "Do NOT infer a link merely because both chunks belong to the same overall system "
    "or product — a shared context is not a relation.\n"
    "- If a chunk is truncated, heavily redacted, or too ambiguous to judge, "
    "set is_reliable=False with a low confidence (≤ 0.4).\n\n"
    "### Evaluation method — use BOTH signals, they must both agree:\n\n"
    "Signal 1 — KEYPHRASES: Do the shared keyphrases provide meaningful evidence "
    "for the relation, beyond generic lexical overlap?\n"
    "  Reliable: keyphrases that are technically significant in context — named entities, "
    "API identifiers, protocol names, specific class or method names, or domain concepts "
    "that would NOT appear in unrelated documentation.\n"
    "  Unreliable: keyphrases that are generic words appearing in ANY technical "
    "documentation (setup, compute, default, output, value, method, function, process, "
    "system, configuration) without domain-specific context.\n\n"
    "Signal 2 — CONTENT: Do both chunks genuinely discuss the same technical concept "
    "or have a real technical dependency?\n"
    "  Reliable: both chunks describe the same named entity, workflow, mechanism, or "
    "concept — a reader would naturally cross-reference them.\n"
    "  Unreliable: chunks discuss unrelated topics despite sharing surface-level words.\n\n"
    "### Decision rule (strict AND):\n"
    "  is_reliable = True  only if BOTH signals are reliable.\n"
    "  is_reliable = False if EITHER signal fails.\n\n"
    "### Output\n"
    "- Exactly one verdict per relation_id, same order.\n"
    "- confidence ∈ [0.0, 1.0] — calibrated as follows:\n"
    "    0.9–1.0 : both signals are unambiguously clear.\n"
    "    0.7–0.9 : one signal is strong, the other is probable.\n"
    "    0.5–0.7 : relation is plausible but evidence is weak or mixed.\n"
    "    0.0–0.5 : chunk content is insufficient or too ambiguous to judge confidently.\n"
    "- reason: one sentence (max 25 words) citing which signal failed or confirmed.\n"
)

_VALIDATOR_EXAMPLES: list = [
    (
        RelationBatchInput(relations=[
            RelationItem(
                relation_id=0,
                relation_type="keyphrases_overlap",
                source_breadcrumb="Kubernetes > Networking > Service",
                target_breadcrumb="Kubernetes > Networking > Ingress Controller",
                source_content=(
                    "A ClusterIP Service exposes pods on an internal IP. kube-proxy programs "
                    "iptables rules to load-balance traffic across the endpoint set. The "
                    "Service selector matches pods by label; endpoints are updated by the "
                    "EndpointSlice controller when pods become ready."
                ),
                target_content=(
                    "The NGINX Ingress Controller watches Ingress resources and configures "
                    "upstream blocks pointing to ClusterIP Services. It resolves Service "
                    "endpoints via the EndpointSlice API and distributes traffic using "
                    "round-robin across ready pod IPs."
                ),
                shared_keyphrases=["ClusterIP Service", "EndpointSlice", "kube-proxy", "pod endpoints"],
            ),
            RelationItem(
                relation_id=1,
                relation_type="keyphrases_overlap",
                source_breadcrumb="Kubernetes > Core > Pod Lifecycle",
                target_breadcrumb="Kubernetes > Storage > PersistentVolumeClaim",
                source_content=(
                    "A pod transitions through Pending, Running, and Succeeded phases. "
                    "The kubelet reports status to the API server after each reconciliation loop."
                ),
                target_content=(
                    "A PersistentVolumeClaim requests storage from available PersistentVolumes. "
                    "The default StorageClass provisions a volume when no match exists."
                ),
                shared_keyphrases=["status", "default", "configuration"],
            ),
        ]),
        RelationBatchOutput(verdicts=[
            RelationVerdict(
                relation_id=0,
                is_reliable=True,
                confidence=0.93,
                reason="Keyphrases are specific API objects (ClusterIP, EndpointSlice) AND both chunks describe the same traffic-routing mechanism.",
            ),
            RelationVerdict(
                relation_id=1,
                is_reliable=False,
                confidence=0.91,
                reason="Keyphrases are generic (status, default) AND content describes unrelated subsystems with no shared technical concept.",
            ),
        ]),
    ),
]


class RelationValidatorPrompt(PydanticPrompt[RelationBatchInput, RelationBatchOutput]):
    instruction: str = _VALIDATOR_INSTRUCTION
    input_model = RelationBatchInput
    output_model = RelationBatchOutput
    examples: List[Tuple[RelationBatchInput, RelationBatchOutput]] = _VALIDATOR_EXAMPLES


class RelationValidator:
    """
    Validates existing semantic relations using BOTH full context AND keyphrases.
    Both signals must agree for a relation to be kept.
    The 'contains' relation type is never touched.
    """

    SEMANTIC_RELATION_TYPES: frozenset = frozenset({
        "keyphrases_overlap", "cosine_similarity",
    })

    def __init__(
        self,
        batch_size: int = 5,
        confidence_threshold: float = 0.5,
        max_content_chars: int = 4000,
        config: Any = None,
    ) -> None:
        self._config = config
        ev  = config.evaluation     if config else None
        kge = config.kg_enrichment  if config else None

        self.batch_size           = batch_size
        self.confidence_threshold = ev.relation_validator_confidence_threshold if ev else confidence_threshold
        self.max_content_chars    = ev.max_content_chars_agents                if ev else max_content_chars

        # Build prompt dynamically from config when available
        if config:
            try:
                from pipeline_config import build_prompt_class, deserialize_few_shots
                instr = config.prompts.relation_validator
                exs   = deserialize_few_shots(
                    "relation_validator", config.few_shots.relation_validator, config
                )
                self.prompt = build_prompt_class(RelationValidatorPrompt, instr, exs)()
            except Exception:
                self.prompt = RelationValidatorPrompt()
        else:
            self.prompt = RelationValidatorPrompt()

        # SEMANTIC_RELATION_TYPES from config (instance attribute overrides class attribute)
        if kge and kge.semantic_relation_types:
            self.SEMANTIC_RELATION_TYPES = frozenset(kge.semantic_relation_types) - {"agent_discovered", "contains"}
        else:
            self.SEMANTIC_RELATION_TYPES = frozenset({
                "keyphrases_overlap", "cosine_similarity",
            })

    def validate(
        self,
        kg: KnowledgeGraph,
        llm: Any,
        batch_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        """Synchronous entry point. Returns (cleaned_kg, stats_dict)."""
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._validate_async(kg, llm, batch_size, confidence_threshold)
        )

    async def _validate_async(
        self,
        kg: KnowledgeGraph,
        llm: Any,
        batch_size: Optional[int],
        confidence_threshold: Optional[float],
    ) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        bs = batch_size or self.batch_size
        threshold = confidence_threshold or self.confidence_threshold

        all_rels = getattr(kg, "relationships", [])
        semantic = [r for r in all_rels if getattr(r, "type", "") in self.SEMANTIC_RELATION_TYPES]
        non_semantic = [r for r in all_rels if getattr(r, "type", "") not in self.SEMANTIC_RELATION_TYPES]

        logging.info(
            "RelationValidator: %d semantic relations to validate (batch=%d, threshold=%.2f)",
            len(semantic), bs, threshold,
        )

        reliable: List[Relationship] = []
        removed = 0
        removal_details: List[Dict] = []

        _TRANSIENT_MARKERS = (
            "invalid model name", "connection error", "timeout", "timed out",
            "temporarily unavailable", "service unavailable", "bad gateway",
            "gateway timeout", "rate limit", "too many requests", "overloaded",
            "internal server error", " 429", " 500", " 502", " 503", " 504",
        )

        for start in range(0, len(semantic), bs):
            batch = semantic[start: start + bs]
            output: Optional[RelationBatchOutput] = None
            _last_exc = None
            # Retry with exponential backoff on transient proxy errors (LiteLLM
            # occasionally rejects a valid model under load — retrying a few
            # seconds later succeeds, as observed: KG enrichment succeeds with
            # the same model while validator batches fail in the same run).
            for _attempt in range(1, 6 + 1):
                try:
                    output = await self.prompt.generate(
                        llm=llm, data=self._build_batch_input(batch)
                    )
                    break
                except Exception as exc:
                    _last_exc = exc
                    _msg = str(exc).lower()
                    if not any(m in _msg for m in _TRANSIENT_MARKERS) or _attempt == 6:
                        break
                    _delay = 3.0 * (2 ** (_attempt - 1))
                    logging.warning(
                        "RelationValidator: batch failed (attempt %d/6, transient): %s — retrying in %.1fs",
                        _attempt, exc, _delay,
                    )
                    await asyncio.sleep(_delay)
            if output is None:
                logging.warning("RelationValidator: batch failed (%s) — keeping all", _last_exc)
                for rel in batch:
                    rel.properties.setdefault("agent_RelationValidator", True)
                reliable.extend(batch)
                continue

            verdict_map = {v.relation_id: v for v in output.verdicts}
            for idx, rel in enumerate(batch):
                v = verdict_map.get(idx)
                keep = v is None or v.is_reliable or v.confidence < threshold
                if v is not None:
                    rel.properties["validation_confidence"] = v.confidence
                rel.properties["agent_RelationValidator"] = keep
                if keep:
                    reliable.append(rel)
                else:
                    removed += 1
                    removal_details.append({
                        "type": getattr(rel, "type", ""),
                        "source": rel.source.properties.get("breadcrumb", "?"),
                        "target": rel.target.properties.get("breadcrumb", "?"),
                        "reason": v.reason,
                        "confidence": v.confidence,
                    })

        kg.relationships = non_semantic + reliable
        stats: Dict[str, Any] = {
            "total_semantic_before": len(semantic),
            "removed": removed,
            "remaining_semantic": len(reliable),
            "removal_details": removal_details,
        }
        logging.info(
            "RelationValidator: %d/%d semantic relations removed", removed, len(semantic)
        )
        return kg, stats

    def _build_batch_input(self, batch: List[Relationship]) -> RelationBatchInput:
        items = []
        for idx, rel in enumerate(batch):
            src = _best_content(rel.source, self.max_content_chars)
            tgt = _best_content(rel.target, self.max_content_chars)
            shared = rel.properties.get("shared_keyphrases", [])
            items.append(RelationItem(
                relation_id=idx,
                relation_type=getattr(rel, "type", ""),
                source_breadcrumb=rel.source.properties.get("breadcrumb", ""),
                target_breadcrumb=rel.target.properties.get("breadcrumb", ""),
                source_content=src[: self.max_content_chars],
                target_content=tgt[: self.max_content_chars],
                shared_keyphrases=shared if isinstance(shared, list) else [],
                overlap_score=rel.properties.get("overlap_score"),
                cosine_score=rel.properties.get("cosine_score"),
            ))
        return RelationBatchInput(relations=items)


# ============================================================================
# SYSTEM 2 — DIRECT RELATION DISCOVERY
# Multi-agent, context-only. No keyword logic. No cosine sim. No O(n²).
#
# Sub-agent 1: DocumentThemeAgent       — semantic themes per document  O(D)
# Sub-agent 2: CrossDocMapAgent         — bridge themes across docs     O(1)
# Sub-agent 3: ChunkLocatorAgent        — best chunks per (theme, doc)  O(T×D)
# Sub-agent 4: DirectPairValidatorAgent — validate (A, B) inter-doc pairs O(P)
# ============================================================================

# ── Sub-agent 1 : DocumentThemeAgent ─────────────────────────────────────────

class DocumentInput(BaseModel):
    doc_id: str
    content_preview: str   # compressed view: breadcrumbs + short excerpts


class Theme(BaseModel):
    name: str
    description: str   # one sentence: what aspect of this theme does the doc cover?


class DocumentThemes(BaseModel):
    doc_id: str
    themes: List[Theme]


_DOC_THEME_INSTRUCTION: str = (
    "You are analyzing a technical documentation page.\n\n"
    "The document is provided as a structured preview: section headings followed by "
    "short excerpts. Read it carefully and extract 3 to 5 SPECIFIC TECHNICAL THEMES.\n\n"
    "Each theme must be:\n"
    "  - Domain-specific: a concrete technical concept unique to this documentation's "
    "subject matter (named algorithms, protocols, APIs, data structures, domain "
    "workflows), NOT generic terms that could appear in any documentation "
    "(e.g. 'configuration', 'data processing', 'best practices', 'Python class').\n"
    "  - Discriminative: something that distinguishes this document from others "
    "in the same corpus.\n"
    "  - Grounded in the content: directly present in the excerpt, not inferred "
    "or assumed from the document title alone.\n\n"
    "For each theme, write one sentence in 'description' explaining what specific "
    "aspect of the theme this document covers.\n"
    "Return the doc_id unchanged.\n"
)

_DOC_THEME_EXAMPLES: list = [
    (
        DocumentInput(
            doc_id="react_hooks.md",
            content_preview=(
                "[React > Hooks > useState]\n"
                "useState returns a stateful value and a setter function. The initial "
                "state is computed once during the first render; subsequent calls to the "
                "setter trigger a re-render with the new value.\n\n"
                "[React > Hooks > useEffect]\n"
                "useEffect schedules a side-effect after render. The dependency array "
                "controls when the effect re-runs; an empty array means it runs only "
                "on mount and unmount.\n\n"
                "[React > Hooks > useRef]\n"
                "useRef holds a mutable .current value that persists across renders "
                "without causing re-renders. Commonly used for DOM element references "
                "and storing previous state."
            ),
        ),
        DocumentThemes(
            doc_id="react_hooks.md",
            themes=[
                Theme(
                    name="useState re-render lifecycle",
                    description="Describes how useState triggers component re-renders when the setter is called with a new value.",
                ),
                Theme(
                    name="useEffect dependency array semantics",
                    description="Explains how the dependency array controls effect re-execution, including the empty-array mount-only pattern.",
                ),
                Theme(
                    name="useRef mutable reference without re-render",
                    description="Details how useRef persists values across renders without triggering the reconciliation process.",
                ),
            ],
        ),
    ),
]


class DocumentThemePrompt(PydanticPrompt[DocumentInput, DocumentThemes]):
    instruction: str = _DOC_THEME_INSTRUCTION
    input_model = DocumentInput
    output_model = DocumentThemes
    examples: List[Tuple[DocumentInput, DocumentThemes]] = _DOC_THEME_EXAMPLES


# ── Sub-agent 2 : CrossDocMapAgent ───────────────────────────────────────────
#
# Input: compact theme registry — one entry per unique theme name + list of docs
# where it was extracted. No descriptions, no full content.
# Token count: O(unique_themes) instead of O(docs × themes × description_length).
# With 264 docs × 3-5 themes ≈ 800-1300 entries × ~7 tokens ≈ 6-10 K tokens.
# Zero information loss: every theme from every doc is represented.

class ThemeEntry(BaseModel):
    theme_name: str
    doc_ids: List[str]   # documents where DocumentThemeAgent extracted this theme


class AllThemeEntries(BaseModel):
    entries: List[ThemeEntry]


class ThemeBridge(BaseModel):
    name: str
    description: str
    doc_ids: List[str]   # ≥2 documents sharing this bridge theme


class ThemeBridges(BaseModel):
    bridges: List[ThemeBridge]


_CROSS_DOC_INSTRUCTION: str = (
    "You receive a compact theme registry for a technical documentation corpus. "
    "Each entry is a technical theme name and the list of documents where it was extracted.\n\n"
    "Your task: identify BRIDGE THEMES — technical concepts that genuinely appear in "
    "MULTIPLE documents (≥2 distinct doc_ids).\n\n"
    "Rules:\n"
    "  1. Merge entries whose theme names express the same concept with different wording "
    "     (e.g. two entries that name the same protocol, algorithm, or technical mechanism "
    "     with slight variations). Pick the clearest, most general name.\n"
    "  2. After merging, accumulate all doc_ids from the merged entries.\n"
    "  3. Keep only bridges that span ≥2 distinct documents.\n"
    "  4. Reject generic themes ('API', 'configuration', 'data processing', "
    "'best practices', 'overview').\n"
    "  5. Be conservative: prefer fewer, high-quality bridges over many weak ones. "
    "A bridge must represent a specific technical concept, not a vague topic.\n\n"
    "For each bridge, write a one-sentence description explaining what the concept is "
    "and why it is technically relevant across documents.\n"
    "Return an empty bridges list if no genuine cross-document bridge exists.\n"
)

_CROSS_DOC_EXAMPLES: list = [
    (
        AllThemeEntries(entries=[
            ThemeEntry(theme_name="OAuth 2.0 authorization code flow",
                       doc_ids=["auth_server.md"]),
            ThemeEntry(theme_name="JWT access token validation",
                       doc_ids=["auth_server.md"]),
            ThemeEntry(theme_name="OAuth2 token exchange for API gateway",
                       doc_ids=["api_gateway.md"]),
            ThemeEntry(theme_name="JWT signature verification middleware",
                       doc_ids=["api_gateway.md"]),
            ThemeEntry(theme_name="refresh token rotation strategy",
                       doc_ids=["auth_server.md"]),
            ThemeEntry(theme_name="JWT claims validation in route guards",
                       doc_ids=["frontend_auth.md"]),
        ]),
        ThemeBridges(bridges=[
            ThemeBridge(
                name="JWT access token validation",
                description="Cryptographic verification of JWT signatures and claims — issued by the auth server, verified by the API gateway middleware, and parsed by frontend route guards.",
                doc_ids=["auth_server.md", "api_gateway.md", "frontend_auth.md"],
            ),
            ThemeBridge(
                name="OAuth 2.0 token exchange",
                description="Protocol flow where the authorization code is exchanged for tokens — described in the auth server spec and consumed by the API gateway for upstream authentication.",
                doc_ids=["auth_server.md", "api_gateway.md"],
            ),
        ]),
    ),
]


class CrossDocMapPrompt(PydanticPrompt[AllThemeEntries, ThemeBridges]):
    instruction: str = _CROSS_DOC_INSTRUCTION
    input_model = AllThemeEntries
    output_model = ThemeBridges
    examples: List[Tuple[AllThemeEntries, ThemeBridges]] = _CROSS_DOC_EXAMPLES


# ── Sub-agent 3 : ChunkLocatorAgent ──────────────────────────────────────────

class ChunkPreview(BaseModel):
    chunk_idx: int
    breadcrumb: str
    excerpt: str   # first ~400 chars of content


class ChunkLocationRequest(BaseModel):
    theme_name: str
    theme_description: str
    doc_id: str
    chunks: List[ChunkPreview]


class LocatedChunks(BaseModel):
    best_indices: List[int]   # 1 to 3 indices, ordered by relevance (0-based)


_CHUNK_LOCATOR_INSTRUCTION: str = (
    "You receive documentation chunks from a single document and a technical theme "
    "to locate within them.\n\n"
    "Identify the 1 to 3 chunks that BEST represent or discuss this theme. "
    "A chunk is relevant if its content substantively covers the theme — not just "
    "mentions the theme's keywords incidentally.\n\n"
    "Return the chunk indices (0-based integers) ordered from most to least relevant. "
    "Return at most 3 indices. Return an empty list only if no chunk is genuinely relevant.\n"
    "Indices must be within [0, len(chunks)-1].\n"
)

_CHUNK_LOCATOR_EXAMPLES: list = [
    (
        ChunkLocationRequest(
            theme_name="JWT access token validation",
            theme_description="Cryptographic verification of JWT signatures and expiration claims in middleware.",
            doc_id="api_gateway.md",
            chunks=[
                ChunkPreview(
                    chunk_idx=0,
                    breadcrumb="API Gateway > Rate Limiting",
                    excerpt="The rate limiter uses a sliding-window counter stored in Redis. Each request increments the counter keyed by client IP.",
                ),
                ChunkPreview(
                    chunk_idx=1,
                    breadcrumb="API Gateway > JWT Middleware",
                    excerpt="The JWT middleware extracts the Bearer token from the Authorization header, verifies the RS256 signature against the JWKS endpoint, and checks exp/iat/iss claims before forwarding the request.",
                ),
                ChunkPreview(
                    chunk_idx=2,
                    breadcrumb="API Gateway > Request Routing",
                    excerpt="Routes are matched by path prefix. The router selects the upstream service and applies path rewriting rules defined in the configuration.",
                ),
            ],
        ),
        LocatedChunks(best_indices=[1]),
    ),
]


class ChunkLocatorPrompt(PydanticPrompt[ChunkLocationRequest, LocatedChunks]):
    instruction: str = _CHUNK_LOCATOR_INSTRUCTION
    input_model = ChunkLocationRequest
    output_model = LocatedChunks
    examples: List[Tuple[ChunkLocationRequest, LocatedChunks]] = _CHUNK_LOCATOR_EXAMPLES


# ── Sub-agent 4 : DirectPairValidatorAgent ───────────────────────────────────

class ChunkInfo(BaseModel):
    breadcrumb: str
    doc_id: str
    content: str


class DirectPairInput(BaseModel):
    chunk_a: ChunkInfo
    chunk_b: ChunkInfo
    shared_theme: str
    theme_description: str


class DirectPairValidation(BaseModel):
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    relation_type: str  # elaboration | contrast | prerequisite | example_of | shared_concept
    rationale: str      # one sentence explaining the decision


_DIRECT_PAIR_VALIDATOR_INSTRUCTION: str = (
    "You receive two documentation chunks (A and B) from TWO DIFFERENT documents "
    "and a shared technical theme that connects them. "
    "Decide whether these two chunks have a GENUINE DIRECT RELATION.\n\n"
    "A direct relation is valid when:\n"
    "  - Both chunks genuinely and substantively discuss the shared theme "
    "(not just a passing mention).\n"
    "  - A reader would benefit from reading both chunks together: B adds real value "
    "to understanding A on this theme, or vice versa.\n\n"
    "Mark is_valid=False if:\n"
    "  - One or both chunks only mention the theme incidentally.\n"
    "  - The chunks discuss the theme from entirely disconnected angles with no "
    "intellectual link.\n"
    "  - The relation would add noise, not insight, to the knowledge graph.\n\n"
    "If is_valid=True, classify the relation as ONE of:\n"
    "  elaboration           — B elaborates, details, or expands on content from A.\n"
    "  contrast              — B presents a contrasting or alternative perspective to A.\n"
    "  prerequisite          — A must be understood before B (or vice versa).\n"
    "  example_of            — B is a concrete example or application of the concept in A.\n"
    "  shared_concept        — Both cover the same concept from complementary angles.\n"
    "  conditional_behavior  — Same mechanism/operation in A and B, but behaves differently\n"
    "                          under different conditions or configurations.\n"
    "  operation_comparison  — Same object or resource is processed by two different\n"
    "                          operations or algorithms described in A and B.\n"
    "  generalization_pattern— One chunk states a general principle; the other instantiates\n"
    "                          or extends it in a specific context.\n"
    "  convergent_goal       — Different mechanisms in A and B produce the same observable\n"
    "                          result or achieve the same goal.\n"
    "  complementary_aspect  — A and B each cover a distinct, orthogonal aspect of the same\n"
    "                          entity or process; together they give a complete picture.\n\n"
    "Prefer the most specific type that fits. Use 'shared_concept' only when none of the\n"
    "more specific types above apply.\n\n"
    "Output:\n"
    "  - is_valid: True/False\n"
    "  - confidence ∈ [0.0, 1.0]\n"
    "  - relation_type: one of the ten labels above (empty string if is_valid=False).\n"
    "  - rationale: one sentence explaining the decision.\n"
)

_DIRECT_PAIR_VALIDATOR_EXAMPLES: list = [
    (
        DirectPairInput(
            chunk_a=ChunkInfo(
                breadcrumb="Networking > TCP > Congestion Control > Slow Start",
                doc_id="tcp_internals.md",
                content=(
                    "TCP slow start initialises the congestion window (cwnd) to one "
                    "maximum segment size (MSS). For each acknowledged segment, cwnd "
                    "doubles until it reaches the slow-start threshold (ssthresh), "
                    "at which point the algorithm switches to congestion avoidance."
                ),
            ),
            chunk_b=ChunkInfo(
                breadcrumb="Networking > TCP > Congestion Control > CUBIC",
                doc_id="tcp_cubic.md",
                content=(
                    "CUBIC replaces the slow-start phase for high-bandwidth networks. "
                    "Its window-growth function is cubic around the last congestion point, "
                    "allowing faster recovery without triggering further packet loss."
                ),
            ),
            shared_theme="TCP congestion window growth",
            theme_description="How TCP grows its congestion window after a connection starts or recovers from loss.",
        ),
        DirectPairValidation(
            is_valid=True,
            confidence=0.92,
            relation_type="conditional_behavior",
            rationale="Both describe cwnd growth but slow start doubles exponentially while CUBIC uses a cubic function — same mechanism behaves differently by algorithm variant.",
        ),
    ),
    (
        DirectPairInput(
            chunk_a=ChunkInfo(
                breadcrumb="PostgreSQL > Indexes > B-tree",
                doc_id="pg_indexes.md",
                content=(
                    "A B-tree index stores column values in sorted order. "
                    "The query planner uses it for equality predicates, range queries, "
                    "and ORDER BY clauses on the leading indexed column."
                ),
            ),
            chunk_b=ChunkInfo(
                breadcrumb="PostgreSQL > Indexes > Hash",
                doc_id="pg_indexes.md",
                content=(
                    "A hash index computes a hash of each indexed value. "
                    "It supports only equality predicates and offers O(1) lookups, "
                    "but cannot serve range queries or sorting."
                ),
            ),
            shared_theme="index lookup for query predicates",
            theme_description="How PostgreSQL index types serve different query predicate shapes.",
        ),
        DirectPairValidation(
            is_valid=True,
            confidence=0.94,
            relation_type="operation_comparison",
            rationale="Both index types operate on the same resource (query predicates on a column) but use different data structures with complementary trade-offs.",
        ),
    ),
    (
        DirectPairInput(
            chunk_a=ChunkInfo(
                breadcrumb="Python > Basics > Functions > Definition",
                doc_id="python_tutorial.md",
                content=(
                    "A function is defined with the def keyword followed by a name "
                    "and a parameter list. The body is indented."
                ),
            ),
            chunk_b=ChunkInfo(
                breadcrumb="Django > ORM > QuerySet > filter",
                doc_id="django_orm.md",
                content=(
                    "QuerySet.filter() accepts keyword arguments that map to database "
                    "column lookups. It returns a new QuerySet, not a list."
                ),
            ),
            shared_theme="function definition",
            theme_description="How functions are defined and called.",
        ),
        DirectPairValidation(
            is_valid=False,
            confidence=0.89,
            relation_type="",
            rationale="Both mention functions but describe entirely unrelated topics — the theme is too generic to constitute a meaningful cross-document link.",
        ),
    ),
]


class DirectPairValidatorPrompt(PydanticPrompt[DirectPairInput, DirectPairValidation]):
    instruction: str = _DIRECT_PAIR_VALIDATOR_INSTRUCTION
    input_model = DirectPairInput
    output_model = DirectPairValidation
    examples: List[Tuple[DirectPairInput, DirectPairValidation]] = _DIRECT_PAIR_VALIDATOR_EXAMPLES


# ── DirectRelationDiscovery ───────────────────────────────────────────────────

class DirectRelationDiscovery:
    """
    Discovers direct A→B relations between chunks from different documents.
    No keyword matching. No cosine similarity. No O(n²) pairwise loop.

    Complexity: O(D + T×D + P) where D = documents, T = bridge themes, P = pairs.
    Outputs 'agent_discovered' relations (A→B) with relation_type and rationale.
    """

    def __init__(
        self,
        max_themes_per_doc: int = 5,
        max_chunks_per_theme: int = 2,
        min_confidence: float = 0.65,
        max_content_chars: int = 4000,
        config: Any = None,
    ) -> None:
        self._config = config
        ev = config.evaluation if config else None

        self.max_themes_per_doc   = max_themes_per_doc
        self.max_chunks_per_theme = max_chunks_per_theme
        self.min_confidence       = ev.discovery_min_confidence   if ev else min_confidence
        self.max_content_chars    = ev.max_content_chars_agents   if ev else max_content_chars

        # Build sub-agent prompts dynamically from config when available
        def _make_prompt(base_cls, prompt_key, few_shot_key):
            if config:
                try:
                    from pipeline_config import build_prompt_class, deserialize_few_shots
                    instr = getattr(config.prompts, prompt_key)
                    exs   = deserialize_few_shots(few_shot_key, getattr(config.few_shots, few_shot_key), config)
                    return build_prompt_class(base_cls, instr, exs)()
                except Exception:
                    pass
            return base_cls()

        self._doc_theme_prompt      = _make_prompt(DocumentThemePrompt,      "doc_theme",            "doc_theme")
        self._cross_doc_prompt      = _make_prompt(CrossDocMapPrompt,         "cross_doc_map",        "cross_doc_map")
        self._chunk_locator_prompt  = _make_prompt(ChunkLocatorPrompt,        "chunk_locator",        "chunk_locator")
        self._pair_validator_prompt = _make_prompt(DirectPairValidatorPrompt, "direct_pair_validator","direct_pair_validator")

    def discover(
        self,
        kg: KnowledgeGraph,
        llm: Any,
        min_confidence: Optional[float] = None,
        on_progress: Optional[callable] = None,
    ) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        """Synchronous entry point. Adds agent_discovered relations to kg."""
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._discover_async(kg, llm, min_confidence, on_progress=on_progress)
        )

    async def _discover_async(
        self,
        kg: KnowledgeGraph,
        llm: Any,
        min_confidence: Optional[float],
        on_progress: Optional[callable] = None,
    ) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        threshold = min_confidence or self.min_confidence
        chunk_by_doc = _group_chunks_by_doc(kg)

        if len(chunk_by_doc) < 2:
            logging.warning(
                "DirectRelationDiscovery: need ≥2 documents, found %d — skipping.",
                len(chunk_by_doc),
            )
            return kg, {"error": f"need ≥2 documents, found {len(chunk_by_doc)}"}

        # ── Sub-agent 1 : DocumentThemeAgent ────────────────────────────────
        logging.info(
            "DirectRelationDiscovery [1/4] DocumentThemeAgent — %d documents",
            len(chunk_by_doc),
        )
        doc_themes: Dict[str, DocumentThemes] = {}
        for doc_id, chunks in chunk_by_doc.items():
            try:
                result: DocumentThemes = await self._doc_theme_prompt.generate(
                    llm=llm,
                    data=DocumentInput(
                        doc_id=doc_id,
                        content_preview=self._build_doc_preview(chunks),
                    ),
                )
                doc_themes[doc_id] = result
            except Exception as exc:
                logging.warning(
                    "DocumentThemeAgent: doc '%s' failed (%s) — skipping", doc_id, exc
                )

        if len(doc_themes) < 2:
            logging.warning(
                "DirectRelationDiscovery: theme extraction succeeded for only %d docs — aborting.",
                len(doc_themes),
            )
            return kg, {"error": "theme extraction insufficient"}

        # ── Sub-agent 2 : CrossDocMapAgent ──────────────────────────────────
        # Build compact theme registry: theme_name → [doc_ids].
        # One entry per unique theme name across all docs — O(unique_themes) tokens
        # instead of O(docs × themes × description_length). No information lost.
        logging.info(
            "DirectRelationDiscovery [2/4] CrossDocMapAgent — building compact registry"
        )
        theme_to_docs: Dict[str, List[str]] = defaultdict(list)
        for doc_id, dt in doc_themes.items():
            for theme in dt.themes[: self.max_themes_per_doc]:
                theme_to_docs[theme.name].append(doc_id)

        all_entries = [
            ThemeEntry(theme_name=name, doc_ids=list(set(doc_ids)))
            for name, doc_ids in theme_to_docs.items()
        ]
        logging.info(
            "  compact registry: %d unique theme entries across %d docs",
            len(all_entries), len(doc_themes),
        )

        # Split into batches to avoid max_tokens overflow on output side.
        # Each batch produces a bounded number of bridge themes.
        _CROSSDOC_BATCH = 120
        all_bridges: List[ThemeBridge] = []
        n_batches = max(1, (len(all_entries) + _CROSSDOC_BATCH - 1) // _CROSSDOC_BATCH)
        logging.info("  splitting into %d batches of ≤%d entries", n_batches, _CROSSDOC_BATCH)
        for batch_idx in range(n_batches):
            batch_entries = all_entries[batch_idx * _CROSSDOC_BATCH:(batch_idx + 1) * _CROSSDOC_BATCH]
            registry = AllThemeEntries(entries=batch_entries)
            try:
                partial: ThemeBridges = await self._cross_doc_prompt.generate(
                    llm=llm, data=registry
                )
                all_bridges.extend(partial.bridges)
                logging.info(
                    "  batch %d/%d → %d bridge(s) found",
                    batch_idx + 1, n_batches, len(partial.bridges),
                )
            except Exception as exc:
                logging.warning(
                    "  CrossDocMapAgent batch %d/%d failed (%s) — skipping batch",
                    batch_idx + 1, n_batches, exc,
                )

        bridges = ThemeBridges(bridges=all_bridges)
        logging.info("  total bridge themes collected: %d", len(bridges.bridges))
        if not bridges.bridges:
            logging.error("CrossDocMapAgent: all batches failed or produced 0 bridges")
            return kg, {"error": "CrossDocMapAgent: 0 bridges after all batches"}

        valid_docs = set(chunk_by_doc.keys())
        bridges.bridges = [
            b for b in bridges.bridges
            if len([d for d in b.doc_ids if d in valid_docs]) >= 2
        ]
        logging.info("  %d valid bridge themes found", len(bridges.bridges))

        if not bridges.bridges:
            return kg, {
                "documents_processed": len(doc_themes),
                "bridge_themes": 0,
                "candidates_evaluated": 0,
                "relations_added": 0,
            }

        # ── Sub-agent 3 : ChunkLocatorAgent ─────────────────────────────────
        logging.info("DirectRelationDiscovery [3/4] ChunkLocatorAgent")
        chunk_locs: Dict[int, Dict[str, List[Node]]] = {}

        for b_idx, bridge in enumerate(bridges.bridges):
            chunk_locs[b_idx] = {}
            for doc_id in bridge.doc_ids:
                if doc_id not in chunk_by_doc:
                    continue
                chunks = chunk_by_doc[doc_id]
                try:
                    located: LocatedChunks = await self._chunk_locator_prompt.generate(
                        llm=llm,
                        data=ChunkLocationRequest(
                            theme_name=bridge.name,
                            theme_description=bridge.description,
                            doc_id=doc_id,
                            chunks=[
                                ChunkPreview(
                                    chunk_idx=i,
                                    breadcrumb=c.properties.get("breadcrumb", ""),
                                    excerpt=c.properties.get("raw_content", "")[:800],
                                )
                                for i, c in enumerate(chunks)
                            ],
                        ),
                    )
                    valid_idx = [i for i in located.best_indices if 0 <= i < len(chunks)]
                    chunk_locs[b_idx][doc_id] = [
                        chunks[i] for i in valid_idx[: self.max_chunks_per_theme]
                    ]
                except Exception as exc:
                    logging.warning(
                        "ChunkLocatorAgent: bridge '%s' / doc '%s' failed (%s) — skipping",
                        bridge.name, doc_id, exc,
                    )

        # ── Sub-agent 4 : DirectPairValidatorAgent ───────────────────────────
        logging.info("DirectRelationDiscovery [4/4] DirectPairValidatorAgent")
        existing_keys = _existing_relation_keys(kg)
        new_relations: List[Relationship] = []
        total_candidates = 0
        total_added = 0

        for b_idx, bridge in enumerate(bridges.bridges):
            doc_ids_in_bridge = [d for d in bridge.doc_ids if d in chunk_locs[b_idx]]
            # Generate all inter-document chunk pairs for this bridge theme
            for i, doc_a in enumerate(doc_ids_in_bridge):
                for doc_b in doc_ids_in_bridge[i + 1:]:
                    for chunk_a in chunk_locs[b_idx].get(doc_a, []):
                        for chunk_b in chunk_locs[b_idx].get(doc_b, []):
                            # Include shared_theme so the same chunk-pair can
                            # produce multiple relations when they share N distinct
                            # bridge themes — previously the second relation was
                            # silently dropped because its key collided.
                            cand_key = (
                                min(_node_id(chunk_a), _node_id(chunk_b)),
                                max(_node_id(chunk_a), _node_id(chunk_b)),
                                f"agent_discovered:{bridge.name}",
                            )
                            if cand_key in existing_keys:
                                continue
                            total_candidates += 1

                            try:
                                validation: DirectPairValidation = (
                                    await self._pair_validator_prompt.generate(
                                        llm=llm,
                                        data=DirectPairInput(
                                            chunk_a=ChunkInfo(
                                                breadcrumb=chunk_a.properties.get("breadcrumb", ""),
                                                doc_id=_doc_id(chunk_a),
                                                content=_best_content(chunk_a, self.max_content_chars),
                                            ),
                                            chunk_b=ChunkInfo(
                                                breadcrumb=chunk_b.properties.get("breadcrumb", ""),
                                                doc_id=_doc_id(chunk_b),
                                                content=_best_content(chunk_b, self.max_content_chars),
                                            ),
                                            shared_theme=bridge.name,
                                            theme_description=bridge.description,
                                        ),
                                    )
                                )
                            except Exception as exc:
                                logging.warning(
                                    "DirectPairValidatorAgent: pair failed (%s) — skipping", exc
                                )
                                continue

                            if not validation.is_valid or validation.confidence < threshold:
                                continue

                            new_relations.append(Relationship(
                                source=chunk_a,
                                target=chunk_b,
                                type="agent_discovered",
                                properties={
                                    "relation_type": validation.relation_type,
                                    "shared_theme": bridge.name,
                                    "rationale": validation.rationale,
                                    "confidence": validation.confidence,
                                },
                            ))
                            existing_keys.add(cand_key)
                            total_added += 1

            # Emit live snapshot after each bridge theme is fully processed
            if on_progress is not None and new_relations:
                kg.relationships.extend(new_relations)
                new_relations = []
                try:
                    on_progress(kg, "agent_discovered")
                except Exception:
                    pass

        kg.relationships.extend(new_relations)
        stats: Dict[str, Any] = {
            "documents_processed": len(doc_themes),
            "bridge_themes": len(bridges.bridges),
            "candidates_evaluated": total_candidates,
            "relations_added": total_added,
        }
        logging.info(
            "DirectRelationDiscovery: %d agent_discovered relations added "
            "(%d candidates, %d bridge themes, %d docs)",
            total_added, total_candidates, len(bridges.bridges), len(doc_themes),
        )
        return kg, stats

    def _build_doc_preview(self, chunks: List[Node]) -> str:
        lines = []
        for chunk in chunks:
            breadcrumb = chunk.properties.get("breadcrumb", "?")
            # Prefer summary (compact, faithful) over truncated raw content
            content = _best_content(chunk, max_chars=600)
            excerpt = " ".join(content.split()[:120])
            lines.append(f"[{breadcrumb}]\n{excerpt}")
        return "\n\n".join(lines)


# ============================================================================
# SYSTEM 3 — SEMANTIC FRAME BRIDGE DISCOVERY
#
# Complements DirectRelationDiscovery (System 2) with bottom-up frame analysis.
#
# System 2 is top-down: document themes → bridge themes → locate chunks.
# System 3 is bottom-up: extract semantic frames per chunk, find frame-level
# structural alignments that theme-discovery would miss (same mechanism under
# different conditions, same object operated on differently, etc.).
#
# Complementarity guarantee: System 3 skips pairs already covered by an
# agent_discovered relation (checked via _existing_relation_keys before any
# LLM call). It only adds genuinely new relations.
#
# Pipeline:
#   Step 1 — FrameExtractionAgent : extract {subject,operation,object,
#             condition,result} frames per chunk batch          O(C/8) calls
#   Step 2 — Structural pre-filter: Jaccard ≥ 0.30 on ≥2 slots (no LLM)
#   Step 3 — Embedding pre-filter : cosine ≥ threshold (optional, fast)
#   Step 4 — DirectPairValidatorAgent (reused): validate & classify the
#             surviving frame-candidate pairs                    O(P) calls
#
# Output: agent_discovered relations (same type as System 2) with
#   relation_type ∈ {conditional_behavior, operation_comparison,
#                    generalization_pattern, convergent_goal,
#                    complementary_aspect, …}
#   shared_theme = bridge_concept (noun phrase from frame alignment)
# ============================================================================

# ── Frame models ─────────────────────────────────────────────────────────────

class SemanticFrame(BaseModel):
    subject: str
    operation: str
    object_: str = Field(alias="object")
    condition: str
    result: str

    model_config = {"populate_by_name": True}


class ChunkFrames(BaseModel):
    chunk_idx: int
    frames: List[SemanticFrame]


class DocumentFrames(BaseModel):
    chunk_frames: List[ChunkFrames]


_FRAME_EXTRACTION_INSTRUCTION: str = (
    "You are analyzing technical documentation chunks to extract semantic frames.\n\n"
    "A semantic frame describes one concrete mechanism or operation in the chunk:\n"
    "  subject   — the named entity or component that performs the action\n"
    "              (e.g. a named algorithm, class, protocol, component, API method)\n"
    "  operation — the specific technical action or mechanism (verb phrase)\n"
    "              (e.g. 'computes', 'indexes', 'routes', 'serializes', 'validates')\n"
    "  object    — what is acted upon, produced, or modified\n"
    "              (e.g. named data structure, query type, payload, resource)\n"
    "  condition — the constraint, mode, or context under which this holds\n"
    "              (use 'always' if unconditional)\n"
    "  result    — the observable output, effect, or consequence\n\n"
    "Rules:\n"
    "  - Extract 1 to 4 frames per chunk. Prefer fewer, high-quality frames.\n"
    "  - Each slot must use domain-specific terms (named entities, identifiers,\n"
    "    technical concepts specific to the subject matter). NEVER use generic\n"
    "    words like 'value', 'result', 'process', 'method', 'data', 'system',\n"
    "    'configuration' alone — these produce false-positive bridges.\n"
    "  - If a slot has no domain-specific content, use 'N/A'.\n"
    "  - A frame is worth keeping only if subject + operation are both domain-specific.\n"
    "  - Return chunk_idx unchanged.\n"
)

_FRAME_EXTRACTION_EXAMPLES: list = [
    (
        ChunkLocationRequest(
            theme_name="",
            theme_description="",
            doc_id="database_guide.md",
            chunks=[
                ChunkPreview(
                    chunk_idx=0,
                    breadcrumb="PostgreSQL > Indexing > B-tree index",
                    excerpt=(
                        "A B-tree index stores values in sorted order, enabling "
                        "range queries and equality lookups in O(log n). "
                        "The planner uses the index when the query predicate matches "
                        "the leading column of the index definition."
                    ),
                ),
                ChunkPreview(
                    chunk_idx=1,
                    breadcrumb="PostgreSQL > Indexing > Hash index",
                    excerpt=(
                        "A hash index stores a hash of each indexed value. "
                        "Lookups are O(1) for equality predicates but hash indexes "
                        "do not support range queries or ORDER BY optimisation."
                    ),
                ),
            ],
        ),
        DocumentFrames(chunk_frames=[
            ChunkFrames(chunk_idx=0, frames=[
                SemanticFrame(**{
                    "subject": "B-tree index",
                    "operation": "stores values in sorted order",
                    "object": "equality and range query predicates",
                    "condition": "leading column matches query predicate",
                    "result": "O(log n) lookup; range queries and ORDER BY supported",
                }),
            ]),
            ChunkFrames(chunk_idx=1, frames=[
                SemanticFrame(**{
                    "subject": "hash index",
                    "operation": "stores hash of each indexed value",
                    "object": "equality query predicates",
                    "condition": "always",
                    "result": "O(1) equality lookup; range queries not supported",
                }),
            ]),
        ]),
    ),
]


class FrameExtractionPrompt(PydanticPrompt[ChunkLocationRequest, DocumentFrames]):
    instruction: str = _FRAME_EXTRACTION_INSTRUCTION
    input_model = ChunkLocationRequest
    output_model = DocumentFrames
    examples: List[Tuple[ChunkLocationRequest, DocumentFrames]] = _FRAME_EXTRACTION_EXAMPLES


# ── SemanticFrameBridgeDiscovery ──────────────────────────────────────────────

_FRAME_GENERIC_TOKENS: frozenset = frozenset({
    "n/a", "always", "value", "result", "method", "data", "system",
    "process", "output", "input", "default", "compute", "function",
    "object", "class", "module", "parameter", "configuration",
    "component", "interface", "service", "handler", "manager",
    "controller", "entity", "model", "instance", "type", "state",
    "request", "response", "returns", "creates", "updates", "the",
    "a", "an", "is", "are", "it", "its", "this", "that", "and", "or",
})


def _slot_tokens(value: str) -> set:
    tokens = {t.lower().strip() for t in value.replace(",", " ").split()}
    return tokens - _FRAME_GENERIC_TOKENS


def _frames_have_structural_overlap(fa: SemanticFrame, fb: SemanticFrame) -> bool:
    """True when ≥2 non-generic frame slots match with Jaccard ≥ 0.30."""
    slots_a = {
        "subject":   _slot_tokens(fa.subject),
        "operation": _slot_tokens(fa.operation),
        "object":    _slot_tokens(fa.object_),
        "result":    _slot_tokens(fa.result),
    }
    slots_b = {
        "subject":   _slot_tokens(fb.subject),
        "operation": _slot_tokens(fb.operation),
        "object":    _slot_tokens(fb.object_),
        "result":    _slot_tokens(fb.result),
    }
    matching = 0
    for slot in ("subject", "operation", "object", "result"):
        sa, sb = slots_a[slot], slots_b[slot]
        if sa and sb and len(sa & sb) / len(sa | sb) >= 0.30:
            matching += 1
    return matching >= 2


class SemanticFrameBridgeDiscovery:
    """
    Bottom-up complement to DirectRelationDiscovery.

    Extracts semantic frames per chunk and discovers cross-document bridges
    based on structural frame alignment — finding pairs that share the same
    mechanism operating under different conditions, or the same object
    processed by different operations.

    Skips pairs already covered by agent_discovered (System 2) to avoid
    redundant LLM calls and duplicate relations.

    Reuses DirectPairValidatorPrompt (same 10-type taxonomy as System 2)
    so relation_type and downstream question generation are identical.

    Output: agent_discovered relations with relation_type ∈ the 10-type
    taxonomy and shared_theme = the frame-derived bridge concept.
    """

    def __init__(
        self,
        min_confidence: float = 0.70,
        max_content_chars: int = 4000,
        embedding_pre_filter_threshold: float = 0.25,
        max_frames_per_chunk: int = 3,
        config: Any = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_content_chars = max_content_chars
        self.embedding_pre_filter_threshold = embedding_pre_filter_threshold
        self.max_frames_per_chunk = max_frames_per_chunk

        self._frame_prompt = FrameExtractionPrompt()
        # Reuse the same validator prompt as DirectRelationDiscovery
        self._pair_validator_prompt = DirectPairValidatorPrompt()

    def discover(
        self,
        kg: KnowledgeGraph,
        llm: Any,
        embedding_model: Any = None,
        on_progress: Optional[callable] = None,
    ) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        """Synchronous entry point. Adds agent_discovered relations to kg."""
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._discover_async(kg, llm, embedding_model, on_progress)
        )

    async def _discover_async(
        self,
        kg: KnowledgeGraph,
        llm: Any,
        embedding_model: Any = None,
        on_progress: Optional[callable] = None,
    ) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        chunk_by_doc = _group_chunks_by_doc(kg)
        if len(chunk_by_doc) < 2:
            return kg, {"error": "need ≥2 documents"}

        all_chunks: List[Node] = [c for chunks in chunk_by_doc.values() for c in chunks]

        # ── Step 1: extract frames ────────────────────────────────────────────
        logging.info(
            "SemanticFrameBridgeDiscovery [1/3] FrameExtractionAgent — %d chunks",
            len(all_chunks),
        )
        _EXTRACT_BATCH = 8
        chunk_frames_map: Dict[str, List[SemanticFrame]] = {}

        for batch_start in range(0, len(all_chunks), _EXTRACT_BATCH):
            batch = all_chunks[batch_start: batch_start + _EXTRACT_BATCH]
            req = ChunkLocationRequest(
                theme_name="", theme_description="", doc_id="",
                chunks=[
                    ChunkPreview(
                        chunk_idx=i,
                        breadcrumb=c.properties.get("breadcrumb", ""),
                        excerpt=_best_content(c, 600)[:600],
                    )
                    for i, c in enumerate(batch)
                ],
            )
            try:
                doc_frames: DocumentFrames = await self._frame_prompt.generate(
                    llm=llm, data=req
                )
                for cf in doc_frames.chunk_frames:
                    if 0 <= cf.chunk_idx < len(batch):
                        node_id = _node_id(batch[cf.chunk_idx])
                        frames = [
                            f for f in cf.frames[: self.max_frames_per_chunk]
                            if f.subject.lower().strip() not in _FRAME_GENERIC_TOKENS
                            or f.operation.lower().strip() not in _FRAME_GENERIC_TOKENS
                        ]
                        if frames:
                            chunk_frames_map[node_id] = frames
            except Exception as exc:
                logging.warning(
                    "FrameExtractionAgent: batch %d failed (%s) — skipping",
                    batch_start // _EXTRACT_BATCH, exc,
                )

        logging.info(
            "  frames extracted for %d/%d chunks", len(chunk_frames_map), len(all_chunks)
        )
        if len(chunk_frames_map) < 2:
            return kg, {"error": "frame extraction insufficient"}

        # ── Step 2: structural pre-filter (no LLM) ───────────────────────────
        chunks_with_frames = [c for c in all_chunks if _node_id(c) in chunk_frames_map]
        node_to_doc = {_node_id(c): _doc_id(c) for c in chunks_with_frames}
        # Keys already covered by agent_discovered — skip to avoid duplicates
        existing_keys = _existing_relation_keys(kg)

        struct_candidates: List[Tuple[Node, Node, str]] = []  # (a, b, bridge_concept)
        for i, ca in enumerate(chunks_with_frames):
            for cb in chunks_with_frames[i + 1:]:
                if node_to_doc[_node_id(ca)] == node_to_doc[_node_id(cb)]:
                    continue  # intra-doc: skip
                # Skip pairs already covered by agent_discovered
                base_key = (
                    min(_node_id(ca), _node_id(cb)),
                    max(_node_id(ca), _node_id(cb)),
                )
                already_covered = any(
                    k for k in existing_keys
                    if k[0] == base_key[0] and k[1] == base_key[1]
                    and k[2].startswith("agent_discovered")
                )
                if already_covered:
                    continue
                # Find the best frame pair with structural overlap
                best_concept = ""
                for fa in chunk_frames_map[_node_id(ca)]:
                    for fb in chunk_frames_map[_node_id(cb)]:
                        if _frames_have_structural_overlap(fa, fb):
                            # Use subject+operation of the overlapping frame as concept name
                            shared = _slot_tokens(fa.subject) & _slot_tokens(fb.subject)
                            if not shared:
                                shared = _slot_tokens(fa.operation) & _slot_tokens(fb.operation)
                            best_concept = " ".join(sorted(shared)[:4]) if shared else fa.subject
                            break
                    if best_concept:
                        break
                if best_concept:
                    struct_candidates.append((ca, cb, best_concept))

        logging.info(
            "SemanticFrameBridgeDiscovery [2/3] structural pre-filter — %d candidates",
            len(struct_candidates),
        )
        if not struct_candidates:
            return kg, {"relations_added": 0, "candidates_evaluated": 0}

        # ── Step 3: optional embedding pre-filter ────────────────────────────
        candidates = struct_candidates
        if embedding_model is not None and len(struct_candidates) > 20:
            try:
                candidates = await self._embedding_filter(
                    struct_candidates, chunk_frames_map, embedding_model
                )
                logging.info(
                    "  embedding filter: %d → %d candidates",
                    len(struct_candidates), len(candidates),
                )
            except Exception as exc:
                logging.warning("Embedding pre-filter failed (%s) — skipping", exc)

        # ── Step 4: DirectPairValidatorAgent (reused) ─────────────────────────
        logging.info(
            "SemanticFrameBridgeDiscovery [3/3] DirectPairValidatorAgent — %d pairs",
            len(candidates),
        )
        new_relations: List[Relationship] = []
        total_added = 0

        for chunk_a, chunk_b, bridge_concept in candidates:
            cand_key = (
                min(_node_id(chunk_a), _node_id(chunk_b)),
                max(_node_id(chunk_a), _node_id(chunk_b)),
                f"agent_discovered:{bridge_concept}",
            )
            if cand_key in existing_keys:
                continue
            try:
                validation: DirectPairValidation = await self._pair_validator_prompt.generate(
                    llm=llm,
                    data=DirectPairInput(
                        chunk_a=ChunkInfo(
                            breadcrumb=chunk_a.properties.get("breadcrumb", ""),
                            doc_id=_doc_id(chunk_a),
                            content=_best_content(chunk_a, self.max_content_chars),
                        ),
                        chunk_b=ChunkInfo(
                            breadcrumb=chunk_b.properties.get("breadcrumb", ""),
                            doc_id=_doc_id(chunk_b),
                            content=_best_content(chunk_b, self.max_content_chars),
                        ),
                        shared_theme=bridge_concept,
                        theme_description=(
                            f"Structural frame bridge: {bridge_concept} — "
                            "two chunks sharing this mechanism or concept from different angles."
                        ),
                    ),
                )
            except Exception as exc:
                logging.warning("DirectPairValidatorAgent (frame): pair failed (%s) — skipping", exc)
                continue

            if not validation.is_valid or validation.confidence < self.min_confidence:
                continue

            new_relations.append(Relationship(
                source=chunk_a,
                target=chunk_b,
                type="agent_discovered",
                properties={
                    "relation_type": validation.relation_type,
                    "shared_theme": bridge_concept,
                    "rationale": validation.rationale,
                    "confidence": validation.confidence,
                    "discovery_method": "semantic_frame_bridge",
                },
            ))
            existing_keys.add(cand_key)
            total_added += 1

        kg.relationships.extend(new_relations)
        if on_progress is not None and new_relations:
            try:
                on_progress(kg, "agent_discovered")
            except Exception:
                pass

        stats: Dict[str, Any] = {
            "chunks_with_frames": len(chunk_frames_map),
            "candidates_evaluated": len(candidates),
            "relations_added": total_added,
        }
        logging.info(
            "SemanticFrameBridgeDiscovery: %d agent_discovered relations added "
            "(%d candidates evaluated)",
            total_added, len(candidates),
        )
        return kg, stats

    async def _embedding_filter(
        self,
        candidates: List[Tuple[Node, Node, str]],
        chunk_frames_map: Dict[str, List[SemanticFrame]],
        embedding_model: Any,
    ) -> List[Tuple[Node, Node, str]]:
        """Keep only candidates whose frame texts are cosine-similar enough."""
        import numpy as np

        def _frame_text(node_id: str) -> str:
            return " | ".join(
                f"{f.subject} {f.operation} {f.object_} {f.condition} {f.result}"
                for f in chunk_frames_map.get(node_id, [])
            )

        unique_nodes = list({_node_id(c): c for pair in candidates for c in (pair[0], pair[1])}.values())
        texts = [_frame_text(_node_id(n)) for n in unique_nodes]
        node_idx = {_node_id(n): i for i, n in enumerate(unique_nodes)}

        if hasattr(embedding_model, "embed_documents"):
            vectors = await asyncio.get_event_loop().run_in_executor(
                None, embedding_model.embed_documents, texts
            )
        else:
            vectors = []
            for t in texts:
                v = await asyncio.get_event_loop().run_in_executor(
                    None, embedding_model.embed_query, t
                )
                vectors.append(v)

        mat = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.where(norms == 0, 1.0, norms)

        kept = []
        for ca, cb, concept in candidates:
            i, j = node_idx[_node_id(ca)], node_idx[_node_id(cb)]
            if float(mat[i] @ mat[j]) >= self.embedding_pre_filter_threshold:
                kept.append((ca, cb, concept))
        return kept


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_agentic_enrichment(
    kg: KnowledgeGraph,
    llm: Any,
    validate: bool = True,
    discover: bool = True,
    frame_bridge: bool = True,
    validator_batch_size: int = 5,
    validator_confidence_threshold: float = 0.5,
    discovery_min_confidence: float = 0.65,
    frame_bridge_min_confidence: float = 0.70,
    embedding_model: Any = None,
    on_progress: Optional[callable] = None,
    config: Any = None,
) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
    """
    Run RelationValidator, DirectRelationDiscovery, and SemanticFrameBridgeDiscovery.

    System 1 (RelationValidator):        validates existing keyphrase/cosine relations.
    System 2 (DirectRelationDiscovery):  top-down theme → chunk discovery.
    System 3 (SemanticFrameBridgeDiscovery): bottom-up frame → pair discovery.
      System 3 skips pairs already covered by System 2 to avoid redundancy.
      Both systems output agent_discovered relations with the same 10-type taxonomy.

    Args:
        kg:                          Fully enriched KG (with keyphrases).
        llm:                         LangchainLLMWrapper.
        validate:                    Run RelationValidator (default True).
        discover:                    Run DirectRelationDiscovery (default True).
        frame_bridge:                Run SemanticFrameBridgeDiscovery (default True).
        validator_batch_size:        Relations per LLM batch for validation.
        validator_confidence_threshold: Min confidence to reject a relation.
        discovery_min_confidence:    Min confidence for DirectRelationDiscovery.
        frame_bridge_min_confidence: Min confidence for SemanticFrameBridgeDiscovery.
        embedding_model:             Optional embedding model for frame pre-filter.
        on_progress:                 Callback(kg, relation_type) after each bridge.
        config:                      PipelineConfig for dynamic prompt building.

    Returns:
        (enriched_kg, {"validation": {...}, "discovery": {...}, "frame_bridge": {...}})
    """
    combined: Dict[str, Any] = {}

    if validate:
        v = RelationValidator(
            batch_size=validator_batch_size,
            confidence_threshold=validator_confidence_threshold,
            config=config,
        )
        kg, combined["validation"] = v.validate(kg, llm)

    if discover:
        d = DirectRelationDiscovery(
            min_confidence=discovery_min_confidence,
            config=config,
        )
        kg, combined["discovery"] = d.discover(kg, llm, on_progress=on_progress)

    if frame_bridge:
        fb = SemanticFrameBridgeDiscovery(
            min_confidence=frame_bridge_min_confidence,
        )
        kg, combined["frame_bridge"] = fb.discover(
            kg, llm, embedding_model=embedding_model, on_progress=on_progress
        )

    return kg, combined
