"""
Enrichissement du Knowledge Graph -- relations `llm_triplet`.

Module domain-agnostic adapte de https://github.com/rahulnyk/knowledge_graph
(`helpers/prompts.py::graphPrompt`). Il ne conserve QUE la production des
relations `llm_triplet`, la meilleure methode multi-hop mesuree (qualite ~0.95).

Pipeline en 2 temps :

1. Extraction des triplets -- pour chaque CHUNK, un appel LLM (`GraphPrompt`,
   reecriture du "network graph maker" de rahulnyk) extrait des triplets
   {node_1, node_2, edge}. Les concepts atomiques (node_1 / node_2) sont
   stockes dans `node.properties["llm_triplets"]`. Ce sont des ATTRIBUTS de
   noeuds, pas encore des aretes du graphe.

2. Construction des relations -- approche "macro" :
   - regroupe les chunks par concept-pont (concept partage) ;
   - filtres qualite : concept present dans >= 2 chunks et pas trop generique
     (ignore s'il apparait dans > `max_concept_doc_freq` des chunks) ;
   - 1 SEUL appel LLM par concept-pont : le LLM recoit la liste de tous les
     chunks partageant le concept et retourne directement les paires reellement
     reliees, avec `confidence` et une description ;
   - cree une `Relationship(type="llm_triplet")` par paire dont
     `confidence >= min_confidence`.

L'orchestrateur `enrich_kg_universal(...)` conserve la signature historique
appelee par `job_runner.py`. Seuls les triplets `llm_triplet` sont reellement
produits ; les autres flags (`add_graph_metrics`, `add_communities`,
`add_concepts`, `add_tfidf_filter`) sont acceptes mais ne font rien (no-op),
afin de rester compatible sans reintroduire les types de relations retires.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship


# ============================================================================
# HELPERS
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
    a = _node_id(r.source)
    b = _node_id(r.target)
    return (min(a, b), max(a, b), getattr(r, "type", ""))


def _existing_relation_keys(kg: KnowledgeGraph) -> set:
    return {_relation_key(r) for r in getattr(kg, "relationships", [])}


def _chunk_nodes(kg: KnowledgeGraph) -> List[Node]:
    return [
        n for n in getattr(kg, "nodes", [])
        if getattr(n, "type", None) == NodeType.CHUNK
    ]


def _best_content(node: Node, max_chars: int) -> str:
    """Return the richest available text for a node."""
    summary = node.properties.get("summary", "")
    if summary:
        return summary
    raw = (
        node.properties.get("raw_content")
        or node.properties.get("page_content", "")
    )
    return raw[:max_chars]


def _normalize_concept(concept: str) -> str:
    """Canonical form for deduplicating concepts (lowercase, collapsed ws)."""
    return " ".join(str(concept).lower().split())


# ============================================================================
# STEP 1 -- TRIPLET EXTRACTION (graphPrompt / "network graph maker")
# ============================================================================

class TripletChunkInput(BaseModel):
    chunk_id: str
    content: str


class Triplet(BaseModel):
    node_1: str = Field(description="A concept mentioned in the text (atomic, concise).")
    node_2: str = Field(description="A second, related concept mentioned in the text.")
    edge: str = Field(description="Short description of the relationship between node_1 and node_2.")


class TripletList(BaseModel):
    triplets: List[Triplet]


# Instruction reprise du network graph maker de rahulnyk/knowledge_graph.
_GRAPH_PROMPT_INSTRUCTION: str = (
    "You are a network graph maker who extracts terms and their relations from a "
    "given context. You are provided with a context chunk. Your task is to extract "
    "the ontology of terms mentioned in the given context. These terms should "
    "represent the key concepts as per the context.\n\n"
    "Thought 1: While traversing through each sentence, think about the key terms "
    "mentioned in it.\n"
    "\tTerms may include object, entity, location, organization, person, condition, "
    "acronym, documents, service, concept, etc.\n"
    "\tTerms should be as atomistic as possible.\n\n"
    "Thought 2: Think about how these terms can have one on one relation with other "
    "terms.\n"
    "\tTerms that are mentioned in the same sentence or the same paragraph are "
    "typically related to each other.\n"
    "\tTerms can be related to many other terms.\n\n"
    "Thought 3: Find out the relation between each such related pair of terms.\n\n"
    "Format your output as a list of triplets. Each triplet has:\n"
    "  node_1 -- a concept from the extracted ontology\n"
    "  node_2 -- a related concept from the extracted ontology\n"
    "  edge   -- a short description of the relationship between node_1 and node_2\n"
)

_GRAPH_PROMPT_EXAMPLES: list = [
    (
        TripletChunkInput(
            chunk_id="example",
            content=(
                "The kubelet is the primary node agent that runs on each node. It "
                "registers the node with the API server and ensures that containers "
                "described in PodSpecs are running and healthy."
            ),
        ),
        TripletList(triplets=[
            Triplet(node_1="kubelet", node_2="node",
                    edge="runs on each node as the primary agent"),
            Triplet(node_1="kubelet", node_2="API server",
                    edge="registers the node with the API server"),
            Triplet(node_1="kubelet", node_2="PodSpec",
                    edge="ensures containers described in PodSpecs are running"),
        ]),
    ),
]


class GraphPrompt(PydanticPrompt[TripletChunkInput, TripletList]):
    instruction: str = _GRAPH_PROMPT_INSTRUCTION
    input_model = TripletChunkInput
    output_model = TripletList
    examples: List[Tuple[TripletChunkInput, TripletList]] = _GRAPH_PROMPT_EXAMPLES


async def _extract_triplets_async(
    kg: KnowledgeGraph,
    llm,
    max_content_chars: int = 4000,
    max_concurrency: int = 8,
) -> Dict[str, Any]:
    prompt = GraphPrompt()
    chunks = _chunk_nodes(kg)
    todo = [n for n in chunks if not n.properties.get("llm_triplets")]

    logging.info(
        "[llm_triplet] extract: %d/%d chunks need triplet extraction",
        len(todo), len(chunks),
    )

    sem = asyncio.Semaphore(max_concurrency)
    n_ok = 0
    n_triplets = 0

    async def _one(node: Node):
        nonlocal n_ok, n_triplets
        content = _best_content(node, max_content_chars)
        if not content.strip():
            return
        async with sem:
            try:
                out: TripletList = await prompt.generate(
                    llm=llm,
                    data=TripletChunkInput(chunk_id=_node_id(node), content=content),
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("[llm_triplet] extraction failed on a chunk: %s", exc)
                return
        # Concepts = ensemble dedupplique des node_1/node_2.
        concepts: List[str] = []
        seen = set()
        for t in out.triplets:
            for concept in (t.node_1, t.node_2):
                key = _normalize_concept(concept)
                if key and key not in seen:
                    seen.add(key)
                    concepts.append(concept.strip())
        node.properties["llm_triplets"] = concepts
        n_ok += 1
        n_triplets += len(out.triplets)

    await asyncio.gather(*[_one(n) for n in todo])
    logging.info(
        "[llm_triplet] extract done: %d chunks enriched, %d triplets total",
        n_ok, n_triplets,
    )
    return {"triplet_chunks_enriched": n_ok, "triplets_extracted": n_triplets}


# ============================================================================
# STEP 2 -- MACRO BRIDGE VALIDATION -> `llm_triplet` RELATIONS
# ============================================================================

class BridgeChunk(BaseModel):
    chunk_id: str
    content: str


class BridgeInput(BaseModel):
    concept: str = Field(description="The shared concept that connects the candidate chunks.")
    chunks: List[BridgeChunk]


class BridgePair(BaseModel):
    chunk_id_a: str
    chunk_id_b: str
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = Field(description="Short description of the concrete relation between the two chunks.")


class BridgePairs(BaseModel):
    pairs: List[BridgePair]


_BRIDGE_INSTRUCTION: str = (
    "You are a knowledge graph relation builder. You are given a shared CONCEPT and "
    "a list of documentation CHUNKS that all mention this concept. Your job is to "
    "identify which PAIRS of chunks are genuinely, substantively connected through "
    "this concept -- i.e. reading both chunks together gives a reader real added "
    "understanding of the concept (definition/use, cause/effect, process/sub-process, "
    "general/specific, prerequisite/application).\n\n"
    "### Rules\n"
    "- Only return pairs with a REAL semantic dependency, not mere co-mention of the "
    "concept.\n"
    "- Compare the chunks against EACH OTHER; use the exact chunk_id values provided.\n"
    "- confidence in [0.0, 1.0]: how strong and useful the cross-reference is.\n"
    "- description: one short sentence explaining the concrete relationship.\n"
    "- Return an empty list if no pair is genuinely connected.\n"
)

_BRIDGE_EXAMPLES: list = [
    (
        BridgeInput(
            concept="EndpointSlice",
            chunks=[
                BridgeChunk(chunk_id="c1", content=(
                    "The EndpointSlice controller watches Services and Pods and writes "
                    "EndpointSlice objects that list ready pod IPs and ports.")),
                BridgeChunk(chunk_id="c2", content=(
                    "kube-proxy reads EndpointSlice objects to program iptables rules "
                    "that load-balance Service traffic across ready endpoints.")),
                BridgeChunk(chunk_id="c3", content=(
                    "A blog post mentions EndpointSlice was introduced to replace the "
                    "older Endpoints API for scalability reasons.")),
            ],
        ),
        BridgePairs(pairs=[
            BridgePair(
                chunk_id_a="c1", chunk_id_b="c2", confidence=0.92,
                description="c1 produces EndpointSlice objects that c2 consumes to route Service traffic.",
            ),
        ]),
    ),
]


class BridgePrompt(PydanticPrompt[BridgeInput, BridgePairs]):
    instruction: str = _BRIDGE_INSTRUCTION
    input_model = BridgeInput
    output_model = BridgePairs
    examples: List[Tuple[BridgeInput, BridgePairs]] = _BRIDGE_EXAMPLES


async def _build_relations_async(
    kg: KnowledgeGraph,
    llm,
    min_confidence: float = 0.6,
    max_concept_doc_freq: float = 0.5,
    require_cross_doc: bool = False,
    max_content_chars: int = 2000,
    max_chunks_per_concept: int = 12,
    max_concurrency: int = 6,
) -> Dict[str, Any]:
    prompt = BridgePrompt()
    chunks = _chunk_nodes(kg)
    n_chunks = len(chunks)
    if n_chunks < 2:
        return {"llm_triplet_relations_added": 0, "concept_bridges": 0}

    id_to_node: Dict[str, Node] = {_node_id(n): n for n in chunks}

    # concept (canonique) -> set d'ids de chunks + libelle d'affichage
    concept_chunks: Dict[str, set] = defaultdict(set)
    concept_label: Dict[str, str] = {}
    for node in chunks:
        for concept in node.properties.get("llm_triplets", []) or []:
            key = _normalize_concept(concept)
            if not key:
                continue
            concept_chunks[key].add(_node_id(node))
            concept_label.setdefault(key, concept)

    # Filtres qualite : >= 2 chunks et pas trop generique.
    freq_ceiling = max(2, int(max_concept_doc_freq * n_chunks))
    bridges = {
        k: ids for k, ids in concept_chunks.items()
        if 2 <= len(ids) <= freq_ceiling
    }
    logging.info(
        "[llm_triplet] %d concept bridges after filtering "
        "(from %d concepts, freq_ceiling=%d)",
        len(bridges), len(concept_chunks), freq_ceiling,
    )

    existing = _existing_relation_keys(kg)
    sem = asyncio.Semaphore(max_concurrency)
    added = 0
    lock = asyncio.Lock()

    async def _one(concept_key: str, ids: set):
        nonlocal added
        node_ids = sorted(ids)[:max_chunks_per_concept]
        bridge_chunks = [
            BridgeChunk(
                chunk_id=nid,
                content=_best_content(id_to_node[nid], max_content_chars),
            )
            for nid in node_ids if nid in id_to_node
        ]
        if len(bridge_chunks) < 2:
            return
        async with sem:
            try:
                out: BridgePairs = await prompt.generate(
                    llm=llm,
                    data=BridgeInput(
                        concept=concept_label.get(concept_key, concept_key),
                        chunks=bridge_chunks,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("[llm_triplet] bridge validation failed: %s", exc)
                return

        for pair in out.pairs:
            if pair.confidence < min_confidence:
                continue
            a = id_to_node.get(pair.chunk_id_a)
            b = id_to_node.get(pair.chunk_id_b)
            if a is None or b is None or a is b:
                continue
            if require_cross_doc and _doc_id(a) == _doc_id(b):
                continue
            rk = (min(_node_id(a), _node_id(b)),
                  max(_node_id(a), _node_id(b)), "llm_triplet")
            async with lock:
                if rk in existing:
                    continue
                existing.add(rk)
                kg.relationships.append(
                    Relationship(
                        source=a,
                        target=b,
                        type="llm_triplet",
                        properties={
                            "shared_concept": concept_label.get(concept_key, concept_key),
                            "confidence": pair.confidence,
                            "description": pair.description,
                            "discovery_method": "llm_triplet_macro_bridge",
                        },
                    )
                )
                added += 1

    await asyncio.gather(*[_one(k, ids) for k, ids in bridges.items()])
    logging.info(
        "[llm_triplet] %d 'llm_triplet' relations added (%d bridges)",
        added, len(bridges),
    )
    return {"llm_triplet_relations_added": added, "concept_bridges": len(bridges)}


# ============================================================================
# ORCHESTRATOR (signature historique, compatible job_runner.py)
# ============================================================================

def enrich_kg_universal(
    kg: KnowledgeGraph,
    llm=None,
    add_proximity: bool = False,          # no-op (type retire)
    add_graph_metrics: bool = False,      # no-op (type retire)
    add_communities: bool = False,        # no-op (type retire)
    add_triplets: bool = False,           # extraction des triplets (attributs)
    add_concepts: bool = False,           # no-op (type retire)
    add_triplet_relations: bool = False,  # cree les relations llm_triplet
    add_tfidf_filter: bool = False,       # no-op
    triplet_min_confidence: float = 0.6,
    triplet_max_concept_doc_freq: float = 0.5,
    triplet_require_cross_doc: bool = False,
    **_ignored,
) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
    """Enrichit `kg` avec les relations `llm_triplet` (seul type conserve).

    Les flags `add_proximity`, `add_graph_metrics`, `add_communities`,
    `add_concepts`, `add_tfidf_filter` sont acceptes pour compatibilite mais
    n'ont plus d'effet (les types de relations correspondants ont ete retires).
    """
    stats: Dict[str, Any] = {}

    # `add_triplet_relations` implique l'extraction prealable des triplets.
    need_triplets = add_triplets or add_triplet_relations
    if need_triplets and llm is None:
        logging.warning(
            "[llm_triplet] llm=None -- extraction/relations impossibles, on saute.")
        return kg, stats

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:  # noqa: BLE001
        pass
    loop = asyncio.get_event_loop()

    if need_triplets:
        stats.update(loop.run_until_complete(_extract_triplets_async(kg, llm)))

    if add_triplet_relations:
        stats.update(loop.run_until_complete(
            _build_relations_async(
                kg,
                llm,
                min_confidence=triplet_min_confidence,
                max_concept_doc_freq=triplet_max_concept_doc_freq,
                require_cross_doc=triplet_require_cross_doc,
            )
        ))

    return kg, stats


# Alias explicite pour l'usage documente (test/debug).
def enrich_kg_with_triplets(
    kg: KnowledgeGraph,
    llm,
    min_confidence: float = 0.6,
    max_concept_doc_freq: float = 0.5,
    require_cross_doc: bool = False,
) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
    """Raccourci : extraction des triplets puis creation des relations llm_triplet."""
    return enrich_kg_universal(
        kg,
        llm=llm,
        add_triplets=True,
        add_triplet_relations=True,
        triplet_min_confidence=min_confidence,
        triplet_max_concept_doc_freq=max_concept_doc_freq,
        triplet_require_cross_doc=require_cross_doc,
    )
