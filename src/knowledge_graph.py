from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Tokenizer hors-ligne (ragas >= 0.2 n'a plus de ragas.tokenizers) ───────
class _WordTokenizer:
    """Tokenizer hors-ligne : encode=split mots, decode=join mots."""

    def encode(self, text: str):
        return text.split()  # retourne les mots (pas des indices)

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)  # rejoint les mots
# ── Fin tokenizer ───────────────────────────────────────────────────────────

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms import (
    OverlapScoreBuilder,
    apply_transforms,
    default_transforms,
)
from ragas.testset.transforms.extractors.llm_based import (
    Keyphrases,
    KeyphrasesExtractor,
    KeyphrasesExtractorPrompt,
    TextWithExtractionLimit,
)


# ---------------------------------------------------------------------------
# filter_chunks : filtre passé à KeyphrasesExtractor pour ne traiter que les
# nœuds de type CHUNK (et non les nœuds DOCUMENT).
# ---------------------------------------------------------------------------
def filter_chunks(node: "Node") -> bool:
    """Return True if the node is a CHUNK node."""
    return getattr(node, "type", None) == NodeType.CHUNK


# ---------------------------------------------------------------------------
# to_jsonable : convertit récursivement une valeur en type JSON-sérialisable.
# ---------------------------------------------------------------------------
def to_jsonable(value: Any) -> Any:
    """Recursively convert a value to a JSON-serialisable type."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    # Pydantic / dataclass / enum
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    if hasattr(value, "value"):  # enum
        return to_jsonable(value.value)
    return str(value)


# ---------------------------------------------------------------------------
# Vérification verbatim tolérante (whitespace + tirets)
# ---------------------------------------------------------------------------
def _normalize_ws(s: str) -> str:
    """Collapse all whitespace runs and lowercase."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _appears_verbatim(kp: str, raw: str) -> bool:
    """Whitespace- and hyphen-tolerant 'verbatim' check."""
    kp_n, raw_n = _normalize_ws(kp), _normalize_ws(raw)
    if kp_n in raw_n:
        return True
    # tolerate "leading-edge radius" vs "leading edge radius"
    return re.sub(r"[\s\-]+", " ", kp_n) in re.sub(r"[\s\-]+", " ", raw_n)


# ============================================================================
# POST-FILTER -- keyphrases g?n?riques bannies
# ============================================================================

_BANNED: set = {
    "method",
    "function",
    "class",
    "instance",
    "attribute",
    "parameter",
    "argument",
    "value",
    "default",
    "setup",
    "compute",
    "return",
    "object",
    "system",
    "type",
    "name",
    "data",
    "code",
    "input",
    "output",
    "example",
}

_BANNED_PHRASES: set = {
    "compute method",
    "setup method",
    "return value",
    "default value",
    "input parameter",
    "output attribute",
    "type attribute",
    "name attribute",
    "system object",
    "child system",
    "parent system",
    "child systems",
    "parent systems",
    "list of dictionaries",
    "parameter named value",
    "function argument",
    "class instance",
    "code example",
}


def _is_banned(kp: str) -> bool:
    """
    Retourne True si la keyphrase est g?n?rique et doit ?tre filtr?e.

    Crit?res :
      1. Mot unique dans _BANNED.
      2. Expression exacte dans _BANNED_PHRASES.
      3. Expression compos?e dont TOUS les tokens sont dans _BANNED.
    """
    kp_lower = kp.lower().strip()
    if kp_lower in _BANNED:
        return True
    if kp_lower in _BANNED_PHRASES:
        return True
    tokens = re.split(r"[\s\-_]+", kp_lower)
    tokens = [t for t in tokens if t]
    if tokens and all(t in _BANNED for t in tokens):
        return True
    return False


def _apply_post_filter(chunk_nodes: list, config=None) -> None:
    """Filtre en place les keyphrases génériques sur les chunk nodes.

    Merge les blacklists statiques (_BANNED / _BANNED_PHRASES) avec les listes
    configurées dans PipelineConfig.kg_enrichment (blacklist_words, blacklist_phrases).

    NOTE: le filtre _appears_verbatim a été supprimé car le LLM génère des
    keyphrases légitimes qui ne sont pas présentes verbatim dans raw_content
    (ex: markdown gras **setup** vs keyphrase "setup method").
    """
    kge = config.kg_enrichment if config is not None else None
    extra_words   = {w.lower() for w in (kge.blacklist_words   if kge else [])}
    extra_phrases = {p.lower() for p in (kge.blacklist_phrases if kge else [])}

    def _is_banned_extended(kp: str) -> bool:
        kp_lower = kp.lower().strip()
        if kp_lower in extra_words or kp_lower in extra_phrases:
            return True
        return _is_banned(kp)

    for node in chunk_nodes:
        kps = node.properties.get("keyphrases", [])
        cleaned = [
            kp
            for kp in kps
            if not _is_banned_extended(kp)
            and len(kp.strip()) >= 3
            and not kp.strip().isdigit()
        ]
        node.properties["keyphrases"] = cleaned


# ============================================================================
# BRIDGE-TERMS PROMPT
# Sous-classe de KeyphrasesExtractorPrompt -- manière idiomatique Ragas :
# on surcharge UNIQUEMENT `instruction` et `examples` comme attributs de classe.
# `input_model` (TextWithExtractionLimit) et `output_model` (Keyphrases) sont
# hérités tels quels de Ragas. On passe ensuite une instance via :
#   KeyphrasesExtractor(prompt=BridgeKeyphrasesExtractorPrompt())
# ============================================================================


class BridgeKeyphrasesExtractorPrompt(KeyphrasesExtractorPrompt):
    """Prompt KeyphrasesExtractor **neutre en domaine** (fallback).

    Hérite de ``KeyphrasesExtractorPrompt`` (Ragas) et ne redéfinit que
    l'instruction et les exemples few-shot. Le modèle d'entrée
    (``TextWithExtractionLimit`` : ``text`` + ``max_num``) et le modèle de
    sortie (``Keyphrases``) restent ceux de Ragas, ce qui garantit la
    cohérence entre les exemples et la requête réelle envoyée au LLM.

    ⚠️ Ce prompt ne sert QUE de repli lorsque la session ne fournit pas
    ``prompts.keyphrase_extractor``. Il doit donc rester strictement neutre :
    aucune mention d'un framework, d'un corpus ou d'un vocabulaire métier
    particulier — sinon un run sur un autre domaine serait biaisé.
    Les exemples ci-dessous utilisent volontairement des identifiants
    fictifs et non signifiants pour illustrer la FORME attendue (identifiant
    composé, terme multi-mots, valeur qualifiée) sans injecter de sémantique.
    """

    instruction: str = (
        "Extract the most discriminant keyphrases from the given text. "
        "The text is an excerpt of technical or reference documentation; do not "
        "assume any particular subject matter — infer what matters from the text "
        "itself.\n\n"
        "### What to extract (in priority order)\n"
        "1. **Proper names and identifiers**: named entities, code identifiers "
        "(CamelCase, snake_case, dotted paths), reference codes, section or "
        "clause labels.\n"
        "2. **Composite domain terms**: multi-word expressions that denote a "
        "single specific concept in this text.\n"
        "3. **Qualified values and constants**: enumerated values, named "
        "thresholds, units-bearing quantities, defined levels or categories.\n"
        "4. **Defined terms**: any expression the text explicitly introduces, "
        "defines, or capitalises as a term of art.\n\n"
        "### What to BAN (never extract these)\n"
        "Generic structural or meta words: method, function, class, instance, "
        "attribute, parameter, argument, variable, value, default, object, type, "
        "name, list, table, figure, section, chapter, overview, introduction, "
        "example, note, see also, description.\n"
        "Generic English stopwords and connectives: the, and, with, from, into, "
        "this, that, these, those, data, code, file, input, output.\n"
        "Single-letter or 2-character tokens. Pure numbers. Common verbs.\n\n"
        "### Rules\n"
        "- Each keyphrase must appear VERBATIM in the source text (case-insensitive).\n"
        "- Prefer specific over general: keep the fully qualified term rather "
        "than its generic head word (e.g. the complete multi-word term rather "
        "than its last word alone).\n"
        "- Keep original casing for identifiers (CamelCase / snake_case preserved).\n"
        "- Multi-word terms are encouraged when they form a single concept.\n"
        "- Return at most 10 keyphrases, sorted by discriminative power "
        "(rarest / most specific first).\n"
        "- If the text contains no substantive content, return an empty list.\n"
    )

    # Exemples few-shot : List[ Tuple[ TextWithExtractionLimit, Keyphrases ] ].
    # On utilise bien le modèle d'entrée réel de Ragas (text + max_num) afin que
    # le schéma des exemples corresponde à celui de la requête d'inférence.
    # Les contenus sont *délibérément abstraits* : ils enseignent la forme
    # (identifiant / terme composé / valeur qualifiée) sans ancrer un domaine.
    examples: List[Tuple[TextWithExtractionLimit, Keyphrases]] = [
        (
            TextWithExtractionLimit(
                text=(
                    "AlphaUnitController is an element that coordinates the primary "
                    "regulation loop for grouped assemblies. A NodeBalancer child "
                    "named 'balancer' is attached, exposing the ports stream_in, "
                    "stream_out, and run_case at the parent level. For each "
                    "(unit_name, stage_index) pair, the parent's rate_{stage_index} "
                    "value is bound to the balancer's {unit_name}_rate value."
                ),
                max_num=10,
            ),
            Keyphrases(
                keyphrases=[
                    "AlphaUnitController",
                    "NodeBalancer",
                    "primary regulation loop",
                    "grouped assemblies",
                    "stream_in",
                    "stream_out",
                    "run_case",
                    "rate_{stage_index}",
                    "stage_index",
                    "unit_name",
                ]
            ),
        ),
        (
            TextWithExtractionLimit(
                text=(
                    "The term Qualified Configuration designates a configuration that "
                    "has passed the Level B acceptance criteria defined in clause "
                    "4.2.1. Each Qualified Configuration must record a nominal "
                    "tolerance band of 0.5 mm and reference either Mode.Continuous "
                    "or Mode.Discrete, depending on whether the governing element "
                    "is declared adaptive."
                ),
                max_num=10,
            ),
            Keyphrases(
                keyphrases=[
                    "Qualified Configuration",
                    "Level B acceptance criteria",
                    "clause 4.2.1",
                    "nominal tolerance band",
                    "0.5 mm",
                    "Mode.Continuous",
                    "Mode.Discrete",
                    "governing element",
                    "adaptive",
                ]
            ),
        ),
    ]



# ============================================================================
# MARKDOWN PARSING & EXTRACTION (rule-based, no LLM, no cost)
# ============================================================================


class MDMetadataExtractor:
    """Rule-based extraction of structural elements from Markdown."""

    @staticmethod
    def extract_headings(content: str) -> List[Tuple[int, str]]:
        """Hierarchical headings as (level, text) tuples."""
        pattern = r"^(#{1,6})\s+(.+)$"
        matches = re.finditer(pattern, content, re.MULTILINE)
        return [(len(m.group(1)), m.group(2)) for m in matches]

    @staticmethod
    def extract_links(content: str) -> List[Dict]:
        """Markdown links [text](url)."""
        pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        matches = re.finditer(pattern, content)
        return [{"text": m.group(1), "url": m.group(2)} for m in matches]

    @staticmethod
    def extract_code_blocks(content: str) -> List[Dict]:
        """Fenced code blocks with optional language tag."""
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)
        return [
            {"language": m.group(1) or "text", "code": m.group(2).strip()}
            for m in matches
        ]

    @staticmethod
    def extract_inline_code(content: str) -> List[str]:
        """Inline code spans (backticks)."""
        pattern = r"`([^`]+)`"
        return re.findall(pattern, content)

    @staticmethod
    def extract_tables(content: str) -> List[List[List[str]]]:
        """
        Markdown tables.
        Verifies: line starts and ends with `|`, and is not a separator row.
        """
        lines = content.split("\n")
        tables: List[List[List[str]]] = []
        current_table: List[List[str]] = []

        for line in lines:
            if line.strip().startswith("|") and line.strip().endswith("|"):
                cells = [cell.strip() for cell in line.split("|")[1:-1]]
                # Skip separator rows like |---|---| or |:---|---:|
                if cells and not all(
                    set(cell.replace("-", "").replace(":", "").strip()) == set()
                    for cell in cells
                ):
                    current_table.append(cells)
            else:
                if current_table:
                    tables.append(current_table)
                    current_table = []

        if current_table:
            tables.append(current_table)

        return tables

    @staticmethod
    def extract_entities_rule_based(content: str) -> List[Dict]:
        """
        Named-entity extraction via regex:
          - emails, URLs, code spans, versions, metrics, file paths.
        """
        entities: List[Dict] = []

        # Emails
        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", content
        )
        entities.extend([{"type": "EMAIL", "value": e} for e in emails])

        # URLs
        urls = re.findall(r"https?://[^\s]+", content)
        entities.extend([{"type": "URL", "value": u} for u in urls])

        # Code references (backticks)
        codes = re.findall(r"`([^`]+)`", content)
        entities.extend([{"type": "CODE", "value": c} for c in codes])

        # Versions (x.y.z)
        versions = re.findall(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", content)
        entities.extend([{"type": "VERSION", "value": v} for v in versions])

        # Numbers with units
        numbers = re.findall(
            r"\b(\d+(?:\.\d+)?)\s*(GB|MB|KB|ms|s|%|?|\$|Hz|GHz)?\b", content
        )
        entities.extend(
            [
                {"type": "METRIC", "value": f"{n[0]}{n[1] if n[1] else ''}"}
                for n in numbers
                if n[0]
            ]
        )

        # File paths
        filepaths = re.findall(r"\./[^\s]+|[a-zA-Z]:\\[^\s]+", content)
        entities.extend([{"type": "FILE_PATH", "value": f} for f in filepaths])

        return entities


class MarkdownChunker:
    """
    Heading-aware chunker that respects fenced code blocks.
    Aggregates content by section (heading + body) instead of arbitrary cuts.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        overlap_ratio: float = 0.1,
        min_chunk_tokens: int = 50,
    ):
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.min_chunk_tokens = min_chunk_tokens

    def _is_in_code_block(self, lines: List[str], line_idx: int) -> bool:
        """Return True if `lines[line_idx]` lies inside a fenced ``` block."""
        in_block = False
        for line in lines[:line_idx]:
            if line.strip().startswith("```"):
                in_block = not in_block
        return in_block

    def chunk_by_sections(self, content: str) -> List[Dict]:
        """
        Split content into sections based on Markdown headings.
        Headings inside fenced code blocks are ignored.
        Sections longer than `max_tokens` are sub-divided with overlap.

        Each returned dict contains:
            title, content, level, breadcrumb
        where `breadcrumb` is the hierarchical path built from a heading stack
        (e.g. "Module Overview > Installation > Dependencies").
        """
        sections: List[Dict] = []
        # Stack maintains the heading hierarchy: [(level, title), ...]
        heading_stack: List[Tuple[int, str]] = []
        current = {
            "title": "Introduction",
            "content": "",
            "level": 0,
            "breadcrumb": "Introduction",
        }
        lines = content.split("\n")

        for idx, line in enumerate(lines):
            if line.startswith("#") and not self._is_in_code_block(lines, idx):
                level = len(line) - len(line.lstrip("#"))
                title = line.lstrip("#").strip()

                if current["content"].strip():
                    sections.append(current)

                # Update heading stack: pop all levels >= current level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

                breadcrumb = " > ".join(t for _, t in heading_stack)
                current = {
                    "title": title,
                    "content": "",
                    "level": level,
                    "breadcrumb": breadcrumb,
                }
            else:
                current["content"] += line + "\n"

        if current["content"].strip():
            sections.append(current)

        # ── Merge consecutive small sections up to max_tokens ─────────────────
        merged: List[Dict] = []
        current_parts: List[Dict] = []
        current_tokens = 0

        for section in sections:
            sec_tokens = len(section["content"].split())
            if sec_tokens > self.max_tokens:
                # Flush pending parts first
                if current_parts:
                    merged.append(self._merge_parts(current_parts))
                    current_parts, current_tokens = [], 0
                # Oversized section: subdivide with overlap
                merged.extend(self._split_long_section(section))
            elif current_tokens + sec_tokens > self.max_tokens:
                # Adding this section would exceed the limit: flush, then start fresh
                merged.append(self._merge_parts(current_parts))
                current_parts = [section]
                current_tokens = sec_tokens
            else:
                current_parts.append(section)
                current_tokens += sec_tokens

        if current_parts:
            merged.append(self._merge_parts(current_parts))

        return merged

    def _merge_parts(self, parts: List[Dict]) -> Dict:
        """Merge a list of consecutive sections into a single chunk dict.

        If there is only one part, it is returned as-is (no copy needed).
        The merged chunk keeps the title and breadcrumb of the *first* part,
        and joins all contents with a blank line between them.
        """
        if len(parts) == 1:
            return parts[0]
        combined_content = "\n\n".join(p["content"] for p in parts)
        titles = [p["title"] for p in parts]
        return {
            "title": parts[0]["title"],
            "content": combined_content,
            "level": parts[0]["level"],
            "breadcrumb": parts[0]["breadcrumb"],
            "merged_titles": titles,
            "is_merged": True,
        }

    def _split_long_section(self, section: Dict) -> List[Dict]:
        """Split a section longer than `max_tokens` with overlap between pieces.

        All subdivisions inherit the same `breadcrumb` as their parent section.
        """
        tokens = section["content"].split()
        overlap = int(self.max_tokens * self.overlap_ratio)
        chunks: List[Dict] = []
        breadcrumb = section.get("breadcrumb", section["title"])

        for i in range(0, len(tokens), self.max_tokens - overlap):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunks.append(
                {
                    "title": section["title"],
                    "content": " ".join(chunk_tokens),
                    "level": section["level"],
                    "is_subdivision": True,
                    "breadcrumb": breadcrumb,
                }
            )

        return chunks


# ============================================================================
# KNOWLEDGE GRAPH BUILDER (merged)
# ============================================================================


class KnowledgeGraphBuilder:
    """
    KG builder with two creation paths:
      - create_from_documents_with_chunking(...)  -- from in-memory dicts {content, metadata}
      - create_from_markdown_files(...)           -- from .md files, with rule-based parsing

    And three enrichment paths:
      - enrich_lightweight(...)          -- no LLM call
      - enrich_prechunked(...)           -- Keyphrases LLM + Overlap + TF-IDF cross-doc
      - enrich_prechunked_official(...)  -- Ragas default_transforms_for_prechunked
    """

    def __init__(
        self,
        llm: Any = None,
        embedding_model: Any = None,
        use_rule_based: bool = True,
        min_tokens: int = 300,
        max_tokens: int = 4096,
        max_keyphrases: int = 10,
        llm_threshold: int = 500,
        min_chunk_tokens: int = 50,
        config: Any = None,
        inter_doc_only: bool = True,
    ):
        """
        Args:
            llm:               Language model (optional, only required for
                               LLM-based enrichment paths).
            embedding_model:   Embedding model (optional).
            use_rule_based:    Use rule-based extraction in enrich_lightweight.
            min_tokens:        Minimum tokens per chunk (LLM HeadlineSplitter).
            max_tokens:        Maximum tokens per chunk.
            max_keyphrases:    Maximum number of keyphrases to extract.
            llm_threshold:     Token-count threshold above which LLM extraction
                               kicks in (reserved for future hybrid mode).
            min_chunk_tokens:  Minimum tokens to keep a chunk (smaller chunks
                               are dropped during MD parsing).
            config:            Optional PipelineConfig — when provided, all
                               domain-specific thresholds and prompts are read
                               from it instead of using the hardcoded defaults.
            inter_doc_only:    If True (default), only create semantic relations
                               between chunks from DIFFERENT documents. If False,
                               also create intra-document semantic relations
                               (allows multi-hop within a single document).
        """
        self._config = config
        self._inter_doc_only = inter_doc_only
        kge = config.kg_enrichment if config else None
        ch  = config.chunking      if config else None

        self.llm = llm
        self.embedding_model = embedding_model
        self.use_rule_based = use_rule_based
        self.min_tokens = min_tokens
        self.max_tokens     = ch.max_tokens       if ch  else max_tokens
        self.max_keyphrases = kge.max_keyphrases  if kge else max_keyphrases
        self.llm_threshold  = llm_threshold
        self.min_chunk_tokens = ch.min_chunk_tokens if ch else min_chunk_tokens
        self.md_extractor = MDMetadataExtractor()
        self.md_chunker = MarkdownChunker(
            max_tokens=self.max_tokens,
            overlap_ratio=ch.overlap_ratio if ch else 0.1,
            min_chunk_tokens=self.min_chunk_tokens,
        )
        logging.info("KnowledgeGraphBuilder initialized")

    # ????????????????????????????????????????????????????????????????????????
    # Creation -- helpers
    # ????????????????????????????????????????????????????????????????????????

    def _create_document_node(self, doc: Dict[str, Any], kg: KnowledgeGraph) -> Node:
        """
        Create a single DOCUMENT node from an in-memory dict and append it to `kg`.

        Returns the created Node (useful for linking CHUNK nodes afterwards).
        """
        metadata = doc.get("metadata", {}) or {}
        filename = metadata.get("filename", "unknown")
        content = doc.get("content", "") or ""

        headings = self.md_extractor.extract_headings(content)
        links = self.md_extractor.extract_links(content)
        code_blocks = self.md_extractor.extract_code_blocks(content)

        doc_node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": content,
                "document_metadata": metadata,
                "filename": filename,
                "headings": headings,
                "links": links,
                "has_code": len(code_blocks) > 0,
                "code_block_count": len(code_blocks),
            },
        )
        kg.nodes.append(doc_node)
        return doc_node

    # ????????????????????????????????????????????????????????????????????????
    # Creation -- public API
    # ????????????????????????????????????????????????????????????????????????

    def create_from_documents_with_chunking(
        self, documents: List[Dict[str, Any]]
    ) -> KnowledgeGraph:
        """
        Create a KG from in-memory documents WITH rule-based chunking.

        For each document:
          1. Creates a DOCUMENT node (via _create_document_node).
          2. Runs MarkdownChunker.chunk_by_sections on the content.
          3. Creates CHUNK nodes with breadcrumb, raw_content, etc.
          4. Adds `contains` relationships (DOCUMENT -> CHUNK).

        This ensures rule-based chunking is used in the batch pipeline
        instead of the LLM-based HeadlineSplitter.
        """
        logging.info("Creating knowledge graph from documents (with chunking)...")
        kg = KnowledgeGraph()

        for doc in documents:
            metadata = doc.get("metadata", {}) or {}
            filename = metadata.get("filename", "unknown")
            content = doc.get("content", "") or ""
            logging.info(f"Adding document + chunks: {filename}")

            doc_node = self._create_document_node(doc, kg)

            # Rule-based heading-aware chunking
            sections = self.md_chunker.chunk_by_sections(content)

            for idx, section in enumerate(sections):
                chunk_content = section["content"]
                token_count = len(chunk_content.split())

                if token_count < self.min_chunk_tokens:
                    logging.debug(
                        f"Skipping small chunk: {section['title']} "
                        f"({token_count} tokens < {self.min_chunk_tokens})"
                    )
                    continue

                chunk_node = Node(
                    type=NodeType.CHUNK,
                    properties={
                        "page_content": chunk_content,
                        "raw_content": chunk_content,
                        "section_title": section["title"],
                        "section_level": section["level"],
                        "breadcrumb": section.get("breadcrumb", section["title"]),
                        "parent_doc": filename,
                        "chunk_index": idx,
                        "is_subdivision": section.get("is_subdivision", False),
                        "token_count": token_count,
                    },
                )
                kg.nodes.append(chunk_node)

                kg.relationships.append(
                    Relationship(
                        source=doc_node,
                        target=chunk_node,
                        type="contains",
                        properties={"order": idx},
                    )
                )

            logging.info(f"  [OK] {filename}: {len(sections)} sections chunked")

        n_docs = sum(1 for n in kg.nodes if n.type == NodeType.DOCUMENT)
        n_chunks = sum(1 for n in kg.nodes if n.type == NodeType.CHUNK)
        logging.info(
            f"Created KG: {n_docs} DOCUMENT + {n_chunks} CHUNK nodes, "
            f"{len(kg.relationships)} relationships"
        )
        return kg

    def create_from_markdown_files(self, md_files: List[Path]) -> KnowledgeGraph:
        """
        Read a list of .md files, extract structural metadata, and build a KG
        with both DOCUMENT nodes and pre-chunked CHUNK nodes (with `contains`
        relationships from doc to chunk).
        """
        logging.info(f"Creating knowledge graph from {len(md_files)} markdown files...")
        kg = KnowledgeGraph()

        for filepath in md_files:
            try:
                content = filepath.read_text(encoding="utf-8")
                logging.info(f"Processing: {filepath.name}")

                # Free rule-based extraction
                headings = self.md_extractor.extract_headings(content)
                links = self.md_extractor.extract_links(content)
                code_blocks = self.md_extractor.extract_code_blocks(content)
                inline_codes = self.md_extractor.extract_inline_code(content)
                tables = self.md_extractor.extract_tables(content)
                doc_node = Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": content,
                        "filename": filepath.name,
                        "file_path": str(filepath),
                        "headings": headings,
                        "links": links,
                        "has_code": len(code_blocks) > 0,
                        "code_block_count": len(code_blocks),
                        "inline_code_count": len(inline_codes),
                        "table_count": len(tables),
                        "extracted_at": datetime.now().isoformat(),
                    },
                )
                kg.nodes.append(doc_node)

                # Heading-aware chunks
                sections = self.md_chunker.chunk_by_sections(content)

                for idx, section in enumerate(sections):
                    chunk_content = section["content"]
                    token_count = len(chunk_content.split())

                    if token_count < self.min_chunk_tokens:
                        logging.debug(
                            f"Skipping small chunk: {section['title']} "
                            f"({token_count} tokens < {self.min_chunk_tokens})"
                        )
                        continue

                    chunk_node = Node(
                        type=NodeType.CHUNK,
                        properties={
                            "page_content": chunk_content,
                            "raw_content": chunk_content,
                            "section_title": section["title"],
                            "section_level": section["level"],
                            "breadcrumb": section.get("breadcrumb", section["title"]),
                            "parent_doc": filepath.name,
                            "chunk_index": idx,
                            "is_subdivision": section.get("is_subdivision", False),
                            "token_count": token_count,
                        },
                    )
                    kg.nodes.append(chunk_node)

                    kg.relationships.append(
                        Relationship(
                            source=doc_node,
                            target=chunk_node,
                            type="contains",
                            properties={"order": idx},
                        )
                    )

                logging.info(
                    f"  [OK] {filepath.name}: "
                    f"{len(headings)} headings, "
                    f"{len(sections)} sections, "
                    f"{len(code_blocks)} code blocks"
                )

            except Exception as e:
                logging.error(f"Error processing {filepath.name}: {e}")
                continue

        logging.info(
            f"Created KG: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships"
        )
        return kg

    # ????????????????????????????????????????????????????????????????????????
    # Enrichment -- rule-based (no LLM)
    # ????????????????????????????????????????????????????????????????????????

    def enrich_lightweight(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Lightweight pass: counts CHUNK nodes and logs stats.
        Keyword extraction is now handled exclusively by the LLM
        (see enrich_prechunked / _extract_keyphrases_llm_async).
        """
        logging.info("Lightweight pass: counting chunks (no rule-based keyphrases)...")
        chunk_count = sum(
            1
            for n in getattr(kg, "nodes", [])
            if getattr(n, "type", None) == NodeType.CHUNK
        )
        logging.info(f"Lightweight pass complete: {chunk_count} CHUNK nodes found.")
        return kg

    # ????????????????????????????????????????????????????????????????????????
    # Enrichment -- Ragas official (Summary + Embedding + Themes + NER + ...)
    # ????????????????????????????????????????????????????????????????????????

    def enrich_prechunked_official(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Ragas-official prechunked pipeline:
          1. SummaryExtractor
          2. CustomNodeFilter
          3. Parallel(EmbeddingExtractor from summary, ThemesExtractor, NERExtractor)
          4. Parallel(CosineSimilarityBuilder on summary_embedding, OverlapScoreBuilder)

        Falls back to enrich_lightweight if llm or embedding_model is missing.
        """
        if self.llm is None or self.embedding_model is None:
            logging.warning(
                "enrich_prechunked_official requires both llm and embedding_model. "
                "Falling back to lightweight enrichment."
            )
            return self.enrich_lightweight(kg)

        logging.info("Starting PRE-CHUNKED enrichment (Ragas official pipeline)...")
        logging.info(
            f"LLM: {type(self.llm).__name__}, "
            f"Embedding Model: {type(self.embedding_model).__name__}"
        )

        try:
            transforms = default_transforms(
                llm=self.llm,
                embedding_model=self.embedding_model,
            )
            apply_transforms(kg, transforms=transforms)
            logging.info(
                "[OK] Pre-chunked enrichment (official default_transforms) completed successfully"
            )
            return kg
        except Exception as e:
            logging.exception(f"Error in official pre-chunked enrichment: {e}")
            logging.warning("Falling back to lightweight enrichment...")
            return self.enrich_lightweight(kg)

    # ????????????????????????????????????????????????????????????????????????
    # Enrichment -- custom prechunked (NER + Keyphrases + Overlap + TF-IDF)
    # ????????????????????????????????????????????????????????????????????????

    def _hydrate_keyphrases_from_store(
        self, kg: KnowledgeGraph, store_dir: Path
    ) -> Set[str]:
        """
        Pre-fill CHUNK keyphrases from a previously saved per-document store.

        For each DOCUMENT node whose content hash matches the `source_hash` stored
        in `store_dir/<stem>.json`, copy the cached keyphrases onto its CHUNK nodes
        (matched by chunk_index). Returns the set of document filenames that were
        hydrated from cache (so the LLM extraction can skip them).
        """
        import hashlib

        store_dir = Path(store_dir)
        if not store_dir.exists():
            return set()

        # Compute current content hash per document filename.
        doc_hash: Dict[str, str] = {}
        for n in getattr(kg, "nodes", []):
            if getattr(n, "type", None) != NodeType.DOCUMENT:
                continue
            props = getattr(n, "properties", {}) or {}
            content = props.get("page_content", "") or ""
            doc_hash[props.get("filename", "unknown")] = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()

        cached: Set[str] = set()
        kp_by_doc: Dict[str, Dict[int, list]] = {}
        for filename, cur_hash in doc_hash.items():
            store_path = store_dir / f"{Path(filename).stem}.json"
            if not store_path.exists():
                continue
            try:
                rec = json.loads(store_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logging.warning("doc store: cannot read %s (%s)", store_path, exc)
                continue
            if rec.get("source_hash") != cur_hash:
                continue  # document changed -> recompute keyphrases
            kp_by_doc[filename] = {
                c.get("chunk_index"): (c.get("keyphrases") or [])
                for c in rec.get("chunks", [])
            }
            cached.add(filename)

        if not cached:
            return set()

        # Copy cached keyphrases onto the matching CHUNK nodes.
        hydrated = 0
        for n in getattr(kg, "nodes", []):
            if getattr(n, "type", None) != NodeType.CHUNK:
                continue
            props = getattr(n, "properties", {}) or {}
            parent = props.get("parent_doc")
            if parent not in cached:
                continue
            kps = kp_by_doc.get(parent, {}).get(props.get("chunk_index"))
            if kps:
                props["keyphrases"] = kps
                hydrated += 1

        logging.info(
            "doc store: hydrated keyphrases for %d chunks from %d cached document(s)",
            hydrated, len(cached),
        )
        return cached

    def enrich_prechunked(
        self,
        kg: KnowledgeGraph,
        store_dir: Optional[Path] = None,
        on_progress: Optional[callable] = None,
    ) -> KnowledgeGraph:
        """
        Custom prechunked enrichment.

        Pipeline :
          0. (optionnel) Réhydrate les keyphrases depuis le store domaine pour les
             documents inchangés -> le LLM ne tourne que sur les nouveaux/modifiés.
          1. KeyphrasesExtractor (Ragas) via apply_transforms -- extraction LLM
             des keyphrases sur les CHUNK nodes uniquement (filter_nodes=filter_chunks).
             Sans LLM -> les chunks n'ont pas de keyphrases, les relations ne
             seront pas cr??es.
          2. OverlapScoreBuilder             -- relations overlap inter-chunks.
          3. enrich_with_rule_based_keyphrases() -- blacklist + IDF pruning +
                                               cosine_similarity.
        """
        logging.info("Enriching knowledge graph (custom prechunked pipeline)...")

        chunk_nodes = [
            n
            for n in getattr(kg, "nodes", [])
            if getattr(n, "type", None) == NodeType.CHUNK
        ]

        if not chunk_nodes:
            logging.warning("enrich_prechunked: no CHUNK nodes found in KG.")
            return kg

        # ?? Step 0 : réhydratation depuis le store (cache keyphrases) ?????????
        if store_dir is not None:
            try:
                self._hydrate_keyphrases_from_store(kg, store_dir)
            except Exception as _hyd_exc:
                logging.warning("doc store hydration failed (%s) — full recompute", _hyd_exc)

        # ?? Step 1 : KeyphrasesExtractor Ragas via apply_transforms ??????????
        if self.llm is None:
            logging.warning(
                "enrich_prechunked: no LLM provided -- chunks will have no keyphrases "
                "and no semantic relations will be built."
            )
        else:
            n_chunks = len(chunk_nodes)
            logging.info(
                "KeyphrasesExtractor (Ragas): extraction sur %d CHUNK nodes...",
                n_chunks,
            )
            try:
                # Build keyphrase prompt: config-driven if available, else the
                # DOMAIN-NEUTRAL built-in fallback.

                _kp_prompt = BridgeKeyphrasesExtractorPrompt()
                if self._config is not None:
                    try:
                        from pipeline_config import build_prompt_class, deserialize_few_shots
                        _kp_instr = self._config.prompts.keyphrase_extractor
                        _kp_exs   = deserialize_few_shots(
                            "keyphrase_extractor",
                            self._config.few_shots.keyphrase_extractor,
                            self._config,
                        )
                        _kp_prompt = build_prompt_class(
                            BridgeKeyphrasesExtractorPrompt, _kp_instr, _kp_exs
                        )()
                    except Exception as _kpe:
                        logging.warning(
                            "enrich_prechunked: keyphrase prompt build failed (%s) — "
                            "falling back to the DOMAIN-NEUTRAL built-in prompt. "
                            "Keyphrase quality will be lower than with a "
                            "domain-tuned prompt from the session config.",
                            _kpe,
                        )


                # Ne (re)calcule que les chunks SANS keyphrases : ceux réhydratés
                # depuis le store domaine sont déjà remplis et sont sautés.
                def _needs_keyphrases(node: "Node") -> bool:
                    return (
                        getattr(node, "type", None) == NodeType.CHUNK
                        and not (getattr(node, "properties", {}) or {}).get("keyphrases")
                    )

                todo = [n for n in chunk_nodes if _needs_keyphrases(n)]
                if not todo:
                    logging.info(
                        "KeyphrasesExtractor : %d/%d chunks déjà en cache — extraction LLM sautée.",
                        n_chunks, n_chunks,
                    )
                else:
                    logging.info(
                        "KeyphrasesExtractor : %d chunks à extraire (%d depuis cache).",
                        len(todo), n_chunks - len(todo),
                    )
                    keyphrase_extractor = KeyphrasesExtractor(
                        llm=self.llm,
                        property_name="keyphrases",
                        max_num=self.max_keyphrases,
                        filter_nodes=_needs_keyphrases,
                        prompt=_kp_prompt,
                    )
                    # Override le tokenizer de l'instance (LLMBasedExtractor utilise
                    # self.tokenizer, pas _ragas_tok._default_tokenizer global)
                    keyphrase_extractor.tokenizer = _WordTokenizer()
                    apply_transforms(kg, transforms=[keyphrase_extractor])

                    # Post-filter : supprimer les keyphrases g?n?riques
                    # (uniquement sur les chunks fraîchement extraits ; ceux du
                    #  cache ont déjà été filtrés lors de leur sauvegarde).
                    _apply_post_filter(todo, config=self._config)
                    logging.info("Post-filter keyphrases g?n?riques appliqu?.")

                # Compter les chunks enrichis avec succ?s
                ok = sum(1 for n in chunk_nodes if n.properties.get("keyphrases"))
                failed = n_chunks - ok
                logging.info(
                    "KeyphrasesExtractor termin? : %d/%d chunks OK, %d sans keyphrases",
                    ok,
                    n_chunks,
                    failed,
                )
            except Exception as e:
                logging.exception(
                    "Erreur lors de l'application de KeyphrasesExtractor: %s", e
                )

        if on_progress is not None:
            try:
                on_progress(kg, "keyphrases_extracted")
            except Exception:
                pass

        # ?? Step 2 : OverlapScoreBuilder (Ragas, rule-based) ?????????????????
        try:
            _kge = self._config.kg_enrichment if self._config else None
            apply_transforms(
                kg,
                transforms=[
                    OverlapScoreBuilder(
                        property_name="keyphrases",
                        new_property_name="overlap_score",
                        threshold=_kge.overlap_score_threshold if _kge else 0.02,
                        distance_threshold=_kge.overlap_distance_threshold if _kge else 0.9,
                        filter_nodes=filter_chunks,
                    )
                ],
            )
            logging.info("OverlapScoreBuilder termin?.")

            # Supprimer les relations keyphrases_overlap INTRA-document
            # (seulement en mode inter_doc_only=True).
            # OverlapScoreBuilder crée des liens entre tous les chunks sans
            # distinction de document source. En mode inter-doc, on ne garde
            # que les liens cross-doc. En mode intra-doc, on les conserve tous.
            if self._inter_doc_only:
                before = len(kg.relationships)
                kg.relationships = [
                    r
                    for r in kg.relationships
                    if not (
                        getattr(r, "type", "") == "keyphrases_overlap"
                        and (
                            getattr(r.source, "properties", {}).get("parent_doc")
                            == getattr(r.target, "properties", {}).get("parent_doc")
                            and getattr(r.source, "properties", {}).get("parent_doc")
                            is not None
                        )
                    )
                ]
                removed = before - len(kg.relationships)
                logging.info(
                    "keyphrases_overlap intra-doc supprimées : %d (restent %d au total)",
                    removed,
                    len(kg.relationships),
                )
            else:
                logging.info(
                    "Mode intra-doc : keyphrases_overlap intra-doc conservées (%d relations totales)",
                    len(kg.relationships),
                )

            # Enrichir les relations keyphrases_overlap restantes avec les
            # keyphrases communes (intersection source ? target) et le score
            enriched_count = 0
            for r in kg.relationships:
                if getattr(r, "type", "") != "keyphrases_overlap":
                    continue
                kps_src = set(r.source.properties.get("keyphrases", []))
                kps_tgt = set(r.target.properties.get("keyphrases", []))
                common = sorted(kps_src & kps_tgt)
                r.properties["shared_keyphrases"] = common
                r.properties["shared_count"] = len(common)
                enriched_count += 1
            logging.info(
                "keyphrases_overlap enrichies avec shared_keyphrases : %d relations",
                enriched_count,
            )
        except Exception as _e:
            logging.warning("OverlapScoreBuilder ?chou?: %s", _e)

        # Step 3 : relations inter-documents (cosine_similarity)
        kg = self.enrich_with_rule_based_keyphrases(kg)
        self._validate_cross_doc_relations(kg)

        if on_progress is not None:
            try:
                on_progress(kg, "rule_based_enriched")
            except Exception:
                pass

        # ?? Step 4 : filtrage par flags enrich_modules (UI toggles) ??????????
        # Si l'utilisateur a désactivé keyphrases/cosine/shared dans l'UI,
        # on retire ces types de relations du KG.
        kg = self._filter_relations_by_modules(kg)
        return kg

    def _filter_relations_by_modules(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Retire les relations dont le module a été désactivé via
        config.enrich_modules (toggles UI). Ne touche jamais aux relations
        structurelles ('contains') ni aux relations d'autres modules
        (agent_discovered, retrospective_entity).
        """
        em = getattr(self._config, "enrich_modules", None) if self._config else None
        if em is None:
            return kg

        # Map type de relation -> flag correspondant
        type_to_flag = {
            "keyphrases_overlap": getattr(em, "keyphrases_overlap", True),
            "cosine_similarity":  getattr(em, "cosine_similarity", True),
        }
        disabled = {t for t, enabled in type_to_flag.items() if not enabled}
        if not disabled:
            return kg

        before = len(kg.relationships)
        kg.relationships = [
            r for r in kg.relationships
            if str(getattr(r, "type", "")) not in disabled
        ]
        removed = before - len(kg.relationships)
        if removed:
            logging.warning(
                "enrich_modules: %d relation(s) retirée(s) — modules désactivés : %s",
                removed, ", ".join(sorted(disabled)),
            )
        return kg

    # ????????????????????????????????????????????????????????????????????????
    # Enrichment -- full pipeline (no chunks yet)
    # ????????????????????????????????????????????????????????????????????????

    # ????????????????????????????????????????????????????????????????????????
    # Cross-document semantic enrichment (TF-IDF + shared_keyphrase)
    # ????????????????????????????????????????????????????????????????????????

    def enrich_with_rule_based_keyphrases(
        self,
        kg: KnowledgeGraph,
        threshold: Optional[int] = None,
        min_keyphrases: Optional[int] = None,
        blacklist: Optional[Set[str]] = None,
        cos_sim_min: Optional[float] = None,
        cos_sim_max: Optional[float] = None,
    ) -> KnowledgeGraph:
        """
        Enrichit le KG avec des relations 'shared_keyphrase' et 'cosine_similarity'
        bas?es sur le champ 'keyphrases' (LLM) de chaque chunk.

        ?tapes :
            1. Blacklist statique (termes connus g?n?riques).
            2. Blacklist regex (num?riques, trop courts, abr?viations).
            3. Relations 'shared_keyphrase' : paires inter-documents partageant
               >= threshold keyphrases apr?s blacklist.
            4. Relations 'cosine_similarity' : paires inter-documents avec
               cos_sim_min <= cosine_sim <= cos_sim_max via TfidfVectorizer sklearn.

        The KG is modified in place and returned.
        When called without arguments the values are read from self._config
        (PipelineConfig) if available, otherwise hardcoded defaults are used.
        """
        # ── Apply config-driven defaults (override None parameters) ──────────
        kge = self._config.kg_enrichment if self._config else None
        if threshold is None:
            threshold = kge.shared_keyphrase_min_count if kge else 3
        if min_keyphrases is None:
            min_keyphrases = kge.shared_keyphrase_min_kps if kge else 3
        if cos_sim_min is None:
            cos_sim_min = kge.cosine_sim_min if kge else 0.6
        if cos_sim_max is None:
            cos_sim_max = kge.cosine_sim_max if kge else 0.9

        # ?? Blacklist statique ??????????????????????????????????????????????
        # The fallback list must stay DOMAIN-NEUTRAL: it only contains
        # structural / editorial boilerplate found in any documentation corpus.
        # Domain-specific terms (framework names, module names, ubiquitous
        # identifiers…) belong in `kg_enrichment.domain_blacklist` of the
        # session config — hardcoding them here would make the filter both
        # biased for one corpus and inoperative for every other one.
        if blacklist is None:
            if kge and kge.domain_blacklist:
                blacklist = set(kge.domain_blacklist)
            else:
                blacklist = {
                    "code description",
                    "output example",
                    "example",
                    "examples",
                    "note",
                    "notes",
                    "see also",
                    "overview",
                    "introduction",
                    "summary",
                    "description",
                    "table",
                    "figure",
                    "section",
                    "chapter",
                    "appendix",
                    "e.g",
                    "i.e",
                }
                logging.warning(
                    "enrich_with_rule_based_keyphrases: no "
                    "'kg_enrichment.domain_blacklist' in session config — using a "
                    "DOMAIN-NEUTRAL fallback blacklist (%d structural terms). "
                    "Ubiquitous domain terms will NOT be filtered; consider "
                    "declaring them in the session config.",
                    len(blacklist),
                )


        # ?? Patterns regex ? rejeter ????????????????????????????????????????
        _base_regex = (
            r"^(\d+\.?\d*"  # purement num?rique (ex: "3.14", "42")
            r"|[a-z0-9]{1,2}"  # 1 ou 2 caract?res
            r"|e\.?g\.?"  # e.g / eg
            r"|i\.?e\.?"  # i.e / ie
            r"|etc\.?"  # etc
            r"|vs\.?"  # vs
            r"|fig\.?"  # fig
            r"|eq\.?"  # eq
            r")$"
        )
        _extra_patterns = (kge.regex_blacklist_patterns if kge else []) or []
        if _extra_patterns:
            _combined_regex = _base_regex[:-1] + "|" + "|".join(
                f"(?:{p})" for p in _extra_patterns
            ) + "$"
        else:
            _combined_regex = _base_regex
        _REGEX_BLACKLIST = re.compile(_combined_regex, re.IGNORECASE)

        def _is_regex_blacklisted(kp: str) -> bool:
            return bool(_REGEX_BLACKLIST.match(kp.strip()))

        # ?? S?lection des chunks ?ligibles ??????????????????????????????????
        chunks_all = [
            n
            for n in getattr(kg, "nodes", [])
            if getattr(n, "type", None) == NodeType.CHUNK
        ]
        eligible = [
            c
            for c in chunks_all
            if len(c.properties.get("keyphrases", [])) > min_keyphrases
        ]
        logging.info(
            "enrich_with_rule_based_keyphrases: %d/%d chunks eligible (> %d keyphrases)",
            len(eligible),
            len(chunks_all),
            min_keyphrases,
        )
        if not eligible:
            logging.warning(
                "enrich_with_rule_based_keyphrases: no eligible chunks, skipping."
            )
            return kg

        # ?? Fonction de filtrage combin? ????????????????????????????????????
        def _keep(kp: str) -> bool:
            kp_lower = kp.lower().strip()
            if kp_lower in blacklist:
                return False
            if _is_regex_blacklisted(kp_lower):
                return False
            return True

        def _normalize(kp: str) -> str:
            """Normalisation morphologique l?g?re :
            - supprime possessifs 's
            - singular: -ies -> -y, -es -> ?, -s -> ? (heuristique)
            - deverbal: -ing -> ?, -ed -> ?
            """
            w = kp.lower().strip()
            w = re.sub(r"'s$", "", w)
            if w.endswith("ies") and len(w) > 4:
                w = w[:-3] + "y"
            elif w.endswith("es") and len(w) > 4:
                w = w[:-2]
            elif w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
                w = w[:-1]
            if w.endswith("ing") and len(w) > 5:
                w = w[:-3]
            if w.endswith("ed") and len(w) > 4:
                w = w[:-2]
            return w.strip()

        # ?? R?cup?rer le document source d'un chunk ?????????????????????????
        def _doc_id(node) -> str:
            meta = node.properties.get("document_metadata", {}) or {}
            fname = (
                meta.get("filename")
                or node.properties.get("filename", "")
                or node.properties.get("parent_doc", "")
            )
            if fname:
                return fname
            return str(getattr(node, "id", id(node)))[:8]

        # ?? Index keyphrase -> chunks (hors blacklist combin?e) ??????????????
        kp_to_chunks: Dict[str, List[Any]] = defaultdict(list)
        chunk_kps: Dict[str, Set[str]] = {}

        for c in eligible:
            raw_id = getattr(c, "id", None)
            if raw_id is None:
                raw_id = c.properties.setdefault("_stable_id", str(uuid.uuid4()))
            c_id = str(raw_id)
            filtered: Set[str] = set()
            for kp in c.properties.get("keyphrases", []):
                if _keep(kp):
                    norm = _normalize(kp)
                    if norm:
                        filtered.add(norm)
            chunk_kps[c_id] = filtered
            for kp in filtered:
                kp_to_chunks[kp].append(c)

        logging.info(
            "enrich_with_rule_based_keyphrases: %d unique keyphrases indexed",
            len(kp_to_chunks),
        )

        # ?? TF-IDF IDF pruning: supprimer les keyphrases trop fr?quentes ??
        import math as _math

        n_chunks_total = len(chunk_kps)
        if n_chunks_total > 1:
            df: Dict[str, int] = defaultdict(int)
            for kps in chunk_kps.values():
                for kp in kps:
                    df[kp] += 1

            idf: Dict[str, float] = {
                kp: _math.log(n_chunks_total / cnt) for kp, cnt in df.items()
            }

            # Seuil : on supprime les keyphrases pr?sentes dans plus de 50 % des chunks
            idf_threshold = kge.idf_threshold if kge else _math.log(2.0)  # ~= 0.693

            before = sum(len(v) for v in chunk_kps.values())
            chunk_kps = {
                c_id: {kp for kp in kps if idf.get(kp, 0.0) >= idf_threshold}
                for c_id, kps in chunk_kps.items()
            }
            after = sum(len(v) for v in chunk_kps.values())
            logging.info(
                "enrich_with_rule_based_keyphrases: IDF pruning removed %d/%d "
                "keyphrase occurrences (threshold IDF >= %.3f)",
                before - after,
                before,
                idf_threshold,
            )

            # Reconstruire l'index kp_to_chunks apr?s pruning IDF
            c_id_to_node = {str(getattr(c, "id", id(c))): c for c in eligible}
            kp_to_chunks = defaultdict(list)
            for c_id, kps in chunk_kps.items():
                node = c_id_to_node.get(c_id)
                if node is None:
                    continue
                for kp in kps:
                    kp_to_chunks[kp].append(node)

            logging.info(
                "enrich_with_rule_based_keyphrases: %d unique keyphrases apr?s IDF pruning",
                len(kp_to_chunks),
            )

        # Relations keyphrases inter-documents (paires partageant des keyphrases)
        # Matching fuzzy : deux keyphrases sont consid?r?es communes si leur
        # similarit? Jaccard sur les tokens est >= JACCARD_KP_THRESHOLD.
        # Ex : "cross-sectional area" et "annular cross-sectional area" -> 3/4 = 0.75 [OK]
        JACCARD_KP_THRESHOLD = kge.jaccard_kp_threshold if kge else 0.6

        def _kp_match(a: str, b: str) -> bool:
            """True si Jaccard(tokens(a), tokens(b)) >= JACCARD_KP_THRESHOLD."""
            ta = set(re.split(r"[\s\-_]+", a))
            tb = set(re.split(r"[\s\-_]+", b))
            if not ta or not tb:
                return False
            inter = len(ta & tb)
            union = len(ta | tb)
            return union > 0 and inter / union >= JACCARD_KP_THRESHOLD

        t0 = time.time()

        # Construire la liste des chunks ?ligibles avec leurs keyphrases apr?s IDF pruning
        c_id_to_node_final = {str(getattr(c, "id", id(c))): c for c in eligible}
        eligible_ids = [cid for cid in chunk_kps if chunk_kps[cid]]
        eligible_nodes_final = [
            c_id_to_node_final[cid] for cid in eligible_ids if cid in c_id_to_node_final
        ]

        # Pairwise matching (inter-doc only or all pairs depending on mode)
        pair_shared: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        pair_nodes: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

        n_elig = len(eligible_nodes_final)
        for i in range(n_elig):
            n_a = eligible_nodes_final[i]
            id_a = str(getattr(n_a, "id", id(n_a)))
            kps_a = list(chunk_kps.get(id_a, set()))
            doc_a = _doc_id(n_a)
            for j in range(i + 1, n_elig):
                n_b = eligible_nodes_final[j]
                # En mode inter_doc_only, skip les paires du même document
                if self._inter_doc_only and _doc_id(n_b) == doc_a:
                    continue
                id_b = str(getattr(n_b, "id", id(n_b)))
                kps_b = list(chunk_kps.get(id_b, set()))
                key = (min(id_a, id_b), max(id_a, id_b))
                if key not in pair_nodes:
                    pair_nodes[key] = (n_a, n_b)
                # Trouver les keyphrases communes via matching Jaccard
                matched_a: set = set()
                for kp_a in kps_a:
                    for kp_b in kps_b:
                        if kp_b in matched_a:
                            continue
                        if kp_a == kp_b or _kp_match(kp_a, kp_b):
                            # Conserver la keyphrase la plus courte comme repr?sentant
                            canonical = kp_a if len(kp_a) <= len(kp_b) else kp_b
                            pair_shared[key].append(canonical)
                            matched_a.add(kp_b)
                            break

        logging.info(
            "enrich_with_rule_based_keyphrases: %d inter-doc candidate pairs "
            "(fuzzy Jaccard >= %.1f) in %.1fs",
            len(pair_shared),
            JACCARD_KP_THRESHOLD,
            time.time() - t0,
        )

        # Les relations "shared_keyphrase" ont ete retirees de la pipeline STARK
        # (type non conserve). Seules subsistent les relations "cosine_similarity"
        # calculees ci-dessous. Le calcul `pair_shared` en amont reste utile pour
        # les statistiques/logs de recouvrement de keyphrases.

        # ?? Relations cosine_similarity ????????????????????????????????????
        # Priorite : embedding_model (titan-embed-v2) sur page_content COMPLET
        # Fallback  : sklearn TF-IDF sur keyphrases si pas d'embedding_model
        if self.embedding_model is not None:
            # ── Chemin A : embedding sur page_content complet (tous les chunks) ─
            logging.info(
                "enrich_with_rule_based_keyphrases: cosine_similarity via "
                "embedding_model (%s) sur page_content.",
                type(self.embedding_model).__name__,
            )
            try:
                # Tous les chunks (pas seulement ceux avec des keyphrases)
                eligible_for_cos = list(chunks_all)
                if len(eligible_for_cos) < 2:
                    logging.warning(
                        "enrich_with_rule_based_keyphrases: "
                        "too few chunks for cosine_similarity (embedding)."
                    )
                else:
                    texts = [
                        (c.properties.get("page_content", "")
                         or c.properties.get("raw_content", "")
                         or "")
                        for c in eligible_for_cos
                    ]
                    emb_model = self.embedding_model
                    if hasattr(emb_model, "embed_documents"):
                        vecs = emb_model.embed_documents(texts)
                    elif hasattr(emb_model, "embeddings") and hasattr(
                        emb_model.embeddings, "embed_documents"
                    ):
                        vecs = emb_model.embeddings.embed_documents(texts)
                    else:
                        raise AttributeError(
                            f"embed_documents non trouve sur {type(emb_model)}"
                        )
                    import math as _math_cos
                    def _dot(a, b):
                        return sum(x * y for x, y in zip(a, b))
                    def _norm(v):
                        return _math_cos.sqrt(sum(x * x for x in v))
                    def _cos(a, b):
                        na, nb = _norm(a), _norm(b)
                        return _dot(a, b) / (na * nb) if na and nb else 0.0

                    n = len(eligible_for_cos)
                    new_rels_cos = 0
                    t1 = time.time()
                    for global_i in range(n):
                        node_a = eligible_for_cos[global_i]
                        doc_a = _doc_id(node_a)
                        for global_j in range(global_i + 1, n):
                            node_b = eligible_for_cos[global_j]
                            if self._inter_doc_only and _doc_id(node_b) == doc_a:
                                continue
                            sim = _cos(vecs[global_i], vecs[global_j])
                            if cos_sim_min <= sim <= cos_sim_max:
                                tokens_a = set(
                                    node_a.properties.get("page_content", "")
                                    .lower().split()
                                )
                                tokens_b = set(
                                    node_b.properties.get("page_content", "")
                                    .lower().split()
                                )
                                union = tokens_a | tokens_b
                                jaccard = (
                                    len(tokens_a & tokens_b) / len(union)
                                    if union else 0.0
                                )
                                _anti_dup_threshold = (
                                    kge.cosine_anti_dup_jaccard if kge else 0.8
                                )
                                if jaccard >= _anti_dup_threshold:
                                    continue
                                kg.relationships.append(
                                    Relationship(
                                        source=node_a,
                                        target=node_b,
                                        type="cosine_similarity",
                                        properties={
                                            "cosine_score": round(sim, 4),
                                            "jaccard": round(jaccard, 4),
                                            "method": "embedding",
                                        },
                                    )
                                )
                                new_rels_cos += 1
                    logging.info(
                        "enrich_with_rule_based_keyphrases: %d new 'cosine_similarity' "
                        "relations via embedding (%.2f<=cos<=%.2f) in %.1fs",
                        new_rels_cos, cos_sim_min, cos_sim_max, time.time() - t1,
                    )
            except Exception as e_emb:
                logging.warning(
                    "enrich_with_rule_based_keyphrases: "
                    "embedding cosine_similarity error: %s — fallback TF-IDF.", e_emb,
                )
                # ── Fallback TF-IDF quand embedding_model échoue (ex: 504) ────
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

                    eligible_for_cos = [
                        c for c in eligible if chunk_kps.get(str(getattr(c, "id", id(c))))
                    ]
                    if len(eligible_for_cos) < 2:
                        logging.warning(
                            "enrich_with_rule_based_keyphrases: too few chunks for cosine_similarity (TF-IDF fallback)."
                        )
                    else:
                        chunk_ids_ordered = [
                            str(getattr(c, "id", id(c))) for c in eligible_for_cos
                        ]
                        corpus = [" ".join(sorted(chunk_kps[cid])) for cid in chunk_ids_ordered]
                        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                        tfidf_matrix = vectorizer.fit_transform(corpus)
                        t1 = time.time()
                        n = len(eligible_for_cos)
                        new_rels_cos = 0
                        BLOCK = 200
                        for i_start in range(0, n, BLOCK):
                            i_end = min(i_start + BLOCK, n)
                            block_sim = sklearn_cosine(
                                tfidf_matrix[i_start:i_end], tfidf_matrix
                            )
                            for local_i, global_i in enumerate(range(i_start, i_end)):
                                node_a = eligible_for_cos[global_i]
                                doc_a = _doc_id(node_a)
                                for global_j in range(global_i + 1, n):
                                    node_b = eligible_for_cos[global_j]
                                    if self._inter_doc_only and _doc_id(node_b) == doc_a:
                                        continue
                                    sim = float(block_sim[local_i, global_j])
                                    if cos_sim_min <= sim <= cos_sim_max:
                                        tokens_a = set(
                                            node_a.properties.get("page_content", "")
                                            .lower().split()
                                        )
                                        tokens_b = set(
                                            node_b.properties.get("page_content", "")
                                            .lower().split()
                                        )
                                        union = tokens_a | tokens_b
                                        jaccard = (
                                            len(tokens_a & tokens_b) / len(union)
                                            if union else 0.0
                                        )
                                        _anti_dup_threshold = (
                                            kge.cosine_anti_dup_jaccard if kge else 0.8
                                        )
                                        if jaccard >= _anti_dup_threshold:
                                            continue
                                        kg.relationships.append(
                                            Relationship(
                                                source=node_a,
                                                target=node_b,
                                                type="cosine_similarity",
                                                properties={
                                                    "cosine_score": round(sim, 4),
                                                    "jaccard": round(jaccard, 4),
                                                    "method": "tfidf_fallback",
                                                },
                                            )
                                        )
                                        new_rels_cos += 1
                        logging.info(
                            "enrich_with_rule_based_keyphrases: %d new 'cosine_similarity' "
                            "relations via TF-IDF fallback (%.2f<=cos<=%.2f) in %.1fs",
                            new_rels_cos, cos_sim_min, cos_sim_max, time.time() - t1,
                        )
                except ImportError:
                    logging.warning(
                        "enrich_with_rule_based_keyphrases: sklearn non disponible, "
                        "TF-IDF fallback impossible."
                    )
                except Exception as e_tfidf:
                    logging.warning(
                        "enrich_with_rule_based_keyphrases: TF-IDF fallback error: %s", e_tfidf,
                    )
        else:
            # ── Chemin B : fallback TF-IDF sklearn sur keyphrases ─────────────
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

                eligible_for_cos = [
                    c for c in eligible if chunk_kps.get(str(getattr(c, "id", id(c))))
                ]
                if len(eligible_for_cos) < 2:
                    logging.warning(
                        "enrich_with_rule_based_keyphrases: too few chunks for cosine_similarity."
                    )
                else:
                    chunk_ids_ordered = [
                        str(getattr(c, "id", id(c))) for c in eligible_for_cos
                    ]
                    corpus = [" ".join(sorted(chunk_kps[cid])) for cid in chunk_ids_ordered]

                    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
                    tfidf_matrix = vectorizer.fit_transform(corpus)

                    t1 = time.time()
                    n = len(eligible_for_cos)
                    new_rels_cos = 0
                    BLOCK = 200

                    for i_start in range(0, n, BLOCK):
                        i_end = min(i_start + BLOCK, n)
                        block_sim = sklearn_cosine(
                            tfidf_matrix[i_start:i_end], tfidf_matrix
                        )
                        for local_i, global_i in enumerate(range(i_start, i_end)):
                            node_a = eligible_for_cos[global_i]
                            doc_a = _doc_id(node_a)
                            for global_j in range(global_i + 1, n):
                                node_b = eligible_for_cos[global_j]
                                if self._inter_doc_only and _doc_id(node_b) == doc_a:
                                    continue
                                sim = float(block_sim[local_i, global_j])
                                if cos_sim_min <= sim <= cos_sim_max:
                                    tokens_a = set(
                                        node_a.properties.get("page_content", "")
                                        .lower().split()
                                    )
                                    tokens_b = set(
                                        node_b.properties.get("page_content", "")
                                        .lower().split()
                                    )
                                    union = tokens_a | tokens_b
                                    jaccard = (
                                        len(tokens_a & tokens_b) / len(union)
                                        if union else 0.0
                                    )
                                    _anti_dup_threshold = (
                                        kge.cosine_anti_dup_jaccard if kge else 0.8
                                    )
                                    if jaccard >= _anti_dup_threshold:
                                        continue
                                    kg.relationships.append(
                                        Relationship(
                                            source=node_a,
                                            target=node_b,
                                            type="cosine_similarity",
                                            properties={
                                                "cosine_score": round(sim, 4),
                                                "jaccard": round(jaccard, 4),
                                            },
                                        )
                                    )
                                    new_rels_cos += 1

                    logging.info(
                        "enrich_with_rule_based_keyphrases: %d new 'cosine_similarity' "
                        "relations (%.2f<=cos<=%.2f) in %.1fs",
                        new_rels_cos, cos_sim_min, cos_sim_max, time.time() - t1,
                    )

            except ImportError:
                logging.warning(
                    "enrich_with_rule_based_keyphrases: sklearn non disponible "
                    "et aucun embedding_model -- 'cosine_similarity' ignorees."
                )
            except Exception as e_cos:
                logging.exception(
                    "enrich_with_rule_based_keyphrases: cosine_similarity error: %s",
                    e_cos,
                )

        return kg

    # ????????????????????????????????????????????????????????????????????????
    # Diagnostic
    # ????????????????????????????????????????????????????????????????????????

    def _validate_cross_doc_relations(self, kg: KnowledgeGraph) -> None:
        """
        Warn if the KG has no cross-document semantic relationships.
        SEMANTIC_RELS lists the relation types considered "semantic".
        """
        SEMANTIC_RELS = {
            "keyphrases_overlap",
            "cosine_similarity",
            "agent_discovered",
            "llm_triplet",
        }
        rel_types = {getattr(r, "type", "") for r in getattr(kg, "relationships", [])}
        found = rel_types.intersection(SEMANTIC_RELS)
        if not found:
            logging.warning(
                "_validate_cross_doc_relations: KG has NO inter-document semantic "
                "relations (found types: %s). TestsetGenerator may produce fewer "
                "multi-hop questions. Check that embedding_model is passed and "
                "that chunks contain 'keyphrases'.",
                rel_types,
            )
        else:
            logging.info(
                "_validate_cross_doc_relations: inter-document relation types found: %s",
                found,
            )

    # ????????????????????????????????????????????????????????????????????????
    # Per-document store (domain-level, shared across sessions)
    # ????????????????????????????????????????????????????????????????????????

    def save_doc_store(self, kg: KnowledgeGraph, store_dir: Path) -> List[Path]:
        """
        Persist one JSON per document into `store_dir` (domain-level store, shared
        across all sessions). Each file gathers the document's chunks (content +
        breadcrumb + index), its structural metadata, and the LLM keyphrases that
        were attached to each CHUNK node during enrich_prechunked().

        Should be called on an ENRICHED knowledge graph so that
        node.properties["keyphrases"] is populated. Returns the list of written paths.
        """
        import hashlib

        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        # Group CHUNK nodes by their parent document filename.
        chunks_by_doc: Dict[str, List[Node]] = defaultdict(list)
        doc_nodes: Dict[str, Node] = {}
        for node in getattr(kg, "nodes", []):
            node_type = getattr(node, "type", None)
            props = getattr(node, "properties", {}) or {}
            if node_type == NodeType.DOCUMENT:
                fname = props.get("filename", "unknown")
                doc_nodes[fname] = node
            elif node_type == NodeType.CHUNK:
                parent = props.get("parent_doc", "unknown")
                chunks_by_doc[parent].append(node)

        written: List[Path] = []
        for filename, dnode in doc_nodes.items():
            dprops = getattr(dnode, "properties", {}) or {}
            content = dprops.get("page_content", "") or ""
            source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            file_path = dprops.get("file_path", "")
            try:
                mtime = Path(file_path).stat().st_mtime if file_path else None
            except OSError:
                mtime = None

            chunk_nodes = sorted(
                chunks_by_doc.get(filename, []),
                key=lambda n: (n.properties or {}).get("chunk_index", 0),
            )
            chunks_out = []
            for cn in chunk_nodes:
                cp = getattr(cn, "properties", {}) or {}
                chunks_out.append(
                    {
                        "chunk_index":   cp.get("chunk_index"),
                        "section_title": cp.get("section_title"),
                        "section_level": cp.get("section_level"),
                        "breadcrumb":    cp.get("breadcrumb"),
                        "token_count":   cp.get("token_count"),
                        "content":       cp.get("page_content", "") or cp.get("raw_content", ""),
                        "keyphrases":    to_jsonable(cp.get("keyphrases", []) or []),
                    }
                )

            doc_record = {
                "filename":     filename,
                "file_path":    file_path,
                "source_hash":  source_hash,
                "mtime":        mtime,
                "extracted_at": datetime.now().isoformat(),
                "metadata": {
                    "headings":          to_jsonable(dprops.get("headings", []) or []),
                    "links":             to_jsonable(dprops.get("links", []) or []),
                    "has_code":          dprops.get("has_code", False),
                    "code_block_count":  dprops.get("code_block_count", 0),
                    "inline_code_count": dprops.get("inline_code_count", 0),
                    "table_count":       dprops.get("table_count", 0),
                },
                "chunks": chunks_out,
            }

            out_path = store_dir / f"{Path(filename).stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(doc_record, f, indent=2, ensure_ascii=False)
            written.append(out_path)

        logging.info(
            "[OK] Doc store updated: %d document JSON(s) written to %s",
            len(written), store_dir,
        )
        return written

    # ????????????????????????????????????????????????????????????????????????
    # Persistence (delegated to KnowledgeGraphStorage; kept here for backward compat)
    # ????????????????????????????????????????????????????????????????????????

    @staticmethod
    def save(kg: KnowledgeGraph, output_path: Path) -> None:
        """Save knowledge graph to a JSON file (delegates to KnowledgeGraphStorage.save)."""
        KnowledgeGraphStorage.save(kg, output_path)


# Backward-compat alias for callers using the old class name
OptimizedKGBuilder = KnowledgeGraphBuilder

# ============================================================================
# KNOWLEDGE GRAPH STORAGE (merged: dict-load + full KG-load + node-type stats)
# ============================================================================


class KnowledgeGraphStorage:
    """Serialize, persist and reload knowledge graphs."""

    @staticmethod
    def save(kg: KnowledgeGraph, output_path: Path) -> None:
        """Serialize a KG to a JSON file with node/relationship metadata."""
        logging.info(f"Saving knowledge graph to {output_path}")
        kg_data: Dict[str, Any] = {"nodes": [], "relationships": []}

        # Serialize nodes
        for node in getattr(kg, "nodes", []):
            node_id = getattr(node, "id", None)
            node_id_json = str(id(node)) if node_id is None else to_jsonable(node_id)

            node_type = getattr(node, "type", "unknown")
            node_type_val = getattr(node_type, "value", None)
            node_type_json = (
                node_type_val if node_type_val is not None else str(node_type)
            )

            props = to_jsonable(getattr(node, "properties", {}) or {})

            kg_data["nodes"].append(
                {
                    "id": node_id_json,
                    "type": node_type_json,
                    "properties": props,
                }
            )

        # Serialize relationships
        for rel in getattr(kg, "relationships", []):
            source = getattr(rel, "source", None)
            target = getattr(rel, "target", None)
            source_id = getattr(source, "id", None) if source is not None else None
            target_id = getattr(target, "id", None) if target is not None else None
            rel_type = getattr(rel, "type", "unknown")
            rel_props = to_jsonable(getattr(rel, "properties", {}) or {})

            kg_data["relationships"].append(
                {
                    "source_id": (
                        to_jsonable(source_id) if source_id is not None else None
                    ),
                    "target_id": (
                        to_jsonable(target_id) if target_id is not None else None
                    ),
                    "type": to_jsonable(rel_type),
                    "properties": rel_props,
                }
            )

        kg_data["metadata"] = {
            "num_nodes": len(getattr(kg, "nodes", [])),
            "num_relationships": len(getattr(kg, "relationships", [])),
            "created_at": datetime.now().isoformat(),
            "node_types": KnowledgeGraphStorage._count_node_types(kg),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(kg_data, f, indent=2, ensure_ascii=False)

        logging.info(
            f"[OK] Knowledge graph saved: "
            f"{kg_data['metadata']['num_nodes']} nodes, "
            f"{kg_data['metadata']['num_relationships']} relationships"
        )

    @staticmethod
    def load(input_path: Path) -> Dict[str, Any]:
        """
        Load a KG JSON as a plain Python dict (lightweight, no Ragas objects).
        Use this for analytics or quick inspection.
        """
        logging.info(f"Loading knowledge graph (dict) from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)
        logging.info(
            f"[OK] Loaded: {len(kg_data['nodes'])} nodes, "
            f"{len(kg_data['relationships'])} relationships"
        )
        return kg_data

    @staticmethod
    def load_as_kg(input_path: Path) -> KnowledgeGraph:
        """
        Load a KG JSON and rebuild a full Ragas KnowledgeGraph object,
        restoring original UUIDs when possible.
        """
        logging.info(f"Loading knowledge graph (full) from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)

        kg = KnowledgeGraph()

        # Build nodes
        id_to_node: Dict[str, Node] = {}
        for node_data in kg_data.get("nodes", []):
            node_type_str = node_data.get("type", "chunk")
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.CHUNK

            node = Node(
                type=node_type,
                properties=node_data.get("properties", {}),
            )
            raw_id = node_data.get("id")
            if raw_id is not None:
                try:
                    from uuid import UUID as _UUID

                    node.id = _UUID(str(raw_id))
                except Exception:
                    pass  # keep auto-generated id
                id_to_node[str(raw_id)] = node

            kg.nodes.append(node)

        # Build relationships
        for rel_data in kg_data.get("relationships", []):
            source_id = str(rel_data.get("source_id", ""))
            target_id = str(rel_data.get("target_id", ""))
            source = id_to_node.get(source_id)
            target = id_to_node.get(target_id)
            if source is None or target is None:
                continue  # skip dangling references

            rel_type = rel_data.get("type", "unknown")
            rel_props = rel_data.get("properties", {})

            kg.relationships.append(
                Relationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    properties=rel_props,
                )
            )

        logging.info(
            f"[OK] Loaded: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships"
        )
        return kg

    @staticmethod
    def _count_node_types(kg: KnowledgeGraph) -> Dict[str, int]:
        """Tally node types for the saved metadata block."""
        type_counts: Dict[str, int] = {}
        for node in getattr(kg, "nodes", []):
            node_type = getattr(node, "type", "unknown")
            type_str = getattr(node_type, "value", str(node_type))
            type_counts[type_str] = type_counts.get(type_str, 0) + 1
        return type_counts


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def optimal_pipeline(
    md_files: List[Path],
    output_path: Path,
    llm: Any = None,
    embedding_model: Any = None,
    use_rule_based: bool = True,
    max_tokens: int = 4096,
    max_keyphrases: int = 10,
    min_chunk_tokens: int = 100,
    enrichment_mode: str = "lightweight",
    log_level: int = logging.INFO,
) -> KnowledgeGraph:
    """
    Orchestrate Markdown -> KG.

    Args:
        md_files:          List of Path objects to .md files.
        output_path:       Where to save the JSON KG.
        llm:               LLM (required for non-lightweight modes).
        embedding_model:   Embedding model (required for `prechunked_official`
                           and recommended for `prechunked_custom`).
        use_rule_based:    Use rule-based extraction in lightweight mode.
        max_tokens:        Max tokens per chunk.
        max_keyphrases:    Max keyphrases per chunk.
        min_chunk_tokens:  Drop chunks smaller than this.
        enrichment_mode:   One of:
                              - "lightweight"          (no LLM, default)
                              - "prechunked_custom"    (NER + Keyphrases LLM
                                                        + TF-IDF cross-doc)
                              - "prechunked_official"  (Ragas summary-based)
        log_level:         Logging level.

    Returns:
        Enriched KnowledgeGraph (also saved to disk).
    """
    logging.basicConfig(level=log_level)

    logging.info("=" * 80)
    logging.info("OPTIMAL MARKDOWN -> KNOWLEDGE GRAPH PIPELINE")
    logging.info("=" * 80)

    builder = KnowledgeGraphBuilder(
        llm=llm,
        embedding_model=embedding_model,
        use_rule_based=use_rule_based,
        max_tokens=max_tokens,
        max_keyphrases=max_keyphrases,
        min_chunk_tokens=min_chunk_tokens,
    )

    logging.info("\n[STEP 1/4] Parsing Markdown files...")
    kg = builder.create_from_markdown_files(md_files)

    logging.info(f"\n[STEP 2/4] Enrichment mode: {enrichment_mode.upper()}")
    if enrichment_mode == "prechunked_official":
        if llm is None or embedding_model is None:
            logging.warning(
                "prechunked_official requires BOTH llm and embedding_model. "
                "Falling back to lightweight."
            )
            kg = builder.enrich_lightweight(kg)
        else:
            kg = builder.enrich_prechunked_official(kg)
    elif enrichment_mode == "prechunked_custom":
        # Rule-based extraction first so chunks have keyphrases even if LLM is
        # missing or fails -- this guarantees TF-IDF can still run.
        kg = builder.enrich_lightweight(kg)
        kg = builder.enrich_prechunked(kg)
    else:  # "lightweight" (default)
        kg = builder.enrich_lightweight(kg)

    logging.info("\n[STEP 3/4] Saving Knowledge Graph...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    KnowledgeGraphStorage.save(kg, output_path)

    logging.info("\n[STEP 4/4] Pipeline complete")
    logging.info("=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Output:        {output_path}")
    logging.info(f"Nodes:         {len(kg.nodes)}")
    logging.info(f"Relationships: {len(kg.relationships)}")
    logging.info("=" * 80)

    return kg