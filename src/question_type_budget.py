"""
question_type_budget.py
-----------------------
Gestion du budget proportionnel des types de questions multi-hop.

Principe :
  - 6 types de questions avec des proportions cibles (somme = 1.0)
  - pick_type(compatible) choisit le type le plus "en retard" par rapport
    à sa proportion cible parmi les types compatibles fournis par le LLM
  - Tie-break aléatoire pour éviter tout biais d'ordre de liste

Types supportés :
  integration, comparison, design_rationale, implementation, enumeration, factual

Utilisation typique :
    budget = QuestionTypeBudget.load(path)
    compatible = await qualify_question_types(ctx1, ctx2, llm_config)
    chosen = budget.pick_type(compatible)
    budget.increment(chosen)
    budget.save(path)
"""

from __future__ import annotations

import json
import logging
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=8)
def _get_openai_client(base_url: Optional[str], api_key: str, model: str) -> ChatOpenAI:
    """Return a cached ChatOpenAI client for the given connection parameters."""
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=256,
        timeout=30,
        max_retries=2,
        http_client=httpx.Client(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )

# ── Proportions cibles (indépendantes du nombre total de questions) ───────────
# Somme = 1.0  — ajustables librement sans changer le reste du code.
QUESTION_TYPE_PROPORTIONS: Dict[str, float] = {
    "integration": 0.30,
    "comparison": 0.20,
    "design_rationale": 0.20,
    "implementation": 0.15,
    "enumeration": 0.10,
    "factual": 0.05,
}

# Ordre canonique (pour affichage reproductible)
QUESTION_TYPES: List[str] = list(QUESTION_TYPE_PROPORTIONS.keys())

# ── Prompt de qualification (DOMAIN-NEUTRAL fallback) ─────────────────────────
# Ces constantes ne servent QUE de repli quand la session ne fournit pas
# `config.prompts.qualify_system` / `qualify_user_template`. Elles doivent donc
# rester strictement neutres en domaine : aucune mention d'un corpus, d'un
# framework ou d'un vocabulaire métier particulier.
_QUALIFY_SYSTEM = (
    "You are an expert analyst of technical and reference documentation. "
    "Respond ONLY with a valid JSON object — no markdown, no explanation, no code fence."
)

# Descriptions neutres des types standard. Utilisées uniquement pour les types
# effectivement présents dans la taxonomie courante ; un type inconnu reçoit une
# description générique (voir _describe_question_type).
_NEUTRAL_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "integration": (
        "do the two segments describe elements that connect, depend on, or "
        "exchange information with each other?"
    ),
    "comparison": (
        "do the two segments describe two elements that can be meaningfully "
        "contrasted (similarities / differences)?"
    ),
    "design_rationale": (
        "does one segment explain WHY something stated in the other is defined, "
        "required, or structured that way?"
    ),
    "implementation": (
        "do both segments together explain HOW to apply, perform, or configure "
        "something concrete?"
    ),
    "enumeration": (
        "do both segments together yield a list of items, steps, conditions, or "
        "attributes?"
    ),
    "factual": (
        "do both segments together yield a precise value, definition, "
        "attribute, or identifier?"
    ),
}


def _describe_question_type(name: str, description: str = "") -> str:
    """Return the description line to use for a question type.

    Priority: the taxonomy's own description (session-provided, therefore
    domain-aware) > neutral built-in description > generic wording.
    """
    desc = (description or "").strip()
    if desc:
        return desc
    neutral = _NEUTRAL_TYPE_DESCRIPTIONS.get(name)
    if neutral:
        return neutral
    return (
        f"can a meaningful '{name}' question be generated from the two segments "
        "taken together?"
    )


def _iter_taxonomy_types(config) -> List[tuple]:
    """Yield (name, description) pairs for the active taxonomy.

    Falls back to the module-level standard types when no config is supplied.
    """
    pairs: List[tuple] = []
    if config is not None:
        try:
            tax = getattr(config, "taxonomy", None)
            for qt in (getattr(tax, "types", None) or []):
                if isinstance(qt, dict):
                    name, desc = qt.get("name"), qt.get("description", "")
                else:
                    name, desc = getattr(qt, "name", None), getattr(qt, "description", "")
                if name:
                    pairs.append((name, desc or ""))
            if not pairs:
                for name in (getattr(tax, "budget", None) or {}):
                    pairs.append((name, ""))
        except Exception:
            pairs = []
    if not pairs:
        pairs = [(n, "") for n in QUESTION_TYPES]
    return pairs


def build_qualify_user_template(config=None) -> str:
    """Build a domain-neutral qualification template for the ACTIVE taxonomy.

    The type list is derived from `config.taxonomy` so that a session with a
    custom taxonomy never sees the six hardcoded standard types, and the JSON
    example always cites types that actually exist.

    Returns a template with the literal placeholders `{context_1}`/`{context_2}`
    (all other braces are escaped) so it can be `.format()`-ed by the caller.
    """
    pairs = _iter_taxonomy_types(config)
    width = max((len(n) for n, _ in pairs), default = 0)
    type_lines = "\n".join(
        f"- {name.ljust(width)} : {_describe_question_type(name, desc)}"
        for name, desc in pairs
    )
    example = ", ".join(f'"{name}"' for name, _ in pairs[:2]) or '"factual"'
    return (
        "Given these two documentation segments:\n"
        "\n"
        "<1-hop> {context_1}\n"
        "\n"
        "<2-hop> {context_2}\n"
        "\n"
        "For each question type below, decide if it is compatible with these two "
        "segments (i.e. a meaningful question of that type CAN be generated from "
        "them together):\n"
        "\n"
        f"{type_lines}\n"
        "\n"
        "Return ONLY this JSON (no markdown, no extra keys):\n"
        f'{{{{"compatible_types": [{example}]}}}}\n'
    )


# Repli statique (taxonomie standard), conservé pour compatibilité d'import.
_QUALIFY_USER_TEMPLATE = build_qualify_user_template(None)

# Max chars envoyés au LLM de qualification (évite les prompts trop longs)
_MAX_QUALIFY_CHARS = 3000



class QuestionTypeBudget:
    """
    Maintient les compteurs de types de questions et sélectionne le type
    le plus sous-représenté par rapport aux proportions cibles.

    Attributes:
        proportions: Dict[str, float]  — proportions cibles (constantes)
        counts:      Dict[str, int]    — compteurs courants
    """

    def __init__(self, counts: Optional[Dict[str, int]] = None, config=None):
        if config is not None:
            # Use taxonomy budget and question types from config
            self.proportions = dict(config.taxonomy.budget)
            _types = config.question_type_names()
        else:
            self.proportions = dict(QUESTION_TYPE_PROPORTIONS)
            _types = QUESTION_TYPES

        self.counts = {t: 0 for t in _types}
        if counts:
            for t, v in counts.items():
                if t in self.counts:
                    self.counts[t] = int(v)

    # ── Algorithme de sélection ───────────────────────────────────────────────

    def pick_type(self, compatible: List[str]) -> str:
        """
        Choisit le type de question le plus sous-représenté parmi `compatible`.

        Algorithme :
          1. Filtrer sur les types connus
          2. Calculer le lag = proportion_cible - proportion_actuelle
          3. Choisir parmi les ex-aequo via tirage aléatoire (pas de biais d'ordre)

        Args:
            compatible: liste des types jugés compatibles par le LLM de qualification.

        Returns:
            Le type choisi (str).
        """
        candidates = [t for t in compatible if t in self.proportions]
        if not candidates:
            # Fallback : si aucun type connu, retourner le premier de la liste
            # ou "integration" par défaut
            return compatible[0] if compatible else "integration"

        total = sum(self.counts.values()) or 1  # évite division par zéro

        def lag(t: str) -> float:
            """Retard : positif = sous-représenté, négatif = sur-représenté."""
            return self.proportions[t] - (self.counts[t] / total)

        max_lag = max(lag(t) for t in candidates)

        # Tous les types partageant le lag maximal (ex-aequo)
        tied = [t for t in candidates if abs(lag(t) - max_lag) < 1e-9]

        # Tirage aléatoire parmi les ex-aequo → pas de biais d'ordre
        return random.choice(tied)

    # ── Compteur ─────────────────────────────────────────────────────────────

    def increment(self, qtype: str) -> None:
        """Incrémente le compteur du type donné (crée-le si inconnu)."""
        if qtype in self.counts:
            self.counts[qtype] += 1
        else:
            logging.warning(
                "QuestionTypeBudget: unknown type %r — incrementing anyway", qtype
            )
            self.counts[qtype] = 1

    # ── Persistance ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path]) -> None:
        """Sauvegarde les compteurs dans un fichier JSON."""
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"counts": self.counts}, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("QuestionTypeBudget: could not save to %s: %s", path, exc)

    @classmethod
    def load(cls, path: Optional[Path], config=None) -> "QuestionTypeBudget":
        """
        Charge les compteurs depuis un fichier JSON existant.
        Si le fichier n'existe pas ou est corrompu, retourne un budget vierge.
        """
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                counts = data.get("counts", {})
                budget = cls(counts=counts, config=config)
                total = sum(budget.counts.values())
                logging.info(
                    "QuestionTypeBudget loaded from %s — %d questions so far: %s",
                    path,
                    total,
                    dict(budget.counts),
                )
                return budget
            except Exception as exc:
                logging.warning(
                    "QuestionTypeBudget: could not load from %s: %s — starting fresh",
                    path,
                    exc,
                )
        return cls(config=config)

    @classmethod
    def load_or_rebuild(
        cls,
        path: Optional[Path],
        existing_questions: List[Dict],
        config=None,
    ) -> "QuestionTypeBudget":
        """
        Charge le budget depuis ``path`` si le fichier existe, sinon le
        reconstruit depuis les ``question_type`` des questions existantes.

        Cas d'usage : le fichier ``_budget.json`` a été perdu (crash disque,
        déplacement du dossier) mais ``checkpoint.json`` contient déjà des
        questions.  Sans cette méthode, le budget repart de zéro et les
        proportions sont faussées pour toute la suite de la génération.

        Args:
            path:               Chemin du fichier budget JSON (peut être None).
            existing_questions: Liste de questions déjà générées (peut être []).
            config:             Optional PipelineConfig — proportions and type
                                names are read from config.taxonomy when provided.

        Returns:
            Un ``QuestionTypeBudget`` chargé ou reconstruit.
        """
        if path and path.exists():
            return cls.load(path, config=config)
        # Le fichier budget est absent — reconstruire depuis les questions existantes
        budget = cls(config=config)
        for q in existing_questions:
            qt = q.get("question_type")
            if qt and qt in budget.counts:
                budget.counts[qt] += 1
        if existing_questions:
            total = sum(budget.counts.values())
            logging.info(
                "Budget reconstructed from %d existing questions (budget file absent): %s",
                total,
                budget.counts,
            )
        return budget

    # ── Rapport ──────────────────────────────────────────────────────────────

    def report(self) -> str:
        """Retourne un rapport lisible des compteurs vs proportions cibles."""
        total = sum(self.counts.values()) or 1
        lines = ["QuestionTypeBudget — current distribution:"]
        for t in self.counts:
            count = self.counts[t]
            actual_pct = 100.0 * count / total
            target_pct = 100.0 * self.proportions.get(t, 0.0)
            bar = "█" * int(actual_pct / 2)
            lines.append(
                f"  {t:<20} {count:>5}  actual={actual_pct:5.1f}%  target={target_pct:5.1f}%  {bar}"
            )
        return "\n".join(lines)


# ── Qualification LLM ─────────────────────────────────────────────────────────


async def qualify_question_types(
    context_1hop: str,
    context_2hop: str,
    llm_config: Optional[dict],
    config=None,
) -> List[str]:
    """
    Appelle le LLM pour déterminer quels types de questions sont compatibles
    avec les deux segments de contexte fournis.

    Args:
        context_1hop: Contenu du segment <1-hop> (peut inclure le label).
        context_2hop: Contenu du segment <2-hop> (peut inclure le label).
        llm_config:   Dict avec base_url / api_key / model (même config que le pipeline).
                      Si None, retourne ["integration"] comme fallback.
        config:       Optional PipelineConfig — when provided, system prompt and
                      user template are read from config.prompts.

    Returns:
        Liste des types compatibles (sous-ensemble de QUESTION_TYPES).
        Garantit au moins ["integration"] si aucun type n'est trouvé.
    """
    if not llm_config:
        logging.debug(
            "qualify_question_types: no llm_config — falling back to all known types"
        )
        _tax_types = []
        if config is not None:
            try:
                tax = getattr(config, "taxonomy", None)
                if tax:
                    budget = getattr(tax, "budget", None) or {}
                    _tax_types = list(budget.keys()) if budget else []
                    if not _tax_types:
                        types_list = getattr(tax, "types", None) or []
                        _tax_types = [
                            (qt.get("name") if isinstance(qt, dict) else getattr(qt, "name", None))
                            for qt in types_list
                        ]
                        _tax_types = [n for n in _tax_types if n]
            except Exception:
                pass
        return sorted(_tax_types) if _tax_types else list(QUESTION_TYPE_PROPORTIONS.keys())

    # Choose prompt templates: session config first, then a DOMAIN-NEUTRAL
    # fallback built from the active taxonomy. The fallback must never carry
    # domain vocabulary — a missing config is signalled loudly instead.
    qualify_system   = (config.prompts.qualify_system        if config else None) or _QUALIFY_SYSTEM
    qualify_template = (config.prompts.qualify_user_template if config else None)
    if not qualify_template:
        qualify_template = build_qualify_user_template(config)
        logging.warning(
            "qualify_question_types: no 'prompts.qualify_user_template' in session "
            "config — using the DOMAIN-NEUTRAL fallback built from the active "
            "taxonomy. Quality will be lower than with a domain-tuned prompt."
        )
    max_qualify_chars = (config.evaluation.max_qualify_chars  if config else None) or _MAX_QUALIFY_CHARS


    # Tronquer les contextes pour ne pas exploser le prompt
    ctx1_clean = _strip_hop_label(context_1hop)[:max_qualify_chars]
    ctx2_clean = _strip_hop_label(context_2hop)[:max_qualify_chars]

    user_msg = qualify_template.format(context_1=ctx1_clean, context_2=ctx2_clean)

    _TRANSIENT_MARKERS = (
        "invalid model name", "connection error", "timeout", "timed out",
        "temporarily unavailable", "service unavailable", "bad gateway",
        "gateway timeout", "rate limit", "too many requests", "overloaded",
        "internal server error", " 429", " 500", " 502", " 503", " 504",
    )

    try:
        client = _get_openai_client(
            base_url=llm_config.get("base_url"),
            api_key=llm_config["api_key"],
            model=llm_config.get("model", "gpt-4o-mini"),
        )
        messages = [
            {"role": "system", "content": qualify_system},
            {"role": "user", "content": user_msg},
        ]
        # Retry with exponential backoff on transient proxy errors (LiteLLM
        # occasionally returns "Invalid model name" / connection errors under
        # load even for a valid, working model — retrying a few seconds later
        # succeeds).
        import asyncio as _asyncio
        _last_exc = None
        response = None
        for _attempt in range(1, 6 + 1):
            try:
                response = await client.ainvoke(messages)
                break
            except Exception as _e:
                _last_exc = _e
                _msg = str(_e).lower()
                if not any(m in _msg for m in _TRANSIENT_MARKERS) or _attempt == 6:
                    raise
                _delay = 3.0 * (2 ** (_attempt - 1))
                logging.warning(
                    "qualify_question_types: transient error (attempt %d/6): %s — retrying in %.1fs",
                    _attempt, _e, _delay,
                )
                await _asyncio.sleep(_delay)
        raw = (response.content or "").strip()

        # Build valid types set.
        # If the session defines a custom taxonomy, use ONLY those types so that
        # the hardcoded defaults (integration, comparison, …) are not accepted as
        # valid responses for a domain-specific session.
        _session_types: set = set()
        if config is not None:
            try:
                tax = getattr(config, "taxonomy", None)
                if tax:
                    budget = getattr(tax, "budget", None) or {}
                    if isinstance(budget, dict) and budget:
                        _session_types = set(budget.keys())
                    types_list = getattr(tax, "types", None) or []
                    for qt in types_list:
                        name = qt.get("name") if isinstance(qt, dict) else getattr(qt, "name", None)
                        if name:
                            _session_types.add(name)
            except Exception:
                pass
        _valid_types = _session_types if _session_types else set(QUESTION_TYPE_PROPORTIONS.keys())

        compatible = _parse_compatible_types(raw, valid_types=_valid_types)
        if compatible:
            logging.debug("qualify_question_types → %s", compatible)
            return compatible

        # LLM returned an empty list or types not in valid set — fall back to
        # all known types so the budget can pick the most under-represented one.
        _fallback = sorted(_valid_types) if _valid_types else list(QUESTION_TYPE_PROPORTIONS.keys())
        logging.warning(
            "qualify_question_types: LLM returned no compatible types (raw=%r) — fallback to all %d known types",
            raw[:200],
            len(_fallback),
        )
        return _fallback

    except Exception as exc:
        # _valid_types may not be bound if the exception occurred before line 361
        _fb_types = locals().get("_valid_types") or locals().get("_session_types")
        _fallback = sorted(_fb_types) if _fb_types else list(QUESTION_TYPE_PROPORTIONS.keys())
        logging.warning(
            "qualify_question_types: LLM call failed (%s) — fallback to all %d known types",
            exc,
            len(_fallback),
        )
        return _fallback


def _strip_hop_label(context: str) -> str:
    """Supprime le label <1-hop> / <2-hop> en tête s'il existe."""
    return re.sub(r"^<\d+-hop>\s*", "", context.strip(), flags=re.IGNORECASE)


def _parse_compatible_types(raw: str, valid_types: Optional[set] = None) -> List[str]:
    """
    Parse la réponse JSON du LLM et retourne la liste des types valides.

    Accepte :
      - {"compatible_types": ["integration", "comparison"]}
      - ```json\n{"compatible_types": [...]}```  (markdown fence — robustesse)

    Args:
        raw:         Raw LLM response string.
        valid_types: Optional set of accepted type names. If None, defaults to
                     QUESTION_TYPE_PROPORTIONS (hardcoded standard types).
    """
    if valid_types is None:
        valid_types = set(QUESTION_TYPE_PROPORTIONS.keys())

    # Supprimer les fences markdown si présentes
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Tentative d'extraction par regex si le JSON est malformé
        match = re.search(r'"compatible_types"\s*:\s*\[([^\]]*)\]', clean)
        if not match:
            return []
        items_raw = match.group(1)
        data = {"compatible_types": re.findall(r'"([^"]+)"', items_raw)}

    raw_types = data.get("compatible_types", [])
    # Filtrer sur les types connus (session taxonomy OU hardcoded)
    valid = [t for t in raw_types if t in valid_types]
    return valid if valid else []
