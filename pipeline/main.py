#!/usr/bin/env python3
"""
main.py — STARK Pipeline Runner
================================
Lance la pipeline STARK depuis la ligne de commande.
Config unique : pipeline/cosapp_v1.yaml

Le fichier cosapp_v1.yaml est une copie de src/sessions/cosapp_v1.yaml
avec deux sections supplémentaires en bas, lues uniquement par ce script :

  tools:      active/désactive chaque outil optionnel
  filters:    restreint la taxonomie, les personas, les longueurs, les styles

Priorité : flags CLI > cosapp_v1.yaml (tools/filters) > .env > defaults

Exemples :
  python main.py                       # tout depuis cosapp_v1.yaml
  python main.py --dry-run             # test rapide (2 questions max)
  python main.py --num-questions 10    # override du nombre de questions
  python main.py --no-discover         # désactiver discover pour ce run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ── Chemins ───────────────────────────────────────────────────────────────────
_PIPELINE_DIR = Path(__file__).parent.resolve()
_DATASET_DIR  = _PIPELINE_DIR.parent
_SRC_DIR      = _DATASET_DIR / "src"
_YAML_FILE    = _PIPELINE_DIR / "cosapp_v1.yaml"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── .env ──────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv as _load_dotenv
_env_file = _DATASET_DIR / ".env"
if _env_file.exists():
    _load_dotenv(_env_file, override=False)


# =============================================================================
# LECTURE DU YAML (brut + PipelineConfig)
# =============================================================================

def read_yaml_raw(path: Path = _YAML_FILE) -> Dict:
    """Charge le YAML complet en dict brut (inclut tools: et filters:)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_session(path: Path = _YAML_FILE):
    """Charge la PipelineConfig depuis le YAML (sections extra ignorées par Pydantic)."""
    from base import PipelineConfig
    return PipelineConfig.from_yaml(path)


# =============================================================================
# ARGPARSE — overrides ponctuels uniquement
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stark-pipeline",
        description="STARK — Pipeline QA  |  config : pipeline/cosapp_v1.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Paramètres de base
    parser.add_argument("--yaml", metavar="FILE", default=str(_YAML_FILE),
                        help=f"Fichier YAML de config (défaut : {_YAML_FILE.name})")
    parser.add_argument("--input-dir",  metavar="DIR",
                        help="Override INPUT_DIR du .env")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Override OUTPUT_DIR du .env")
    parser.add_argument("--num-questions", type=int, metavar="N",
                        help="Override NUM_QUESTIONS du .env")
    parser.add_argument("--env", metavar="FILE",
                        help="Override du fichier .env")

    # Overrides outils (sans toucher au YAML)
    grp = parser.add_argument_group("Overrides outils (priorité sur cosapp_v1.yaml)")
    grp.add_argument("--dry-run",       action="store_true", help="Test : 2 questions max")
    grp.add_argument("--validate",      action="store_true", help="Activer RelationValidator")
    grp.add_argument("--no-validate",   action="store_true", help="Désactiver RelationValidator")
    grp.add_argument("--discover",      action="store_true", help="Activer DirectRelationDiscovery")
    grp.add_argument("--no-discover",   action="store_true", help="Désactiver DirectRelationDiscovery")
    grp.add_argument("--qa-eval",       action="store_true", help="Activer QA Evaluator")
    grp.add_argument("--no-qa-eval",    action="store_true", help="Désactiver QA Evaluator")

    return parser.parse_args()


# =============================================================================
# FUSION YAML + CLI → paramètres effectifs
# =============================================================================

def build_params(raw: Dict, args: argparse.Namespace) -> argparse.Namespace:
    """Produit un Namespace avec tous les paramètres effectifs."""
    tools   = raw.get("tools",   {})
    filters = raw.get("filters", {})

    def _tool(on_flag, off_flag, key, default=False):
        if on_flag:  return True
        if off_flag: return False
        return bool(tools.get(key, default))

    # Chemins & nombre
    input_dir  = args.input_dir  or os.environ.get("INPUT_DIR",  "./docs")
    output_dir = args.output_dir or os.environ.get("OUTPUT_DIR")   # résolu après chargement session
    num_q      = args.num_questions or int(os.environ.get("NUM_QUESTIONS", "10"))

    return argparse.Namespace(
        yaml_file           = Path(args.yaml),
        input_dir           = Path(input_dir),
        output_dir          = Path(output_dir) if output_dir else None,
        num_questions       = num_q,
        # outils
        vlm                 = bool(tools.get("vlm", False)),
        cosine_relations    = bool(tools.get("cosine_relations", False)),
        keyphrase_relations = bool(tools.get("keyphrase_relations", False)),
        validate_relations  = _tool(args.validate,  args.no_validate,  "validate_relations"),
        discover_relations  = _tool(args.discover,  args.no_discover,  "discover_relations"),
        qa_eval             = _tool(args.qa_eval,   args.no_qa_eval,   "qa_eval"),
        dry_run             = args.dry_run or bool(tools.get("dry_run", False)),
        # filtres
        filter_taxonomy     = filters.get("taxonomy"),   # None ou list[str]
        filter_personas     = filters.get("personas"),   # None ou list[str]
        filter_lengths      = filters.get("lengths"),    # None ou list[str]
        filter_styles       = filters.get("styles"),     # None ou list[str]
    )


# =============================================================================
# APPLICATION DES FILTRES
# =============================================================================

def apply_filters(config, p: argparse.Namespace) -> None:
    """Modifie config en place selon les filtres définis dans cosapp_v1.yaml."""

    # ── Taxonomie ─────────────────────────────────────────────────────────────
    if p.filter_taxonomy:
        allowed = set(p.filter_taxonomy)
        new_types = [t for t in config.taxonomy.types if t.name in allowed]
        if new_types:
            config.taxonomy.types = new_types
            kept = {k: v for k, v in config.taxonomy.budget.items() if k in allowed}
            total = sum(kept.values()) or 1.0
            config.taxonomy.budget = {k: round(v / total, 6) for k, v in kept.items()}
            logging.info("filter.taxonomy : %s types conservés", [t.name for t in new_types])
        else:
            logging.warning("filter.taxonomy : aucun type valide parmi %s — filtre ignoré", p.filter_taxonomy)

    # ── Lengths ───────────────────────────────────────────────────────────────
    if p.filter_lengths:
        config.query_params.lengths = list(p.filter_lengths)
        logging.info("filter.lengths : %s", p.filter_lengths)

    # ── Styles ────────────────────────────────────────────────────────────────
    if p.filter_styles:
        config.query_params.styles = list(p.filter_styles)
        logging.info("filter.styles : %s", p.filter_styles)

    # (Personas filtrées directement dans run_pipeline sur la liste RAGAS)


# =============================================================================
# LLM / EMBEDDING
# =============================================================================

def _http_clients(verify: bool):
    import httpx
    return httpx.Client(verify=verify), httpx.AsyncClient(verify=verify)


def build_llm(model: Optional[str] = None) -> Any:
    from langchain_openai import ChatOpenAI
    _m   = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    _url = os.environ.get("OPENAI_BASE_URL")
    _key = os.environ.get("OPENAI_API_KEY", "")
    _tok = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
    _ssl = os.environ.get("OPENAI_VERIFY_SSL", "true").lower() != "false"
    kw: dict = dict(model=_m, openai_api_key=_key, max_tokens=_tok)
    if _url: kw["openai_api_base"] = _url
    if not _ssl:
        sc, ac = _http_clients(False)
        kw["http_client"] = sc
        kw["http_async_client"] = ac
    return ChatOpenAI(**kw)


def build_embedding() -> Any:
    from langchain_openai import OpenAIEmbeddings
    _url = os.environ.get("EMBEDDING_BASE_URL")
    _key = os.environ.get("EMBEDDING_API_KEY", "")
    _m   = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    _ssl = os.environ.get("OPENAI_VERIFY_SSL", "true").lower() != "false"
    kw: dict = dict(model=_m, openai_api_key=_key)
    if _url: kw["openai_api_base"] = _url
    if not _ssl:
        sc, ac = _http_clients(False)
        kw["http_client"] = sc
        kw["http_async_client"] = ac
    return OpenAIEmbeddings(**kw)


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_header(config, p: argparse.Namespace, n_files: int, output_dir: Path) -> None:
    active = []
    if p.vlm:                  active.append("vlm")
    if p.cosine_relations:     active.append("cosine-relations")
    if p.keyphrase_relations:  active.append("keyphrase-relations")
    if p.validate_relations:   active.append("validate-relations")
    if p.discover_relations:   active.append("discover-relations")
    if p.qa_eval:              active.append("qa-eval")
    if p.dry_run:              active.append("DRY-RUN")

    active_filters = []
    if p.filter_taxonomy: active_filters.append(f"taxonomy={p.filter_taxonomy}")
    if p.filter_personas: active_filters.append(f"personas={p.filter_personas}")
    if p.filter_lengths:  active_filters.append(f"lengths={p.filter_lengths}")
    if p.filter_styles:   active_filters.append(f"styles={p.filter_styles}")

    print()
    print("=" * 64)
    print(f"  STARK Pipeline — {config.domain.name}")
    print("=" * 64)
    print(f"  Config       : {p.yaml_file.name}")
    print(f"  Documents    : {n_files} fichier(s) dans {p.input_dir}")
    print(f"  Sortie       : {output_dir}")
    print(f"  Questions    : {p.num_questions}" + (" → max 2 (dry-run)" if p.dry_run else ""))
    print(f"  Outils       : {', '.join(active) or 'base seulement'}")
    if active_filters:
        print(f"  Filtres      : {', '.join(active_filters)}")
    print("=" * 64)
    print()


# =============================================================================
# PIPELINE PRINCIPALE
# =============================================================================

async def run_pipeline(config, p: argparse.Namespace, llm, emb, files: List[Path], out: Path) -> List[dict]:
    from base import KnowledgeGraphBuilder, KnowledgeGraphStorage, build_ragas_personas, QuestionGenerator

    out.mkdir(parents=True, exist_ok=True)
    kg_store = Path(os.environ.get("STARK_KG_STORE_DIR", str(_DATASET_DIR / "kg_store")))
    t0 = time.time()

    # ── BASE 1 : Construction du KG ───────────────────────────────────────────
    print("[BASE 1/3] Construction du graphe de connaissances…")
    kb = KnowledgeGraphBuilder(llm=llm, embedding_model=emb, config=config)
    kg = kb.create_from_markdown_files(files)
    kg = kb.enrich_prechunked(kg, store_dir=kg_store)
    try:
        written = kb.save_doc_store(kg, kg_store)
        logging.info("[DOC_STORE] %d doc(s) dans %s", len(written), kg_store)
    except Exception as e:
        logging.warning("[DOC_STORE] %s", e)
    print(f"  → {len(getattr(kg,'nodes',[]))} nœuds, {len(getattr(kg,'relationships',[]))} relations")

    # ── OUTIL : RelationValidator ──────────────────────────────────────────────
    if p.validate_relations:
        print("[OUTIL] RelationValidator — élagage des relations faibles…")
        from tools import RelationValidator
        kg, vs = await RelationValidator(config=config)._validate_async(kg, llm, None, None)
        print(f"  → {vs.get('removed',0)}/{vs.get('total_semantic_before',0)} supprimées")

    # ── OUTIL : DirectRelationDiscovery ───────────────────────────────────────
    if p.discover_relations:
        print("[OUTIL] DirectRelationDiscovery — nouvelles relations inter-docs…")
        from tools import DirectRelationDiscovery
        kg, ds = await DirectRelationDiscovery(config=config)._discover_async(kg, llm, None)
        print(f"  → {ds.get('relations_added',0)} relations ajoutées ({ds.get('bridge_themes',0)} bridges, {ds.get('documents_processed',0)} docs)")

    # Sauvegarder le KG
    KnowledgeGraphStorage.save(kg, out / "knowledge_graph.json")
    print(f"  → KG sauvegardé : knowledge_graph.json")

    # ── BASE 2 : Génération des questions ──────────────────────────────────────
    n = min(p.num_questions, 2) if p.dry_run else p.num_questions
    print(f"\n[BASE 2/3] Génération de {n} question(s)…")

    all_personas = build_ragas_personas(config)
    # Filtrer les personas si demandé
    if p.filter_personas:
        allowed_p = set(p.filter_personas)
        personas = [pe for pe in all_personas if pe.name in allowed_p]
        if not personas:
            logging.warning("filter.personas : aucun persona valide — tous utilisés")
            personas = all_personas
    else:
        personas = all_personas

    qgen = QuestionGenerator(llm=llm, config=config)
    questions = await qgen.generate(
        kg=kg,
        persona_list=personas,
        num_questions=n,
        checkpoint_path=out / "questions_checkpoint.json",
    )
    print(f"  → {len(questions)} question(s) générée(s)")

    # ── OUTIL : QA Evaluator ──────────────────────────────────────────────────
    qa_scores = None
    if p.qa_eval and questions:
        print("[OUTIL] QA Evaluator — scoring LLM-as-judge…")
        try:
            from tools import QAEvaluator, _HAS_QA_EVAL
            if _HAS_QA_EVAL:
                scoring_llm = build_llm(os.environ.get("SCORING_MODEL", "gpt-4.1-mini"))
                ev = QAEvaluator(llm=scoring_llm, config=config)
                qa_scores = await ev.evaluate(questions)
                n_ok = sum(1 for s in qa_scores if s.get("overall", 0) >= config.evaluation.qa_eval_threshold)
                print(f"  → {n_ok}/{len(questions)} au-dessus du seuil {config.evaluation.qa_eval_threshold}")
            else:
                print("  ⚠ QAEvaluator non disponible")
        except Exception as e:
            logging.warning("[QA_EVAL] %s", e)

    # ── BASE 3 : Sauvegarde ───────────────────────────────────────────────────
    print("\n[BASE 3/3] Sauvegarde du dataset…")
    dataset = {
        "version": "2.0",
        "config":  p.yaml_file.name,
        "domain":  config.domain.name,
        "num_questions": len(questions),
        "tools_used": {
            k: getattr(p, k)
            for k in ("vlm","cosine_relations","keyphrase_relations",
                      "validate_relations","discover_relations","qa_eval","dry_run")
        },
        "filters_used": {
            "taxonomy": p.filter_taxonomy,
            "personas": p.filter_personas,
            "lengths":  p.filter_lengths,
            "styles":   p.filter_styles,
        },
        "questions": questions,
    }
    if qa_scores:
        dataset["qa_scores"] = qa_scores

    result = out / "dataset.json"
    with open(result, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    elapsed = round(time.time() - t0, 1)
    print()
    print("=" * 64)
    print(f"  ✅  Terminé en {elapsed}s")
    print(f"  📄  {len(questions)} questions → {result}")
    print("=" * 64)
    print()
    return questions


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    # Override .env si --env fourni
    if args.env:
        ep = Path(args.env)
        if ep.exists():
            _load_dotenv(ep, override=True)
        else:
            print(f"⚠  .env introuvable : {args.env}", file=sys.stderr)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Charger le YAML (brut + PipelineConfig) ───────────────────────────────
    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"❌  YAML introuvable : {yaml_path}", file=sys.stderr)
        sys.exit(1)

    raw    = read_yaml_raw(yaml_path)
    config = load_session(yaml_path)

    # ── Fusionner CLI + YAML → paramètres effectifs ───────────────────────────
    p = build_params(raw, args)

    # ── Résoudre output_dir si non fourni ────────────────────────────────────
    if p.output_dir is None:
        default_out = _DATASET_DIR / "src" / "output" / config.meta.session_id
        p.output_dir = Path(os.environ.get("OUTPUT_DIR", str(default_out)))

    # ── Lister les fichiers d'entrée ──────────────────────────────────────────
    EXTS = {".md", ".txt", ".rst", ".ipynb", ".pdf", ".docx", ".html", ".asciidoc"}
    if p.input_dir.is_dir():
        files = sorted(f for f in p.input_dir.rglob("*") if f.suffix.lower() in EXTS)
    elif p.input_dir.is_file():
        files = [p.input_dir]
    else:
        print(f"❌  Répertoire introuvable : {p.input_dir}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"❌  Aucun document dans : {p.input_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Appliquer les filtres sur la config ───────────────────────────────────
    apply_filters(config, p)

    # ── Afficher le résumé ────────────────────────────────────────────────────
    print_header(config, p, len(files), p.output_dir)

    # ── Construire LLM & embedding ────────────────────────────────────────────
    try:
        llm = build_llm()
        emb = build_embedding()
    except Exception as e:
        print(f"❌  LLM init : {e}", file=sys.stderr)
        sys.exit(1)

    # ── Lancer ───────────────────────────────────────────────────────────────
    try:
        asyncio.run(run_pipeline(config, p, llm, emb, files, p.output_dir))
    except KeyboardInterrupt:
        print("\n⚠  Interrompu.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logging.exception("Pipeline échouée")
        print(f"\n❌  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()