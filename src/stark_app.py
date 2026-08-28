"""
stark_app.py
STARK — Synthetic Training And RAG Knowledge configurator
Run: cd dataset_generator/src && streamlit run stark_app.py --server.port 8501
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import nest_asyncio
import streamlit as st

# nest_asyncio ne peut pas patcher uvloop.Loop (utilisé par Streamlit/uvicorn).
# On force la politique asyncio standard avant l'application du patch.
try:
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.DefaultEventLoopPolicy())
    nest_asyncio.apply()
except Exception:
    pass  # Dégradation gracieuse si le patch échoue

# ── Paths absolus basés sur __file__ (avant tout chargement de .env) ──────────
_SRC_DIR  = Path(__file__).resolve().parent   # dataset_generator/src/
_APP_DIR  = _SRC_DIR.parent                   # dataset_generator/

# STARK utilise toujours ses propres répertoires — indépendants du .env CLI
# (le .env a OUTPUT_DIR=./output_merged_chunks_v3 destiné au pipeline batch)
SESSIONS_DIR  = _SRC_DIR / "sessions"
OUTPUT_DIR    = _SRC_DIR / "output"       # ← toujours src/output/, jamais output_merged_chunks_v3/
DOCUMENTS_DIR = _APP_DIR / "docs"
KG_STORE_DIR  = _APP_DIR / "kg_store"     # store domaine : 1 JSON par doc, partagé entre sessions

# Surcharge possible via variables d'environnement STARK-spécifiques
# (distinctes des variables CLI du .env)
if os.environ.get("STARK_SESSIONS_DIR"):
    SESSIONS_DIR = Path(os.environ["STARK_SESSIONS_DIR"])
if os.environ.get("STARK_OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["STARK_OUTPUT_DIR"])
if os.environ.get("STARK_DOCUMENTS_DIR"):
    DOCUMENTS_DIR = Path(os.environ["STARK_DOCUMENTS_DIR"])
if os.environ.get("STARK_KG_STORE_DIR"):
    KG_STORE_DIR = Path(os.environ["STARK_KG_STORE_DIR"])

# ── Chargement du .env — uniquement les variables LLM/API ────────────────────
# On charge le .env mais on NE laisse PAS les variables de chemin (.env CLI)
# écraser les valeurs STARK déjà définies ci-dessus.
try:
    from dotenv import load_dotenv as _load_dotenv
    _ENV_PATH = _APP_DIR / ".env"
    # override=False : ne pas écraser les variables déjà définies dans l'env système
    # Cela charge OPENAI_API_KEY, OPENAI_MODEL, etc. sans toucher OUTPUT_DIR
    _load_dotenv(_ENV_PATH, override=False)
    # Force-charge uniquement les clés API si elles sont absentes
    if not os.environ.get("OPENAI_API_KEY"):
        _env_text = _ENV_PATH.read_text(encoding="utf-8") if _ENV_PATH.exists() else ""
        for _line in _env_text.splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                if _k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
                          "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL",
                          "SCORING_MODEL", "USE_KG_AGENT", "HAIKU_MODEL"):
                    _v = _v.strip().strip('"').strip("'")
                    if _v and not os.environ.get(_k):
                        os.environ[_k] = _v
except (ImportError, Exception):
    pass

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── sys.path pour les imports locaux ──────────────────────────────────────────
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Pipeline imports (dégradation gracieuse si dépendances absentes) ──────────
try:
    from session_manager import SessionManager
    from pipeline_config import PipelineConfig
    from config_agent import run_config_agent, validate_config
    _PIPELINE_OK = True
    _PIPELINE_ERR = ""
except Exception as _e:
    _PIPELINE_OK = False
    _PIPELINE_ERR = str(_e)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STARK",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* STARK brand — soft */
.stark-wordmark {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 1.85rem;
    font-weight: 300;
    letter-spacing: 0.22em;
    color: #c9d4e8;
    margin: 0;
    line-height: 1.1;
}
.stark-wordmark strong {
    font-weight: 600;
    background: linear-gradient(120deg, #93b4f0 0%, #b8a9f5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stark-tagline {
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    color: #6b7a99;
    text-transform: uppercase;
    margin-top: 3px;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.stark-tagline em {
    color: #8fa8d4;
    font-style: normal;
}
.stark-wrap {
    padding: 10px 0 6px 0;
    border-bottom: 1px solid rgba(147,180,240,0.12);
    margin-bottom: 2px;
}
.stark-sub { font-size: 0.75rem; color: #888; letter-spacing: 1px; margin-top: -8px; }

/* Session cards */
.session-card {
    border: 1px solid #2a2a3e; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 8px;
    transition: border-color 0.2s;
}
.session-card:hover { border-color: #4f8ef7; }
.session-card .domain { font-size: 0.78rem; color: #888; margin-top: 2px; }

/* Wizard steps */
.step-indicator {
    display: flex; align-items: center; gap: 8px; margin-bottom: 24px;
}
.step-dot {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
}
.step-active   { background: #4f8ef7; color: white; }
.step-done     { background: #22c55e; color: white; }
.step-inactive { background: #333; color: #888; }
.step-line     { flex: 1; height: 2px; background: #333; }
.step-line-done{ flex: 1; height: 2px; background: #22c55e; }

/* Badges */
.badge-error   { background:#ef4444; color:white; border-radius:4px; padding:2px 7px; font-size:0.75rem; }
.badge-warning { background:#f59e0b; color:white; border-radius:4px; padding:2px 7px; font-size:0.75rem; }
.badge-ok      { background:#22c55e; color:white; border-radius:4px; padding:2px 7px; font-size:0.75rem; }

/* Budget bar */
.budget-bar {
    height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #4f8ef7, #a259ff);
    margin-bottom: 4px;
}

/* Home cards */
.home-card {
    border: 1px solid #2a2a3e; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 12px;
    background: #0e1117;
    transition: border-color 0.2s, background 0.2s;
}
.home-card:hover { border-color: #4f8ef7; background: #131720; }
.home-card h4 { margin: 0 0 4px 0; font-size: 1rem; color: #e2e8f0; }
.home-card .meta { font-size: 0.78rem; color: #888; }
.home-card .stats { font-size: 0.82rem; color: #a0aec0; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Singletons ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_sm() -> "SessionManager":
    return SessionManager(sessions_dir=SESSIONS_DIR)

# ── State ─────────────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "page":             "home",
        "session_id":       None,
        "wizard_step":      1,
        "wizard_data":      {},
        "wizard_cfg_dict":  None,
        "wizard_issues":    [],
        "n_samples":        1,
        "edit_cfg":         None,
        "edit_cfg_sid":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _nav(page: str, session_id: Optional[str] = None):
    st.session_state.page = page
    if session_id is not None:
        st.session_state.session_id = session_id
    st.rerun()

# ── LLM helpers ───────────────────────────────────────────────────────────────
def _llm_config() -> Dict[str, Any]:
    return {
        "base_url": os.environ.get("OPENAI_BASE_URL"),
        "api_key":  os.environ.get("OPENAI_API_KEY", ""),
        "model":    os.environ.get("OPENAI_MODEL", "claude-haiku-4-5-20251001"),
    }

def _build_llm():
    cfg = _llm_config()
    if not cfg["api_key"]:
        return None
    try:
        import httpx
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(
            ChatOpenAI(
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                temperature=0.0,
                max_tokens=4096,
                timeout=120,
                http_client=httpx.Client(verify=False),
                http_async_client=httpx.AsyncClient(verify=False),
            )
        )
    except Exception as e:
        st.error(f"LLM indisponible : {e}")
        return None

def _build_embeddings():
    """
    Construit le modèle d'embedding (titan-embed-v2 via litellm) pour RAKG.

    Utilise EMBEDDING_BASE_URL / EMBEDDING_MODEL si définis, sinon retombe
    sur OPENAI_BASE_URL / text-embedding-ada-002. Retourne None si pas de clé API.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        import httpx
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        emb_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        emb_key = os.environ.get("EMBEDDING_API_KEY") or api_key
        return LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                base_url=emb_url,
                api_key=emb_key,
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
                http_client=httpx.Client(verify=False),
                http_async_client=httpx.AsyncClient(verify=False),
            )
        )
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
_STARK_LOGO_HTML = """
<div class="stark-wrap">
  <div class="stark-wordmark"><strong>STARK</strong></div>
  <div class="stark-tagline"><em>S</em>ynthetic <em>T</em>estset for <em>A</em>ssessing <em>R</em>etrieval <em>K</em>nowledge</div>
</div>
"""

_STARK_HERO_HTML = """
<div class="stark-wrap" style="border-bottom:none;padding-bottom:0;">
  <div class="stark-wordmark" style="font-size:2.4rem;"><strong>STARK</strong></div>
  <div class="stark-tagline" style="font-size:0.72rem;margin-top:5px;"><em>S</em>ynthetic <em>T</em>estset for <em>A</em>ssessing <em>R</em>etrieval <em>K</em>nowledge</div>
</div>
"""

def _sidebar():
    with st.sidebar:
        st.markdown(_STARK_LOGO_HTML, unsafe_allow_html=True)
        st.divider()

        # Navigation principale
        col1, col2 = st.columns(2)
        with col1:
            if st.button("＋ Nouvelle Session", use_container_width=True, type="primary"):
                st.session_state.wizard_step = 1
                st.session_state.wizard_data = {}
                st.session_state.wizard_cfg_dict = None
                st.session_state.n_samples = 1
                _nav("wizard")
        with col2:
            if st.button("📚 Documents", use_container_width=True):
                _nav("documents")

        if st.button("🏠 Accueil", use_container_width=True):
            _nav("home")

        st.markdown("---")
        st.markdown("**Sessions sauvegardées**")

        if not _PIPELINE_OK:
            st.warning(f"Pipeline indisponible :\n{_PIPELINE_ERR}")
            return

        sm = get_sm()
        sessions = sm.list_sessions()
        if not sessions:
            st.caption("Aucune session. Créez-en une ci-dessus.")
        for s in sessions:
            sid = s["session_id"]
            active = (st.session_state.page == "session"
                      and st.session_state.session_id == sid)
            border = "#4f8ef7" if active else "#2a2a3e"
            st.markdown(
                f'<div class="session-card" style="border-color:{border}">'
                f'<strong>{sid}</strong>'
                f'<div class="domain">{s.get("domain_name","?")} · '
                f'{(s.get("created_at","")[:10])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Ouvrir", key=f"open_{sid}", use_container_width=True):
                sm_cfg = sm.get(sid).to_dict()
                st.session_state.edit_cfg = sm_cfg
                st.session_state.edit_cfg_sid = sid
                _nav("session", session_id=sid)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
def _home_page():
    st.markdown(_STARK_HERO_HTML, unsafe_allow_html=True)
    st.caption("Configurez et générez des datasets QA multi-hop adaptés à vos documents.")
    st.divider()

    # Statuts rapides
    md_count = len(list(DOCUMENTS_DIR.glob("**/*.md")))
    col1, col2, col3 = st.columns(3)
    with col1:
        sessions_count = 0
        if _PIPELINE_OK:
            try:
                sessions_count = len(get_sm().list_sessions())
            except Exception:
                pass
        st.metric("Sessions sauvegardées", sessions_count)
    with col2:
        st.metric("Documents disponibles", md_count)
    with col3:
        status = "✅ OK" if _PIPELINE_OK else "❌ Manquant"
        st.metric("Pipeline", status)

    if not _PIPELINE_OK:
        st.warning(
            f"⚠️ Certains modules sont manquants : `{_PIPELINE_ERR}`\n\n"
            "La navigation et la visualisation des sessions restent disponibles."
        )

    st.divider()

    # Actions rapides
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✨ Nouvelle Session", use_container_width=True, type="primary"):
            st.session_state.wizard_step = 1
            st.session_state.wizard_data = {}
            st.session_state.wizard_cfg_dict = None
            st.session_state.n_samples = 1
            _nav("wizard")
    with col2:
        if st.button("📚 Gérer les Documents", use_container_width=True):
            _nav("documents")
    with col3:
        st.info(f"Docs : `{DOCUMENTS_DIR}`")

    # Liste des sessions existantes
    if _PIPELINE_OK:
        st.subheader("Sessions récentes")
        sm = get_sm()
        sessions = sm.list_sessions()
        if not sessions:
            st.info("Aucune session. Cliquez sur **Nouvelle Session** pour commencer.")
        else:
            for s in sessions:
                sid = s["session_id"]
                n_jobs = len(list((SESSIONS_DIR / sid / "jobs").glob("*.json"))) \
                    if (SESSIONS_DIR / sid / "jobs").exists() else 0
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f'<div class="home-card">'
                        f'<h4>⚡ {sid}</h4>'
                        f'<div class="meta">{s.get("domain_name","?")} · '
                        f'Créé le {(s.get("created_at","")[:10])}</div>'
                        f'<div class="stats">{n_jobs} job(s) · '
                        f'{s.get("description","")[:80]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.write("")
                    if st.button("Ouvrir →", key=f"home_open_{sid}",
                                 use_container_width=True, type="primary"):
                        cfg_dict = sm.get(sid).to_dict()
                        st.session_state.edit_cfg = cfg_dict
                        st.session_state.edit_cfg_sid = sid
                        _nav("session", session_id=sid)

# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD — INDICATEUR D'ÉTAPES
# ══════════════════════════════════════════════════════════════════════════════
def _wizard_steps(current: int):
    labels = ["Décrire", "Générer", "Réviser"]
    parts = []
    for i, label in enumerate(labels, 1):
        if i < current:
            dot  = f'<div class="step-dot step-done">✓</div>'
            line = '<div class="step-line-done"></div>' if i < 3 else ""
        elif i == current:
            dot  = f'<div class="step-dot step-active">{i}</div>'
            line = '<div class="step-line"></div>' if i < 3 else ""
        else:
            dot  = f'<div class="step-dot step-inactive">{i}</div>'
            line = '<div class="step-line"></div>' if i < 3 else ""
        parts.append(
            f'{dot}<span style="font-size:0.8rem;color:#aaa">{label}</span>'
        )
        if i < 3:
            parts.append(line)
    html = '<div class="step-indicator">' + "".join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD — ÉTAPE 1 : DÉCRIRE LE CORPUS
# ══════════════════════════════════════════════════════════════════════════════
def _wizard_step1():
    st.title("Nouvelle Session")
    _wizard_steps(1)
    st.subheader("1 — Décrivez votre corpus")
    st.caption(
        "Plus vous fournissez de contexte, meilleure sera la configuration générée."
    )

    # Sélection de documents existants
    md_files = sorted(DOCUMENTS_DIR.glob("**/*.md"))
    if md_files:
        with st.expander("📂 Sélectionner des documents depuis la bibliothèque", expanded=False):
            selected_docs = st.multiselect(
                "Documents disponibles",
                options=[str(f) for f in md_files],
                format_func=lambda p: Path(p).name,
                key="wizard_selected_docs",
            )
            if selected_docs and st.button("📖 Pré-remplir les échantillons depuis les docs"):
                samples_from_docs = []
                for dp in selected_docs[:3]:
                    try:
                        content = Path(dp).read_text(encoding="utf-8")
                        samples_from_docs.append(content[:800])
                    except Exception:
                        pass
                st.session_state._prefill_samples = samples_from_docs
                st.session_state.n_samples = max(st.session_state.n_samples, len(samples_from_docs))
                for i, val in enumerate(samples_from_docs):
                    st.session_state[f"sample_{i}"] = val
                st.rerun()

    with st.form("wizard_step1"):
        domain = st.text_input(
            "Nom du domaine *",
            placeholder="ex. FEniCS FEM, Kubernetes, OpenFOAM, CoSApp-Turbo…",
        )
        description = st.text_area(
            "Décrivez ce que couvrent vos documents *",
            height=130,
            placeholder=(
                "ex. Documentation technique pour FEniCS, une bibliothèque Python FEM "
                "couvrant FunctionSpace, formes variationnelles, DirichletBC, raffinement "
                "de maillage et solveurs EDP."
            ),
        )

        st.markdown("**Extraits représentatifs** *(optionnel mais fortement recommandé)*")
        st.caption(
            "Collez 1 à 3 extraits de votre documentation pour que l'agent "
            "génère des prompts encore plus précis."
        )
        samples = []
        prefill = st.session_state.get("_prefill_samples", [])
        for i in range(st.session_state.n_samples):
            default_val = prefill[i] if i < len(prefill) else ""
            s = st.text_area(
                f"Extrait {i+1}",
                height=100,
                key=f"sample_{i}",
                value=default_val,
                placeholder="Collez un extrait représentatif de votre documentation…",
            )
            if s.strip():
                samples.append(s.strip())

        col1, col2 = st.columns([1, 5])
        with col1:
            add = st.form_submit_button("＋ Extrait")
        with col2:
            quality = st.checkbox(
                "🔬 Mode optimal — adapter chaque prompt individuellement (plus lent, meilleure qualité)",
                value=True,
            )

        st.divider()
        go = st.form_submit_button("Continuer →", type="primary")

    if add:
        st.session_state.n_samples += 1
        st.rerun()

    if go:
        if not domain.strip() or not description.strip():
            st.error("Le nom du domaine et la description sont obligatoires.")
            return
        st.session_state.wizard_data = {
            "domain":      domain.strip(),
            "description": description.strip(),
            "samples":     samples,
            "quality":     quality,
        }
        st.session_state.wizard_step = 2
        st.session_state.wizard_cfg_dict = None
        st.session_state._prefill_samples = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD — ÉTAPE 2 : GÉNÉRATION IA
# ══════════════════════════════════════════════════════════════════════════════
def _wizard_step2():
    st.title("Nouvelle Session")
    _wizard_steps(2)
    st.subheader("2 — Génération de la configuration")

    if not _PIPELINE_OK:
        st.error(f"Imports pipeline échoués : {_PIPELINE_ERR}")
        if st.button("← Retour"):
            st.session_state.wizard_step = 1
            st.rerun()
        return

    llm = _build_llm()
    if not llm:
        st.error(
            "Aucun LLM disponible. "
            "Configurez `OPENAI_API_KEY` dans votre environnement ou fichier `.env`."
        )
        if st.button("← Retour"):
            st.session_state.wizard_step = 1
            st.rerun()
        return

    data = st.session_state.wizard_data

    stages = [
        "🔍 Analyse du corpus",
        "📊 Conception de la taxonomie des questions",
        "👤 Création des personas",
        "✍️ Génération des prompts adaptés au domaine",
        "✅ Validation de la configuration",
    ]

    placeholder = st.empty()
    with placeholder.container():
        for s in stages:
            st.write(f"⏳ {s}…")

    with st.spinner("Agent de configuration en cours — 30 à 90 secondes…"):
        try:
            cfg, issues = asyncio.run(run_config_agent(
                llm,
                description=data["description"],
                samples=data.get("samples", []),
            ))
            cfg_dict = cfg.to_dict()

            st.session_state.wizard_cfg_dict = cfg_dict
            st.session_state.wizard_issues   = [str(i) for i in issues]
            st.session_state.wizard_step     = 3

        except Exception as e:
            st.session_state.wizard_cfg_dict = None
            placeholder.empty()
            st.error(f"Échec de la configuration : {e}")
            with st.expander("Détails de l'erreur"):
                st.code(traceback.format_exc())
            if st.button("← Retour"):
                st.session_state.wizard_step = 1
                st.rerun()
            return

    placeholder.empty()
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD — ÉTAPE 3 : RÉVISION & PERSONNALISATION
# ══════════════════════════════════════════════════════════════════════════════
def _wizard_step3():
    st.title("Nouvelle Session")
    _wizard_steps(3)
    st.subheader("3 — Révision & personnalisation")

    cfg_dict = st.session_state.wizard_cfg_dict
    if not cfg_dict:
        st.error("Aucune configuration trouvée. Retournez à l'étape 1.")
        if st.button("← Retour"):
            st.session_state.wizard_step = 1
            st.rerun()
        return

    issues   = st.session_state.get("wizard_issues", [])
    errors   = [i for i in issues if "ERROR"   in i.upper()]
    warnings = [i for i in issues if "WARNING" in i.upper()]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Types de questions", len(cfg_dict.get("taxonomy", {}).get("types", [])))
    with col2:
        st.metric("Personas", len(cfg_dict.get("personas", [])))
    with col3:
        if errors:
            st.markdown(f'<span class="badge-error">{len(errors)} erreur(s)</span>',
                        unsafe_allow_html=True)
        elif warnings:
            st.markdown(f'<span class="badge-warning">{len(warnings)} avertissement(s)</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-ok">✓ Valide</span>', unsafe_allow_html=True)

    if errors or warnings:
        with st.expander("⚠ Problèmes de validation"):
            for i in issues:
                st.write(i)

    tabs = st.tabs([
        "🌐 Vue d'ensemble", "📊 Taxonomie", "👤 Personas",
        "✍ Prompts", "📝 Few-shots", "⚙ Avancé",
    ])
    # scope="wizard" : état des widgets isolé de celui des pages de session.
    with tabs[0]: _tab_overview_edit(cfg_dict, scope="wizard")
    with tabs[1]: _tab_taxonomy_edit(cfg_dict, scope="wizard")
    with tabs[2]: _tab_personas_edit(cfg_dict, scope="wizard")
    with tabs[3]: _tab_prompts_edit(cfg_dict, scope="wizard")
    with tabs[4]: _tab_fewshots_edit(cfg_dict, scope="wizard")
    with tabs[5]: _tab_advanced_edit(cfg_dict, scope="wizard")

    st.divider()
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        if st.button("← Regénérer", use_container_width=True):
            st.session_state.wizard_step = 1
            st.session_state.wizard_cfg_dict = None
            st.rerun()
    with col2:
        sid = st.text_input(
            "Nom de la session",
            value=cfg_dict.get("meta", {}).get("session_id", "ma_session"),
            label_visibility="collapsed",
            placeholder="Nom de la session…",
        )
    with col3:
        if st.button("💾 Sauvegarder ✓", use_container_width=True, type="primary"):
            _save_wizard_session(cfg_dict, sid.strip())

def _save_wizard_session(cfg_dict: dict, sid: str):
    if not sid:
        st.error("Entrez un nom de session.")
        return
    sm = get_sm()
    # SessionManager assainit l'id (minuscules, alphanum + tirets) au moment de
    # créer le fichier YAML. On applique la MÊME transformation ici pour que
    # session_state, la navigation et le nom réel du fichier restent cohérents
    # (sinon l'ouverture juste après échoue avec "session introuvable").
    clean_sid = SessionManager._clean_id(sid)
    if clean_sid != sid:
        st.info(f"Nom assaini : `{sid}` → `{clean_sid}` (le fichier YAML utilisera ce nom).")
    if sm.exists(clean_sid):
        st.error(f"La session '{clean_sid}' existe déjà. Choisissez un autre nom.")
        return
    try:
        cfg_dict.setdefault("meta", {})["session_id"] = clean_sid
        sm.create(cfg_dict, session_id=clean_sid)
        st.success(f"✅ Session '{clean_sid}' sauvegardée !")
        st.session_state.edit_cfg = cfg_dict
        st.session_state.edit_cfg_sid = clean_sid
        st.session_state.wizard_step = 1
        st.session_state.wizard_cfg_dict = None
        _nav("session", session_id=clean_sid)
    except Exception as e:
        st.error(f"Impossible de sauvegarder : {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  ONGLETS D'ÉDITION PARTAGÉS
# ══════════════════════════════════════════════════════════════════════════════

# ── Isolation de l'état des widgets par session ───────────────────────────────
# Streamlit mémorise la valeur d'un widget par sa `key` et, quand cette key
# existe déjà dans st.session_state, IGNORE le paramètre `value=`. Les onglets
# d'édition ci-dessous utilisaient des keys identiques pour TOUTES les sessions
# (`_pn_0`, `_qt_name_0`, `_bgt_<type>`…). En ouvrant la session A puis la
# session B dans le même onglet navigateur, les widgets de B réaffichaient les
# valeurs mémorisées de A, puis les lignes `personas[idx] = {...}` /
# `types[idx] = {...}` réécrivaient ces valeurs dans le cfg_dict de B (muté
# in-place, donc aussi dans st.session_state.edit_cfg). Un job lancé ensuite
# tournait avec la config d'une AUTRE session.
# Bug observé : job `96373f70` de la session `dungeons___dragons_5e_draft`
# exécuté avec les 5 personas « Turbomachinery … » et les 5 types de questions
# de la session `cosapp_x`.
# Correctif : préfixer toutes les keys par un `scope` propre à la session (ou au
# wizard), ce qui rend l'état des widgets étanche entre sessions.
def _scope_key(scope: str, name: str) -> str:
    return f"{scope}__{name}"


def _tab_overview_edit(cfg: dict, scope: str = "wizard"):
    k = lambda n: _scope_key(scope, n)
    domain = cfg.get("domain", {})
    st.text_input("Nom du domaine",       value=domain.get("name", ""),        key=k("_ov_name"))
    st.text_area("Description",           value=domain.get("description", ""), key=k("_ov_desc"), height=100)
    st.text_input("Langue (ISO-639)",     value=domain.get("language", "en"),  key=k("_ov_lang"))
    vocab = domain.get("domain_vocabulary", [])
    st.text_area(
        "Vocabulaire du domaine (un terme par ligne)",
        value="\n".join(vocab), key=k("_ov_vocab"), height=120,
    )
    if st.button("Appliquer les modifications", key=k("_ov_apply")):
        cfg.setdefault("domain", {})
        cfg["domain"]["name"]              = st.session_state[k("_ov_name")]
        cfg["domain"]["description"]       = st.session_state[k("_ov_desc")]
        cfg["domain"]["language"]          = st.session_state[k("_ov_lang")]
        cfg["domain"]["domain_vocabulary"] = [
            t.strip() for t in st.session_state[k("_ov_vocab")].splitlines() if t.strip()
        ]
        st.success("Vue d'ensemble mise à jour.")

def _tab_taxonomy_edit(cfg: dict, scope: str = "wizard"):
    k = lambda n: _scope_key(scope, n)
    tax    = cfg.get("taxonomy", {})
    types  = tax.get("types", [])
    budget = tax.get("budget", {})

    st.markdown("**Types de questions**")
    st.caption("Modifiez les noms et descriptions. Le budget s'ajuste automatiquement.")

    for idx, qt in enumerate(types):
        c1, c2 = st.columns([2, 5])
        with c1:
            new_name = st.text_input(
                "Nom", value=qt["name"], key=k(f"_qt_name_{idx}"),
                label_visibility="collapsed",
            )
        with c2:
            new_desc = st.text_input(
                "Description", value=qt.get("description", ""),
                key=k(f"_qt_desc_{idx}"), label_visibility="collapsed",
            )
        types[idx] = {"name": new_name, "description": new_desc}

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Ajouter un type", key=k("_qt_add")):
            types.append({"name": "nouveau_type", "description": "Description…"})
            cfg.setdefault("taxonomy", {})["types"] = types
            st.rerun()
    with col2:
        if len(types) > 1 and st.button("➖ Supprimer le dernier", key=k("_qt_rm")):
            types.pop()
            cfg.setdefault("taxonomy", {})["types"] = types
            st.rerun()

    st.divider()
    st.markdown("**Budget** (doit sommer à 1.0)")
    type_names = [qt["name"] for qt in types]

    new_budget: Dict[str, float] = {}
    total = 0.0
    for name in type_names:
        default = float(budget.get(name, 1.0 / max(len(type_names), 1)))
        val = st.number_input(
            name, min_value=0.0, max_value=1.0,
            value=round(default, 3), step=0.05,
            format="%.3f", key=k(f"_bgt_{name}"),
        )
        new_budget[name] = val
        total += val
        pct = int(val * 100)
        st.markdown(
            f'<div class="budget-bar" style="width:{max(pct,2)}%"></div>',
            unsafe_allow_html=True,
        )

    color = "#22c55e" if abs(total - 1.0) < 0.02 else "#ef4444"
    st.markdown(
        f"**Total : <span style='color:{color}'>{total:.3f}</span>**",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Auto-normaliser", key=k("_bgt_norm")):
            if total > 0:
                # NB: on n'utilise PAS `k` comme variable de boucle ici — `k` est
                # le helper de scoping des keys de widgets défini plus haut.
                new_budget = {_n: round(_v / total, 6) for _n, _v in new_budget.items()}
                diff = 1.0 - sum(new_budget.values())
                first = next(iter(new_budget))
                new_budget[first] = round(new_budget[first] + diff, 6)
            cfg.setdefault("taxonomy", {})["budget"] = new_budget
            cfg["taxonomy"]["types"]  = types
            st.success("Budget normalisé.")
            st.rerun()
    with col2:
        if st.button("Sauvegarder la taxonomie", key=k("_tax_save"), type="primary"):
            cfg.setdefault("taxonomy", {})["types"]  = types
            cfg["taxonomy"]["budget"] = new_budget
            st.success("Taxonomie sauvegardée.")

def _tab_personas_edit(cfg: dict, scope: str = "wizard"):
    k = lambda n: _scope_key(scope, n)
    personas = cfg.get("personas", [])
    st.markdown("**Personas**")
    for idx, p in enumerate(personas):
        with st.expander(f"👤 {p.get('name', f'Persona {idx+1}')}", expanded=False):
            n = st.text_input("Nom",       value=p.get("name", ""),              key=k(f"_pn_{idx}"))
            r = st.text_area("Description du rôle", value=p.get("role_description", ""),
                             key=k(f"_prole_{idx}"), height=160)
            personas[idx] = {"name": n, "role_description": r}
            if st.button("Supprimer", key=k(f"_prm_{idx}")):
                personas.pop(idx)
                cfg["personas"] = personas
                st.rerun()

    if st.button("➕ Ajouter un persona", key=k("_pa_add")):
        personas.append({"name": "Nouveau Persona", "role_description": "Décrivez le rôle ici…"})
        cfg["personas"] = personas
        st.rerun()

    if st.button("💾 Sauvegarder les personas", key=k("_p_save"), type="primary"):
        cfg["personas"] = personas
        st.success("Personas sauvegardés.")

def _tab_prompts_edit(cfg: dict, scope: str = "wizard"):
    k = lambda n: _scope_key(scope, n)
    prompts = cfg.get("prompts", {})

    # Seuls les 3 prompts session-spécifiques sont éditables.
    # Les 10 autres sont universels (hardcodés, GEPA-optimized).
    labels = {
        "keyphrase_extractor":     "🔑 Extracteur de keyphrases (domaine-spécifique)",
        "qualify_system":          "✅ Qualification — système (domaine-spécifique)",
        "qualify_user_template":   "📝 Qualification — template utilisateur (domaine-spécifique)",
    }

    if not prompts:
        st.info("Aucun prompt dans cette configuration.")
        return

    st.caption(
        "Seuls les prompts adaptés au domaine sont éditables. "
        "Les 10 autres (génération, évaluation, KG) sont universels et optimisés par GEPA."
    )

    changed = {}
    for key, label in labels.items():
        if key not in prompts:
            continue
        with st.expander(f"✍ {label}", expanded=False):
            val = st.text_area(
                label, value=prompts[key], height=300,
                key=k(f"_prompt_{key}"), label_visibility="collapsed",
            )
            changed[key] = val
            # Validations
            if key == "qualify_user_template":
                if "{context_1}" not in val:
                    st.warning("⚠ Placeholder `{context_1}` manquant !")
                if "{context_2}" not in val:
                    st.warning("⚠ Placeholder `{context_2}` manquant !")

    if st.button("💾 Sauvegarder les prompts", key=k("_prall_save"), type="primary"):
        cfg.setdefault("prompts", {}).update(changed)
        st.success("Prompts sauvegardés.")

def _tab_fewshots_edit(cfg: dict, scope: str = "wizard"):
    k = lambda n: _scope_key(scope, n)
    fs = cfg.get("few_shots", {})
    keys = [
        "qa_evaluator", "query_generation", "answer_generation", "two_hop_bridge",
        "relation_validator", "doc_theme", "cross_doc_map", "chunk_locator",
        "direct_pair_validator", "keyphrase_extractor",
    ]
    st.caption(
        "Les few-shots sont stockés au format JSON `[{\"input\":{…}, \"output\":{…}}, …]`. "
        "Modifiez directement. Laissez `[]` pour n'en utiliser aucun."
    )
    for key in keys:
        with st.expander(key.replace("_", " ").title(), expanded=False):
            raw    = json.dumps(fs.get(key, []), ensure_ascii=False, indent=2)
            edited = st.text_area(
                key, value=raw, height=200,
                key=k(f"_fs_{key}"), label_visibility="collapsed",
            )
            if st.button("Sauvegarder", key=k(f"_fs_save_{key}")):
                try:
                    parsed = json.loads(edited)
                    cfg.setdefault("few_shots", {})[key] = parsed
                    st.success(f"{key} sauvegardé ({len(parsed)} exemple(s)).")
                except json.JSONDecodeError as e:
                    st.error(f"JSON invalide : {e}")

def _tab_advanced_edit(cfg: dict, scope: str = "wizard"):
    k = lambda n: _scope_key(scope, n)
    st.caption(
        "Paramètres avancés — les valeurs par défaut sont soigneusement calibrées. "
        "Ne les modifiez que si vous savez ce que vous faites."
    )
    ev  = cfg.get("evaluation", {})
    kg  = cfg.get("kg_enrichment", {})
    ch  = cfg.get("chunking", {})
    mdl = cfg.get("models", {})
    qp  = cfg.get("query_params", {})

    _ALL_STYLES  = ["perfect_grammar", "web_search_like", "misspelled", "poor_grammar"]
    _ALL_LENGTHS = ["long", "medium", "short"]

    with st.expander("Styles & longueurs de requêtes", expanded=False):
        st.caption(
            "Laissez vide pour utiliser les valeurs par défaut de la pipeline "
            "(`perfect_grammar` + `web_search_like`, longueurs `long / medium / short`)."
        )
        active_styles = st.multiselect(
            "Styles de requête actifs",
            options=_ALL_STYLES,
            default=qp.get("styles") or [],
            format_func=lambda s: {
                "perfect_grammar":  "Perfect grammar",
                "web_search_like":  "Web search like queries",
                "misspelled":       "Misspelled queries",
                "poor_grammar":     "Poor grammar",
            }.get(s, s),
            help="Vide = défaut pipeline (perfect_grammar + web_search_like)",
            key=k("_qp_styles"),
        )
        active_lengths = st.multiselect(
            "Longueurs de requête actives",
            options=_ALL_LENGTHS,
            default=qp.get("lengths") or [],
            format_func=lambda l: {"long": "Long (≥ 20 mots)", "medium": "Medium (10-19 mots)", "short": "Short (≤ 9 mots)"}.get(l, l),
            help="Vide = défaut pipeline (long + medium + short)",
            key=k("_qp_lengths"),
        )
        if st.button("Sauvegarder les styles/longueurs", key=k("_qp_save")):
            cfg.setdefault("query_params", {})
            cfg["query_params"]["styles"]  = active_styles
            cfg["query_params"]["lengths"] = active_lengths
            st.success("Styles et longueurs sauvegardés.")

    with st.expander("Évaluation", expanded=False):
        ev["qa_eval_threshold"] = st.slider(
            "Seuil d'évaluation QA", 0.0, 1.0,
            float(ev.get("qa_eval_threshold", 0.6)), 0.05,
            key=k("_adv_qa_thr"),
        )
        ev["max_retry"] = st.number_input(
            "Nb max de retries", 0, 10, int(ev.get("max_retry", 2)),
            key=k("_adv_max_retry"),
        )
        ev["max_context_chars"] = st.number_input(
            "Nb max de chars de contexte", 1000, 50000,
            int(ev.get("max_context_chars", 12000)), 1000,
            key=k("_adv_max_ctx"),
        )

    with st.expander("Enrichissement KG", expanded=False):
        kg["cosine_sim_min"] = st.slider(
            "Similarité cosine min", 0.0, 1.0, float(kg.get("cosine_sim_min", 0.6)), 0.05,
            key=k("_adv_cos_min"),
        )
        kg["cosine_sim_max"] = st.slider(
            "Similarité cosine max", 0.0, 1.0, float(kg.get("cosine_sim_max", 0.9)), 0.05,
            key=k("_adv_cos_max"),
        )
        kg["idf_threshold"] = st.number_input(
            "Seuil IDF", 0.0, 10.0, float(kg.get("idf_threshold", 0.693)), 0.1,
            key=k("_adv_idf"),
        )

    with st.expander("Chunking", expanded=False):
        ch["max_tokens"] = st.number_input(
            "Tokens max/chunk", 100, 8192, int(ch.get("max_tokens", 1024)), 128,
            key=k("_adv_max_tok"),
        )
        ch["min_chunk_tokens"] = st.number_input(
            "Tokens min/chunk", 10, 500, int(ch.get("min_chunk_tokens", 50)), 10,
            key=k("_adv_min_tok"),
        )




    if st.button("💾 Sauvegarder les paramètres avancés", key=k("_adv_save"), type="primary"):
        cfg["evaluation"]    = ev
        cfg["kg_enrichment"] = kg
        cfg["chunking"]      = ch
        cfg["models"]        = mdl
        st.success("Paramètres avancés sauvegardés.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE SESSION
# ══════════════════════════════════════════════════════════════════════════════
def _session_page(sid: str):
    if not _PIPELINE_OK:
        st.error(f"Pipeline indisponible : {_PIPELINE_ERR}")
        return

    sm = get_sm()
    if not sm.exists(sid):
        st.error(f"Session '{sid}' introuvable.")
        if st.button("← Accueil"):
            _nav("home")
        return

    # `edit_cfg` est un cache mémoire UNIQUE pour toute l'app. Il faut donc
    # vérifier qu'il correspond bien à la session affichée : sinon on éditerait
    # (et on lancerait un job avec) la config d'une autre session. On mémorise le
    # session_id associé au cache et on recharge depuis le YAML dès qu'il diffère.
    cfg_dict = st.session_state.edit_cfg
    if cfg_dict is None or st.session_state.get("edit_cfg_sid") != sid:
        cfg_dict = sm.get(sid).to_dict()
        st.session_state.edit_cfg = cfg_dict
        st.session_state.edit_cfg_sid = sid

    # En-tête
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    with col1:
        domain  = cfg_dict.get("domain", {}).get("name", sid)
        created = cfg_dict.get("meta", {}).get("created_at", "")[:10]
        st.title(f"⚡ {sid}")
        st.caption(f"{domain} · Créé le {created}")
    with col2:
        if st.button("💾 Sauver", use_container_width=True, type="primary"):
            _persist_session(sid, cfg_dict, sm)
    with col3:
        if st.button("🔀 Dupliquer", use_container_width=True):
            new_id = f"{sid}_fork"
            try:
                sm.fork(sid, new_id=new_id)
                st.success(f"Dupliqué → {new_id}")
            except Exception as e:
                st.error(str(e))
    with col4:
        if st.button("🗑 Supprimer", use_container_width=True):
            sm.delete(sid)
            st.session_state.edit_cfg = None
            _nav("home")

    # Bandeau de validation
    try:
        cfg_obj = PipelineConfig.from_dict(cfg_dict)
        issues  = validate_config(cfg_obj)
        n_err   = sum(1 for i in issues if i.severity == "error")
        n_warn  = sum(1 for i in issues if i.severity == "warning")
        if n_err:
            st.error(f"⚠ {n_err} erreur(s) — corrigez avant de générer.")
        elif n_warn:
            st.warning(f"{n_warn} avertissement(s).")
        else:
            st.success("✓ Configuration valide.")
    except Exception:
        pass

    st.divider()
    tabs = st.tabs([
        "🌐 Vue d'ensemble", "📊 Taxonomie", "👤 Personas",
        "✍ Prompts", "📝 Few-shots", "⚙ Avancé",
        "🚀 Générer", "📦 Résultats", "🔍 YAML",
    ])

    # scope unique par session : empêche les valeurs de widgets d'une session de
    # « fuiter » dans une autre (et donc d'écraser sa config avant un job).
    _scope = f"sess_{sid}"
    with tabs[0]: _tab_overview_edit(cfg_dict, scope=_scope)
    with tabs[1]: _tab_taxonomy_edit(cfg_dict, scope=_scope)
    with tabs[2]: _tab_personas_edit(cfg_dict, scope=_scope)
    with tabs[3]: _tab_prompts_edit(cfg_dict, scope=_scope)
    with tabs[4]: _tab_fewshots_edit(cfg_dict, scope=_scope)
    with tabs[5]: _tab_advanced_edit(cfg_dict, scope=_scope)
    with tabs[6]: _tab_generate(sid, cfg_dict)
    with tabs[7]: _tab_results(sid)
    with tabs[8]: _tab_inspect_yaml(sid, sm)

def _tab_inspect_yaml(sid: str, sm: "SessionManager"):
    """Affiche le contenu YAML brut réellement persisté sur disque pour la session.

    Permet de vérifier/inspecter ce qui a été écrit (utile pour diagnostiquer les
    problèmes de création : id assaini, champs manquants…). Le YAML est lu depuis
    le fichier `sessions/{sid}.yaml`, indépendamment de `edit_cfg` en mémoire.
    """
    st.subheader("Inspecter le YAML de la session")
    yaml_path = SESSIONS_DIR / f"{sid}.yaml"
    st.caption(f"Fichier : `{yaml_path}`")

    if not yaml_path.exists():
        st.error(
            f"Fichier YAML introuvable : `{yaml_path}`.\n\n"
            "La session n'a peut-être pas été correctement persistée sur disque."
        )
        return

    try:
        raw_yaml = yaml_path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Impossible de lire le YAML : {e}")
        return

    n_lines = len(raw_yaml.splitlines())
    size_kb = round(len(raw_yaml.encode("utf-8")) / 1024, 1)
    st.caption(f"{n_lines} lignes · {size_kb} KB")

    st.code(raw_yaml, language="yaml")

    st.download_button(
        "📥 Télécharger le YAML",
        data=raw_yaml.encode("utf-8"),
        file_name=f"{sid}.yaml",
        mime="text/yaml",
        key=f"dl_yaml_{sid}",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Recharger la config depuis le disque", key=f"reload_yaml_{sid}"):
            # Recharge la config depuis le YAML disque et remplace edit_cfg en mémoire
            # (permet de repartir de l'état réellement persisté après une édition externe).
            try:
                st.session_state.edit_cfg = sm.get(sid).to_dict()
                st.session_state.edit_cfg_sid = sid
                st.success("Config rechargée depuis le disque.")
                st.rerun()
            except Exception as e:
                st.error(f"Rechargement impossible : {e}")

    # ── Resynchronisation des 11 prompts universels ──────────────────────────
    st.divider()
    st.markdown("**⚠️ Prompts universels obsolètes ?**")
    st.caption(
        "Les 11 prompts universels (query_generation, answer_generation, two_hop_bridge, "
        "qa_evaluator, "
        "relation_validator, etc.) sont figés dans le code mais **copiés dans le YAML** "
        "à la création de la session. Si le code a évolué depuis, la copie stockée est "
        "périmée. Ce bouton réinjecte les versions actuelles **sans toucher** aux 3 "
        "prompts spécifiques au domaine (keyphrase_extractor, qualify_system, qualify_user_template)."
    )

    # Détecter la dérive : comparer les prompts stockés aux prompts actuels du code
    try:
        from config_agent import universal_prompts as _universal_prompts
        stored_prompts = (st.session_state.edit_cfg or {}).get("prompts", {})
        current_universal = _universal_prompts()
        drifted = [
            k for k, v in current_universal.items()
            if stored_prompts.get(k) != v
        ]
        has_legacy = "query_answer_generation" in stored_prompts
        if drifted or has_legacy:
            msg = []
            if drifted:
                msg.append(f"{len(drifted)} prompt(s) universel(s) obsolète(s) : {', '.join(drifted)}")
            if has_legacy:
                msg.append("champ legacy `query_answer_generation` présent (sera migré)")
            st.warning("• " + "\n\n• ".join(msg))
        else:
            st.success("✓ Les prompts universels sont à jour.")
    except Exception as _e:
        st.caption(f"(Détection de dérive indisponible : {_e})")

    with col_b:
        pass

    if st.button("♻️ Resynchroniser les prompts universels", key=f"resync_yaml_{sid}",
                 type="primary"):
        try:
            from config_agent import resync_universal_prompts
            cfg_dict = st.session_state.edit_cfg or sm.get(sid).to_dict()
            new_cfg_dict, changed = resync_universal_prompts(cfg_dict)
            # Persiste immédiatement sur disque via PipelineConfig pour valider le schéma
            cfg_obj = PipelineConfig.from_dict(new_cfg_dict)
            cfg_obj.meta.session_id = sid
            sm.save(cfg_obj)
            st.session_state.edit_cfg = cfg_obj.to_dict()
            if changed:
                st.success(
                    f"✅ {len(changed)} prompt(s) universel(s) resynchronisé(s) et sauvegardé(s) : "
                    f"{', '.join(changed)}"
                )
            else:
                st.info("Aucun changement nécessaire — déjà à jour.")
            st.rerun()
        except Exception as e:
            st.error(f"Resynchronisation impossible : {e}")


def _persist_session(sid: str, cfg_dict: dict, sm: "SessionManager"):
    try:
        cfg_obj = PipelineConfig.from_dict(cfg_dict)
        cfg_obj.meta.session_id = sid
        sm.save(cfg_obj)
        st.success("Session sauvegardée.")
    except Exception as e:
        st.error(f"Échec de la sauvegarde : {e}")

def _mark_job_interrupted(sid: str, job_id: str):
    """Marque un job zombie (running sans thread actif) comme interrompu."""
    p = SESSIONS_DIR / sid / "jobs" / f"{job_id}.json"
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        data["status"] = "failed"
        data["finished_at"] = datetime.now().isoformat()
        data.setdefault("errors", []).append({
            "type": "Interrupted",
            "message": "Job interrompu (app redémarrée ou thread tué).",
            "traceback": "",
        })
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET GÉNÉRER
# ══════════════════════════════════════════════════════════════════════════════
def _tab_generate(sid: str, cfg_dict: dict):
    st.subheader("Générer des questions")

    all_docs = sorted(
        f for glob in _SUPPORTED_GLOBS
        for f in DOCUMENTS_DIR.glob(f"**/{glob}")
        if "_originals" not in f.parts
    )
    if not all_docs:
        st.warning(
            f"Aucun document dans `{DOCUMENTS_DIR}`. "
            "Uploadez des documents dans la **Bibliothèque de Documents** d'abord."
        )
        if st.button("📚 Aller à la bibliothèque"):
            _nav("documents")
        return

    non_md = [f for f in all_docs if f.suffix.lower() not in {".md", ".markdown"}]
    if non_md:
        st.info(
            f"ℹ️ {len(non_md)} fichier(s) non-Markdown détecté(s) "
            f"({', '.join(sorted({f.suffix for f in non_md}))}) — "
            "ils seront convertis automatiquement via **Docling Serve** avant la génération."
        )

    file_paths = [str(f) for f in all_docs]

    selected = st.multiselect(
        "Documents à inclure",
        options=file_paths,
        default=file_paths,
        format_func=lambda p: f"{_MARKDOWN_ICON.get(Path(p).suffix.lower().lstrip('.'), '📄')} {Path(p).name}",
    )


    num_q = st.number_input(
        "Nombre de questions à générer",
        min_value=5, max_value=5000, value=50, step=5,
    )

    # ── Mode multi-hop ─────────────────────────────────────────────────────
    hop_mode = st.radio(
        "Mode multi-hop",
        options=["inter_doc", "intra_doc"],
        format_func=lambda x: (
            "🔗 Inter-documents (strict) — questions entre fichiers différents"
            if x == "inter_doc"
            else "📄 Intra-document autorisé — questions dans un même fichier"
        ),
        horizontal=True,
        index=1,  # Par défaut intra-doc pour éviter le piège du 0-questions
        help=(
            "**Inter-documents** : les questions relient des chunks de fichiers "
            "différents (nécessite ≥ 2 documents). "
            "**Intra-document** : autorise aussi les questions reliant des chunks "
            "du même fichier (fonctionne avec 1 seul document)."
        ),
    )
    inter_doc_only = (hop_mode == "inter_doc")

    st.divider()
    enable_ragas_eval = st.checkbox(
        "🧪 QA Eval (RAGAS + 2-Hop judge)",
        value=False,
        help=(
            "Active les 3 juges d'évaluation sur chaque question générée : "
            "**ResponseGroundedness** (réponse ancrée dans les contextes), "
            "**AnswerAccuracy** (cohérence réponse / référence LLM sans contexte), "
            "**TwoHopJudge** (question et réponse exploitent vraiment les 2 contextes). "
            "Ralentit la génération (~2-5s par question)."
        ),
    )
    if enable_ragas_eval:
        _ev = (cfg_dict.get("evaluation") or {})
        _default_retries = int(_ev.get("max_retry", 2))
        _inline_max_retry = st.number_input(
            "↳ Retries QA Eval max",
            min_value=0, max_value=10,
            value=_default_retries,
            step=1,
            help=(
                "Nombre de tentatives de régénération si une question échoue l'évaluation QA. "
                "0 = pas de retry (on garde la première génération). "
                "Si la **question** est jugée single-hop, la question ET la réponse sont régénérées. "
                "Si seule la **réponse** est fautive, la question est conservée et seule la réponse est retentée. "
                "Surcharge temporairement le paramètre `evaluation.max_retry` de la config de session."
            ),
        )
        # Write back into cfg_dict so the launch thread picks it up
        cfg_dict.setdefault("evaluation", {})["max_retry"] = _inline_max_retry

    # ── Modules d'enrichissement du KG ──────────────────────────────────────
    _em = (cfg_dict.get("enrich_modules") or {})
    with st.expander("⚙️ Modules d'enrichissement du KG", expanded=False):
        st.caption(
            "Activez/désactivez chaque module avant le lancement. "
            "Les modules coûteux en LLM sont désactivés par défaut."
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            _em_keyphrases = st.checkbox(
                "🔑 Keyphrases",
                value=bool(_em.get("keyphrases_overlap", True)),
                help="Relations entre chunks basées sur les mots-clés (TF-IDF). Rapide, aucun appel LLM.",
            )
            _em_cosine = st.checkbox(
                "🧲 Similarité cosine (embeddings)",
                value=bool(_em.get("cosine_similarity", False)),
                help=(
                    "Relations **cosine_similarity** entre chunks, calculées sur le contenu complet "
                    "vectorisé par le modèle d'embedding. Seuils réglables dans l'onglet ⚙ Avancé. "
                    "0 appel LLM."
                ),
            )
            _em_kg_agent = st.checkbox(
                "🤖 KG Agent (inter-documents)",
                value=bool(_em.get("kg_agent", False)),
                help=(
                    "Relations **agent_discovered** via `DirectRelationDiscovery` (inter-documents). "
                    "Nécessite plusieurs appels LLM — coûteux mais produit des relations riches."
                ),
            )
        with col_m2:
            _em_validator = st.checkbox(
                "✅ Validation des relations",
                value=bool(_em.get("relation_validator", True)),
                help="Valide les relations existantes via `RelationValidator`.",
            )
            _em_univ_triplet_relations = st.checkbox(
                "🔺 Relations llm_triplet",
                value=bool(_em.get("universal_triplet_relations", False)),
                help=(
                    "Extrait des triplets {concept, relation, concept} par chunk (LLM), "
                    "puis crée des relations **llm_triplet** entre chunks partageant un concept-pont "
                    "(1 appel LLM de validation par concept). "
                    "Meilleur type de relation multi-hop mesuré (qualité ~0.95). Coûteux en LLM."
                ),
            )

        # Write back enrichment module flags into cfg_dict
        cfg_dict["enrich_modules"] = {
            "keyphrases_overlap": _em_keyphrases,
            "cosine_similarity": _em_cosine,
            "kg_agent": _em_kg_agent,
            "relation_validator": _em_validator and _em_kg_agent,
            "frame_bridge": False,
            "retrospective": False,
            "qa_eval": enable_ragas_eval,
            "universal_triplets": _em_univ_triplet_relations,
            "universal_triplet_relations": _em_univ_triplet_relations,
            "graphify": False,
        }

    # ── (Graphify supprimé de l'UI) ───────────────────────────────────────────
    cfg_dict["graphify_reuse_path"] = None
    _em_graphify = False  # used below only to satisfy _em_graphify reference
    if False:  # block kept for structural compatibility, never executed
        _em_graphify = st.checkbox(
            "🕸️ Activer Graphify",
            value=False,
            key="_em_graphify_standalone",
            help=(
                "Lance l'ÉTAPE 3.7/4 après le rétrospectif. "
                "Non-bloquant : un échec Graphify n'interrompt pas la génération."
            ),
        )

        _graphify_reuse_path = None
        if _em_graphify:
            # ── Source du graphe : reconstruire (LLM) ou réutiliser un existant ──
            # On liste les graphify_graph.json (graphe Graphify NATIF, pas le snapshot
            # KG RAGAS kg_04_graphify.json) dans src/output/*/.
            _existing_graphs = []
            for _gp in sorted(
                list(OUTPUT_DIR.glob("*/graphify_graph.json"))
                + list(OUTPUT_DIR.glob("*/graphify_graph.json"))
            ):
                if _gp in [_x[0] for _x in _existing_graphs]:  # dédupliquer
                    continue
                try:
                    _st = _gp.stat()
                    _sess = _gp.parent.name
                    _size_mb = round(_st.st_size / (1024 * 1024), 1)
                    _mtime = datetime.fromtimestamp(_st.st_mtime).strftime("%Y-%m-%d %H:%M")
                    _existing_graphs.append((str(_gp), f"{_sess} · {_mtime} · {_size_mb} MB (graphe natif)"))
                except Exception:
                    pass

            _mode = st.radio(
                "Source du graphe Graphify",
                options=["rebuild", "reuse"],
                format_func=lambda m: (
                    "🔨 Reconstruire (extraction LLM, coûteux)"
                    if m == "rebuild"
                    else "♻️ Réutiliser un graphe existant (rapide, 0 LLM)"
                ),
                horizontal=True,
                index=0,
                key="_gfy_source_mode",
                help=(
                    "Réutiliser un graphe déjà produit évite de relancer l'extraction LLM. "
                    "Le mapping chunk↔relation est recalculé automatiquement pour le corpus courant "
                    "(fonctionne tant que les mêmes documents/chunks sont présents)."
                ),
            )

            if _mode == "reuse":
                if _existing_graphs:
                    _labels = [lbl for _, lbl in _existing_graphs]
                    _paths = [p for p, _ in _existing_graphs]
                    _sel_idx = st.selectbox(
                        "Graphe Graphify à réutiliser",
                        options=list(range(len(_paths))),
                        format_func=lambda i: _labels[i],
                        key="_gfy_reuse_select",
                    )
                    _graphify_reuse_path = _paths[_sel_idx]
                    st.caption(f"📂 `{_graphify_reuse_path}`")
                else:
                    st.info(
                        "Aucun graphe Graphify existant trouvé dans "
                        f"`{OUTPUT_DIR}/*/kg_04_graphify.json`. "
                        "Lancez d'abord une génération en mode « Reconstruire »."
                    )

                # Import manuel d'un graphe depuis un autre emplacement
                _uploaded_graph = st.file_uploader(
                    "…ou importer un graphe Graphify (JSON)",
                    type=["json"],
                    key="_gfy_reuse_upload",
                    help="Fichier kg_04_graphify.json ou graph.json (format NetworkX).",
                )
                if _uploaded_graph is not None:
                    _imp_dir = OUTPUT_DIR / sid
                    _imp_dir.mkdir(parents=True, exist_ok=True)
                    _imp_path = _imp_dir / "kg_04_graphify_imported.json"
                    _imp_path.write_bytes(_uploaded_graph.read())
                    _graphify_reuse_path = str(_imp_path)
                    st.success(f"✅ Graphe importé : `{_imp_path.name}`")

        # Propagate into cfg_dict
        cfg_dict.setdefault("enrich_modules", {})["graphify"] = _em_graphify
        # Chemin de réutilisation (None = reconstruire). Lu par l'étape 3.7.
        cfg_dict["graphify_reuse_path"] = _graphify_reuse_path

    if inter_doc_only and len(selected) < 2:
        st.warning(
            "⚠️ Le mode **inter-documents** nécessite au moins **2 fichiers** sélectionnés. "
            "Ajoutez des documents ou passez en mode **intra-document autorisé**."
        )

    st.divider()
    nohup_mode = st.checkbox(
        "🛡️ Mode nohup (résiste à la mise en veille / déconnexion)",
        value=False,
        help=(
            "Lance la génération dans un processus détaché (`nohup`). "
            "Le job continue même si votre PC se met en veille ou si la connexion VSCode est coupée. "
            "Sans cette option, le job est un thread dans le process Streamlit et s'arrête avec lui."
        ),
    )

    job_key = f"job_{sid}"

    # ── Reprise après redémarrage : si pas de job en session_state,
    # chercher le dernier job sur disque (nohup ou thread interrompu)
    if not st.session_state.get(job_key):
        jobs_dir = SESSIONS_DIR / sid / "jobs"
        if jobs_dir.exists():
            candidates = sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in candidates:
                try:
                    d = json.loads(p.read_text())
                    if d.get("session_id") == sid:
                        st.session_state[job_key] = d["job_id"]
                        break
                except Exception:
                    pass

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🚀 Lancer la génération", use_container_width=True,
            type="primary", disabled=not selected,
        ):
            job_id = str(uuid.uuid4())[:8]
            _write_initial_job(
                sid, job_id, selected, num_q,
                cfg_dict=cfg_dict,
                inter_doc_only=inter_doc_only,
                enable_ragas_eval=enable_ragas_eval,
                output_dir=str(OUTPUT_DIR / sid),
                kg_store_dir=str(KG_STORE_DIR),
            )
            if nohup_mode:
                _launch_nohup(sid, job_id)
            else:
                t = threading.Thread(
                    target=_generation_thread,
                    args=(sid, job_id, selected, num_q, cfg_dict, inter_doc_only,
                          enable_ragas_eval),
                    daemon=True,
                )
                t.start()
            st.session_state[job_key] = job_id
            mode_label = "nohup (détaché)" if nohup_mode else "arrière-plan"
            st.success(f"✅ Job `{job_id}` lancé en {mode_label}.")
            st.rerun()
    with col2:
        if st.button("🔄 Actualiser le statut", use_container_width=True):
            st.rerun()

    job_id = st.session_state.get(job_key)
    if job_id:
        # Affichage auto-rafraîchi via st.fragment : ne bloque plus le thread
        # serveur (fini le time.sleep) et ne relance plus toute la page (fini le
        # saut d'onglet). Seul le fragment se rafraîchit toutes les 3 s.
        _render_job_status_live(sid, job_id, zombie_key_prefix="zombie")

def _write_initial_job(
    sid: str, job_id: str, files: List[str], n: int,
    cfg_dict: Optional[dict] = None,
    inter_doc_only: bool = False,
    enable_ragas_eval: bool = False,
    output_dir: Optional[str] = None,
    kg_store_dir: Optional[str] = None,
):
    jobs_dir = SESSIONS_DIR / sid / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "job_id": job_id, "session_id": sid,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "started_at": None, "finished_at": None,
        "progress": {"done": 0, "total": n},
        "input_files": files, "num_questions_requested": n,
        "num_questions_generated": 0, "errors": [], "result_path": None,
        # Extra params needed by run_job_detached.py
        "cfg_dict": cfg_dict or {},
        "inter_doc_only": inter_doc_only,
        "enable_ragas_eval": enable_ragas_eval,
        "output_dir": output_dir or str(OUTPUT_DIR / sid),
        "kg_store_dir": kg_store_dir or str(KG_STORE_DIR),
        "nohup_mode": False,  # updated by _launch_nohup if used
    }
    dest = jobs_dir / f"{job_id}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)

def _read_job_status(sid: str, job_id: str) -> Optional[dict]:
    p = SESSIONS_DIR / sid / "jobs" / f"{job_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None

def _read_job_log(sid: str, job_id: str) -> str:
    """Lit le fichier de log du job, retourne '' si absent."""
    log_path = SESSIONS_DIR / sid / "jobs" / f"{job_id}.log"
    if log_path.exists():
        try:
            return log_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""

def _count_checkpoint_questions(sid: str, job_id: str) -> Optional[int]:
    """Return the live question count from the generator's checkpoint file.

    QuestionGenerator rewrites ``questions_checkpoint_{job_id}.json`` (a JSON
    list) after every question. The job-status JSON's ``progress`` field is only
    refreshed after the blocking generate() call returns, so it stays at 0 for
    the whole run; counting the checkpoint gives real-time progress instead.
    Returns None if the checkpoint is absent or unreadable.
    """
    ckpt = OUTPUT_DIR / sid / f"questions_checkpoint_{job_id}.json"
    if not ckpt.exists():
        return None
    try:
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data.get("questions", []))
    except Exception:
        return None
    return None

_PIPELINE_STAGES = [
    ("converting",    "Conversion",       "PDF/DOCX → Markdown via Docling"),
    ("kg_build",      "Indexation",       "Découpage du corpus en chunks, construction du graphe de documents"),
    ("kg_enrich",     "Relations",        "Extraction des mots-clés et liens sémantiques entre chunks"),
    ("kg_agent",      "Agent KG",         "Découverte de relations inter-documents par LLM"),
    ("retrospective", "Rétrospectif",     "Enrichissement centré-entités (RAKG)"),
    ("generation",    "Génération Q&A",   "Qualification des paires → génération question + réponse"),
    ("done",          "Terminé",          ""),
]

_STAGE_ACTIVITY = {
    "converting":    "🔄 Conversion des documents non-Markdown en cours…",
    "kg_build":      "🔄 Découpage du corpus et construction du graphe brut…",
    "kg_enrich":     "🔄 Extraction des keyphrases TF-IDF et calcul des relations…",
    "kg_agent":      "🔄 L'agent LLM cherche de nouvelles relations entre documents…",
    "retrospective": "🔄 Enrichissement rétrospectif par entités…",
    "generation":    "🔄 Génération des questions et réponses en cours…",
}

def _render_pipeline_stages(status: dict):
    """Affiche une barre de progression par étape de la pipeline."""
    stage      = status.get("pipeline_stage", "")
    s          = status.get("status", "")
    stage_keys = [k for k, _, _ in _PIPELINE_STAGES]
    durations  = status.get("stage_durations", {})

    try:
        current_idx = stage_keys.index(stage)
    except ValueError:
        current_idx = -1

    cols = st.columns(len(_PIPELINE_STAGES))
    for i, (key, label, desc) in enumerate(_PIPELINE_STAGES):
        with cols[i]:
            if s == "failed" and i == current_idx:
                icon = "❌"
            elif s == "failed" and i > current_idx:
                icon = "⬜"
            elif i < current_idx or (key == "done" and s == "done"):
                icon = "✅"
            elif i == current_idx:
                icon = "🔄"
            else:
                icon = "⬜"
            dt = durations.get(key)
            timing = f" `{dt:.0f}s`" if dt else ""
            st.caption(f"{icon} **{label}**{timing}")
            if desc and (i == current_idx or i < current_idx):
                st.caption(f"<small style='color:#888'>{desc}</small>", unsafe_allow_html=True)

    # Ligne contextuelle : ce qui se passe en ce moment
    if s == "running":
        activity = _STAGE_ACTIVITY.get(stage, "🔄 En cours…")
        st.info(activity, icon=None)


def _render_kg_downloads(status: dict, key_prefix: str = ""):
    """Affiche les boutons de téléchargement des snapshots KG avec leurs stats."""
    snapshots = [
        ("kg_raw_path",      "kg_01_raw.json",      "KG brut",      "kg_raw_stats"),
        ("kg_enriched_path", "kg_02_enriched.json",  "KG enrichi",   "kg_enriched_stats"),
        ("kg_agent_path",    "kg_03_agent.json",      "KG agent",     "kg_agent_stats"),
        ("kg_final_path",    "knowledge_graph.json",  "KG final",     "kg_final_stats"),
    ]
    available = [(lbl, status[pk], status.get(sk, {}))
                 for pk, _, lbl, sk in snapshots if status.get(pk) and Path(status[pk]).exists()]
    if not available:
        return

    jid = status.get("job_id", "")
    _kp = key_prefix or jid

    with st.expander(f"📦 Snapshots Knowledge Graph ({len(available)} disponibles)", expanded=False):
        for lbl, path_str, stats in available:
            path = Path(path_str)
            col_dl, col_stats = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    f"📥 {lbl}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/json",
                    key=f"kg_dl_{_kp}_{lbl}",
                )
            with col_stats:
                if stats:
                    st.caption(
                        f"{stats.get('nodes_docs',0)} doc(s) · "
                        f"{stats.get('nodes_chunks',0)} chunk(s) · "
                        f"{stats.get('relations_total',0)} relation(s)"
                    )
                    by_type = stats.get("relations_by_type", {})
                    if by_type:
                        top = sorted(by_type.items(), key=lambda x: -x[1])[:4]
                        st.caption("  " + "  ·  ".join(f"{k}: {v}" for k, v in top))


def _render_kg_live(status: dict, out_dir: Path):
    """Render an inline D3 KG visualization using the freshest available snapshot."""
    # Priority: live (intra-step) > agent > enriched > raw
    stage = status.get("pipeline_stage", "")
    if stage not in ("kg_build", "kg_enrich", "kg_agent", "retrospective", "generation", "done"):
        return

    candidates = [
        out_dir / "kg_live.json",
        out_dir / "kg_03_agent.json",
        out_dir / "kg_02_enriched.json",
        out_dir / "kg_01_raw.json",
    ]
    # During generation/done, prefer the final KG
    if stage in ("generation", "done"):
        kg_final = out_dir / "knowledge_graph.json"
        if kg_final.exists():
            candidates.insert(0, kg_final)

    kg_path = next((p for p in candidates if p.exists() and p.stat().st_size > 100), None)
    if kg_path is None:
        return

    try:
        from visualize_kg import preprocess_kg_for_viz, build_visualization_html
        kg_dict = json.loads(kg_path.read_text(encoding="utf-8"))
    except Exception:
        return

    # Try to load Q&A checkpoint for question annotations
    job_id  = status.get("job_id", "")
    sid     = status.get("session_id", "")
    qa_list = None
    if stage in ("generation", "done") and job_id:
        ckpt = out_dir / f"questions_checkpoint_{job_id}.json"
        if ckpt.exists():
            try:
                qa_list = json.loads(ckpt.read_text(encoding="utf-8"))
            except Exception:
                pass

    nodes, links, qa_by_node = preprocess_kg_for_viz(kg_dict, qa_list)
    if not nodes:
        return

    n_nodes = len(nodes)
    n_links = len(links)
    snap_label = kg_path.name.replace(".json", "")
    stage_labels = {
        "kg_build":    "Construction KG",
        "kg_enrich":   "Enrichissement KG",
        "kg_agent":    "Agent KG (découverte)",
        "retrospective": "Rétrospectif",
        "generation":  "Génération questions",
        "done":        "Terminé",
    }
    stage_str = stage_labels.get(stage, stage)

    with st.expander(f"🕸️ Knowledge Graph — {n_nodes} nœuds · {n_links} relations · {stage_str}", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Nœuds", n_nodes)
        col2.metric("Relations", n_links)
        col3.caption(f"Source : `{snap_label}`")

        html = build_visualization_html(nodes, links, qa_by_node, height=620)
        st.components.v1.html(html, height=630, scrolling=False)


@st.fragment(run_every=3)
def _render_job_status_live(sid: str, job_id: str, zombie_key_prefix: str = "zombie"):
    """Fragment auto-rafraîchi (toutes les 3 s) affichant le statut d'un job.

    Remplace l'ancienne boucle `time.sleep(4) + st.rerun()` qui gelait tout le
    thread serveur Streamlit et relançait la page entière (donc renvoyait
    l'utilisateur au premier onglet). Un st.fragment isole le rafraîchissement :
    seul ce bloc est re-exécuté, sans toucher aux onglets ni bloquer l'UI.
    Quand le job n'est plus `running`, on stoppe l'auto-refresh du fragment.
    """
    status = _read_job_status(sid, job_id)
    if not status:
        return

    # ── Détection zombie : job "running" mais app/thread interrompu ──────────
    job_is_nohup = status.get("nohup_mode", False)
    if (status.get("status") == "running"
            and status.get("started_at")
            and not status.get("finished_at")
            and not job_is_nohup):
        st.warning(
            "⚠️ Ce job est marqué **en cours** mais l'app a peut-être redémarré. "
            "S'il ne progresse plus, marquez-le comme interrompu."
        )
        if st.button("🛑 Marquer comme interrompu", key=f"{zombie_key_prefix}_{job_id}"):
            _mark_job_interrupted(sid, job_id)
            st.rerun(scope="fragment")

    # ── Bouton Kill pour les jobs nohup en cours ──────────────────────────────
    if (job_is_nohup
            and status.get("status") == "running"
            and status.get("pid")):
        if st.button(
            "🔴 Tuer le processus (kill)",
            key=f"kill_nohup_{job_id}",
            type="primary",
            help=f"Envoie SIGTERM au processus nohup (PID {status.get('pid')})",
        ):
            ok = _kill_nohup_process(sid, job_id)
            if ok:
                st.success(f"✅ Processus PID {status.get('pid')} tué — job annulé.")
            else:
                st.error("❌ Impossible de tuer le processus (PID introuvable ou permission refusée).")
            st.rerun(scope="fragment")

    _render_job_status(status)

    # Stoppe l'auto-refresh du fragment dès que le job est terminé/échoué.
    # (st.fragment(run_every=3) continuerait sinon indéfiniment.)
    if status.get("status") != "running":
        return


def _render_job_status(status: dict):
    s    = status["status"]
    icon = {"queued": "⏳", "running": "🔄", "done": "✅",
            "failed": "❌", "cancelled": "🚫"}.get(s, "?")

    # ── En-tête ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.markdown(f"**Statut :** {icon} `{s}`")
    with c2:
        elapsed = status.get("elapsed_seconds")
        if elapsed:
            st.caption(f"⏱ Durée totale : {elapsed:.0f}s")
    with c3:
        fin = status.get("finished_at", "")
        if fin:
            st.caption(f"Terminé : {fin[:19]}")

    # ── Progression par étape ─────────────────────────────────────────────────
    _render_pipeline_stages(status)

    # ── Progression conversion Docling ────────────────────────────────────────
    if s == "running" and status.get("pipeline_stage") == "converting":
        conv = status.get("converting_progress", {})
        c_done  = conv.get("done", 0)
        c_total = conv.get("total", 1)
        c_name  = conv.get("current", "…")
        st.progress(
            min(max(c_done / max(c_total, 1), 0.0), 1.0),
            text=f"Conversion : {c_done}/{c_total} — {c_name}",
        )
        st.caption("🔄 Conversion via Docling Serve en cours…")

    # ── Barre de progression questions ────────────────────────────────────────
    if s == "running" and status.get("pipeline_stage") != "converting":
        prog = status.get("progress", {})
        done, total = prog.get("done", 0), prog.get("total", 1)
        # Live progress: the job JSON's `progress` is only refreshed AFTER the
        # (blocking) generate() call returns, so it stays at 0 for the whole run.
        # The generator, however, rewrites its checkpoint file after every
        # question. Count its entries to reflect real-time progress.
        _live_done = _count_checkpoint_questions(
            status.get("session_id", ""), status.get("job_id", "")
        )
        if _live_done is not None and _live_done > done:
            done = _live_done
        # Le checkpoint peut contenir PLUS d'entrées que `total` : le générateur
        # produit un buffer de scénarios et conserve les questions recalées par
        # l'évaluation QA (qui ne comptent pas dans l'objectif). Le ratio peut
        # donc dépasser 1.0, ce que st.progress refuse
        # (StreamlitAPIException: Progress Value has invalid value…).
        # On clampe la fraction dans [0, 1] tout en affichant le compte réel.
        _frac = min(max(done / max(total, 1), 0.0), 1.0)
        st.progress(_frac, text=f"Questions : {done}/{total}")
        st.caption("🔄 Actualisation automatique toutes les 3 secondes…")

    # ── Résumé final ──────────────────────────────────────────────────────────
    if s == "done":
        n_gen = status.get("num_questions_generated", 0)
        n_req = status.get("num_questions_requested", "?")
        type_dist = status.get("question_type_distribution", {})
        st.success(f"✅ {n_gen}/{n_req} question(s) générée(s)")

        if type_dist:
            dist_cols = st.columns(min(len(type_dist), 6))
            for col, (qt, cnt) in zip(dist_cols,
                                       sorted(type_dist.items(), key=lambda x: -x[1])):
                col.metric(qt, cnt)

        rp = status.get("result_path")
        if rp and Path(rp).exists():
            st.download_button(
                "📥 Télécharger dataset.json",
                data=Path(rp).read_bytes(),
                file_name="dataset.json",
                mime="application/json",
                key=f"dl_main_{status.get('job_id','')}",
            )

    # ── Visualisation KG live ────────────────────────────────────────────────
    _out_dir = Path(status.get("output_dir", "")) if status.get("output_dir") else None
    if _out_dir is None:
        _sid = status.get("session_id", "")
        if _sid:
            _out_dir = OUTPUT_DIR / _sid
    if _out_dir is not None:
        _render_kg_live(status, _out_dir)

    # ── Snapshots KG ──────────────────────────────────────────────────────────
    _render_kg_downloads(status, key_prefix=f"live_{status.get('job_id','')}")

    # ── Erreurs ───────────────────────────────────────────────────────────────
    if status.get("errors"):
        with st.expander("❌ Erreurs", expanded=True):
            for e in status["errors"]:
                st.error(e.get("message", str(e)))
                if e.get("traceback"):
                    st.code(e["traceback"], language="python")

    # ── Logs en temps réel ────────────────────────────────────────────────────
    sid_    = status.get("session_id", "")
    job_id_ = status.get("job_id", "")
    log_content = _read_job_log(sid_, job_id_)
    if log_content:
        lines = log_content.splitlines()
        n_lines = len(lines)
        with st.expander(
            f"📋 Logs ({n_lines} lignes)",
            expanded=(s in ("running", "failed")),
        ):
            if n_lines > 300:
                st.caption(f"Affichage des 300 dernières lignes sur {n_lines}")
                lines = lines[-300:]

            # Rendu stylé : sépare les lignes structurelles des lignes de détail
            rendered = []
            for line in lines:
                # Extraire la partie message (après le préfixe timestamp [LEVEL]).
                # Deux formats possibles :
                #   thread : "HH:MM:SS [LEVEL] message"
                #   nohup  : "HH:MM:SS [LEVEL] logger.name — message"
                # On tolère les deux : le groupe "name — " optionnel est retiré.
                import re as _re
                m = _re.match(r"^\d{2}:\d{2}:\d{2} \[(\w+)\] (.*)$", line)
                if m:
                    level, msg = m.group(1), m.group(2)
                    # Retire le préfixe "logger.name — " ajouté par le root logger
                    # (mode nohup / imports pipeline) pour un affichage homogène.
                    _m2 = _re.match(r"^[\w.]+ — (.*)$", msg)
                    if _m2:
                        msg = _m2.group(1)
                else:
                    level, msg = "INFO", line

                if level == "ERROR":
                    rendered.append(f"🔴 {msg}")
                elif level == "WARNING":
                    rendered.append(f"🟡 {msg}")
                elif msg.startswith("━"):
                    rendered.append(f"\n{'─'*60}")
                elif msg.startswith("┌─ ÉTAPE") or msg.startswith("┌─"):
                    rendered.append(f"\n▶ {msg[2:].strip()}")
                elif msg.startswith("└─ ✓"):
                    rendered.append(f"  ✅ {msg[4:].strip()}")
                elif msg.startswith("└─"):
                    rendered.append(f"  {msg[2:].strip()}")
                elif msg.startswith("  →") or msg.startswith("  ✓") or msg.startswith("  💾") or msg.startswith("  📊"):
                    rendered.append(f"    {msg.strip()}")
                elif msg.startswith("🚀") or msg.startswith("✅") or msg.startswith("❌"):
                    rendered.append(f"\n{msg}")
                else:
                    rendered.append(f"  {msg}")

            st.code("\n".join(rendered), language=None)
            if s == "running":
                st.caption("↑ Les logs se mettent à jour — cliquez sur 🔄 Actualiser")

def _launch_nohup(sid: str, job_id: str):
    """Launch run_job_detached.py as a detached nohup process."""
    import subprocess
    job_path = SESSIONS_DIR / sid / "jobs" / f"{job_id}.json"
    log_path = SESSIONS_DIR / sid / "jobs" / f"{job_id}.log"
    # Persiste le flag avant de lancer le process détaché
    try:
        d = json.loads(job_path.read_text())
        d["nohup_mode"] = True
        tmp = job_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(job_path)
    except Exception:
        pass
    script = Path(__file__).parent / "run_job_detached.py"
    cmd = [sys.executable, str(script), str(job_path)]
    with open(log_path, "a") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from parent process group
            cwd=str(Path(__file__).parent),
        )
    # Sauvegarde le PID pour permettre le kill depuis l'UI
    try:
        d = json.loads(job_path.read_text())
        d["pid"] = proc.pid
        tmp = job_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(job_path)
    except Exception:
        pass


def _kill_nohup_process(sid: str, job_id: str) -> bool:
    """Tue le processus nohup d'un job en cours via son PID sauvegardé.

    Envoie SIGTERM puis met à jour le statut du job à 'cancelled'.
    Retourne True si le processus a été tué, False sinon.
    """
    import signal as _signal
    job_path = SESSIONS_DIR / sid / "jobs" / f"{job_id}.json"
    try:
        d = json.loads(job_path.read_text())
    except Exception:
        return False

    pid = d.get("pid")
    if not pid:
        return False

    killed = False
    try:
        import os as _os
        _os.kill(int(pid), _signal.SIGTERM)
        killed = True
    except ProcessLookupError:
        killed = True  # déjà terminé
    except Exception:
        pass

    # Met à jour le statut dans le JSON
    try:
        d["status"] = "cancelled"
        d["finished_at"] = datetime.now().isoformat()
        d["cancelled_by"] = "user_ui"
        tmp = job_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(job_path)
    except Exception:
        pass

    return killed


def _generation_thread(
    sid: str, job_id: str, input_files: List[str],
    num_questions: int, cfg_dict: dict,
    inter_doc_only: bool = True,
    enable_ragas_eval: bool = False,
):
    """Thread d'arrière-plan : possède sa propre event loop, écrit le statut en JSON."""
    import logging as _logging

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    jobs_dir = SESSIONS_DIR / sid / "jobs"
    job_path = jobs_dir / f"{job_id}.json"
    log_path = jobs_dir / f"{job_id}.log"

    # ── Logger dédié à ce job ────────────────────────────────────────────────
    # Loggers are singletons keyed by name: a relaunch/resume of the same job_id
    # returns the SAME logger object, and Python's root logger persists for the
    # lifetime of the Streamlit process. Without cleanup, each (re)launch stacks
    # another FileHandler on both loggers → every log line written N times.
    # Remove any pre-existing FileHandler pointing at this job's log file first.
    _abs_log = str(log_path.resolve())

    def _purge_stale_handlers(_logger):
        for _h in list(_logger.handlers):
            if isinstance(_h, _logging.FileHandler):
                try:
                    if os.path.abspath(getattr(_h, "baseFilename", "")) == os.path.abspath(_abs_log):
                        _logger.removeHandler(_h)
                        _h.close()
                except Exception:
                    pass

    job_logger = _logging.getLogger(f"stark.job.{job_id}")
    job_logger.setLevel(_logging.DEBUG)
    job_logger.propagate = False
    _purge_stale_handlers(job_logger)
    fh = _logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                        datefmt="%H:%M:%S"))
    job_logger.addHandler(fh)

    # Redirige aussi le root logger vers ce fichier (capture les imports du pipeline)
    _purge_stale_handlers(_logging.getLogger())
    root_fh = _logging.FileHandler(log_path, encoding="utf-8")
    root_fh.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                                             datefmt="%H:%M:%S"))
    _logging.getLogger().addHandler(root_fh)

    def _log(msg: str, level: str = "INFO"):
        getattr(job_logger, level.lower(), job_logger.info)(msg)

    def _update(patch: dict):
        try:
            data = json.loads(job_path.read_text())
            data.update(patch)
            tmp = job_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(job_path)
        except Exception:
            pass

    _time = time  # alias local pour compatibilité avec le reste de la fonction

    def _kg_stats(kg) -> dict:
        """Return a compact stats dict for a KG snapshot."""
        if kg is None:
            return {}
        nodes     = getattr(kg, "nodes", [])
        rels      = getattr(kg, "relationships", [])
        n_docs    = sum(1 for n in nodes if str(getattr(n, "type", "")) in ("NodeType.DOCUMENT", "document"))
        n_chunks  = sum(1 for n in nodes if str(getattr(n, "type", "")) in ("NodeType.CHUNK", "chunk"))
        rel_types: dict = {}
        for r in rels:
            rt = str(getattr(r, "type", "unknown"))
            rel_types[rt] = rel_types.get(rt, 0) + 1
        return {
            "nodes_total":  len(nodes),
            "nodes_docs":   n_docs,
            "nodes_chunks": n_chunks,
            "relations_total": len(rels),
            "relations_by_type": rel_types,
        }

    def _save_kg_snapshot(kg, path: Path, label: str):
        try:
            KnowledgeGraphStorage.save(kg, path)
            _log(f"  💾 KG snapshot '{label}' sauvegardé → {path.name}")
        except Exception as _e:
            _log(f"  ⚠️  Snapshot KG '{label}' échoué : {_e}", "WARNING")

    def _write_kg_live_snapshot(kg, label: str):
        """Atomic overwrite of kg_live.json and bump live_snapshot_count in the job JSON."""
        live_path = out_dir / "kg_live.json"
        try:
            KnowledgeGraphStorage.save(kg, live_path)
            current = json.loads(job_path.read_text()) if job_path.exists() else {}
            count = current.get("live_snapshot_count", 0) + 1
            _update({"live_snapshot_count": count, "live_snapshot_label": label})
        except Exception as _e:
            _log(f"  ⚠️  Live KG snapshot échoué ({label}) : {_e}", "WARNING")

    def _log_kg_stats(kg, label: str):
        stats = _kg_stats(kg)
        _log(f"  📊 [{label}] {stats['nodes_docs']} doc(s) · "
             f"{stats['nodes_chunks']} chunk(s) · "
             f"{stats['relations_total']} relation(s)")
        by_type = stats.get("relations_by_type", {})
        if by_type:
            for rt, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
                _log(f"       {rt}: {cnt}")
        semantic = sum(v for k, v in by_type.items()
                       if k in ("keyphrases_overlap", "cosine_similarity",
                                "agent_discovered", "llm_triplet", "retrospective_entity"))
        if semantic == 0:
            _log("  ⚠️  Aucune relation sémantique détectée — "
                 "les questions multi-hop nécessitent des relations entre chunks", "WARNING")
        return stats

    try:
        t_start = _time.time()
        _log("━" * 60)
        _log(f"🚀 Job {job_id} démarré — session: {sid}")
        _log(f"   Heure   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _log(f"   Fichiers: {len(input_files)}")
        for f in input_files:
            _log(f"     • {Path(f).name}")
        _log(f"   Objectif: {num_questions} questions")
        _log("━" * 60)
        _update({"status": "running", "started_at": datetime.now().isoformat(),
                 "pipeline_stage": "init"})

        from knowledge_graph import KnowledgeGraphBuilder, KnowledgeGraphStorage
        from question_generator import QuestionGenerator
        from pipeline_config import PipelineConfig as _PC, build_ragas_personas

        import httpx
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        llm_cfg = _llm_config()
        llm = LangchainLLMWrapper(
            ChatOpenAI(
                base_url=llm_cfg.get("base_url"),
                api_key=llm_cfg["api_key"],
                model=llm_cfg.get("model", "claude-haiku-4-5-20251001"),
                temperature=0.0, max_tokens=4096, timeout=120,
                http_client=httpx.Client(verify=False),
                http_async_client=httpx.AsyncClient(verify=False),
            )
        )
        _log(f"🤖 LLM : {llm_cfg.get('model','?')}  |  base_url: {llm_cfg.get('base_url') or 'OpenAI'}")

        # ── Embedding model (titan-embed-v2 via litellm) ─────────────────────
        embedding_model = _build_embeddings()
        if embedding_model is not None:
            _log(f"🔢 Embeddings : {os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002')}")
        else:
            _log("⚠️  Embeddings indisponibles — RAKG rétrospectif limité", "WARNING")

        existing_files = [f for f in input_files if Path(f).exists()]
        missing  = [f for f in input_files if not Path(f).exists()]
        if missing:
            _log(f"  ⚠️  Fichiers introuvables : {missing}", "WARNING")
        if not existing_files:
            raise ValueError("Aucun fichier d'entrée trouvé.")

        cfg = _PC.from_dict(cfg_dict)
        out_dir = OUTPUT_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        _mode_label = "inter-documents strict" if inter_doc_only else "intra-document autorisé"
        _log(f"📋 Mode multi-hop  : {_mode_label}")
        _log(f"🧪 QA Eval (RAGAS) : {'activé' if enable_ragas_eval else 'désactivé'}")
        _log(f"📁 Répertoire sortie: {out_dir}")

        # ── Étape 0 : Conversion Docling (documents non-Markdown) ────────────
        _MARKDOWN_EXTS = {".md", ".markdown"}
        needs_conversion = [f for f in existing_files if Path(f).suffix.lower() not in _MARKDOWN_EXTS]
        if needs_conversion:
            _log("")
            _log(f"┌─ ÉTAPE 0 : Conversion de {len(needs_conversion)} document(s) via Docling Serve")
            _update({"pipeline_stage": "converting",
                     "converting": [Path(f).name for f in needs_conversion]})
            try:
                from document_preprocessor import ensure_markdown_files

                def _conv_progress(name: str, done: int, total: int):
                    _log(f"  [{done}/{total}] '{name}' converti")
                    _update({"converting_progress": {"current": name, "done": done, "total": total}})

                md_dir = out_dir / "markdown"
                md_paths = ensure_markdown_files(existing_files, md_dir, progress_callback=_conv_progress)
                _log(f"└─ ✓ {len(md_paths)} fichier(s) markdown prêt(s)")
            except Exception as _conv_exc:
                _log(f"  ⚠️  Conversion Docling échouée : {_conv_exc}", "WARNING")
                _log(traceback.format_exc(), "WARNING")
                # Fallback : ne garder que les .md déjà présents
                md_paths = [Path(f) for f in existing_files if Path(f).suffix.lower() in _MARKDOWN_EXTS]
                if not md_paths:
                    raise
        else:
            md_paths = [Path(f) for f in existing_files]

        md_files = [p for p in md_paths if p.exists()]
        if not md_files:
            raise ValueError("Aucun fichier markdown exploitable après conversion.")

        # ── Détection checkpoint existant → skip étapes KG ───────────────────
        checkpoint    = out_dir / f"questions_checkpoint_{job_id}.json"
        kg_final_path = out_dir / "knowledge_graph.json"
        _has_checkpoint = checkpoint.exists() and checkpoint.stat().st_size > 10
        _checkpoint_n   = 0
        if _has_checkpoint:
            try:
                _checkpoint_n = len(json.loads(checkpoint.read_text()))
            except Exception:
                _has_checkpoint = False
        _has_kg = kg_final_path.exists() and kg_final_path.stat().st_size > 1000

        if _has_checkpoint and _has_kg:
            _log("")
            _log(f"⏩ Checkpoint détecté : {_checkpoint_n} questions déjà générées")
            _log("┌─ ÉTAPES 1-3/4 : IGNORÉES (KG final existant rechargé)")
            _update({"pipeline_stage": "generation"})
            kg = KnowledgeGraphStorage.load_as_kg(kg_final_path)
            _stats_cached = _kg_stats(kg)
            stats1 = _stats_cached
            stats2 = _stats_cached
            stats3 = _stats_cached
            _log(f"└─ ✓ KG chargé — {_stats_cached.get('nodes_chunks','?')} chunks · "
                 f"{_stats_cached.get('relations_total','?')} relations")
        else:
            # ── Étape 1/4 : Construction du KG brut ──────────────────────────
            _log("")
            _log("┌─ ÉTAPE 1/4 : Construction du Knowledge Graph brut")
            t1 = _time.time()
            _update({"pipeline_stage": "kg_build"})

            kg_builder = KnowledgeGraphBuilder(
                llm=llm, embedding_model=embedding_model,
                config=cfg, inter_doc_only=inter_doc_only,
            )
            kg = kg_builder.create_from_markdown_files(md_files)

            dt1 = _time.time() - t1
            stats1 = _log_kg_stats(kg, "KG brut")
            _log(f"└─ ✓ terminé en {dt1:.1f}s")

            _save_kg_snapshot(kg, out_dir / "kg_01_raw.json", "kg_brut")
            _update({"kg_raw_path": str(out_dir / "kg_01_raw.json"),
                     "kg_raw_stats": stats1,
                     "stage_durations": {**json.loads(job_path.read_text()).get("stage_durations", {}), "kg_build": round(dt1, 1)}})

            # ── Étape 2/4 : Enrichissement ────────────────────────────────────
            _log("")
            _log("┌─ ÉTAPE 2/4 : Enrichissement du KG (keyphrases, entités, relations sémantiques)")
            t2 = _time.time()
            _update({"pipeline_stage": "kg_enrich"})

            def _on_enrich_progress(kg_state, label: str):
                _write_kg_live_snapshot(kg_state, f"enrich_{label}")

            kg = kg_builder.enrich_prechunked(
                kg, store_dir=KG_STORE_DIR, on_progress=_on_enrich_progress
            )

            dt2 = _time.time() - t2
            stats2 = _log_kg_stats(kg, "KG enrichi")
            _log(f"└─ ✓ terminé en {dt2:.1f}s")

            _save_kg_snapshot(kg, out_dir / "kg_02_enriched.json", "kg_enrichi")
            _update({"kg_enriched_path": str(out_dir / "kg_02_enriched.json"),
                     "kg_enriched_stats": stats2,
                     "stage_durations": {**json.loads(job_path.read_text()).get("stage_durations", {}), "kg_enrich": round(dt2, 1)}})

            try:
                written = kg_builder.save_doc_store(kg, KG_STORE_DIR)
                _log(f"  💾 Store domaine mis à jour → {KG_STORE_DIR} ({len(written)} docs)")
            except Exception as _store_exc:
                _log(f"  ⚠️  Store domaine échoué : {_store_exc}", "WARNING")

            # ── Étape 2.5/4 : Enrichissement universel (knowledge_graph/rahulnyk) ──
            # Méthodes domain-agnostic, activables via les flags universal_* de
            # enrich_modules. Bloc try/except : n'interrompt jamais le pipeline.
            _em_cfg_u = cfg.enrich_modules if hasattr(cfg, "enrich_modules") else None
            _u_prox = getattr(_em_cfg_u, "universal_proximity", False) if _em_cfg_u else False
            _u_metr = getattr(_em_cfg_u, "universal_metrics", False) if _em_cfg_u else False
            _u_comm = getattr(_em_cfg_u, "universal_communities", False) if _em_cfg_u else False
            _u_trip = getattr(_em_cfg_u, "universal_triplets", False) if _em_cfg_u else False
            _u_conc = getattr(_em_cfg_u, "universal_concepts", False) if _em_cfg_u else False
            _u_trel = getattr(_em_cfg_u, "universal_triplet_relations", False) if _em_cfg_u else False
            # universal_triplet_relations nécessite les triplets extraits d'abord.
            if _u_trel and not _u_trip:
                _log("  ↳ universal_triplet_relations=True force universal_triplets=True")
                _u_trip = True
            if any([_u_prox, _u_metr, _u_comm, _u_trip, _u_conc, _u_trel]):
                _log("")
                _log(f"┌─ ÉTAPE 2.5/4 : Enrichissement universel "
                     f"[proximity={_u_prox}, metrics={_u_metr}, communities={_u_comm}, "
                     f"triplets={_u_trip}, concepts={_u_conc}, triplet_relations={_u_trel}]")
                t_univ = _time.time()
                try:
                    from kg_enrich_universal import enrich_kg_universal
                    kg, u_stats = enrich_kg_universal(
                        kg,
                        llm=llm if (_u_trip or _u_conc or _u_trel) else None,
                        add_graph_metrics=_u_metr,
                        add_communities=_u_comm,
                        add_triplets=_u_trip,
                        add_concepts=_u_conc,
                        add_triplet_relations=_u_trel,
                        add_tfidf_filter=False,  # STARK a déjà son propre filtre IDF
                    )
                    dt_univ = _time.time() - t_univ
                    _log(f"└─ ✓ Enrichissement universel terminé en {dt_univ:.1f}s — {u_stats}")
                    _update({"universal_enrichment_stats": u_stats,
                             "stage_durations": {**json.loads(job_path.read_text()).get("stage_durations", {}), "universal": round(dt_univ, 1)}})
                except Exception as _univ_exc:
                    _log(f"└─ ⚠️  Enrichissement universel échoué ({_univ_exc}) — on continue sans", "WARNING")
                    _log(traceback.format_exc(), "WARNING")

            # ── Étape 3/4 : KG Agent ──────────────────────────────────────────
            # Lire les flags depuis enrich_modules (priorité) ou USE_KG_AGENT env
            _em_cfg = cfg.enrich_modules if hasattr(cfg, "enrich_modules") else None
            use_kg_agent = (
                _em_cfg.kg_agent if _em_cfg is not None
                else os.environ.get("USE_KG_AGENT", "false").lower() == "true"
            )
            use_relation_validator = (
                _em_cfg.relation_validator if _em_cfg is not None else True
            )
            use_frame_bridge = (
                _em_cfg.frame_bridge if _em_cfg is not None else False
            )
            _log("")
            _log(f"┌─ ÉTAPE 3/4 : KG Agent [kg_agent={use_kg_agent}, relation_validator={use_relation_validator}, frame_bridge={use_frame_bridge}]")
            t3 = _time.time()
            _update({"pipeline_stage": "kg_agent"})

            if use_kg_agent:
                try:
                    from kg_agent import RelationValidator, DirectRelationDiscovery

                    if use_relation_validator:
                        _log("  → Validation des relations existantes…")
                        validator = RelationValidator(config=cfg)
                        kg, val_stats = loop.run_until_complete(
                            validator._validate_async(kg, llm, None, None)
                        )
                        _log(f"  ✓ Validation terminée")
                        if isinstance(val_stats, dict):
                            for k, v in val_stats.items():
                                _log(f"       {k}: {v}")
                        else:
                            _log(f"       {val_stats}")
                    else:
                        _log("  → Validation désactivée (relation_validator=false)")

                    _log("  → Découverte de nouvelles relations inter-documents…")
                    discoverer = DirectRelationDiscovery(config=cfg)
                    kg, disc_stats = loop.run_until_complete(
                        discoverer._discover_async(
                            kg, llm, None,
                            on_progress=lambda _kg, lbl: _write_kg_live_snapshot(_kg, lbl),
                        )
                    )
                    _log(f"  ✓ Découverte terminée")
                    if isinstance(disc_stats, dict):
                        for k, v in disc_stats.items():
                            _log(f"       {k}: {v}")
                    else:
                        _log(f"       {disc_stats}")

                    if use_frame_bridge:
                        _log("  → Frame Bridge (bottom-up, SemanticFrameBridgeDiscovery)…")
                        from kg_agent import SemanticFrameBridgeDiscovery
                        ev_cfg = getattr(cfg, "evaluation", None)
                        fb_conf = getattr(ev_cfg, "frame_bridge_min_confidence", 0.70) if ev_cfg else 0.70
                        fb = SemanticFrameBridgeDiscovery(min_confidence=fb_conf)
                        kg, fb_stats = loop.run_until_complete(
                            fb._discover_async(
                                kg, llm,
                                embedding_model=embedding_model if "embedding_model" in dir() else None,
                                on_progress=lambda _kg, lbl: _write_kg_live_snapshot(_kg, lbl),
                            )
                        )
                        _log(f"  ✓ Frame Bridge terminé")
                        if isinstance(fb_stats, dict):
                            for k, v in fb_stats.items():
                                _log(f"       {k}: {v}")
                    else:
                        _log("  → Frame Bridge désactivé (frame_bridge=false)")

                    stats3 = _log_kg_stats(kg, "KG final (post-agent)")
                    _save_kg_snapshot(kg, out_dir / "kg_03_agent.json", "kg_agent")
                    _update({"kg_agent_path": str(out_dir / "kg_03_agent.json"),
                             "kg_agent_stats": stats3})
                except Exception as _agent_exc:
                    _log(f"  ⚠️  KG Agent échoué ({_agent_exc}) — on continue sans", "WARNING")
                    _log(traceback.format_exc(), "WARNING")
            else:
                _log("  → Désactivé (kg_agent=false)")
                stats3 = stats2

            dt3 = _time.time() - t3
            _log(f"└─ ✓ terminé en {dt3:.1f}s")
            _update({"stage_durations": {**json.loads(job_path.read_text()).get("stage_durations", {}), "kg_agent": round(dt3, 1)}})

            # ── Étape 3.5/4 : Enrichissement rétrospectif centré-entité (RAKG §III-D) ──
            _em_retro = (_em_cfg.retrospective if _em_cfg is not None else None)
            _retro_sub = getattr(cfg, "retrospective", None)
            _retro_enabled = (
                _em_retro if _em_retro is not None
                else (getattr(_retro_sub, "enabled", False) if _retro_sub is not None else False)
            )
            _log("")
            if _retro_enabled:
                _log("┌─ ÉTAPE 3.5/4 : Enrichissement rétrospectif RAKG (centré-entité)")
                _update({"pipeline_stage": "retrospective"})
                t_retro = _time.time()
                try:
                    from entity_retrospective import discover_relations_retrospective
                    retro_checkpoint = out_dir / f"retrospective_checkpoint_{job_id}.jsonl"
                    _emb_model = getattr(kg_builder, "embedding_model", None)
                    kg = loop.run_until_complete(discover_relations_retrospective(
                        kg=kg,
                        config=cfg,
                        llm=llm,
                        checkpoint_path=retro_checkpoint,
                        embedding_model=_emb_model or embedding_model,
                    ))
                    stats_retro = _kg_stats(kg)
                    retro_added = stats_retro.get("relations_total", 0) - stats3.get("relations_total", 0)
                    dt_retro = _time.time() - t_retro
                    _log(f"└─ ✓ Rétrospectif terminé en {dt_retro:.1f}s — "
                         f"{max(0, retro_added)} relations retrospective_entity ajoutées")
                    _update({"retrospective_relations_added": max(0, retro_added),
                             "retrospective_checkpoint": str(retro_checkpoint),
                             "stage_durations": {**json.loads(job_path.read_text()).get("stage_durations", {}), "retrospective": round(dt_retro, 1)}})
                except Exception as _retro_exc:
                    _log(f"└─ ⚠️  Rétrospectif échoué ({_retro_exc}) — on continue sans", "WARNING")
                    _log(traceback.format_exc(), "WARNING")
            else:
                _log("┌─ ÉTAPE 3.5/4 : Rétrospectif RAKG désactivé (retrospective=false)")

            # ── Étape 3.7/4 : Enrichissement Graphify (sémantique LLM + code AST) ──
            # Extraction Graphify EN MÉMOIRE sur les chunks du KG, puis injection
            # de relations graphify_semantic (multi-hop) et graphify_code (calls,
            # imports, inherits, references…). Bloc try/except : n'interrompt jamais
            # le pipeline. Snapshot kg_04_graphify.json (métadonnées recalculées
            # automatiquement par KnowledgeGraphStorage.save).
            _gfy_enabled = getattr(_em_cfg, "graphify", False) if _em_cfg is not None else False
            # Chemin de réutilisation d'un graphe Graphify existant (None = reconstruire).
            _gfy_reuse_path = cfg_dict.get("graphify_reuse_path")
            _log("")
            if _gfy_enabled:
                _update({"pipeline_stage": "graphify"})
                t_gfy = _time.time()
                try:
                    from kg_enrich_graphify import add_graphify_relations_from_graph
                    _stats_before_gfy = _kg_stats(kg)

                    if _gfy_reuse_path and Path(_gfy_reuse_path).exists():
                        # ── Mode RÉUTILISER : charge un graphe existant (0 LLM) ──
                        _log(f"┌─ ÉTAPE 3.7/4 : Graphify — RÉUTILISATION du graphe existant")
                        _log(f"  → Source : {_gfy_reuse_path}")
                        from graphify_integration import load_graphify_graph_for_kg
                        _G, _chunk_map = load_graphify_graph_for_kg(
                            kg, _gfy_reuse_path, directed=True, verbose=True,
                        )
                        if _G is None or not _chunk_map:
                            _log("  ⚠️  Réutilisation impossible (graphe vide ou corpus différent) — "
                                 "bascule sur reconstruction LLM", "WARNING")
                            from kg_enrich_graphify import enrich_kg_with_graphify_in_memory
                            kg, gfy_stats = enrich_kg_with_graphify_in_memory(
                                kg, max_hops=2, require_cross_doc=True,
                                include_code_relations=True, directed=True, verbose=True,
                            )
                        else:
                            kg, gfy_stats = add_graphify_relations_from_graph(
                                kg, _G, _chunk_map,
                                max_hops=2, require_cross_doc=True,
                                include_code_relations=True, verbose=True,
                            )
                    else:
                        # ── Mode RECONSTRUIRE : extraction LLM en mémoire ────────
                        _log("┌─ ÉTAPE 3.7/4 : Enrichissement Graphify (extraction en mémoire)")
                        from kg_enrich_graphify import enrich_kg_with_graphify_in_memory
                        kg, gfy_stats = enrich_kg_with_graphify_in_memory(
                            kg,
                            max_hops=2,
                            require_cross_doc=True,
                            include_code_relations=True,
                            directed=True,
                            batch_size=30,
                            cache_graph_path=str(out_dir / "graphify_graph.json"),
                            verbose=True,
                        )

                    dt_gfy = _time.time() - t_gfy
                    _log(f"└─ ✓ Graphify terminé en {dt_gfy:.1f}s — "
                         f"{gfy_stats.get('graphify_semantic', 0)} graphify_semantic, "
                         f"{gfy_stats.get('graphify_code', 0)} graphify_code")
                    _save_kg_snapshot(kg, out_dir / "kg_04_graphify.json", "kg_graphify")
                    _update({"graphify_path": str(out_dir / "kg_04_graphify.json"),
                             "graphify_stats": gfy_stats,
                             "graphify_reused": bool(_gfy_reuse_path),
                             "stage_durations": {**json.loads(job_path.read_text()).get("stage_durations", {}), "graphify": round(dt_gfy, 1)}})
                except Exception as _gfy_exc:
                    _log(f"└─ ⚠️  Graphify échoué ({_gfy_exc}) — on continue sans", "WARNING")
                    _log(traceback.format_exc(), "WARNING")
            else:
                _log("┌─ ÉTAPE 3.7/4 : Enrichissement Graphify désactivé (graphify=false)")

            _save_kg_snapshot(kg, kg_final_path, "kg_final")
            _update({"kg_final_path": str(kg_final_path), "kg_final_stats": _kg_stats(kg)})

        # ── Étape 4/4 : Génération des questions ─────────────────────────────
        _log("")
        _log(f"┌─ ÉTAPE 4/4 : Génération des questions ({num_questions} demandées)")
        t4 = _time.time()
        _update({"pipeline_stage": "generation",
                 "progress": {"done": 0, "total": num_questions}})

        question_gen = QuestionGenerator(
            llm=llm, config=cfg, inter_doc_only=inter_doc_only,
            enable_ragas_eval=enable_ragas_eval,
        )
        personas = build_ragas_personas(cfg)

        # Résumé contextuel avant génération
        kg_final_stats = _kg_stats(kg)
        n_chunks   = kg_final_stats.get("nodes_chunks", "?")
        n_rels     = kg_final_stats.get("relations_total", "?")
        rel_by_type = kg_final_stats.get("relations_by_type", {})
        sem_rels   = sum(v for k, v in rel_by_type.items()
                         if k not in ("child", "next", "document"))
        _log(f"  → KG final : {n_chunks} chunks · {n_rels} relations dont {sem_rels} sémantiques")
        _log(f"  → {len(personas)} persona(s) : {', '.join(p.name for p in personas)}")
        type_names = [t.name for t in cfg.taxonomy.types] if hasattr(cfg.taxonomy, 'types') else []
        if type_names:
            _log(f"  → {len(type_names)} type(s) de questions : {', '.join(type_names)}")
        _log(f"  → Mode : {'inter-documents strict' if inter_doc_only else 'intra-document autorisé'}")
        _log(f"  → QA Eval : {'activé' if enable_ragas_eval else 'désactivé'}")
        _log(f"  → Checkpoint : {checkpoint.name} {'(reprise)' if _has_checkpoint else '(nouveau)'}")

        questions = loop.run_until_complete(
            question_gen.generate(
                kg=kg, persona_list=personas,
                num_questions=num_questions,
                checkpoint_path=checkpoint,
            )
        )

        dt4 = _time.time() - t4
        _log(f"  ✓ {len(questions)} question(s) générée(s) en {dt4:.1f}s")
        _update({"stage_durations": {**json.loads(job_path.read_text()).get("stage_durations", {}), "generation": round(dt4, 1)}})

        # Stats par type de question
        type_dist: dict = {}
        passed_n = 0
        for q in questions:
            qt = q.get("question_type", "unknown")
            type_dist[qt] = type_dist.get(qt, 0) + 1
            if (q.get("qa_eval") or {}).get("passed"):
                passed_n += 1
        if type_dist:
            _log("  → Distribution des types :")
            for qt, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
                _log(f"       {qt}: {cnt}")
        if any(q.get("qa_eval") for q in questions):
            _log(f"  → QA Eval passé : {passed_n}/{len(questions)}")
        _log(f"└─ ✓ terminé en {dt4:.1f}s")

        result_path = out_dir / "dataset.json"
        result_path.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "session_id": sid,
                    "job_id": job_id,
                    "num_questions": len(questions),
                    "questions": questions,
                },
                ensure_ascii=False, indent=2,
            )
        )

        t_total = _time.time() - t_start
        _log("")
        _log("━" * 60)
        _log(f"✅ Pipeline terminée en {t_total:.1f}s")
        _log(f"   KG brut      : {stats1.get('nodes_chunks','?')} chunks · {stats1.get('relations_total','?')} relations")
        _log(f"   KG enrichi   : {stats2.get('nodes_chunks','?')} chunks · {stats2.get('relations_total','?')} relations")
        _log(f"   KG final     : {stats3.get('nodes_chunks','?')} chunks · {stats3.get('relations_total','?')} relations")
        _log(f"   Questions    : {len(questions)}/{num_questions} générées")
        _log(f"   Dataset      : {result_path.name}")
        _log("━" * 60)

        _update({
            "status": "done",
            "pipeline_stage": "done",
            "finished_at": datetime.now().isoformat(),
            "num_questions_generated": len(questions),
            "progress": {"done": len(questions), "total": num_questions},
            "result_path": str(result_path),
            "elapsed_seconds": round(t_total, 1),
            "question_type_distribution": type_dist,
        })

    except Exception as exc:
        _log("", "ERROR")
        _log("━" * 60, "ERROR")
        _log(f"❌ ERREUR : {type(exc).__name__}: {exc}", "ERROR")
        _log(traceback.format_exc(), "ERROR")
        _log("━" * 60, "ERROR")
        _update({
            "status": "failed",
            "pipeline_stage": "failed",
            "finished_at": datetime.now().isoformat(),
            "errors": [{
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }],
        })
    finally:
        _logging.getLogger().removeHandler(root_fh)
        fh.close()
        root_fh.close()
        loop.close()
# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════
def _tab_results(sid: str):
    st.subheader("Résultats")

    jobs_dir = SESSIONS_DIR / sid / "jobs"
    if not jobs_dir.exists():
        st.info("Aucun job pour cette session.")
        return

    jobs: List[dict] = []
    for p in sorted(jobs_dir.glob("*.json"), reverse=True):
        try:
            jobs.append(json.loads(p.read_text()))
        except Exception:
            pass

    if not jobs:
        st.info("Aucun job trouvé.")
        return

    for job in jobs[:10]:
        jid    = job["job_id"]
        s      = job["status"]
        icon   = {"done": "✅", "failed": "❌", "running": "🔄",
                  "queued": "⏳", "cancelled": "🚫"}.get(s, "?")
        elapsed = job.get("elapsed_seconds")
        elapsed_str = f" · {elapsed:.0f}s" if elapsed else ""
        with st.expander(
            f"{icon} Job `{jid}` — "
            f"{job.get('num_questions_generated', 0)}/{job.get('num_questions_requested','?')} questions"
            f"{elapsed_str}  ({job.get('created_at','')[:10]})",
            expanded=(s == "done"),
        ):
            # ── Résumé ──────────────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Statut :** {s}")
                st.write(f"**Demandé :** {job.get('num_questions_requested','?')}")
                st.write(f"**Généré :** {job.get('num_questions_generated', 0)}")
            with col2:
                st.write(f"**Créé :** {job.get('created_at','')[:19]}")
                st.write(f"**Terminé :** {(job.get('finished_at') or '')[:19] or '—'}")
                if elapsed:
                    st.write(f"**Durée :** {elapsed:.0f}s")
            with col3:
                # Stats KG final
                kg_stats = job.get("kg_final_stats") or job.get("kg_enriched_stats") or {}
                if kg_stats:
                    st.write(f"**KG chunks :** {kg_stats.get('nodes_chunks', '?')}")
                    st.write(f"**KG relations :** {kg_stats.get('relations_total', '?')}")

            # ── Distribution des types ───────────────────────────────────────
            type_dist = job.get("question_type_distribution", {})
            if type_dist:
                st.caption("Distribution des types :")
                tc = st.columns(min(len(type_dist), 6))
                for col, (qt, cnt) in zip(tc, sorted(type_dist.items(), key=lambda x: -x[1])):
                    col.metric(qt, cnt)

            # ── Téléchargements ──────────────────────────────────────────────
            dl_cols = st.columns(2)
            with dl_cols[0]:
                rp = job.get("result_path")
                if rp and Path(rp).exists():
                    st.download_button(
                        "📥 Dataset JSON",
                        data=Path(rp).read_bytes(),
                        file_name=f"{sid}_{jid}_dataset.json",
                        mime="application/json",
                        key=f"dl_{jid}",
                    )
            with dl_cols[1]:
                # Log du job
                log_path = SESSIONS_DIR / sid / "jobs" / f"{jid}.log"
                if log_path.exists():
                    st.download_button(
                        "📋 Télécharger logs",
                        data=log_path.read_bytes(),
                        file_name=f"{jid}.log",
                        mime="text/plain",
                        key=f"dl_log_{jid}",
                    )

            # ── Snapshots KG ─────────────────────────────────────────────────
            _render_kg_downloads(job, key_prefix=f"hist_{jid}")

            # ── Questions ────────────────────────────────────────────────────
            rp = job.get("result_path")
            if rp and Path(rp).exists():
                try:
                    result = json.loads(Path(rp).read_bytes())
                    qs = result.get("questions", [])
                    if qs:
                        _render_dataset_questions(qs, jid)
                except Exception:
                    pass

            errs = job.get("errors", [])
            if errs:
                with st.expander("❌ Erreurs"):
                    for e in errs:
                        st.error(e.get("message", str(e)))
                        if e.get("traceback"):
                            st.code(e["traceback"], language="python")


def _render_dataset_questions(qs: list, job_id: str) -> None:
    """Render the full question list with QA details, metadata, and contexts."""

    # ── Summary metrics ──────────────────────────────────────────────────────
    total = len(qs)
    type_counts: dict = {}
    passed_count = 0
    for q in qs:
        qt = q.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1
        qe = q.get("qa_eval") or {}
        if qe.get("passed"):
            passed_count += 1

    m1, m2, m3 = st.columns(3)
    m1.metric("Questions", total)
    m2.metric("Types distincts", len(type_counts))
    qa_eval_present = any(q.get("qa_eval") for q in qs)
    if qa_eval_present:
        m3.metric("QA Eval — passé", f"{passed_count}/{total}")

    # Type distribution bar
    if type_counts:
        st.caption("Distribution des types :")
        bar_cols = st.columns(len(type_counts))
        for col, (qt, cnt) in zip(bar_cols, sorted(type_counts.items(), key=lambda x: -x[1])):
            col.metric(qt, cnt, delta=f"{100*cnt//total}%", delta_color="off")

    st.divider()

    # ── Per-question cards ────────────────────────────────────────────────────
    # Pagination: 10 questions per page
    page_size = 10
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1,
        key=f"page_{job_id}",
        help=f"{total_pages} page(s) · {page_size} questions/page",
    )
    start = (page - 1) * page_size
    page_qs = qs[start: start + page_size]

    for i, q in enumerate(page_qs, start + 1):
        qtype  = q.get("question_type", "?")
        nhops  = q.get("num_hops", "?")
        gp     = q.get("generation_params") or {}
        style  = gp.get("style", "")
        length = gp.get("length", "")
        persona= gp.get("persona", "")
        rel    = gp.get("relation_type", "")
        themes = gp.get("themes") or []
        qe     = q.get("qa_eval") or {}
        passed = qe.get("passed")

        badge = "✅" if passed is True else ("❌" if passed is False else "—")
        label = (
            f"{badge} Q{i} · **{qtype}** · {nhops}-hop"
            + (f" · {style}" if style else "")
            + (f" / {length}" if length else "")
        )

        with st.expander(label, expanded=False):

            # Row 1: question text
            st.markdown(f"##### ❓ Question")
            st.info(q.get("question", "—"))

            # Row 2: answer
            st.markdown(f"##### 💬 Réponse de référence")
            st.success(q.get("reference", "—") or "—")

            # Row 3: QA Eval scores + generation metadata (side by side)
            left, right = st.columns(2)

            with left:
                st.markdown("**QA Eval**")
                if qe:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Groundedness", _fmt_score(qe.get("groundedness_score")))
                    s2.metric("Answer Accuracy", _fmt_score(qe.get("answer_accuracy_score")))
                    s3.metric("2-Hop", _fmt_score(qe.get("two_hop_score")))
                    attempts = qe.get("attempts", 1)
                    is_faulty = qe.get("question_is_faulty")
                    faulty_tag = " · ⚠️ question single-hop" if is_faulty else ""
                    st.caption(
                        f"Résultat : {'✅ passé' if passed else '❌ échoué'}"
                        f"  · {attempts} tentative(s){faulty_tag}"
                    )
                    fb = qe.get("feedback", "")
                    if fb and not passed:
                        st.error(fb, icon="🔍")
                else:
                    st.caption("Pas d'évaluation QA disponible.")

            with right:
                st.markdown("**Métadonnées**")
                st.caption(
                    f"**Persona :** {persona or '—'}  \n"
                    f"**Relation :** {rel or '—'}  \n"
                    f"**Thèmes :** {', '.join(themes) if themes else '—'}  \n"
                    f"**Sources :** {', '.join(q.get('source_documents') or []) or '—'}"
                )

            # Row 4: contexts
            ctx1 = q.get("context_1hop", "")
            ctx2 = q.get("context_2hop", "")
            if ctx1 or ctx2:
                st.markdown("**Contextes**")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("1-hop")
                    st.text_area(
                        label="ctx1", value=ctx1, height=140,
                        disabled=True, label_visibility="collapsed",
                        key=f"ctx1_{job_id}_{i}",
                    )
                with c2:
                    st.caption("2-hop")
                    st.text_area(
                        label="ctx2", value=ctx2, height=140,
                        disabled=True, label_visibility="collapsed",
                        key=f"ctx2_{job_id}_{i}",
                    )


def _fmt_score(v) -> str:
    """Format a float score as a percentage string, or '—' if None."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.0%}"
    except (TypeError, ValueError):
        return str(v)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
_SUPPORTED_UPLOAD_TYPES = ["md", "txt", "rst", "pdf", "docx", "pptx", "xlsx", "html", "png", "jpg", "jpeg"]
_SUPPORTED_GLOBS = ["*.md", "*.txt", "*.rst", "*.pdf", "*.docx", "*.pptx", "*.xlsx", "*.html", "*.png", "*.jpg", "*.jpeg"]
_MARKDOWN_ICON = {"md": "📝", "txt": "📄", "rst": "📘", "pdf": "📕",
                  "docx": "📃", "pptx": "📊", "xlsx": "📈",
                  "html": "🌐", "png": "🖼", "jpg": "🖼", "jpeg": "🖼"}

def _documents_page():
    st.title("📚 Bibliothèque de Documents")
    st.caption(
        f"Répertoire : `{DOCUMENTS_DIR}`  —  "
        "Déposez vos documents ici (`.md`, `.pdf`, `.docx`, `.pptx`, `.rst`, `.txt`, images…). "
        "Les formats non-Markdown sont convertis automatiquement via **Docling Serve** avant la génération."
    )

    # ── Upload ─────────────────────────────────────────────────────────────────
    # Garde une trace des fichiers déjà traités (nom + taille) pour éviter
    # de les reconvertir sur chaque rerun Streamlit (le file_uploader persiste).
    _PROCESSED_KEY = "_docs_processed_uploads"
    if _PROCESSED_KEY not in st.session_state:
        st.session_state[_PROCESSED_KEY] = set()

    with st.expander("⬆ Uploader des documents", expanded=True):
        uploaded = st.file_uploader(
            "Sélectionnez un ou plusieurs fichiers",
            type=_SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            _MD_EXTS = {".md", ".markdown"}
            originals_dir = DOCUMENTS_DIR / "_originals"
            originals_dir.mkdir(parents=True, exist_ok=True)

            # Seulement les fichiers pas encore traités dans cette session
            unprocessed = [
                uf for uf in uploaded
                if (uf.name, uf.size) not in st.session_state[_PROCESSED_KEY]
            ]

            if unprocessed:
                converted: List[str] = []
                failed: List[str] = []

                with st.spinner("Traitement des fichiers (conversion Docling si nécessaire)…"):
                    progress = st.progress(0.0, text="Préparation…")
                    n = len(unprocessed)
                    for idx, uf in enumerate(unprocessed, 1):
                        name = uf.name or "unknown"
                        ext = Path(name).suffix.lower()
                        data = uf.read()
                        try:
                            if ext in _MD_EXTS:
                                dest = DOCUMENTS_DIR / name
                                dest.write_bytes(data)
                                converted.append(dest.name)
                            else:
                                src = originals_dir / name
                                src.write_bytes(data)
                                progress.progress(
                                    (idx - 1) / max(n, 1),
                                    text=f"Conversion Docling : {name}",
                                )
                                from document_preprocessor import ensure_markdown_files
                                md_paths = ensure_markdown_files(
                                    [str(src)], out_dir=DOCUMENTS_DIR
                                )
                                converted.extend(p.name for p in md_paths)
                        except Exception as exc:
                            failed.append(f"{name} — {exc}")
                        st.session_state[_PROCESSED_KEY].add((uf.name, uf.size))
                        progress.progress(idx / max(n, 1), text=f"{idx}/{n} traité(s)")
                    progress.empty()

                if converted:
                    st.success(
                        f"✅ {len(converted)} document(s) prêt(s) : {', '.join(converted)}"
                    )
                if failed:
                    st.error("❌ Échec de conversion :")
                    for msg in failed:
                        st.write(f"• {msg}")
                if converted:
                    st.rerun()

    st.divider()

    # ── Parcourir les fichiers existants ───────────────────────────────────────
    # Note : on exclut le sous-dossier _originals/ (documents source bruts
    # conservés après conversion Docling) pour n'afficher que les .md exploitables.
    md_files = sorted(
        f for glob in _SUPPORTED_GLOBS
        for f in DOCUMENTS_DIR.glob(f"**/{glob}")
        if "_originals" not in f.parts
    )

    if not md_files:
        st.info(
            "Aucun document dans la bibliothèque.\n\n"
            "Uploadez des fichiers ci-dessus ou placez-les dans :\n"
            f"`{DOCUMENTS_DIR}`"
        )
        return

    # Stats
    total_size = sum(f.stat().st_size for f in md_files) / 1024
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fichiers disponibles", len(md_files))
    with col2:
        st.metric("Taille totale", f"{total_size:.1f} KB")

    # Filtre
    search = st.text_input("🔍 Filtrer par nom", placeholder="rechercher…")
    filtered = [f for f in md_files if not search or search.lower() in f.name.lower()]
    st.caption(f"{len(filtered)} fichier(s) affiché(s)")

    for f in filtered:
        try:
            rel = f.relative_to(DOCUMENTS_DIR)
        except ValueError:
            rel = f.name
        size_kb = round(f.stat().st_size / 1024, 1)

        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            ext_icon = _MARKDOWN_ICON.get(f.suffix.lower().lstrip("."), "📄")
            with st.expander(f"{ext_icon} {rel}  ({size_kb} KB)", expanded=False):
                if f.suffix.lower() in (".md", ".txt", ".rst", ".html"):
                    try:
                        content = f.read_text(encoding="utf-8", errors="replace")
                        preview = content[:3000]
                        if len(content) > 3000:
                            preview += "\n\n*(aperçu tronqué)*"
                        st.markdown(preview)
                    except Exception as e:
                        st.error(f"Impossible de lire : {e}")
                else:
                    st.caption(f"Aperçu non disponible pour les fichiers `{f.suffix}`. Sera converti via Docling lors de la génération.")
        with col2:
            st.download_button(
                "📥", data=f.read_bytes(),
                file_name=f.name, mime="text/markdown",
                key=f"dl_doc_{f}",
            )
        with col3:
            if st.button("🗑", key=f"rm_doc_{f}", help="Supprimer ce fichier"):
                try:
                    f.unlink()
                    st.success(f"'{f.name}' supprimé.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _init()
    _sidebar()

    page = st.session_state.page
    sid  = st.session_state.session_id

    if page == "home":
        _home_page()
    elif page == "wizard":
        step = st.session_state.wizard_step
        if step == 1:
            _wizard_step1()
        elif step == 2:
            _wizard_step2()
        elif step == 3:
            _wizard_step3()
        else:
            st.session_state.wizard_step = 1
            st.rerun()
    elif page == "session" and sid:
        _session_page(sid)
    elif page == "documents":
        _documents_page()
    else:
        _nav("home")

if __name__ == "__main__":
    main()