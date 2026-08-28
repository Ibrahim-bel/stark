"""
document_preprocessor.py
------------------------
Étape de prétraitement des documents : convertit tout document non-Markdown
(PDF, DOCX, PPTX, XLSX, HTML, images, ...) en Markdown **structuré** via
Docling Serve, afin que la suite du pipeline (qui ne consomme que des .md)
puisse les traiter de façon homogène.

Post-traitement du markdown :
  1. fix_heading_hierarchy()       — corrige la hiérarchie ##/###
  2. clean_markdown()              — supprime headers/footers/footnotes/références
  3. describe_images_with_vlm()   — remplace les images base64 par des descriptions VLM

Variables d'environnement :
  DESCRIBE_IMAGES        true/false  — activer la description VLM (défaut true)
  STRIP_REFERENCES       true/false  — supprimer ## References/Appendix (défaut true)
  REPAIR_UNICODE         true/false  — réparer les artefacts Unicode/ligatures PDF (défaut true)
  STRIP_TECHNICAL_LINKS  true/false  — supprimer mailto/URL nues/footnotes-liens (défaut true)
                                       Note : peut retirer des listes de liens légitimes
                                       (pages de ressources, etc.) — désactiver si besoin.
  VLM_MIN_IMAGE_BYTES    int         — taille mini (octets décodés) sous laquelle une image
                                       est considérée décorative et ignorée par le VLM (défaut 2048)
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
from pathlib import Path
from typing import Callable, List, Optional

from docling_client import DoclingServeClient

# Extensions déjà en texte/markdown : pas besoin de conversion.
_MARKDOWN_EXTENSIONS = {".md", ".markdown"}
# Extensions de texte brut lisibles directement (réécrites en .md).
_PLAIN_TEXT_EXTENSIONS = {".txt"}

# ── Réparation Unicode ────────────────────────────────────────────────────────

_LIGATURE_MAP = str.maketrans({
    'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
    'ﬅ': 'st', 'ﬆ': 'st',
})
_RE_HYPHEN_BREAK = re.compile(r'(\w)-\n(\w)')


def _repair_unicode(text: str) -> str:
    """Normalise les artefacts Unicode fréquents dans les PDF convertis."""
    text = text.translate(_LIGATURE_MAP)
    # Caractères invisibles : soft hyphen, zero-width space/non-joiner/joiner, BOM
    for ch in ('­', '​', '‌', '‍', '﻿'):
        text = text.replace(ch, '')
    # Espace insécable → espace normale
    text = text.replace('\xa0', ' ')
    # Mots coupés en fin de ligne : "exam-\nple" → "example"
    text = _RE_HYPHEN_BREAK.sub(r'\1\2', text)
    return text


# ── Suppression des liens techniques ─────────────────────────────────────────

_RE_MAILTO_LINK = re.compile(r'\[([^\]]*)\]\(mailto:[^\)]+\)')
_RE_BARE_URL_LINE = re.compile(r'^\s*https?://\S+\s*$', re.MULTILINE)
_RE_FOOTNOTE_LINK_LINE = re.compile(
    r'^\s*\[\d+\s+https?://\S*\]\(https?://[^\)]*\)\s*$', re.MULTILINE
)


def _strip_technical_links(text: str) -> str:
    """Supprime les liens sans valeur sémantique (mailto, URL nues, footnotes-liens).

    Note : la suppression des lignes ne contenant qu'une URL nue peut retirer
    des listes de liens légitimes (pages de ressources, bibliographies d'URLs).
    Désactivable via STRIP_TECHNICAL_LINKS=false.
    Note : la suppression d'un lien mailto au milieu d'une phrase peut laisser
    un résidu de ponctuation (ex. "Écrivez à .") — acceptable dans ce contexte.
    """
    def _keep_display(m: re.Match) -> str:
        display = m.group(1).strip()
        # Email brut comme texte d'affichage → supprimer entièrement
        if re.match(r'^\S+@\S+\.\S+$', display):
            return ''
        return display

    text = _RE_MAILTO_LINK.sub(_keep_display, text)
    text = _RE_BARE_URL_LINE.sub('', text)
    text = _RE_FOOTNOTE_LINK_LINE.sub('', text)
    return text


# ── Correction de la hiérarchie des titres ────────────────────────────────────

def fix_heading_hierarchy(text: str) -> str:
    """
    Corrige la hiérarchie des titres produits par Docling.

    Docling utilise ## pour tout (titre principal, sections, sous-sections).
    Cette fonction rétablit une hiérarchie cohérente :
      - La première ligne ## non numérotée → # (titre principal du document)
      - Sections numérotées "## N Titre" → ## (sections H2)
      - Sous-sections numérotées "## N.M Titre" → ### (H3)
      - Sous-sous-sections "## N.M.P Titre" → #### (H4)
      - Annexes/Appendix → ## (H2)
    """
    lines = text.split("\n")
    result = []
    title_found = False

    for line in lines:
        # Détecter les lignes de heading Docling (## ou ###)
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if not m:
            result.append(line)
            continue

        hashes = m.group(1)
        content = m.group(2).strip()

        # On ne traite que les ## (Docling flatten tout en ##)
        if hashes != "##":
            result.append(line)
            continue

        # Numérotation de section : "1 Introduction", "2.1 Related Work", etc.
        num_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', content)
        if num_match:
            num = num_match.group(1)
            depth = num.count('.') + 2  # "1" → H2, "1.1" → H3, "1.1.1" → H4
            result.append('#' * min(depth, 4) + ' ' + content)
            continue

        # Pas de numéro → c'est soit le titre principal, soit une section spéciale
        # (Abstract, References, Conclusion, Appendix…)
        special_sections = {
            'abstract', 'introduction', 'conclusion', 'conclusions',
            'references', 'bibliography', 'acknowledgments', 'acknowledgements',
            'appendix', 'supplementary', 'related work', 'background',
            'discussion', 'limitations', 'future work', 'methodology',
            'experiments', 'results', 'evaluation',
        }
        content_lower = content.lower().strip()

        # Si c'est une section spéciale connue → H2
        if any(content_lower.startswith(s) for s in special_sections):
            result.append('## ' + content)
            continue

        # Premier ## sans numéro → titre principal (H1)
        if not title_found:
            result.append('# ' + content)
            title_found = True
            continue

        # Autres ## sans numéro → H2
        result.append('## ' + content)

    return "\n".join(result)


# ── Patterns de nettoyage headers/footers ────────────────────────────────────

_RE_PAGE_NUMBER = re.compile(
    r"^\s*(?:Page\s*\d+(?:\s+of\s+.{0,80})?|-\s*\d+\s*-|\d+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_RE_PAGE_OF = re.compile(
    r"^.*\bPage\s*\d+\s+of\s+.{3,100}$",
    re.MULTILINE | re.IGNORECASE,
)

_RE_FOOTNOTE = re.compile(
    r"^\s*[\*†‡§¹²³⁴⁵⁶⁷⁸⁹⁰]\s+\S.{0,200}$",
    re.MULTILINE,
)

# Sections à supprimer (références, annexes, remerciements)
_STRIP_SECTION_PATTERN = re.compile(
    r'^#{1,4}\s+(?:References|Bibliography|Acknowledgments?|Acknowledgements?'
    r'|Appendix\b.*|Supplementary\b.*)',
    re.MULTILINE | re.IGNORECASE,
)


def _strip_trailing_sections(text: str) -> str:
    """
    Supprime ## References, ## Appendix, ## Acknowledgments et tout ce qui suit
    (ces sections polluent le KG sans apporter de valeur sémantique).
    """
    # Trouver la position de la première section à supprimer
    match = _STRIP_SECTION_PATTERN.search(text)
    if match:
        return text[:match.start()].rstrip() + "\n"
    return text


def _remove_author_affiliations(text: str) -> str:
    """
    Supprime le bloc auteurs/affiliations situé entre le titre (H1) et l'Abstract.
    Ces lignes contiennent souvent des noms, emails, numéros d'affiliation
    collés ensemble par Docling — elles polluent le début du document.
    """
    lines = text.split("\n")
    result = []
    in_header_block = False
    abstract_found = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Détection du début du bloc header (après le titre)
        if result and result[-1].lstrip().startswith("# ") and not in_header_block:
            in_header_block = True

        # Fin du bloc header : on rencontre ## Abstract ou ## 1 Introduction
        if in_header_block and re.match(r'^#{1,4}\s+(Abstract|Introduction|\d+\s+)', stripped, re.IGNORECASE):
            in_header_block = False
            abstract_found = True

        if in_header_block and not abstract_found:
            # Garder seulement les lignes qui semblent être du contenu réel
            # (pas des lignes d'auteurs/affiliations)
            is_affiliation = (
                # Ligne courte avec numéros d'affiliation collés
                (len(stripped) < 200 and re.search(r'\d\s+\w', stripped) and
                 re.search(r'University|Institute|School|Lab|Department|Corp|Inc\.',
                           stripped, re.IGNORECASE))
                or
                # Ligne contenant des emails
                re.search(r'\S+@\S+\.\S+', stripped)
                or
                # Ligne avec symboles d'auteurs (*†‡) au milieu
                (re.search(r'[†‡§\*]', stripped) and len(stripped) < 150)
            )
            if is_affiliation:
                continue  # Supprimer cette ligne

        result.append(line)

    return "\n".join(result)


def _remove_repeated_headers(text: str, min_repeats: int = 3) -> str:
    """Supprime les lignes courtes qui se répètent de façon suspecte (headers)."""
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 5 < len(stripped) < 80 and not stripped.startswith("#"):
            counts[stripped] = counts.get(stripped, 0) + 1
    to_remove = {k for k, v in counts.items() if v >= min_repeats}
    if not to_remove:
        return text
    filtered = [line for line in lines if line.strip() not in to_remove]
    return "\n".join(filtered)


def clean_markdown(
    markdown: str,
    source_name: str = "",
    strip_references: Optional[bool] = None,
    strip_technical_links: Optional[bool] = None,
) -> str:
    """
    Nettoie le markdown produit par Docling :
    0. Décode les entités HTML résiduelles (&amp; → &, etc.)
    0b. Répare les artefacts Unicode (ligatures, mots coupés, invisibles) [REPAIR_UNICODE]
    1. Corrige la hiérarchie des titres (## → #/##/###)
    2. Supprime les numéros de page parasites
    3. Supprime les références "Page N of <titre>"
    4. Supprime les notes de bas de page flottantes
    5. Supprime les liens techniques (mailto, URL nues, footnotes-liens) [STRIP_TECHNICAL_LINKS]
    6. Supprime les lignes courtes répétées (headers de page)
    7. Supprime le bloc auteurs/affiliations
    8. Supprime ## References, ## Appendix, ## Acknowledgments (si activé) [STRIP_REFERENCES]
    9. Normalise les espaces et lignes vides
    10. Garantit au moins un titre #
    """
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")

    # 0. Décoder les entités HTML résiduelles (avant la détection des titres)
    text = html.unescape(text)

    # 0b. Réparer les artefacts Unicode (ligatures, césures PDF, espaces insécables)
    if os.environ.get("REPAIR_UNICODE", "true").lower() == "true":
        text = _repair_unicode(text)

    # 1. Corriger la hiérarchie des titres
    text = fix_heading_hierarchy(text)

    # 2-4. Supprimer les patterns de pages/footers/footnotes
    text = _RE_PAGE_OF.sub("", text)
    text = _RE_PAGE_NUMBER.sub("", text)
    text = _RE_FOOTNOTE.sub("", text)

    # 5. Supprimer les liens techniques sans valeur sémantique
    _strip_links = (
        strip_technical_links
        if strip_technical_links is not None
        else os.environ.get("STRIP_TECHNICAL_LINKS", "true").lower() == "true"
    )
    if _strip_links:
        text = _strip_technical_links(text)

    # 6. Supprimer les headers répétés
    text = _remove_repeated_headers(text, min_repeats=3)

    # 7. Supprimer le bloc auteurs/affiliations
    text = _remove_author_affiliations(text)

    # 8. Supprimer les sections de références/annexes
    _strip = (
        strip_references
        if strip_references is not None
        else os.environ.get("STRIP_REFERENCES", "true").lower() == "true"
    )
    if _strip:
        text = _strip_trailing_sections(text)

    # 9. Normaliser les espaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()

    # 10. Garantir au moins un titre
    if text and not any(line.lstrip().startswith("#") for line in text.split("\n")):
        title = Path(source_name).stem.replace("_", " ").strip() or "Document"
        text = f"# {title}\n\n{text}"

    return text + "\n"


# ── Description VLM des images ────────────────────────────────────────────────

_RE_IMAGE_DATA_URI = re.compile(
    r'!\[([^\]]*)\]\((data:image/[^;]+;base64,[A-Za-z0-9+/=]+)\)'
)
_RE_IMAGE_PLACEHOLDER = re.compile(r'<!--\s*image\s*-->', re.IGNORECASE)

# Patterns de titres auto-générés par le VLM à supprimer
_RE_VLM_TITLE_PREFIX = re.compile(
    r'^(?:#+\s*)?(?:(?:Figure|Image)\s+)?'
    r'(?:Description|Summary|Overview|Caption)\s*[:\-]?\s*\n?',
    re.IGNORECASE | re.MULTILINE,
)
# Supprime tout heading Markdown résiduel en première ligne d'une réponse VLM
_RE_VLM_LEADING_HEADING = re.compile(r'^\s*#{1,6}\s+[^\n]*\n?')
# Pattern pour détecter une légende de figure dans le contexte
_RE_FIGURE_CAPTION = re.compile(
    r'Figure\s+(\d+[a-z]?)\s*[:.]?\s*([^>📷\n]{10,150})',
    re.IGNORECASE,
)
# Extrait le payload base64 d'un data URI
_RE_BASE64_PAYLOAD = re.compile(r'base64,([A-Za-z0-9+/=]+)')


def _is_decorative_image(data_uri: str) -> bool:
    """Retourne True si l'image est probablement décorative (taille estimée < VLM_MIN_IMAGE_BYTES).

    Critère purement dimensionnel — aucune analyse du contenu de l'image.
    Une icône, un logo, un badge ou un séparateur tient typiquement en quelques centaines
    d'octets ; une figure informative dépasse quasi-systématiquement 2 Ko.
    """
    m = _RE_BASE64_PAYLOAD.search(data_uri)
    if not m:
        return False
    estimated_bytes = len(m.group(1)) * 3 // 4
    threshold = int(os.environ.get("VLM_MIN_IMAGE_BYTES", "2048"))
    return estimated_bytes < threshold


def _call_vlm(data_uri: str, context: str = "", model: Optional[str] = None) -> str:
    """
    Appelle le VLM pour décrire une image. Nettoie la réponse.
    """
    import requests as req

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = (os.environ.get("OPENAI_BASE_URL", "") or "").rstrip("/")
    vlm_model = model or os.environ.get("OPENAI_MODEL", "claude-haiku-4-5-20251001")

    if not api_key or not base_url:
        return "[Image — VLM non configuré]"

    prompt = (
        "Describe this figure in 2-3 concise sentences, focusing on what it shows. "
        "Do NOT start with 'Figure Description', 'Image Description' or any heading. "
        "Go straight to the description."
    )
    if context.strip():
        prompt += f" Context clue: {context.strip()[:200]}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": vlm_model,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    try:
        r = req.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            verify=False,
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        # Nettoyer les titres auto-générés par le VLM (préfixes connus + tout heading résiduel)
        raw = _RE_VLM_TITLE_PREFIX.sub("", raw).strip()
        raw = _RE_VLM_LEADING_HEADING.sub("", raw, count=1).strip()
        # Limiter à 3 phrases
        sentences = re.split(r'(?<=[.!?])\s+', raw)
        return " ".join(sentences[:3]).strip()
    except Exception as exc:
        logging.warning("[VLM] Échec description image : %s", exc)
        return f"[Image — description indisponible ({type(exc).__name__})]"


def describe_images_with_vlm(
    markdown: str,
    max_images: int = 50,
    model: Optional[str] = None,
) -> str:
    """
    Remplace les images base64 dans le markdown par des descriptions textuelles VLM.
    Les `<!-- image -->` (mode placeholder) sont simplement supprimés.
    """
    # Mode placeholder → simple suppression
    text = _RE_IMAGE_PLACEHOLDER.sub("", markdown)

    # Mode embedded → description VLM
    images = list(_RE_IMAGE_DATA_URI.finditer(text))
    if not images:
        return text

    logging.info("[VLM] %d image(s) trouvées dans le markdown.", len(images))

    result_parts = []
    last_end = 0
    vlm_count = 0

    for idx, match in enumerate(images):
        result_parts.append(text[last_end:match.start()])
        last_end = match.end()

        alt_text = match.group(1)
        data_uri = match.group(2)

        # Filtrer les images décoratives avant tout appel VLM
        if _is_decorative_image(data_uri):
            logging.debug("[VLM] Image %d/%d ignorée (décorative, trop petite).", idx + 1, len(images))
            continue

        if vlm_count >= max_images:
            result_parts.append(f"> 📷 [Figure {idx + 1} — limite atteinte]")
        else:
            # Contexte autour de l'image (300 chars avant)
            context_before = text[max(0, match.start() - 300):match.start()]

            # Extraire la légende de figure du contexte (ex: "Figure 1: Pipeline overview")
            cap_match = _RE_FIGURE_CAPTION.search(context_before)
            if cap_match:
                fig_num = cap_match.group(1)
                label = f"Figure {fig_num}"
                # Transmettre la légende comme contexte VLM
                context_for_vlm = cap_match.group(2).strip()
            else:
                label = alt_text.strip() if (alt_text.strip() and
                        alt_text.lower() not in ("image", "figure", "")) else f"Figure {idx + 1}"
                context_for_vlm = context_before

            desc = _call_vlm(data_uri, context=context_for_vlm, model=model)
            result_parts.append(f"> 📷 **{label}** : {desc}")
            logging.info("[VLM] Image %d/%d décrite.", idx + 1, len(images))
            vlm_count += 1

    result_parts.append(text[last_end:])
    return "".join(result_parts)


# ── Normalisation finale ──────────────────────────────────────────────────────

def _normalize_markdown(markdown: str, source_name: str) -> str:
    """Normalise les fins de ligne, réduit les lignes vides excessives, garantit un titre."""
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip()

    if text and not any(line.lstrip().startswith("#") for line in text.split("\n")):
        title = Path(source_name).stem.replace("_", " ").strip() or "Document"
        text = f"# {title}\n\n{text}"

    return text + "\n"


# ── Conversion de document ─────────────────────────────────────────────────────

def convert_document_to_markdown(
    src_path: Path,
    out_dir: Path,
    client: Optional[DoclingServeClient] = None,
    describe_images: Optional[bool] = None,
    strip_references: Optional[bool] = None,
) -> Path:
    """Convertit un document unique en Markdown avec post-traitement complet."""
    src_path = Path(src_path)
    ext = src_path.suffix.lower()

    if ext in _MARKDOWN_EXTENSIONS:
        logging.info("[PREPROCESS] '%s' déjà Markdown — conservé.", src_path.name)
        return src_path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _hash6 = hashlib.md5(str(src_path).encode()).hexdigest()[:6]
    out_path = out_dir / f"{src_path.stem}_{_hash6}.md"

    if ext in _PLAIN_TEXT_EXTENSIONS:
        raw = src_path.read_text(encoding="utf-8", errors="replace")
        out_path.write_text(_normalize_markdown(raw, src_path.name), encoding="utf-8")
        return out_path

    logging.info("[PREPROCESS] Conversion Docling de '%s' ...", src_path.name)
    client = client or DoclingServeClient()
    markdown = client.convert_to_markdown(src_path)

    markdown = clean_markdown(markdown, src_path.name, strip_references=strip_references)

    _describe = (
        describe_images if describe_images is not None
        else os.environ.get("DESCRIBE_IMAGES", "true").lower() == "true"
    )
    if _describe:
        markdown = describe_images_with_vlm(markdown)
    else:
        markdown = _RE_IMAGE_DATA_URI.sub(
            lambda m: "> 📷 [Image — description VLM désactivée]", markdown
        )

    markdown = _normalize_markdown(markdown, src_path.name)
    out_path.write_text(markdown, encoding="utf-8")
    logging.info("[PREPROCESS] '%s' → '%s' (%d chars).", src_path.name, out_path.name, len(markdown))
    return out_path


def ensure_markdown_files(
    input_files: List[str],
    out_dir: Path,
    client: Optional[DoclingServeClient] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    describe_images: Optional[bool] = None,
    strip_references: Optional[bool] = None,
) -> List[Path]:
    """Convertit tous les fichiers d'entrée en .md via Docling Serve."""
    out_dir = Path(out_dir)
    md_paths: List[Path] = []
    shared_client = client
    _total = sum(1 for f in input_files if Path(f).suffix.lower() not in _MARKDOWN_EXTENSIONS)
    _done = 0

    for f in input_files:
        src = Path(f)
        if not src.exists():
            logging.warning("[PREPROCESS] Introuvable, ignoré : %s", src)
            continue
        ext = src.suffix.lower()
        if ext in _MARKDOWN_EXTENSIONS:
            md_paths.append(src)
            continue
        if ext not in _PLAIN_TEXT_EXTENSIONS and shared_client is None:
            shared_client = DoclingServeClient()
        md_path = convert_document_to_markdown(
            src, out_dir, client=shared_client,
            describe_images=describe_images, strip_references=strip_references,
        )
        md_paths.append(md_path)
        _done += 1
        if progress_callback is not None:
            progress_callback(src.name, _done, _total)

    if not md_paths:
        raise FileNotFoundError(f"Aucun fichier exploitable parmi : {input_files}")
    return md_paths
