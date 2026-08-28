"""
docling_client.py
-----------------
Client pour l'API **Docling Serve** (conversion de documents en Markdown).

Workflow asynchrone Docling Serve (3 étapes) :
  1. Soumission  : POST /v1/convert/file/async  -> renvoie un task_id
  2. Polling     : GET  /v1/status/poll/{task_id} -> attend SUCCESS/FAILURE
  3. Résultat    : GET  /v1/result/{task_id}      -> renvoie le document converti

Configuration via variables d'environnement :
  DOCLING_SERVE_URL    URL de base du service (ex: http://host:5001)
  DOCLING_API_KEY      clé API optionnelle
  DOCLING_TIMEOUT      délai max d'attente du polling (secondes, défaut 600)
  DOCLING_OCR_ENABLED  activer l'OCR (true/false, défaut false)
  DOCLING_OCR_ENGINE   moteur OCR (défaut easyocr)
  DOCLING_OCR_LANG     langues OCR séparées par des virgules (défaut en,fr,de,es)
  DOCLING_PDF_BACKEND  backend PDF (défaut dlparse_v4)
  DOCLING_TABLE_MODE   mode tableaux (défaut fast)
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

import requests

warnings.filterwarnings("ignore")  # masque les warnings SSL (verify=False)

# Types MIME supportés par Docling Serve.
MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "md": "text/markdown",
    "txt": "text/plain",
    "csv": "text/csv",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "json": "application/json",
    "rst": "text/x-rst",
}

# Statuts renvoyés par l'API.
TERMINAL_STATUSES = {
    "SUCCESS", "FAILURE", "success", "failure", "completed", "failed", "error",
}
SUCCESS_STATUSES = {"SUCCESS", "success", "completed"}

POLL_INTERVAL = 5.0  # secondes entre deux vérifications de statut


class DoclingServeError(RuntimeError):
    """Erreur levée en cas d'échec de conversion Docling Serve."""


class DoclingServeClient:
    """Client minimal pour convertir des documents via Docling Serve."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        ocr_enabled: Optional[bool] = None,
        ocr_engine: Optional[str] = None,
        ocr_langs: Optional[List[str]] = None,
        pdf_backend: Optional[str] = None,
        table_mode: Optional[str] = None,
        image_export_mode: str = "embedded",
    ) -> None:
        self.base_url = (base_url or os.getenv("DOCLING_SERVE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("DOCLING_API_KEY")
        self.timeout = (
            timeout if timeout is not None else float(os.getenv("DOCLING_TIMEOUT", "600"))
        )
        self.ocr_enabled = (
            ocr_enabled
            if ocr_enabled is not None
            else os.getenv("DOCLING_OCR_ENABLED", "false").lower() == "true"
        )
        self.ocr_engine = ocr_engine or os.getenv("DOCLING_OCR_ENGINE", "easyocr")
        if ocr_langs is not None:
            self.ocr_langs = ocr_langs
        else:
            self.ocr_langs = [
                l.strip()
                for l in os.getenv("DOCLING_OCR_LANG", "en,fr,de,es").split(",")
                if l.strip()
            ]
        self.pdf_backend = pdf_backend or os.getenv("DOCLING_PDF_BACKEND", "dlparse_v4")
        self.table_mode = table_mode or os.getenv("DOCLING_TABLE_MODE", "accurate")
        self.image_export_mode = image_export_mode

        if not self.base_url:
            raise DoclingServeError(
                "DOCLING_SERVE_URL n'est pas configuré (variable d'environnement)."
            )

    # ── Session HTTP ───────────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.verify = False  # certificats internes non vérifiés
        if self.api_key:
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "X-API-Key": self.api_key,
                }
            )
        return session

    def _build_form_data(self, to_formats: List[str]):
        form_data = [
            ("target_type", "inbody"),
            ("image_export_mode", self.image_export_mode),
            ("do_ocr", str(self.ocr_enabled).lower()),
            ("force_ocr", "false"),
            ("ocr_engine", self.ocr_engine),
            ("pdf_backend", self.pdf_backend),
            ("table_mode", self.table_mode),
            ("abort_on_error", "false"),
        ]
        for fmt in to_formats:
            form_data.append(("to_formats", fmt))
        for lang in self.ocr_langs:
            form_data.append(("ocr_lang", lang))
        return form_data

    # ── Étapes du workflow ──────────────────────────────────────────────────────

    def _submit_async(self, session: requests.Session, path: Path, mime: str, form_data) -> str:
        logging.info("[DOCLING] Envoi de '%s' à %s ...", path.name, self.base_url)
        with open(path, "rb") as f:
            resp = session.post(
                f"{self.base_url}/v1/convert/file/async",
                files=[("files", (path.name, f, mime))],
                data=form_data,
                timeout=120,
            )
        resp.raise_for_status()
        task_id = resp.json().get("task_id")
        if not task_id:
            raise DoclingServeError(f"Pas de task_id dans la réponse : {resp.text}")
        logging.info("[DOCLING] Tâche soumise : %s", task_id)
        return task_id

    def _poll_status(self, session: requests.Session, task_id: str) -> None:
        start = time.time()
        last_status = None
        while (time.time() - start) < self.timeout:
            resp = session.get(
                f"{self.base_url}/v1/status/poll/{task_id}", timeout=30
            )
            resp.raise_for_status()
            status = resp.json().get("task_status", "")
            if status != last_status:
                logging.info("[DOCLING] Statut : %s", status)
                last_status = status
            if status in TERMINAL_STATUSES:
                if status in SUCCESS_STATUSES:
                    return
                raise DoclingServeError(
                    f"Conversion échouée (statut={status})"
                )
            time.sleep(POLL_INTERVAL)
        raise DoclingServeError(
            f"Conversion expirée après {int(self.timeout)}s (task_id={task_id})"
        )

    def _get_result(self, session: requests.Session, task_id: str) -> dict:
        resp = session.get(f"{self.base_url}/v1/result/{task_id}", timeout=60)
        resp.raise_for_status()
        return resp.json()

    # ── API publique ──────────────────────────────────────────────────────────

    def convert_to_markdown(self, file_path: Path) -> str:
        """
        Convertit un document en Markdown via Docling Serve.

        Args:
            file_path: chemin du document à convertir.

        Returns:
            Le contenu Markdown extrait.

        Raises:
            DoclingServeError: si la conversion échoue.
        """
        path = Path(file_path)
        if not path.exists():
            raise DoclingServeError(f"Fichier introuvable : {path}")

        mime = MIME_TYPES.get(
            path.suffix.lower().lstrip("."), "application/octet-stream"
        )
        session = self._build_session()
        form_data = self._build_form_data(to_formats=["md"])

        t0 = time.time()
        try:
            task_id = self._submit_async(session, path, mime, form_data)
            self._poll_status(session, task_id)
            result = self._get_result(session, task_id)
        except requests.RequestException as exc:
            raise DoclingServeError(f"Erreur HTTP Docling Serve : {exc}") from exc

        doc = result.get("document", {})
        markdown = doc.get("md_content") or doc.get("text_content", "")
        logging.info(
            "[DOCLING] Conversion de '%s' terminée en %.0fs — %d caractères.",
            path.name,
            time.time() - t0,
            len(markdown),
        )
        if not markdown.strip():
            raise DoclingServeError(
                f"Conversion vide pour '{path.name}' (aucun contenu extrait)."
            )
        return markdown