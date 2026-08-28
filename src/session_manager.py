"""
session_manager.py
------------------
Filesystem-backed CRUD for pipeline sessions.

Each session is stored as a YAML file: sessions/{session_id}.yaml
A session is a PipelineConfig with an associated MetaConfig.

Usage:
    sm = SessionManager(sessions_dir=Path("sessions"))
    cfg = sm.create(config_dict)
    cfg = sm.get("cosapp_v1")
    sm.update("cosapp_v1", {"evaluation.qa_eval_threshold": 0.7})
    sm.delete("cosapp_v1")
    clone = sm.fork("cosapp_v1", new_id="cosapp_v2")
    all_ids = sm.list_sessions()
"""
from __future__ import annotations

import copy
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_config import PipelineConfig

class SessionManager:
    """
    Manages PipelineConfig sessions persisted as YAML files.

    Args:
        sessions_dir: Directory where *.yaml session files are stored.
                      Created automatically if it does not exist.
    """

    def __init__(self, sessions_dir: Path = Path("sessions")) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.yaml"

    @staticmethod
    def _clean_id(raw: str) -> str:
        """Sanitise a session id: lowercase, alphanum + hyphens only."""
        clean = re.sub(r"[^a-z0-9_\-]", "_", raw.lower())
        return clean[:64]

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(
        self,
        config_dict: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> PipelineConfig:
        """
        Create a new session from a config dict.

        Args:
            config_dict: Full PipelineConfig dict (or partial — defaults filled in).
            session_id:  Desired session ID.  Auto-generated if not provided.

        Returns:
            PipelineConfig — the created config (also saved to disk).

        Raises:
            ValueError: If a session with the same ID already exists.
        """
        sid = self._clean_id(session_id or str(uuid.uuid4())[:8])
        if self._path(sid).exists():
            raise ValueError(f"Session '{sid}' already exists. Use update() or fork().")

        # Stamp meta
        config_dict = copy.deepcopy(config_dict)
        meta = config_dict.setdefault("meta", {})
        meta["session_id"] = sid
        meta["created_at"] = datetime.now().isoformat()

        cfg = PipelineConfig.from_dict(config_dict)
        cfg.to_yaml(self._path(sid))
        return cfg

    def get(self, session_id: str) -> PipelineConfig:
        """
        Load and return a session by ID.

        Raises:
            FileNotFoundError: If the session does not exist.
        """
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found")
        return PipelineConfig.from_yaml(path)

    def update(
        self,
        session_id: str,
        patch: Dict[str, Any],
    ) -> PipelineConfig:
        """
        Apply a partial update to an existing session.

        ``patch`` may be:
          - A nested dict matching PipelineConfig structure (deep-merged).
          - A flat dict with dot-notation keys, e.g. {"evaluation.qa_eval_threshold": 0.7}.

        Returns the updated PipelineConfig.
        """
        cfg = self.get(session_id)
        current = cfg.to_dict()

        # Expand dot-notation keys into nested dicts
        expanded = _expand_dot_keys(patch)
        _deep_merge(current, expanded)

        # Preserve session_id and timestamps
        current.setdefault("meta", {})["session_id"] = session_id

        updated = PipelineConfig.from_dict(current)
        updated.to_yaml(self._path(session_id))
        return updated

    def delete(self, session_id: str) -> None:
        """Delete a session and all its associated job files."""
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found")
        path.unlink()
        # Remove associated session directory (jobs, results, etc.)
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def fork(
        self,
        source_id: str,
        new_id: Optional[str] = None,
    ) -> PipelineConfig:
        """
        Clone an existing session with a new session_id.

        Args:
            source_id: ID of the session to clone.
            new_id:    ID for the new session. Auto-generated if not provided.

        Returns:
            PipelineConfig of the new session.
        """
        source = self.get(source_id)
        d = source.to_dict()
        new_sid = self._clean_id(new_id or f"{source_id}-fork-{str(uuid.uuid4())[:4]}")
        if self._path(new_sid).exists():
            raise ValueError(f"Session '{new_sid}' already exists")
        d["meta"]["session_id"] = new_sid
        d["meta"]["created_at"] = datetime.now().isoformat()
        d["meta"]["description"] = f"Fork of {source_id}"
        new_cfg = PipelineConfig.from_dict(d)
        new_cfg.to_yaml(self._path(new_sid))
        return new_cfg

    def list_sessions(self) -> List[Dict[str, str]]:
        """
        Return a list of all sessions as dicts with id, description, created_at.
        """
        result = []
        for path in sorted(self.sessions_dir.glob("*.yaml")):
            sid = path.stem
            try:
                cfg = PipelineConfig.from_yaml(path)
                result.append({
                    "session_id": sid,
                    "domain_name": cfg.domain.name,
                    "description": cfg.meta.description,
                    "created_at": cfg.meta.created_at,
                    "schema_version": cfg.meta.schema_version,
                })
            except Exception:
                result.append({
                    "session_id": sid,
                    "domain_name": "?",
                    "description": "(parse error)",
                    "created_at": "",
                    "schema_version": "?",
                })
        return result

    def save(self, cfg: PipelineConfig) -> None:
        """Persist a PipelineConfig under its own session_id."""
        cfg.to_yaml(self._path(cfg.meta.session_id))

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _expand_dot_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a flat dict with dot-notation keys to a nested dict.

    Example:
        {"evaluation.qa_eval_threshold": 0.7, "domain.name": "X"}
        → {"evaluation": {"qa_eval_threshold": 0.7}, "domain": {"name": "X"}}

    Non-dotted keys are left as-is.
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # If value is itself a dict and the leaf already is a dict, merge deeper
        if isinstance(value, dict) and isinstance(node.get(parts[-1]), dict):
            _deep_merge(node[parts[-1]], value)
        else:
            node[parts[-1]] = value
    return result

def _deep_merge(base: Dict, patch: Dict) -> None:
    """Recursively merge `patch` into `base` in-place."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

