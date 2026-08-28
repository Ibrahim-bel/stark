"""
job_runner.py
-------------
Async job execution for the pipeline server.

Jobs are background asyncio tasks. Progress and status are written to
sessions/{session_id}/jobs/{job_id}.json so the server can poll them
without blocking.

Usage (from server.py):
    runner = JobRunner(sessions_dir=Path("sessions"))
    job_id = await runner.submit_generate(session_id, input_files, num_questions, llm, ...)
    status = runner.get_status(session_id, job_id)
    runner.cancel(session_id, job_id)
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Job status schema ─────────────────────────────────────────────────────────

class JobStatus:
    QUEUED    = "queued"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"

def _now() -> str:
    return datetime.now().isoformat()

# ── JobRunner ─────────────────────────────────────────────────────────────────

class JobRunner:
    """
    Manages background generation jobs.

    Progress is persisted to JSON so the server can serve status queries
    without holding state in memory.
    """

    def __init__(self, sessions_dir: Path = Path("sessions")) -> None:
        self.sessions_dir = Path(sessions_dir)
        self._tasks: Dict[str, asyncio.Task] = {}  # job_id → task

    # ── Paths ─────────────────────────────────────────────────────────────────

    def _jobs_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id / "jobs"

    def _job_path(self, session_id: str, job_id: str) -> Path:
        return self._jobs_dir(session_id) / f"{job_id}.json"

    def _results_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id / "results"

    # ── Status I/O ────────────────────────────────────────────────────────────

    def _write_status(self, session_id: str, job_id: str, data: dict) -> None:
        path = self._job_path(session_id, job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def get_status(self, session_id: str, job_id: str) -> Optional[Dict]:
        """Return the current job status dict, or None if not found."""
        path = self._job_path(session_id, job_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_jobs(self, session_id: str) -> List[Dict]:
        """Return all job status dicts for a session, newest first."""
        jobs_dir = self._jobs_dir(session_id)
        if not jobs_dir.exists():
            return []
        result = []
        for p in sorted(jobs_dir.glob("*.json"), reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    result.append(json.load(f))
            except Exception:
                pass
        return result

    # ── Submit ────────────────────────────────────────────────────────────────

    async def submit_generate(
        self,
        session_id: str,
        input_files: List[str],
        num_questions: int,
        llm: Any,
        embedding_model: Any,
        pipeline_config: Any,
        output_dir: Path,
    ) -> str:
        """
        Start a generation job in the background.

        Returns the job_id immediately (job runs asynchronously).
        """
        job_id = str(uuid.uuid4())[:8]
        initial = {
            "job_id":     job_id,
            "session_id": session_id,
            "type":       "generate",
            "status":     JobStatus.QUEUED,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "progress":   {"done": 0, "total": num_questions},
            "input_files": input_files,
            "num_questions_requested": num_questions,
            "num_questions_generated": 0,
            "errors":     [],
            "result_path": None,
            "output_dir": str(output_dir),
        }
        self._write_status(session_id, job_id, initial)

        task = asyncio.create_task(
            self._run_generate(
                session_id, job_id, input_files, num_questions,
                llm, embedding_model, pipeline_config, output_dir,
            )
        )
        self._tasks[job_id] = task
        return job_id

    # ── Cancel ────────────────────────────────────────────────────────────────

    def cancel(self, session_id: str, job_id: str) -> bool:
        """
        Request cancellation of a running job.

        Returns True if the task was found and cancelled, False otherwise.
        """
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            status = self.get_status(session_id, job_id) or {}
            status["status"] = JobStatus.CANCELLED
            status["finished_at"] = _now()
            self._write_status(session_id, job_id, status)
            return True
        return False

    # ── Background coroutine ──────────────────────────────────────────────────

    async def _run_generate(
        self,
        session_id: str,
        job_id: str,
        input_files: List[str],
        num_questions: int,
        llm: Any,
        embedding_model: Any,
        pipeline_config: Any,
        output_dir: Path,
    ) -> None:
        status = self.get_status(session_id, job_id) or {}
        status["status"]     = JobStatus.RUNNING
        status["started_at"] = _now()
        self._write_status(session_id, job_id, status)

        try:
            # Import pipeline components lazily to avoid circular deps
            from knowledge_graph import KnowledgeGraphBuilder, KnowledgeGraphStorage
            from question_generator import QuestionGenerator
            from pipeline_config import build_ragas_personas
            from document_preprocessor import ensure_markdown_files

            # Étape de prétraitement : convertir tout document non-markdown
            # (PDF, DOCX, PPTX, ...) en .md structuré via Docling Serve, afin
            # que le reste du pipeline ne manipule que des .md.
            existing = [f for f in input_files if Path(f).exists()]
            if not existing:
                raise ValueError(f"None of the input files exist: {input_files}")

            _MD_EXTS = {".md", ".markdown"}
            needs_conversion = [
                f for f in existing
                if Path(f).suffix.lower() not in _MD_EXTS
            ]
            if needs_conversion:
                status["status"] = JobStatus.RUNNING
                status["stage"] = "converting_documents"
                status["converting"] = [Path(f).name for f in needs_conversion]
                self._write_status(session_id, job_id, status)
                logging.warning(
                    "[PREPROCESS] Converting %d non-markdown document(s) via Docling: %s",
                    len(needs_conversion),
                    needs_conversion,
                )

            def _progress(name: str, done: int, total: int) -> None:
                status["converting_progress"] = {
                    "current": name,
                    "done": done,
                    "total": total,
                }
                self._write_status(session_id, job_id, status)

            output_dir.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_event_loop()
            _convert = functools.partial(
                ensure_markdown_files,
                existing,
                output_dir / "markdown",
                None,
                _progress,
            )
            md_files = await loop.run_in_executor(None, _convert)
            status["stage"] = "building_knowledge_graph"
            status["markdown_files"] = [str(p) for p in md_files]
            self._write_status(session_id, job_id, status)
            if not md_files:
                raise ValueError(f"None of the input files exist: {input_files}")

            kg_builder = KnowledgeGraphBuilder(
                llm=llm,
                embedding_model=embedding_model,
                config=pipeline_config,
            )

            # Store domaine : 1 JSON par doc (chunks + métadonnées + keyphrases),
            # partagé entre toutes les sessions. Réutilisé pour réhydrater les
            # keyphrases des docs inchangés (le LLM ne tourne que sur les nouveaux).
            kg_store_dir = Path(os.environ.get("STARK_KG_STORE_DIR", "kg_store"))

            kg = kg_builder.create_from_markdown_files(md_files)
            kg = kg_builder.enrich_prechunked(kg, store_dir=kg_store_dir)

            try:
                written = kg_builder.save_doc_store(kg, kg_store_dir)
                logging.warning("[DOC_STORE] %d document JSON(s) written to %s", len(written), kg_store_dir)
            except Exception as _store_exc:
                logging.warning("[DOC_STORE] failed: %s — continuing.", _store_exc)

            # Step 2.5 : Enrichissement universel (knowledge_graph/rahulnyk)
            # Méthodes domain-agnostic, pilotées par les flags universal_* de
            # enrich_modules. Bloc try/except : n'interrompt jamais le pipeline.
            _emu = getattr(pipeline_config, "enrich_modules", None)
            _u_prox = getattr(_emu, "universal_proximity", False) if _emu else False
            _u_metr = getattr(_emu, "universal_metrics", False) if _emu else False
            _u_comm = getattr(_emu, "universal_communities", False) if _emu else False
            _u_trip = getattr(_emu, "universal_triplets", False) if _emu else False
            _u_conc = getattr(_emu, "universal_concepts", False) if _emu else False
            _u_trel = getattr(_emu, "universal_triplet_relations", False) if _emu else False
            # universal_triplet_relations nécessite les triplets extraits d'abord.
            if _u_trel and not _u_trip:
                logging.warning(
                    "[UNIVERSAL] universal_triplet_relations=True force "
                    "universal_triplets=True (les triplets doivent être extraits).")
                _u_trip = True
            if any([_u_prox, _u_metr, _u_comm, _u_trip, _u_conc, _u_trel]):
                logging.warning(
                    "[UNIVERSAL] Enrichissement universel [proximity=%s, metrics=%s, "
                    "communities=%s, triplets=%s, concepts=%s, triplet_relations=%s]...",
                    _u_prox, _u_metr, _u_comm, _u_trip, _u_conc, _u_trel)
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
                        add_tfidf_filter=False,
                    )
                    logging.warning("[UNIVERSAL] Done: %s", u_stats)
                except Exception as _univ_exc:
                    logging.warning(
                        "[UNIVERSAL] Enrichment failed: %s — continuing without.",
                        _univ_exc)
                    logging.warning(traceback.format_exc())

            # Step 3: KG Agent enrichment (DirectRelationDiscovery)
            # Must call async methods directly since we're already in an async context
            _use_kg_agent = os.environ.get("USE_KG_AGENT", "false").lower() == "true"
            logging.warning("[KG_AGENT] USE_KG_AGENT=%s", _use_kg_agent)
            if _use_kg_agent:
                logging.warning("[KG_AGENT] Running agentic enrichment (async)...")
                try:
                    from kg_agent import RelationValidator, DirectRelationDiscovery
                    # Validate existing relations
                    validator = RelationValidator(config=pipeline_config)
                    kg, val_stats = await validator._validate_async(kg, llm, None, None)
                    logging.warning("[KG_AGENT] Validation done: %s", val_stats)
                    # Discover new inter-document relations
                    discoverer = DirectRelationDiscovery(config=pipeline_config)
                    kg, disc_stats = await discoverer._discover_async(kg, llm, None)
                    logging.warning("[KG_AGENT] Discovery done: %s", disc_stats)
                except Exception as _agent_exc:
                    logging.warning("[KG_AGENT] Enrichment failed: %s — continuing without agent relations.", _agent_exc)
                    logging.warning(traceback.format_exc())
            else:
                logging.warning("[KG_AGENT] Skipping agentic enrichment (USE_KG_AGENT=false).")

            # Save KG
            output_dir.mkdir(parents=True, exist_ok=True)
            kg_path = output_dir / "knowledge_graph.json"
            KnowledgeGraphStorage.save(kg, kg_path)

            # Step 4 : RAKG Retrospective Entity-Centric KG enrichment
            # Ajoute des relations retrospective_entity dans kg.relationships.
            # Aucun autre composant STARK modifié.
            _retro_cfg = getattr(pipeline_config, "retrospective", None)
            if _retro_cfg is not None and getattr(_retro_cfg, "enabled", False):
                status["stage"] = "retrospective"
                status.setdefault("progress", {})["pct"] = 50
                self._write_status(session_id, job_id, status)
                logging.warning("[RETROSPECTIVE] Running entity-centric KG enrichment (RAKG §III-D)...")
                try:
                    from entity_retrospective import discover_relations_retrospective
                    retro_checkpoint = output_dir / "retrospective_checkpoint.jsonl"
                    kg = await discover_relations_retrospective(
                        kg=kg,
                        config=pipeline_config,
                        llm=llm,
                        checkpoint_path=retro_checkpoint,
                        embedding_model=embedding_model,
                    )
                    # Re-sauvegarder le KG enrichi
                    KnowledgeGraphStorage.save(kg, kg_path)
                    logging.warning("[RETROSPECTIVE] Done — KG re-saved.")
                except Exception as _retro_exc:
                    logging.warning(
                        "[RETROSPECTIVE] Enrichment failed: %s — continuing without retrospective relations.",
                        _retro_exc,
                    )
                    logging.warning(traceback.format_exc())
            else:
                logging.warning("[RETROSPECTIVE] Skipping (retrospective.enabled=false).")

            status["progress"]["total"] = num_questions
            self._write_status(session_id, job_id, status)

            question_gen = QuestionGenerator(llm=llm, config=pipeline_config)
            personas     = build_ragas_personas(pipeline_config)
            checkpoint   = output_dir / "questions_checkpoint.json"

            questions = await question_gen.generate(
                kg=kg,
                persona_list=personas,
                num_questions=num_questions,
                checkpoint_path=checkpoint,
            )

            result_path = output_dir / "dataset.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": "2.0",
                        "session_id": session_id,
                        "job_id": job_id,
                        "num_questions": len(questions),
                        "questions": questions,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            status["status"]                 = JobStatus.DONE
            status["finished_at"]            = _now()
            status["num_questions_generated"] = len(questions)
            status["progress"]["done"]        = len(questions)
            status["result_path"]             = str(result_path)

        except asyncio.CancelledError:
            status["status"]     = JobStatus.CANCELLED
            status["finished_at"] = _now()
        except Exception as exc:
            status["status"]     = JobStatus.FAILED
            status["finished_at"] = _now()
            status["errors"].append({
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            logging.error("Job %s/%s failed: %s", session_id, job_id, exc)
        finally:
            self._write_status(session_id, job_id, status)
            self._tasks.pop(job_id, None)

    # ── Dry-run ───────────────────────────────────────────────────────────────

    async def dry_run(
        self,
        session_id: str,
        chunks: List[str],
        n_questions: int,
        llm: Any,
        pipeline_config: Any,
    ) -> Dict:
        """
        Run the pipeline on a small subset (2 triplets max) and return sample Q&As.

        Returns:
            {questions: [...], issues: [...], elapsed_seconds: float}
        """
        import time
        t0 = time.time()

        try:
            from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
            from knowledge_graph import KnowledgeGraphBuilder
            from question_generator import QuestionGenerator
            from pipeline_config import build_ragas_personas

            cfg = pipeline_config

            # Build a tiny KG from the provided chunks
            kg = KnowledgeGraph()
            doc_node = Node(
                type=NodeType.DOCUMENT,
                properties={"filename": "dry_run.md", "page_content": "\n\n".join(chunks)},
            )
            kg.nodes.append(doc_node)
            for idx, chunk_text in enumerate(chunks):
                chunk_node = Node(
                    type=NodeType.CHUNK,
                    properties={
                        "raw_content": chunk_text,
                        "page_content": chunk_text,
                        "breadcrumb": f"Chunk {idx + 1}",
                        "parent_doc": "dry_run.md",
                        "chunk_index": idx,
                        "token_count": len(chunk_text.split()),
                        "is_subdivision": False,
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

            # Run a lightweight enrichment (keyphrases only — no cosine matrix on 2 chunks)
            kb = KnowledgeGraphBuilder(llm=llm, config=cfg)
            try:
                kg = kb.enrich_prechunked(kg)
            except Exception as e:
                logging.warning("Dry-run: enrich_prechunked failed (%s) — proceeding anyway", e)

            personas = build_ragas_personas(cfg)
            question_gen = QuestionGenerator(llm=llm, config=cfg)
            questions = await question_gen.generate(
                kg=kg,
                persona_list=personas,
                num_questions=min(n_questions, 2),
            )

            return {
                "questions": questions,
                "num_generated": len(questions),
                "elapsed_seconds": round(time.time() - t0, 1),
                "error": None,
            }

        except Exception as exc:
            return {
                "questions": [],
                "num_generated": 0,
                "elapsed_seconds": round(time.time() - t0, 1),
                "error": str(exc),
            }

