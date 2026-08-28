"""
server.py
---------
FastAPI server for the multi-domain RAGAS synthetic-data pipeline.

Endpoints:

  Session lifecycle:
    POST   /api/sessions
    GET    /api/sessions
    GET    /api/sessions/{id}
    PUT    /api/sessions/{id}
    DELETE /api/sessions/{id}
    POST   /api/sessions/{id}/fork

  Config agent:
    POST   /api/sessions/{id}/auto-configure

  Validation & test:
    POST   /api/sessions/{id}/validate
    POST   /api/sessions/{id}/dry-run

  Generation:
    POST   /api/sessions/{id}/generate
    GET    /api/sessions/{id}/jobs/{job_id}
    DELETE /api/sessions/{id}/jobs/{job_id}
    GET    /api/sessions/{id}/jobs
    GET    /api/sessions/{id}/results

Run:
    uvicorn server:app --reload --port 8080
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File as FastAPIFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config_agent import ConfigIssue, run_config_agent, validate_config
from job_runner import JobRunner
from session_manager import SessionManager

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAGAS Synthetic Data Pipeline",
    description=(
        "Multi-domain synthetic QA dataset generation server. "
        "Each session carries a full PipelineConfig that specifies prompts, "
        "personas, question taxonomy, and enrichment thresholds."
    ),
    version="1.0.0",
)

# CORS — allow frontend dev server and same-origin production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", "sessions"))
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR",   "output"))

sm = SessionManager(sessions_dir=SESSIONS_DIR)
jr = JobRunner(sessions_dir=SESSIONS_DIR)

# ── Request/Response schemas ──────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    samples: Optional[List[str]] = None
    session_id: Optional[str] = None

class UpdateSessionRequest(BaseModel):
    patch: Dict[str, Any]

class ForkRequest(BaseModel):
    new_id: Optional[str] = None

class AutoConfigureRequest(BaseModel):
    description: Optional[str] = None
    samples: Optional[List[str]] = None
    chunk_pairs: Optional[List[Dict[str, str]]] = None
    save: bool = True

class DryRunRequest(BaseModel):
    chunks: List[str]
    n_questions: int = 2

class GenerateRequest(BaseModel):
    input_files: List[str]
    num_questions: int = 10

def _issue_list(issues: List[ConfigIssue]) -> List[Dict]:
    return [
        {"severity": i.severity, "field": i.field, "message": i.message}
        for i in issues
    ]

def _get_llm():
    """Build an LLM wrapper from environment variables (best-effort)."""
    try:
        import httpx
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return LangchainLLMWrapper(
            ChatOpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=api_key,
                model=os.environ.get("OPENAI_MODEL", "claude-haiku-4-5-20251001"),
                temperature=0.0,
                max_tokens=4096,
                timeout=120,
                http_client=httpx.Client(verify=False),
                http_async_client=httpx.AsyncClient(verify=False),
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM not available: {exc}")

def _get_embedding_model():
    """Build an embedding model wrapper from environment variables."""
    try:
        import httpx
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        api_key = os.environ.get("OPENAI_API_KEY", "")
        emb_url = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            return None
        return LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                base_url=emb_url,
                api_key=api_key,
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
                http_client=httpx.Client(verify=False),
                http_async_client=httpx.AsyncClient(verify=False),
            )
        )
    except Exception:
        return None

# ── Session lifecycle ─────────────────────────────────────────────────────────

@app.post("/api/sessions", status_code=201)
async def create_session(req: CreateSessionRequest):
    """
    Create a session.

    Two modes:
      1. ``config`` is provided → validate and persist directly.
      2. ``description`` and/or ``samples`` provided → run auto-configure first.
    """
    if req.config:
        try:
            cfg = sm.create(req.config, session_id=req.session_id)
            issues = validate_config(cfg)
            return {
                "session_id": cfg.meta.session_id,
                "domain_name": cfg.domain.name,
                "issues": _issue_list(issues),
            }
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    if req.description or req.samples:
        llm = _get_llm()
        try:
            cfg, issues = await run_config_agent(
                llm,
                description=req.description or "",
                samples=req.samples or [],
            )
            if req.session_id:
                cfg.meta.session_id = req.session_id
            sm.save(cfg)
            return {
                "session_id": cfg.meta.session_id,
                "domain_name": cfg.domain.name,
                "issues": _issue_list(issues),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    raise HTTPException(
        status_code=422,
        detail="Provide either 'config' or at least one of 'description'/'samples'",
    )

@app.get("/api/sessions")
def list_sessions():
    """List all sessions."""
    return sm.list_sessions()

@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Return the full PipelineConfig for a session."""
    try:
        cfg = sm.get(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return cfg.to_dict()

@app.put("/api/sessions/{session_id}")
def update_session(session_id: str, req: UpdateSessionRequest):
    """Apply a partial update to a session (nested dict or dot-notation keys)."""
    try:
        cfg = sm.update(session_id, req.patch)
        issues = validate_config(cfg)
        return {
            "session_id": cfg.meta.session_id,
            "issues": _issue_list(issues),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    """Delete a session and all its job files."""
    try:
        sm.delete(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

@app.post("/api/sessions/{session_id}/fork", status_code=201)
def fork_session(session_id: str, req: ForkRequest):
    """Clone a session with a new session_id."""
    try:
        cfg = sm.fork(session_id, new_id=req.new_id)
        return {"session_id": cfg.meta.session_id, "forked_from": session_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

# ── Config agent ──────────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/auto-configure")
async def auto_configure(session_id: str, req: AutoConfigureRequest):
    """
    Run the 4-stage config agent on corpus samples and update the session.

    The existing session is used as a base (defaults for non-generated fields).
    Pass ``save=false`` to return the draft without persisting it.
    """
    try:
        base_cfg = sm.get(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not req.description and not req.samples:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of 'description' or 'samples'",
        )

    llm = _get_llm()
    try:
        cfg, issues = await run_config_agent(
            llm,
            description=req.description or "",
            samples=req.samples or [],
            chunk_pairs=req.chunk_pairs or [],
            base_config=base_cfg,
        )
        cfg.meta.session_id = session_id
        if req.save:
            sm.save(cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "session_id": session_id,
        "config": cfg.to_dict(),
        "issues": _issue_list(issues),
        "saved": req.save,
    }

# ── Validation & dry-run ──────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/validate")
def validate_session(session_id: str):
    """Run the rule-based ValidationLayer. Returns a list of issues."""
    try:
        cfg = sm.get(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    issues = validate_config(cfg)
    return {
        "session_id": session_id,
        "valid": len([i for i in issues if i.severity == "error"]) == 0,
        "issues": _issue_list(issues),
    }

@app.post("/api/sessions/{session_id}/dry-run")
async def dry_run(session_id: str, req: DryRunRequest):
    """
    Run the pipeline on the provided chunks (max 2 questions).

    Returns sample questions + QA eval scores.
    """
    try:
        cfg = sm.get(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not req.chunks:
        raise HTTPException(status_code=422, detail="Provide at least 1 chunk")

    llm = _get_llm()
    result = await jr.dry_run(
        session_id=session_id,
        chunks=req.chunks,
        n_questions=req.n_questions,
        llm=llm,
        pipeline_config=cfg,
    )
    return result

# ── Generation ────────────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/generate", status_code=202)
async def generate(session_id: str, req: GenerateRequest):
    """
    Start a generation job.

    Returns immediately with a job_id. Poll /jobs/{job_id} for progress.
    """
    try:
        cfg = sm.get(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    issues = validate_config(cfg)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Config has errors — run /validate first",
                "issues": _issue_list(errors),
            },
        )

    llm = _get_llm()
    emb = _get_embedding_model()
    output_dir = OUTPUT_DIR / session_id

    job_id = await jr.submit_generate(
        session_id=session_id,
        input_files=req.input_files,
        num_questions=req.num_questions,
        llm=llm,
        embedding_model=emb,
        pipeline_config=cfg,
        output_dir=output_dir,
    )
    return {"job_id": job_id, "session_id": session_id, "status": "queued"}

@app.get("/api/sessions/{session_id}/jobs")
def list_jobs(session_id: str):
    """List all jobs for a session."""
    if not sm.exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return jr.list_jobs(session_id)

@app.get("/api/sessions/{session_id}/jobs/{job_id}")
def get_job(session_id: str, job_id: str):
    """Return the current status of a job."""
    status = jr.get_status(session_id, job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return status

@app.delete("/api/sessions/{session_id}/jobs/{job_id}", status_code=204)
def cancel_job(session_id: str, job_id: str):
    """Cancel a running job."""
    if not jr.cancel(session_id, job_id):
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or already finished",
        )

@app.get("/api/sessions/{session_id}/results")
def get_results(session_id: str, job_id: Optional[str] = Query(default=None)):
    """
    Download the generated dataset JSON.

    If job_id is provided, returns that job's result.
    Otherwise returns the most recent successful job's result.
    """
    if not sm.exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    result_path: Optional[Path] = None

    if job_id:
        status = jr.get_status(session_id, job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        rp = status.get("result_path")
        if rp and Path(rp).exists():
            result_path = Path(rp)
    else:
        # Find the most recent done job
        for job_status in jr.list_jobs(session_id):
            rp = job_status.get("result_path")
            if rp and Path(rp).exists():
                result_path = Path(rp)
                break

    if result_path is None:
        raise HTTPException(
            status_code=404,
            detail="No results available. Run /generate first.",
        )
    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename=f"{session_id}_dataset.json",
    )

# ── File Upload ───────────────────────────────────────────────────────────────

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = FastAPIFile(...)):
    """
    Upload document files. Returns filenames and extracted text samples.

    Supports .md, .txt, .pdf. Text is extracted for the agent mode.
    """
    filenames: List[str] = []
    samples: List[str] = []

    for f in files:
        content = await f.read()
        dest = UPLOAD_DIR / (f.filename or "unknown")
        dest.write_bytes(content)
        filenames.append(str(dest))

        # Extract sample text (first 2000 chars)
        try:
            if f.filename and f.filename.endswith(('.md', '.txt', '.rst')):
                text = content.decode('utf-8', errors='ignore')[:2000]
                samples.append(text)
            else:
                # For PDF etc, just note the filename
                samples.append(f"[Binary file: {f.filename}]")
        except Exception:
            samples.append(f"[Could not extract text from {f.filename}]")

    return {"filenames": filenames, "samples": samples}

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sm.list_sessions())}

# ── Serve frontend (production: built React app) ─────────────────────────────

_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    # Serve index.html for all non-API routes (SPA fallback)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't serve SPA for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        file_path = _frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_frontend_dist / "index.html"))

