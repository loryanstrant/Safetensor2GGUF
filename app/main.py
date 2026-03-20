"""Safetensor-to-GGUF LoRA converter – FastAPI web application."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="LoRA → GGUF Converter")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

CONVERT_SCRIPT = os.getenv(
    "CONVERT_SCRIPT_PATH",
    "/opt/llama.cpp/convert_lora_to_gguf.py",
)

# In-memory store for running / completed jobs (kept simple on purpose).
jobs: dict[str, dict[str, Any]] = {}


# ── HTML front-end ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


# ── File / directory browser (AJAX) ────────────────────────────────────────
@app.get("/api/browse")
async def browse(path: str = "/") -> JSONResponse:
    """Return the contents of *path* so the UI can offer a file-picker."""
    target = Path(path)
    if not target.exists():
        return JSONResponse({"error": "Path does not exist"}, status_code=404)

    if target.is_file():
        return JSONResponse({"type": "file", "path": str(target)})

    entries: list[dict[str, str]] = []
    parent = str(target.parent) if str(target) != "/" else "/"
    entries.append({"name": "..", "path": parent, "type": "directory"})
    try:
        for child in sorted(target.iterdir()):
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "type": "directory" if child.is_dir() else "file",
                }
            )
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)

    return JSONResponse({"type": "directory", "path": str(target), "entries": entries})


# ── Conversion endpoint ────────────────────────────────────────────────────
@app.post("/api/convert")
async def convert(request: Request) -> JSONResponse:
    """Launch ``convert_lora_to_gguf.py`` as an async subprocess."""
    body: dict[str, Any] = await request.json()

    lora_path: str = body.get("lora_path", "").strip()
    base_model: str = body.get("base_model", "").strip()
    outtype: str = body.get("outtype", "f16").strip()
    outfile: str = body.get("outfile", "").strip()
    verbose: bool = body.get("verbose", False)

    if not lora_path:
        return JSONResponse({"error": "lora_path is required"}, status_code=400)

    if outtype not in {"f32", "f16", "bf16", "q8_0", "auto"}:
        return JSONResponse({"error": "Invalid outtype"}, status_code=400)

    lora_dir = Path(lora_path)
    if not lora_dir.is_dir():
        return JSONResponse({"error": "lora_path must be an existing directory"}, status_code=400)

    if base_model:
        base_path = Path(base_model)
        if not base_path.exists():
            return JSONResponse({"error": "base model path does not exist"}, status_code=400)

    cmd: list[str] = ["python3", CONVERT_SCRIPT]
    if base_model:
        cmd += ["--base", base_model]
    cmd += ["--outtype", outtype]
    if outfile:
        cmd += ["--outfile", outfile]
    if verbose:
        cmd.append("--verbose")
    cmd.append(lora_path)

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running", "log": "", "cmd": " ".join(cmd)}

    asyncio.create_task(_run_job(job_id, cmd))

    return JSONResponse({"job_id": job_id})


async def _run_job(job_id: str, cmd: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace")
        output_lines.append(line)
        jobs[job_id]["log"] = "".join(output_lines)

    await proc.wait()
    jobs[job_id]["status"] = "success" if proc.returncode == 0 else "failed"
    jobs[job_id]["returncode"] = proc.returncode


# ── Job status polling ──────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(
        {
            "status": job["status"],
            "log": job["log"],
            "returncode": job.get("returncode"),
        }
    )
