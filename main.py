import copy
import inspect
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel, Field

TORCH_IMPORT_ERROR: Optional[str] = None
try:
    import torch
    from diffusers import DPMSolverMultistepScheduler, I2VGenXLPipeline, TextToVideoSDPipeline
    from diffusers.utils import export_to_video
    from PIL import Image
except Exception as exc:  # pragma: no cover
    TORCH_IMPORT_ERROR = str(exc)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "paths": {
        "models_dir": "models",
        "outputs_dir": "outputs",
        "tmp_dir": "tmp",
    },
    "huggingface": {
        "token": "",
    },
    "defaults": {
        "text2video_model": "damo-vilab/text-to-video-ms-1.7b",
        "image2video_model": "ali-vilab/i2vgen-xl",
        "num_inference_steps": 30,
        "num_frames": 16,
        "guidance_scale": 9.0,
        "fps": 8,
        "width": 512,
        "height": 512,
    },
}

TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
PIPELINES: Dict[str, Any] = {}
PIPELINES_LOCK = threading.Lock()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate.resolve()


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def desanitize_repo_id(name: str) -> str:
    return name.replace("--", "/")


def safe_in_directory(target: Path, root: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class SettingsStore:
    def __init__(self, path: Path, defaults: Dict[str, Any]) -> None:
        self._path = path
        self._defaults = copy.deepcopy(defaults)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load()
        self._write(self._settings)

    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return copy.deepcopy(self._defaults)
        try:
            content = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                return copy.deepcopy(self._defaults)
            return deep_merge(self._defaults, content)
        except Exception:
            return copy.deepcopy(self._defaults)

    def _write(self, payload: Dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._settings)

    def update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            merged = deep_merge(self._settings, updates)
            self._settings = deep_merge(self._defaults, merged)
            self._write(self._settings)
            return copy.deepcopy(self._settings)


settings_store = SettingsStore(DATA_DIR / "settings.json", DEFAULT_SETTINGS)


def ensure_runtime_dirs(settings: Dict[str, Any]) -> None:
    for key in ("models_dir", "outputs_dir", "tmp_dir"):
        resolve_path(settings["paths"][key]).mkdir(parents=True, exist_ok=True)


ensure_runtime_dirs(settings_store.get())


def detect_runtime() -> Dict[str, Any]:
    if TORCH_IMPORT_ERROR:
        return {
            "diffusers_ready": False,
            "cuda_available": False,
            "rocm_available": False,
            "device": "cpu",
            "import_error": TORCH_IMPORT_ERROR,
        }
    cuda_available = bool(torch.cuda.is_available())
    rocm_available = bool(getattr(torch.version, "hip", None))
    return {
        "diffusers_ready": True,
        "cuda_available": cuda_available,
        "rocm_available": rocm_available,
        "device": "cuda" if cuda_available else "cpu",
        "torch_version": torch.__version__,
    }


def get_device_and_dtype() -> tuple[str, Any]:
    if TORCH_IMPORT_ERROR:
        raise RuntimeError(f"Diffusers runtime is not available: {TORCH_IMPORT_ERROR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def create_task(task_type: str, message: str = "Queued") -> str:
    task_id = str(uuid.uuid4())
    now = utc_now()
    with TASKS_LOCK:
        TASKS[task_id] = {
            "id": task_id,
            "task_type": task_type,
            "status": "queued",
            "progress": 0.0,
            "message": message,
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
    return task_id


def update_task(task_id: str, **updates: Any) -> None:
    with TASKS_LOCK:
        if task_id not in TASKS:
            return
        TASKS[task_id].update(updates)
        TASKS[task_id]["updated_at"] = utc_now()


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        return copy.deepcopy(task) if task else None


def resolve_model_source(model_ref: str, settings: Dict[str, Any]) -> str:
    direct = Path(model_ref).expanduser()
    if direct.exists():
        return str(direct.resolve())
    models_dir = resolve_path(settings["paths"]["models_dir"])
    local_dir = models_dir / sanitize_repo_id(model_ref)
    if local_dir.exists():
        return str(local_dir.resolve())
    return model_ref


def get_pipeline(kind: Literal["text-to-video", "image-to-video"], model_ref: str, settings: Dict[str, Any]) -> Any:
    source = resolve_model_source(model_ref, settings)
    cache_key = f"{kind}:{source}"
    with PIPELINES_LOCK:
        if cache_key in PIPELINES:
            return PIPELINES[cache_key]
    device, dtype = get_device_and_dtype()
    if kind == "text-to-video":
        pipe = TextToVideoSDPipeline.from_pretrained(source, torch_dtype=dtype)
        if hasattr(pipe, "scheduler") and hasattr(pipe.scheduler, "config"):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe = I2VGenXLPipeline.from_pretrained(source, torch_dtype=dtype)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    with PIPELINES_LOCK:
        PIPELINES[cache_key] = pipe
    return pipe


def call_with_supported_kwargs(pipe: Any, kwargs: Dict[str, Any]) -> Any:
    accepted = inspect.signature(pipe.__call__).parameters.keys()
    filtered = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
    return pipe(**filtered)


def resolve_preview_url(model: Any) -> Optional[str]:
    model_id = getattr(model, "id", None)
    if not model_id:
        return None

    def build_url(file_name: str) -> str:
        return f"https://huggingface.co/{quote(model_id, safe='/')}/resolve/main/{quote(file_name, safe='/')}"

    card_data = getattr(model, "cardData", None)
    if isinstance(card_data, dict):
        thumbnail = card_data.get("thumbnail")
        if isinstance(thumbnail, str) and thumbnail.strip():
            thumb = thumbnail.strip()
            if thumb.startswith("http://") or thumb.startswith("https://"):
                return thumb
            return build_url(thumb.lstrip("./"))

    candidates = []
    siblings = getattr(model, "siblings", None) or []
    for sibling in siblings:
        file_name = None
        if isinstance(sibling, dict):
            file_name = sibling.get("rfilename")
        else:
            file_name = getattr(sibling, "rfilename", None)
        if isinstance(file_name, str) and file_name:
            candidates.append(file_name)

    priority = [
        "thumbnail.png",
        "thumbnail.jpg",
        "thumbnail.jpeg",
        "preview.png",
        "preview.jpg",
        "preview.jpeg",
    ]

    lower_map = {name.lower(): name for name in candidates}
    for name in priority:
        if name in lower_map:
            return build_url(lower_map[name])

    for name in candidates:
        lower = name.lower()
        if ("thumbnail" in lower or "preview" in lower) and lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return build_url(name)

    return None


def find_local_preview_relpath(model_dir: Path, models_dir: Path) -> Optional[str]:
    priority = [
        "thumbnail.png",
        "thumbnail.jpg",
        "thumbnail.jpeg",
        "preview.png",
        "preview.jpg",
        "preview.jpeg",
        "preview.webp",
    ]
    files = []
    for candidate in priority:
        path = model_dir / candidate
        if path.exists() and path.is_file():
            files.append(path)
    if not files:
        for path in model_dir.iterdir():
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                lower = path.name.lower()
                if "thumbnail" in lower or "preview" in lower:
                    files.append(path)
    if not files:
        return None
    rel = files[0].resolve().relative_to(models_dir.resolve())
    return str(rel).replace("\\", "/")


def search_hf_models(
    task: Literal["text-to-video", "image-to-video"],
    query: str,
    limit: int,
    token: Optional[str],
) -> list[Dict[str, Any]]:
    api = HfApi(token=token)
    capped_limit = min(max(limit, 1), 50)
    normalized_query = query.strip()
    if normalized_query:
        found = api.list_models(
            search=normalized_query,
            sort="downloads",
            direction=-1,
            limit=capped_limit * 6,
            full=True,
            cardData=True,
        )
    else:
        found = api.list_models(
            filter=task,
            sort="downloads",
            direction=-1,
            limit=capped_limit * 3,
            full=True,
            cardData=True,
        )
    results: list[Dict[str, Any]] = []
    for model in found:
        pipeline_tag = getattr(model, "pipeline_tag", None)
        tags = getattr(model, "tags", []) or []
        matches_task = pipeline_tag == task or task in tags
        if not matches_task:
            continue
        model_id = getattr(model, "id", "")
        if not model_id:
            continue
        results.append(
            {
                "id": model_id,
                "pipeline_tag": pipeline_tag,
                "downloads": getattr(model, "downloads", None),
                "likes": getattr(model, "likes", None),
                "private": getattr(model, "private", False),
                "model_url": f"https://huggingface.co/{quote(model_id, safe='/')}",
                "preview_url": resolve_preview_url(model),
            }
        )
        if len(results) >= capped_limit:
            break

    if not results and not normalized_query:
        fallback = api.list_models(
            search=task,
            sort="downloads",
            direction=-1,
            limit=capped_limit * 4,
            full=True,
            cardData=True,
        )
        for model in fallback:
            pipeline_tag = getattr(model, "pipeline_tag", None)
            tags = getattr(model, "tags", []) or []
            matches_task = pipeline_tag == task or task in tags
            if not matches_task:
                continue
            model_id = getattr(model, "id", "")
            if not model_id:
                continue
            results.append(
                {
                    "id": model_id,
                    "pipeline_tag": pipeline_tag,
                    "downloads": getattr(model, "downloads", None),
                    "likes": getattr(model, "likes", None),
                    "private": getattr(model, "private", False),
                    "model_url": f"https://huggingface.co/{quote(model_id, safe='/')}",
                    "preview_url": resolve_preview_url(model),
                }
            )
            if len(results) >= capped_limit:
                break

    return results


def get_filesystem_roots() -> list[Path]:
    if os.name == "nt":
        roots: list[Path] = []
        for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{drive}:/")
            if root.exists():
                roots.append(root.resolve())
        return roots
    return [Path("/").resolve()]


def resolve_browsing_directory(requested_path: str, fallback_path: str) -> Path:
    requested = (requested_path or "").strip()
    fallback = resolve_path(fallback_path)
    candidate = resolve_path(requested) if requested else fallback
    if candidate.is_file():
        candidate = candidate.parent

    probe = candidate
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent

    if probe.exists():
        return probe.resolve()

    roots = get_filesystem_roots()
    if roots:
        return roots[0]
    return fallback


class Text2VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_id: Optional[str] = None
    num_inference_steps: int = Field(default=30, ge=1, le=120)
    num_frames: int = Field(default=16, ge=8, le=128)
    guidance_scale: float = Field(default=9.0, ge=0.0, le=30.0)
    fps: int = Field(default=8, ge=1, le=60)
    seed: Optional[int] = Field(default=None, ge=0)


class DownloadRequest(BaseModel):
    repo_id: str = Field(min_length=3)
    revision: Optional[str] = None
    target_dir: Optional[str] = None


def text2video_worker(task_id: str, payload: Text2VideoRequest) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload.model_id or settings["defaults"]["text2video_model"]
    try:
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline("text-to-video", model_ref, settings)
        update_task(task_id, progress=0.35, message="Generating frames")
        device, _ = get_device_and_dtype()
        generator = None
        if payload.seed is not None:
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(payload.seed)
        out = call_with_supported_kwargs(
            pipe,
            {
                "prompt": payload.prompt,
                "negative_prompt": payload.negative_prompt or None,
                "num_inference_steps": payload.num_inference_steps,
                "num_frames": payload.num_frames,
                "guidance_scale": payload.guidance_scale,
                "generator": generator,
            },
        )
        frames = out.frames[0]
        update_task(task_id, progress=0.9, message="Encoding mp4")
        output_name = f"text2video_{task_id}.mp4"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        export_to_video(frames, str(output_path), fps=payload.fps)
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={"video_file": output_name, "model": model_ref},
        )
    except Exception as exc:
        update_task(task_id, status="error", message="Generation failed", error=str(exc))


def image2video_worker(task_id: str, payload: Dict[str, Any]) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    model_ref = payload["model_id"] or settings["defaults"]["image2video_model"]
    image_path = Path(payload["image_path"])
    try:
        update_task(task_id, status="running", progress=0.05, message="Loading model")
        pipe = get_pipeline("image-to-video", model_ref, settings)
        update_task(task_id, progress=0.35, message="Preparing image")
        image = Image.open(image_path).convert("RGB")
        width = int(payload["width"])
        height = int(payload["height"])
        if width > 0 and height > 0:
            image = image.resize((width, height))
        device, _ = get_device_and_dtype()
        generator = None
        if payload["seed"] is not None:
            gen_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(int(payload["seed"]))
        update_task(task_id, progress=0.45, message="Generating frames")
        out = call_with_supported_kwargs(
            pipe,
            {
                "prompt": payload["prompt"],
                "negative_prompt": payload["negative_prompt"] or None,
                "image": image,
                "height": height,
                "width": width,
                "target_fps": int(payload["fps"]),
                "num_inference_steps": int(payload["num_inference_steps"]),
                "num_frames": int(payload["num_frames"]),
                "guidance_scale": float(payload["guidance_scale"]),
                "generator": generator,
            },
        )
        frames = out.frames[0]
        update_task(task_id, progress=0.9, message="Encoding mp4")
        output_name = f"image2video_{task_id}.mp4"
        output_path = resolve_path(settings["paths"]["outputs_dir"]) / output_name
        export_to_video(frames, str(output_path), fps=int(payload["fps"]))
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Done",
            result={"video_file": output_name, "model": model_ref},
        )
    except Exception as exc:
        update_task(task_id, status="error", message="Generation failed", error=str(exc))
    finally:
        if image_path.exists():
            image_path.unlink(missing_ok=True)


def download_model_worker(task_id: str, repo_id: str, revision: Optional[str], target_dir: Optional[str]) -> None:
    settings = settings_store.get()
    ensure_runtime_dirs(settings)
    clean_target_dir = (target_dir or "").strip()
    base_dir_raw = clean_target_dir or settings["paths"]["models_dir"]
    models_dir = resolve_path(base_dir_raw)
    models_dir.mkdir(parents=True, exist_ok=True)
    token = settings["huggingface"].get("token") or None
    model_dir = models_dir / sanitize_repo_id(repo_id)
    try:
        update_task(task_id, status="running", progress=0.1, message=f"Downloading {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
        update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Download complete",
            result={"repo_id": repo_id, "local_path": str(model_dir.resolve()), "base_dir": str(models_dir.resolve())},
        )
    except Exception as exc:
        update_task(task_id, status="error", message="Download failed", error=str(exc))


app = FastAPI(title="ROCm VideoGen")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/system/info")
def system_info() -> Dict[str, Any]:
    return detect_runtime()


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return settings_store.get()


@app.put("/api/settings")
def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = settings_store.update(payload)
    ensure_runtime_dirs(updated)
    return updated


@app.get("/api/models/local")
def list_local_models(dir: str = "") -> Dict[str, Any]:
    settings = settings_store.get()
    models_dir = resolve_path(dir.strip() or settings["paths"]["models_dir"])
    items = []
    if models_dir.exists():
        for child in sorted(models_dir.iterdir()):
            if child.is_dir():
                items.append(
                    {
                        "name": child.name,
                        "repo_hint": desanitize_repo_id(child.name),
                        "path": str(child.resolve()),
                    }
                )
    return {"items": items, "base_dir": str(models_dir)}


@app.get("/api/models/preview")
def get_local_model_preview(rel: str) -> FileResponse:
    settings = settings_store.get()
    models_dir = resolve_path(settings["paths"]["models_dir"])
    requested = (models_dir / rel).resolve()
    if not safe_in_directory(requested, models_dir):
        raise HTTPException(status_code=400, detail="Invalid preview path")
    if requested.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(status_code=400, detail="Unsupported preview type")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    media_type = "image/webp" if requested.suffix.lower() == ".webp" else "image/jpeg"
    if requested.suffix.lower() == ".png":
        media_type = "image/png"
    return FileResponse(requested, media_type=media_type)


@app.get("/api/models/catalog")
def model_catalog(task: Literal["text-to-video", "image-to-video"], limit: int = 30) -> Dict[str, Any]:
    settings = settings_store.get()
    token = settings["huggingface"].get("token") or None
    models_dir = resolve_path(settings["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    local_items = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        repo_hint = desanitize_repo_id(child.name)
        preview_rel = find_local_preview_relpath(child, models_dir)
        preview_url = None
        if preview_rel:
            preview_url = f"/api/models/preview?rel={quote(preview_rel, safe='/')}"
        local_items.append(
            {
                "source": "local",
                "label": f"[local] {repo_hint}",
                "value": str(child.resolve()),
                "id": repo_hint,
                "preview_url": preview_url,
                "model_url": f"https://huggingface.co/{quote(repo_hint, safe='/')}",
            }
        )

    remote_items = search_hf_models(task=task, query="", limit=limit, token=token)
    dedup_remote = []
    local_repo_hints = {item["id"] for item in local_items}
    for item in remote_items:
        model_id = item.get("id")
        if not model_id:
            continue
        if model_id in local_repo_hints:
            continue
        dedup_remote.append(
            {
                "source": "remote",
                "label": f"[hf] {model_id}",
                "value": model_id,
                "id": model_id,
                "preview_url": item.get("preview_url"),
                "model_url": item.get("model_url"),
            }
        )

    default_model = settings["defaults"]["text2video_model"] if task == "text-to-video" else settings["defaults"]["image2video_model"]
    return {"items": local_items + dedup_remote, "default_model": default_model}


@app.get("/api/fs/directories")
def list_directories(path: str = "") -> Dict[str, Any]:
    settings = settings_store.get()
    current_dir = resolve_browsing_directory(path, settings["paths"]["models_dir"])
    roots = [str(root) for root in get_filesystem_roots()]

    directories = []
    try:
        for child in sorted(current_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir():
                directories.append({"name": child.name, "path": str(child.resolve())})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {current_dir}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Directory not found: {current_dir}") from exc

    parent = str(current_dir.parent.resolve()) if current_dir.parent != current_dir else None
    return {
        "current_path": str(current_dir),
        "parent_path": parent,
        "roots": roots,
        "directories": directories,
    }


@app.get("/api/models/search")
def search_models(
    task: Literal["text-to-video", "image-to-video"],
    query: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    settings = settings_store.get()
    token = settings["huggingface"].get("token") or None
    results = search_hf_models(task=task, query=query, limit=limit, token=token)
    return {"items": results}


@app.post("/api/models/download")
def download_model(req: DownloadRequest) -> Dict[str, str]:
    task_id = create_task("download", "Download queued")
    thread = threading.Thread(
        target=download_model_worker,
        args=(task_id, req.repo_id, req.revision, req.target_dir),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id}


@app.post("/api/generate/text2video")
def generate_text2video(req: Text2VideoRequest) -> Dict[str, str]:
    if TORCH_IMPORT_ERROR:
        raise HTTPException(status_code=500, detail=f"Diffusers runtime is not available: {TORCH_IMPORT_ERROR}")
    task_id = create_task("text2video", "Generation queued")
    thread = threading.Thread(target=text2video_worker, args=(task_id, req), daemon=True)
    thread.start()
    return {"task_id": task_id}


@app.post("/api/generate/image2video")
async def generate_image2video(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    model_id: str = Form(""),
    num_inference_steps: int = Form(30),
    num_frames: int = Form(16),
    guidance_scale: float = Form(9.0),
    fps: int = Form(8),
    width: int = Form(512),
    height: int = Form(512),
    seed: Optional[int] = Form(None),
) -> Dict[str, str]:
    if TORCH_IMPORT_ERROR:
        raise HTTPException(status_code=500, detail=f"Diffusers runtime is not available: {TORCH_IMPORT_ERROR}")
    settings = settings_store.get()
    tmp_dir = resolve_path(settings["paths"]["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "input.png").suffix or ".png"
    tmp_path = tmp_dir / f"{uuid.uuid4()}{suffix}"
    with tmp_path.open("wb") as out_file:
        shutil.copyfileobj(image.file, out_file)
    payload = {
        "image_path": str(tmp_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_id": model_id.strip(),
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
    }
    task_id = create_task("image2video", "Generation queued")
    thread = threading.Thread(target=image2video_worker, args=(task_id, payload), daemon=True)
    thread.start()
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> Dict[str, Any]:
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/api/videos/{file_name}")
def get_video(file_name: str) -> FileResponse:
    settings = settings_store.get()
    outputs_dir = resolve_path(settings["paths"]["outputs_dir"])
    requested = (outputs_dir / file_name).resolve()
    if not safe_in_directory(requested, outputs_dir):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(requested, media_type="video/mp4")
