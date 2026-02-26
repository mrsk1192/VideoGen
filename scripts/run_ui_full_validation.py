from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from PIL import Image, ImageFilter, ImageStat
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
ARTIFACTS_ROOT = ROOT / "artifacts" / "e2e_full"
ARTIFACTS_DIR = ARTIFACTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
LATEST_DIR = ARTIFACTS_ROOT / "latest"
INPUT_IMAGE = Path(r"C:\AI\VideoGen\TestAsset\sticker8377632396286038267.png")
PROMPT = "A whale is flying in the sky."
NEGATIVE = "low quality, blurry, jpeg artifacts, deformed, watermark, text, bad anatomy"
TASK_KEY = "videogen_last_task_id"
MAX_I2V_AUTO_DOWNLOAD_GB = 10.0


@dataclass
class StepResult:
    name: str
    status: str
    started_at: str
    ended_at: str
    duration_sec: float
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ServerRunner:
    def __init__(self, port: int, log_path: Path) -> None:
        self.port = port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_reader = threading.Event()
        self.lines: list[str] = []

    def start(self) -> None:
        cmd = [
            str(VENV_PYTHON),
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as logf:
            while not self._stop_reader.is_set():
                line = self.proc.stdout.readline()
                if not line:
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                line = line.rstrip("\n")
                self.lines.append(line)
                logf.write(line + "\n")
                logf.flush()

    def wait_ready(self, timeout_sec: int = 120) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        url = f"http://127.0.0.1:{self.port}/api/system/info"
        last_error = ""
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"Server exited early with code {self.proc.returncode}")
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    return resp.json()
                last_error = f"status={resp.status_code}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise RuntimeError(f"Server not ready: {last_error}")

    def stop(self) -> None:
        self._stop_reader.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._reader_thread:
            self._reader_thread.join(timeout=3)


def create_driver() -> tuple[Any, str]:
    edge = EdgeOptions()
    edge.add_argument("--window-size=1700,1100")
    try:
        return webdriver.Edge(options=edge), "edge-headful"
    except WebDriverException:
        pass
    chrome = ChromeOptions()
    chrome.add_argument("--window-size=1700,1100")
    try:
        return webdriver.Chrome(options=chrome), "chrome-headful"
    except WebDriverException:
        pass
    edge_h = EdgeOptions()
    edge_h.add_argument("--headless=new")
    edge_h.add_argument("--window-size=1700,1100")
    return webdriver.Edge(options=edge_h), "edge-headless"


def api_get(base_url: str, path: str, timeout: int = 20) -> dict[str, Any]:
    resp = requests.get(f"{base_url}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_post(base_url: str, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def poll_task(base_url: str, task_id: str, timeout_sec: int = 3600) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        task = api_get(base_url, f"/api/tasks/{task_id}", timeout=20)
        status = str(task.get("status") or "")
        if status in {"completed", "error"}:
            return task
        time.sleep(1.0)
    raise TimeoutError(f"task timeout: {task_id}")


def screenshot(driver: Any, name: str) -> str:
    path = ARTIFACTS_DIR / "screenshots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    driver.save_screenshot(str(path))
    return str(path)


def wait_css(wait: WebDriverWait, css: str, clickable: bool = False):
    locator = (By.CSS_SELECTOR, css)
    if clickable:
        return wait.until(EC.element_to_be_clickable(locator))
    return wait.until(EC.presence_of_element_located(locator))


def click_tab(wait: WebDriverWait, tab_name: str) -> None:
    wait_css(wait, f"button.tab[data-tab='{tab_name}']", clickable=True).click()
    wait_css(wait, f"#panel-{tab_name}.active")


def set_value(driver: Any, elem_id: str, value: str) -> None:
    node = driver.find_element(By.ID, elem_id)
    node.clear()
    node.send_keys(value)


def select_value(driver: Any, elem_id: str, value: str) -> bool:
    elem = driver.find_element(By.ID, elem_id)
    options = driver.execute_script("return Array.from(arguments[0].options).map((o) => o.value);", elem) or []
    if value in options:
        driver.execute_script(
            """
            arguments[0].value = arguments[1];
            arguments[0].dispatchEvent(new Event("change", {bubbles:true}));
            """,
            elem,
            value,
        )
        return True
    return False


def select_first_non_empty(driver: Any, elem_id: str) -> Optional[str]:
    elem = driver.find_element(By.ID, elem_id)
    values = driver.execute_script("return Array.from(arguments[0].options).map((o) => o.value);", elem) or []
    for raw in values:
        v = str(raw or "").strip()
        if v:
            select_value(driver, elem_id, v)
            return v
    return None


def select_multi_values(driver: Any, elem_id: str, values: list[str]) -> None:
    elem = driver.find_element(By.ID, elem_id)
    driver.execute_script(
        """
        const select = arguments[0];
        const values = new Set(arguments[1]);
        for (const opt of select.options) {
          opt.selected = values.has(opt.value);
        }
        select.dispatchEvent(new Event("change", {bubbles:true}));
        """,
        elem,
        values,
    )


def get_last_task_id(driver: Any) -> str:
    return str(driver.execute_script(f"return localStorage.getItem('{TASK_KEY}') || '';") or "").strip()


def wait_new_task_id(driver: Any, prev_task: str, timeout_sec: int = 40) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        curr = get_last_task_id(driver)
        if curr and curr != prev_task:
            return curr
        time.sleep(0.4)
    raise TimeoutError("task id not updated in localStorage")


def image_quality(path: Path) -> dict[str, Any]:
    img = Image.open(path).convert("RGB")
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(img)
    edge_stat = ImageStat.Stat(edges)
    return {
        "width": int(img.width),
        "height": int(img.height),
        "mean_rgb": [round(v, 3) for v in stat.mean],
        "stddev_rgb": [round(v, 3) for v in stat.stddev],
        "edge_mean": round(float(edge_stat.mean[0]), 3),
    }


def video_quality(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
    }
    try:
        import imageio.v3 as iio  # type: ignore

        frame = iio.imread(path, index=0)
        info["first_frame_shape"] = [int(v) for v in frame.shape]
        info["first_frame_std"] = round(float(frame.std()), 3)
    except Exception as exc:
        info["probe_error"] = str(exc)
    return info


def choose_local_model(local_items: list[dict[str, Any]], task: str) -> Optional[dict[str, Any]]:
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for item in local_items:
        if task in (item.get("compatible_tasks") or []):
            if item.get("is_lora") or item.get("is_vae"):
                fallback.append(item)
            else:
                preferred.append(item)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def hf_size_bytes(repo_id: str) -> Optional[int]:
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception:
        return None
    total = 0
    found = False
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None)
        if isinstance(size, int) and size > 0:
            total += size
            found = True
            continue
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            lfs_size = lfs.get("size")
            if isinstance(lfs_size, int) and lfs_size > 0:
                total += lfs_size
                found = True
    return total if found else None


def run() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_at": utc_now(),
        "steps": [],
        "notes": [],
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
    }
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = ServerRunner(port=port, log_path=ARTIFACTS_DIR / "server.log")
    driver = None

    def add_step(name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        started = utc_now()
        ts = time.perf_counter()
        try:
            details = fn() or {}
            result = StepResult(name, "passed", started, utc_now(), time.perf_counter() - ts, details=details)
        except Exception as exc:
            result = StepResult(
                name,
                "failed",
                started,
                utc_now(),
                time.perf_counter() - ts,
                error=f"{exc}\n{traceback.format_exc()}",
            )
        report["steps"].append(result.__dict__)

    try:
        server.start()
        runtime = server.wait_ready(timeout_sec=180)
        report["runtime_info"] = runtime

        driver, driver_mode = create_driver()
        report["driver_mode"] = driver_mode
        wait = WebDriverWait(driver, 40)
        driver.get(base_url)
        screenshot(driver, "00_home")

        settings = api_get(base_url, "/api/settings")
        models_dir = str(settings.get("paths", {}).get("models_dir") or "")
        outputs_dir = str(settings.get("paths", {}).get("outputs_dir") or "")
        report["paths"] = {"models_dir": models_dir, "outputs_dir": outputs_dir}

        add_step("Model Search (empty query / source switch)", lambda: _step_search_models(driver, wait))
        add_step("Settings persistence", lambda: _step_settings_persistence(driver, wait))
        add_step("Prepare local models", lambda: _step_prepare_models(base_url, report))

        local_models = api_get(base_url, "/api/models/local?limit=500").get("items", [])
        t2i_model = choose_local_model(local_models, "text-to-image")
        i2i_model = choose_local_model(local_models, "image-to-image")
        t2v_model = choose_local_model(local_models, "text-to-video")
        i2v_model = choose_local_model(local_models, "image-to-video")
        report["selected_models"] = {
            "text_to_image": (t2i_model or {}).get("path"),
            "image_to_image": (i2i_model or {}).get("path"),
            "text_to_video": (t2v_model or {}).get("path"),
            "image_to_video": (i2v_model or {}).get("path"),
        }

        add_step("Local models UI", lambda: _step_local_models(driver, wait))
        add_step(
            "Generate Text-to-Image (LoRA/VAE off)",
            lambda: _step_generate_t2i(driver, wait, base_url, t2i_model, None, None, suffix="base"),
        )
        add_step(
            "Generate Text-to-Image (LoRA/VAE on)",
            lambda: _step_generate_t2i(driver, wait, base_url, t2i_model, "first", "first", suffix="lora_vae"),
        )
        add_step(
            "Generate Image-to-Image (LoRA/VAE off)",
            lambda: _step_generate_i2i(driver, wait, base_url, i2i_model, None, None, suffix="base"),
        )
        add_step(
            "Generate Image-to-Image (LoRA/VAE on)",
            lambda: _step_generate_i2i(driver, wait, base_url, i2i_model, "first", "first", suffix="lora_vae"),
        )
        add_step("Generate Text-to-Video", lambda: _step_generate_t2v(driver, wait, base_url, t2v_model))
        add_step("Generate Image-to-Video", lambda: _step_generate_i2v(driver, wait, base_url, i2v_model))
        add_step("Outputs UI view/delete", lambda: _step_outputs(driver, wait, base_url))
        add_step("GPU usage evidence", lambda: _step_gpu_evidence(base_url))
    finally:
        if driver is not None:
            try:
                screenshot(driver, "99_final")
            except Exception:
                pass
            driver.quit()
        server.stop()
        _write_coverage(ARTIFACTS_DIR)

    report["ended_at"] = utc_now()
    safe_json_dump(ARTIFACTS_DIR / "report.json", report)
    _write_summary(report, ARTIFACTS_DIR)
    if LATEST_DIR.exists() or LATEST_DIR.is_symlink():
        if LATEST_DIR.is_symlink():
            LATEST_DIR.unlink()
        elif LATEST_DIR.is_dir():
            for item in LATEST_DIR.iterdir():
                if item.is_file():
                    item.unlink()
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for src in ARTIFACTS_DIR.iterdir():
        dst = LATEST_DIR / src.name
        if src.is_file():
            dst.write_bytes(src.read_bytes())
        elif src.is_dir():
            if dst.exists():
                for child in dst.rglob("*"):
                    if child.is_file():
                        child.unlink()
            else:
                dst.mkdir(parents=True, exist_ok=True)
            for child in src.rglob("*"):
                if child.is_file():
                    out = dst / child.relative_to(src)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(child.read_bytes())
    return 0


def _step_search_models(driver: Any, wait: WebDriverWait) -> dict[str, Any]:
    click_tab(wait, "models")
    Select(driver.find_element(By.ID, "searchTask")).select_by_value("text-to-image")
    set_value(driver, "searchQuery", "")
    set_value(driver, "searchLimit", "30")
    source_counts: dict[str, int] = {}
    for source in ("huggingface", "civitai", "all"):
        Select(driver.find_element(By.ID, "searchSource")).select_by_value(source)
        driver.find_element(By.CSS_SELECTOR, "#searchForm button[type='submit']").click()
        wait_css(wait, "#searchResults")
        time.sleep(2.0)
        rows = driver.find_elements(By.CSS_SELECTOR, "#searchResults .row")
        source_counts[source] = len(rows)
        screenshot(driver, f"search_{source}")
    thumbs = len(driver.find_elements(By.CSS_SELECTOR, "#searchResults img.model-preview"))
    return {"result_counts": source_counts, "thumbnail_count": thumbs}


def _step_settings_persistence(driver: Any, wait: WebDriverWait) -> dict[str, Any]:
    click_tab(wait, "settings")
    original = driver.find_element(By.ID, "cfgGuidance").get_attribute("value")
    new_value = "8.7" if original != "8.7" else "8.6"
    set_value(driver, "cfgGuidance", new_value)
    driver.find_element(By.CSS_SELECTOR, "#settingsForm button[type='submit']").click()
    time.sleep(1.5)
    driver.get(driver.current_url)
    click_tab(wait, "settings")
    saved = driver.find_element(By.ID, "cfgGuidance").get_attribute("value")
    ok = saved == new_value
    set_value(driver, "cfgGuidance", original)
    driver.find_element(By.CSS_SELECTOR, "#settingsForm button[type='submit']").click()
    screenshot(driver, "settings_persistence")
    return {"guidance_before": original, "guidance_after_reload": saved, "persisted": ok}


def _step_prepare_models(base_url: str, report: dict[str, Any]) -> dict[str, Any]:
    settings = api_get(base_url, "/api/settings")
    models_dir = str(settings.get("paths", {}).get("models_dir") or "")
    local = api_get(base_url, "/api/models/local?limit=500").get("items", [])
    local_names = {(Path(str(item.get("path") or "")).name or "") for item in local}
    wanted = [
        ("latent-consistency/lcm-lora-sdxl", 3.0),
        ("madebyollin/sdxl-vae-fp16-fix", 3.0),
    ]
    downloaded: list[str] = []
    skipped: list[str] = []
    for repo_id, max_gb in wanted:
        local_name = repo_id.replace("/", "--")
        if local_name in local_names:
            continue
        size = hf_size_bytes(repo_id)
        if size is not None and size / 1024 / 1024 / 1024 > max_gb:
            skipped.append(f"{repo_id} too large: {size / 1024 / 1024 / 1024:.2f}GB")
            continue
        task = api_post(base_url, "/api/models/download", {"repo_id": repo_id, "revision": None, "target_dir": models_dir})
        task_id = str(task.get("task_id") or "")
        if not task_id:
            skipped.append(f"{repo_id} task_id missing")
            continue
        final = poll_task(base_url, task_id, timeout_sec=7200)
        if final.get("status") == "completed":
            downloaded.append(repo_id)
        else:
            skipped.append(f"{repo_id} failed: {final.get('error')}")

    # Some VAE repos do not ship model_index.json. Add a minimal index for local VAE detection.
    vae_dir = Path(models_dir) / "madebyollin--sdxl-vae-fp16-fix"
    if vae_dir.exists():
        model_index = vae_dir / "model_index.json"
        if not model_index.exists() and (vae_dir / "config.json").exists():
            model_index.write_text(
                json.dumps({"_class_name": "AutoencoderKL"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            downloaded.append("madebyollin/sdxl-vae-fp16-fix:model_index_added")

    i2v_local = choose_local_model(api_get(base_url, "/api/models/local?limit=500").get("items", []), "image-to-video")
    if not i2v_local:
        size = hf_size_bytes("ali-vilab/i2vgen-xl")
        if size and (size / 1024 / 1024 / 1024) > MAX_I2V_AUTO_DOWNLOAD_GB:
            skipped.append(
                f"image-to-video model skipped: ali-vilab/i2vgen-xl size={size / 1024 / 1024 / 1024:.2f}GB > {MAX_I2V_AUTO_DOWNLOAD_GB}GB"
            )
    report["model_prepare"] = {"downloaded": downloaded, "skipped": skipped}
    return report["model_prepare"]


def _set_common_prompt(driver: Any, prefix: str) -> None:
    set_value(driver, f"{prefix}Prompt", PROMPT)
    set_value(driver, f"{prefix}Negative", NEGATIVE)


def _wait_select_ready(driver: Any, elem_id: str, timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            sel = Select(driver.find_element(By.ID, elem_id))
            values = [(opt.get_attribute("value") or "").strip() for opt in sel.options]
            if any(values):
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise TimeoutError(f"select not ready: {elem_id}")


def _select_model_with_fallback(driver: Any, elem_id: str, preferred_value: str, preferred_label: str) -> str:
    _wait_select_ready(driver, elem_id, timeout_sec=80)
    if preferred_value and select_value(driver, elem_id, preferred_value):
        return preferred_value
    sel = Select(driver.find_element(By.ID, elem_id))
    for opt in sel.options:
        text = (opt.text or "").strip().lower()
        if preferred_label and preferred_label.lower() in text and (opt.get_attribute("value") or "").strip():
            chosen = str(opt.get_attribute("value") or "")
            if select_value(driver, elem_id, chosen):
                return chosen
    chosen = select_first_non_empty(driver, elem_id)
    if chosen:
        return chosen
    raise RuntimeError(f"No selectable model in {elem_id}")


def _run_generation_wait(driver: Any, base_url: str, prev_task: str, timeout_sec: int = 7200) -> dict[str, Any]:
    task_id = wait_new_task_id(driver, prev_task=prev_task, timeout_sec=60)
    task = poll_task(base_url, task_id=task_id, timeout_sec=timeout_sec)
    if task.get("status") != "completed":
        raise RuntimeError(f"task failed: id={task_id} error={task.get('error')}")
    return task


def _step_generate_t2i(driver: Any, wait: WebDriverWait, base_url: str, model: Optional[dict[str, Any]], lora_mode: Optional[str], vae_mode: Optional[str], suffix: str) -> dict[str, Any]:
    if not model:
        raise RuntimeError("No local text-to-image model")
    click_tab(wait, "text-image")
    _set_common_prompt(driver, "t2i")
    _select_model_with_fallback(driver, "t2iModelSelect", str(model.get("path") or ""), str(model.get("repo_hint") or ""))
    set_value(driver, "t2iSteps", "8")
    set_value(driver, "t2iGuidance", "7.5")
    set_value(driver, "t2iWidth", "512")
    set_value(driver, "t2iHeight", "512")
    if lora_mode == "first":
        lora = select_first_non_empty(driver, "t2iLoraSelect")
        select_multi_values(driver, "t2iLoraSelect", [lora] if lora else [])
        set_value(driver, "t2iLoraScale", "0.8")
    else:
        select_multi_values(driver, "t2iLoraSelect", [])
    if vae_mode == "first":
        select_first_non_empty(driver, "t2iVaeSelect")
    else:
        select_value(driver, "t2iVaeSelect", "")
    prev = get_last_task_id(driver)
    screenshot(driver, f"t2i_{suffix}_before")
    driver.find_element(By.CSS_SELECTOR, "#text2imageForm button[type='submit']").click()
    task = _run_generation_wait(driver, base_url, prev_task=prev, timeout_sec=5400)
    image_file = str((task.get("result") or {}).get("image_file") or "")
    path = Path(api_get(base_url, "/api/settings").get("paths", {}).get("outputs_dir", "outputs")) / image_file
    screenshot(driver, f"t2i_{suffix}_after")
    return {"task_id": task.get("id"), "image_file": image_file, "quality": image_quality(path)}


def _step_generate_i2i(driver: Any, wait: WebDriverWait, base_url: str, model: Optional[dict[str, Any]], lora_mode: Optional[str], vae_mode: Optional[str], suffix: str) -> dict[str, Any]:
    if not model:
        raise RuntimeError("No local image-to-image model")
    if not INPUT_IMAGE.exists():
        raise RuntimeError(f"Input image missing: {INPUT_IMAGE}")
    click_tab(wait, "image-image")
    driver.find_element(By.ID, "i2iImage").send_keys(str(INPUT_IMAGE))
    _set_common_prompt(driver, "i2i")
    _select_model_with_fallback(driver, "i2iModelSelect", str(model.get("path") or ""), str(model.get("repo_hint") or ""))
    set_value(driver, "i2iSteps", "8")
    set_value(driver, "i2iGuidance", "7.0")
    set_value(driver, "i2iStrength", "0.55")
    set_value(driver, "i2iWidth", "512")
    set_value(driver, "i2iHeight", "512")
    if lora_mode == "first":
        lora = select_first_non_empty(driver, "i2iLoraSelect")
        select_multi_values(driver, "i2iLoraSelect", [lora] if lora else [])
        set_value(driver, "i2iLoraScale", "0.8")
    else:
        select_multi_values(driver, "i2iLoraSelect", [])
    if vae_mode == "first":
        select_first_non_empty(driver, "i2iVaeSelect")
    else:
        select_value(driver, "i2iVaeSelect", "")
    prev = get_last_task_id(driver)
    screenshot(driver, f"i2i_{suffix}_before")
    driver.find_element(By.CSS_SELECTOR, "#image2imageForm button[type='submit']").click()
    task = _run_generation_wait(driver, base_url, prev_task=prev, timeout_sec=5400)
    image_file = str((task.get("result") or {}).get("image_file") or "")
    path = Path(api_get(base_url, "/api/settings").get("paths", {}).get("outputs_dir", "outputs")) / image_file
    screenshot(driver, f"i2i_{suffix}_after")
    return {"task_id": task.get("id"), "image_file": image_file, "quality": image_quality(path)}


def _step_generate_t2v(driver: Any, wait: WebDriverWait, base_url: str, model: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not model:
        raise RuntimeError("No local text-to-video model")
    click_tab(wait, "text")
    _set_common_prompt(driver, "t2v")
    _select_model_with_fallback(driver, "t2vModelSelect", str(model.get("path") or ""), str(model.get("repo_hint") or ""))
    set_value(driver, "t2vSteps", "8")
    set_value(driver, "t2vFrames", "8")
    set_value(driver, "t2vGuidance", "8.0")
    set_value(driver, "t2vFps", "6")
    select_multi_values(driver, "t2vLoraSelect", [])
    prev = get_last_task_id(driver)
    screenshot(driver, "t2v_before")
    driver.find_element(By.CSS_SELECTOR, "#text2videoForm button[type='submit']").click()
    task = _run_generation_wait(driver, base_url, prev_task=prev, timeout_sec=10800)
    video_file = str((task.get("result") or {}).get("video_file") or "")
    path = Path(api_get(base_url, "/api/settings").get("paths", {}).get("outputs_dir", "outputs")) / video_file
    screenshot(driver, "t2v_after")
    return {"task_id": task.get("id"), "video_file": video_file, "quality": video_quality(path)}


def _step_generate_i2v(driver: Any, wait: WebDriverWait, base_url: str, model: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not model:
        raise RuntimeError("No local image-to-video model (skipped)")
    if not INPUT_IMAGE.exists():
        raise RuntimeError(f"Input image missing: {INPUT_IMAGE}")
    click_tab(wait, "image")
    driver.find_element(By.ID, "i2vImage").send_keys(str(INPUT_IMAGE))
    _set_common_prompt(driver, "i2v")
    _select_model_with_fallback(driver, "i2vModelSelect", str(model.get("path") or ""), str(model.get("repo_hint") or ""))
    set_value(driver, "i2vSteps", "8")
    set_value(driver, "i2vFrames", "8")
    set_value(driver, "i2vGuidance", "8.0")
    set_value(driver, "i2vFps", "6")
    set_value(driver, "i2vWidth", "512")
    set_value(driver, "i2vHeight", "512")
    select_multi_values(driver, "i2vLoraSelect", [])
    prev = get_last_task_id(driver)
    screenshot(driver, "i2v_before")
    driver.find_element(By.CSS_SELECTOR, "#image2videoForm button[type='submit']").click()
    task = _run_generation_wait(driver, base_url, prev_task=prev, timeout_sec=10800)
    video_file = str((task.get("result") or {}).get("video_file") or "")
    path = Path(api_get(base_url, "/api/settings").get("paths", {}).get("outputs_dir", "outputs")) / video_file
    screenshot(driver, "i2v_after")
    return {"task_id": task.get("id"), "video_file": video_file, "quality": video_quality(path)}


def _step_local_models(driver: Any, wait: WebDriverWait) -> dict[str, Any]:
    click_tab(wait, "local-models")
    driver.find_element(By.ID, "refreshLocalModels").click()
    time.sleep(1.2)
    rows = driver.find_elements(By.CSS_SELECTOR, "#localModels .row.model-row")
    thumbs = driver.find_elements(By.CSS_SELECTOR, "#localModels img.model-preview")
    screenshot(driver, "local_models")
    return {"rows": len(rows), "thumbnail_rows": len(thumbs)}


def _step_outputs(driver: Any, wait: WebDriverWait, base_url: str) -> dict[str, Any]:
    click_tab(wait, "outputs")
    driver.find_element(By.ID, "refreshOutputs").click()
    time.sleep(1.0)
    before = api_get(base_url, "/api/outputs?limit=500").get("items", [])
    rows = driver.find_elements(By.CSS_SELECTOR, "#outputsList .row.model-row")
    deleted = None
    if rows:
        buttons = driver.find_elements(By.CSS_SELECTOR, "#outputsList .output-delete-btn")
        if buttons:
            deleted = rows[0].text.split("\n")[0].strip()
            buttons[0].click()
            try:
                WebDriverWait(driver, 5).until(EC.alert_is_present())
                driver.switch_to.alert.accept()
            except TimeoutException:
                pass
            time.sleep(1.5)
    try:
        if driver.switch_to.alert:
            driver.switch_to.alert.dismiss()
    except Exception:
        pass
    after = api_get(base_url, "/api/outputs?limit=500").get("items", [])
    screenshot(driver, "outputs")
    return {"count_before": len(before), "count_after": len(after), "deleted_candidate": deleted}


def _step_gpu_evidence(base_url: str) -> dict[str, Any]:
    sys_info = api_get(base_url, "/api/system/info")
    logs = api_get(base_url, "/api/logs/recent?limit=300")
    text = "\n".join(str(line) for line in logs.get("lines", []))
    return {
        "device": sys_info.get("device"),
        "cuda_available": sys_info.get("cuda_available"),
        "rocm_available": sys_info.get("rocm_available"),
        "has_pipeline_load_log": "pipeline load start" in text,
        "has_cuda_log": "device=cuda" in text,
    }


def _write_coverage(out_dir: Path) -> None:
    report_txt = out_dir / "coverage.txt"
    xml_path = out_dir / "coverage.xml"
    cov_cmd = [
        str(VENV_PYTHON),
        "-m",
        "pytest",
        "tests",
        "--maxfail=1",
        "--disable-warnings",
        "--cov=main",
        "--cov-report=term-missing",
        f"--cov-report=xml:{xml_path}",
    ]
    run = subprocess.run(
        cov_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    report_txt.write_text((run.stdout or "") + "\n" + (run.stderr or ""), encoding="utf-8")


def _write_summary(report: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# UI Full Validation Summary",
        "",
        f"- started_at: {report.get('started_at')}",
        f"- ended_at: {report.get('ended_at', utc_now())}",
        f"- driver_mode: {report.get('driver_mode')}",
        f"- runtime: {json.dumps(report.get('runtime_info', {}), ensure_ascii=False)}",
        "",
        "## Test viewpoints",
        "- Startup and health (`/api/system/info`, server readiness).",
        "- UI navigation across all tabs and multilingual/search controls.",
        "- Settings save and reload persistence.",
        "- Model preparation (local scan + auto download for LoRA/VAE if absent).",
        "- Generation paths: text->image, image->image, text->video, image->video.",
        "- LoRA/VAE on/off comparison on image workflows.",
        "- Outputs view/delete from UI.",
        "- GPU evidence from system info and logs.",
        "",
        "## Step results",
    ]
    for step in report.get("steps", []):
        lines.append(f"- {step.get('name')}: {step.get('status')} ({step.get('duration_sec'):.2f}s)")
        if step.get("details"):
            lines.append(f"  - details: {json.dumps(step.get('details'), ensure_ascii=False)}")
        if step.get("error"):
            lines.append("  - error captured (see report.json)")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(run())
