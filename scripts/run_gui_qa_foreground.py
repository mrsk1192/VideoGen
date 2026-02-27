from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

import requests
from PIL import Image, ImageFilter, ImageStat
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
START_BAT = ROOT / "start.bat"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
INPUT_IMAGE = Path(r"C:\AI\VideoGen\TestAsset\sticker8377632396286038267.png")
PROMPT_BASE = "A majestic humpback whale flying in a bright blue sky with soft clouds, cinematic, ultra detailed."
TASK_KEY = "videogen_last_task_id"
ART_ROOT = ROOT / "artifacts" / "gui_qa_foreground"
ART_DIR = ART_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
LATEST_DIR = ART_ROOT / "latest"
VIDEO_TARGET_SECONDS = 5.0
VIDEO_TARGET_FRAMES = 30
VIDEO_TARGET_FPS = 6

RECOMMENDED_MODEL_KEYS: dict[str, list[str]] = {
    "t2i": ["realvis", "sg161222--realvisxl_v5.0", "stable-diffusion-xl-base-1.0"],
    "i2i": ["realvis", "sg161222--realvisxl_v5.0", "stable-diffusion-xl-base-1.0"],
    "t2v": ["damo-vilab--text-to-video-ms-1.7b", "ali-vilab--text-to-video-ms-1.7b", "text-to-video-ms-1.7b"],
    "i2v": ["ali-vilab--i2vgen-xl", "i2vgen-xl"],
}


@dataclass
class StepResult:
    name: str
    status: str
    started_at: str
    ended_at: str
    duration_sec: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ensure_coverage_bootstrap(bootstrap_dir: Path, coverage_file: Path) -> tuple[Path, Path]:
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = bootstrap_dir / "sitecustomize.py"
    sitecustomize.write_text(
        "import os\n"
        "if os.getenv('COVERAGE_PROCESS_START'):\n"
        "    try:\n"
        "        import coverage\n"
        "        coverage.process_startup()\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    rc = bootstrap_dir / ".coveragerc"
    rc.write_text(
        "[run]\n"
        "branch = True\n"
        "parallel = True\n"
        "source = main\n"
        f"data_file = {str(coverage_file).replace(chr(92), '/')}\n"
        "\n"
        "[report]\n"
        "skip_covered = False\n",
        encoding="utf-8",
    )
    return sitecustomize, rc


class BatchServer:
    def __init__(self, log_path: Path, env: dict[str, str]) -> None:
        self.log_path = log_path
        self.env = env
        self.proc: Optional[subprocess.Popen[str]] = None
        self.port: Optional[int] = None
        self.lines: list[str] = []
        self.stop_read = threading.Event()
        self.reader: Optional[threading.Thread] = None

    def start(self) -> None:
        cmd = ["cmd.exe", "/c", str(START_BAT)]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self.env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _read_loop(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as f:
            for raw in self.proc.stdout:
                line = raw.rstrip("\n")
                self.lines.append(line)
                m = re.search(r"Starting server at http://localhost:(\d+)", line)
                if m:
                    self.port = int(m.group(1))
                f.write(line + "\n")
                f.flush()
                if self.stop_read.is_set():
                    break

    def wait_ready(self, timeout_sec: int = 240) -> str:
        end = time.time() + timeout_sec
        last_err = ""
        while time.time() < end:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"start.bat terminated early: code={self.proc.returncode}")
            if self.port:
                base = f"http://127.0.0.1:{self.port}"
                try:
                    r = requests.get(f"{base}/api/system/info", timeout=2)
                    if r.status_code == 200:
                        return base
                    last_err = f"status={r.status_code}"
                except Exception as exc:
                    last_err = str(exc)
            time.sleep(0.5)
        raise TimeoutError(f"server not ready: {last_err}")

    def stop(self) -> None:
        self.stop_read.set()
        if self.proc and self.proc.poll() is None:
            try:
                # Prefer graceful Ctrl+C so coverage data can be flushed.
                self.proc.send_signal(signal.CTRL_C_EVENT)
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                    self.proc.wait(timeout=10)
                except Exception:
                    self.proc.terminate()
            if self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        if self.reader:
            self.reader.join(timeout=3)


def ping_system_info(base_url: str, timeout_sec: float = 2.0) -> tuple[bool, str]:
    try:
        r = requests.get(f"{base_url}/api/system/info", timeout=timeout_sec)
        if r.status_code == 200:
            return True, "ok"
        return False, f"status={r.status_code}"
    except Exception as exc:
        return False, str(exc)


def create_driver() -> tuple[Any, str]:
    opts = EdgeOptions()
    opts.add_argument("--window-size=1700,1100")
    try:
        return webdriver.Edge(options=opts), "edge-headful"
    except WebDriverException as exc:
        raise RuntimeError("Edge headful 起動に失敗しました。") from exc


def screenshot(driver: Any, name: str) -> str:
    path = ART_DIR / "screenshots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    driver.save_screenshot(str(path))
    return str(path)


def click_tab(wait: WebDriverWait, tab: str) -> None:
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"button.tab[data-tab='{tab}']"))).click()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"#panel-{tab}.active")))


def set_value(driver: Any, elem_id: str, value: str) -> None:
    node = driver.find_element(By.ID, elem_id)
    node.clear()
    node.send_keys(value)


def select_value_js(driver: Any, elem_id: str, value: str) -> bool:
    elem = driver.find_element(By.ID, elem_id)
    values = driver.execute_script("return Array.from(arguments[0].options).map(o => o.value);", elem) or []
    if value not in values:
        return False
    driver.execute_script(
        "arguments[0].value=arguments[1]; arguments[0].dispatchEvent(new Event('change',{bubbles:true}));",
        elem,
        value,
    )
    return True


def select_first_non_empty(driver: Any, elem_id: str) -> Optional[str]:
    elem = driver.find_element(By.ID, elem_id)
    values = driver.execute_script("return Array.from(arguments[0].options).map(o => o.value);", elem) or []
    for raw in values:
        value = str(raw or "").strip()
        if value and select_value_js(driver, elem_id, value):
            return value
    return None


def select_preferred_option(driver: Any, elem_id: str, preferred_keys: list[str]) -> Optional[str]:
    elem = driver.find_element(By.ID, elem_id)
    rows = (
        driver.execute_script(
            "return Array.from(arguments[0].options).map(o => ({value: o.value || '', text: o.text || ''}));",
            elem,
        )
        or []
    )
    keys = [str(k).strip().lower() for k in preferred_keys if str(k).strip()]
    for row in rows:
        value = str((row or {}).get("value") or "").strip()
        if not value:
            continue
        label = str((row or {}).get("text") or "").strip()
        haystack = f"{value} {label}".lower()
        if any(key in haystack for key in keys) and select_value_js(driver, elem_id, value):
            return value
    return select_first_non_empty(driver, elem_id)


def select_multi(driver: Any, elem_id: str, values: list[str]) -> None:
    elem = driver.find_element(By.ID, elem_id)
    driver.execute_script(
        """
        const select = arguments[0];
        const wanted = new Set(arguments[1]);
        for (const opt of select.options) opt.selected = wanted.has(opt.value);
        select.dispatchEvent(new Event("change",{bubbles:true}));
        """,
        elem,
        values,
    )


def read_task_id(driver: Any) -> str:
    return str(driver.execute_script(f"return localStorage.getItem('{TASK_KEY}') || '';") or "")


def wait_new_task_id(driver: Any, prev: str, timeout_sec: int = 120) -> str:
    end = time.time() + timeout_sec
    while time.time() < end:
        cur = read_task_id(driver)
        if cur and cur != prev:
            return cur
        time.sleep(0.4)
    raise TimeoutError("task id was not updated")


def wait_task_done_ui(driver: Any, timeout_sec: int = 21600) -> str:
    end = time.time() + timeout_sec
    node = driver.find_element(By.ID, "taskStatus")
    while time.time() < end:
        text = (node.text or "").strip()
        lower = text.lower()
        if ("status=完了" in text) or ("status=completed" in lower):
            return text
        if ("status=エラー" in text) or ("status=error" in lower) or ("error=" in lower):
            raise RuntimeError(f"task ended with error: {text}")
        time.sleep(1.0)
    raise TimeoutError(f"task timeout: {node.text}")


def parse_filename_from_url(url: str) -> str:
    if not url:
        return ""
    name = Path(urlparse(url).path).name
    return unquote(name) if name else ""


class ServerHealthMonitor:
    def __init__(self, base_url: str, log_path: Path, interval_sec: float = 2.0) -> None:
        self.base_url = base_url
        self.log_path = log_path
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.total_checks = 0
        self.fail_checks = 0
        self.last_error = ""

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as f:
            while not self.stop_event.is_set():
                ok, msg = ping_system_info(self.base_url, timeout_sec=2.0)
                self.total_checks += 1
                if not ok:
                    self.fail_checks += 1
                    self.last_error = msg
                f.write(
                    json.dumps(
                        {
                            "ts": utc_now(),
                            "ok": ok,
                            "message": msg,
                            "total_checks": self.total_checks,
                            "fail_checks": self.fail_checks,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()
                time.sleep(self.interval_sec)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "fail_checks": self.fail_checks,
            "last_error": self.last_error,
        }


def image_quality(path: Path) -> dict[str, Any]:
    img = Image.open(path).convert("RGB")
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(img)
    edge = ImageStat.Stat(edges)
    return {
        "width": int(img.width),
        "height": int(img.height),
        "mean_rgb": [round(v, 3) for v in stat.mean],
        "stddev_rgb": [round(v, 3) for v in stat.stddev],
        "edge_mean": round(float(edge.mean[0]), 3),
    }


def video_quality(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"size_mb": round(path.stat().st_size / 1024 / 1024, 3)}
    try:
        import imageio.v3 as iio  # type: ignore

        frame = iio.imread(path, index=0)
        out["first_frame_shape"] = [int(v) for v in frame.shape]
        out["first_frame_std"] = round(float(frame.std()), 3)
    except Exception as exc:
        out["probe_error"] = str(exc)
    return out


def coverage_report(coverage_file: Path, out_dir: Path) -> None:
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_file)
    subprocess.run(
        [str(PYTHON), "-m", "coverage", "combine", str(coverage_file.parent)],
        cwd=str(ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    rep = subprocess.run(
        [str(PYTHON), "-m", "coverage", "report", "-m"],
        cwd=str(ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    (out_dir / "coverage_gui.txt").write_text((rep.stdout or "") + "\n" + (rep.stderr or ""), encoding="utf-8")
    subprocess.run(
        [str(PYTHON), "-m", "coverage", "xml", "-o", str(out_dir / "coverage_gui.xml")],
        cwd=str(ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def run() -> int:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "started_at": utc_now(),
        "input_image": str(INPUT_IMAGE),
        "requirements": {
            "gui_only": True,
            "selenium_foreground": True,
            "start_from_batch": True,
            "keep_models": True,
            "keep_outputs": True,
        },
        "steps": [],
    }
    if not INPUT_IMAGE.exists():
        raise RuntimeError(f"input image missing: {INPUT_IMAGE}")

    bootstrap_dir = ART_DIR / "coverage_bootstrap"
    coverage_file = ART_DIR / ".coverage"
    _, rc = ensure_coverage_bootstrap(bootstrap_dir, coverage_file)

    env = os.environ.copy()
    env["AUTO_OPEN_BROWSER"] = "0"
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(free_port())
    env["QA_COVERAGE"] = "1"
    env["COVERAGE_PROCESS_START"] = str(rc)
    env["COVERAGE_FILE"] = str(coverage_file)
    env["PYTHONPATH"] = str(bootstrap_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    server = BatchServer(log_path=ART_DIR / "server.log", env=env)
    monitor: Optional[ServerHealthMonitor] = None
    driver = None
    base_url = ""

    def step(name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        st = utc_now()
        t0 = time.perf_counter()
        try:
            details = fn() or {}
            result["steps"].append(
                StepResult(
                    name=name,
                    status="passed",
                    started_at=st,
                    ended_at=utc_now(),
                    duration_sec=time.perf_counter() - t0,
                    details=details,
                ).__dict__
            )
        except Exception as exc:
            result["steps"].append(
                StepResult(
                    name=name,
                    status="failed",
                    started_at=st,
                    ended_at=utc_now(),
                    duration_sec=time.perf_counter() - t0,
                    details={},
                    error=f"{exc}\n{traceback.format_exc()}",
                ).__dict__
            )

    try:
        server.start()
        base_url = server.wait_ready(timeout_sec=300)
        result["base_url"] = base_url
        result["start_bat_port"] = server.port
        monitor = ServerHealthMonitor(base_url=base_url, log_path=ART_DIR / "server_monitor.log", interval_sec=5.0)
        monitor.start()

        driver, driver_mode = create_driver()
        result["driver_mode"] = driver_mode
        wait = WebDriverWait(driver, 60)
        driver.get(base_url)
        screenshot(driver, "00_home")

        def wait_task_done_monitored(task_id: str, timeout_sec: int = 21600, stale_sec: int = 600) -> str:
            end = time.time() + timeout_sec
            node = driver.find_element(By.ID, "taskStatus")
            next_health = 0.0
            unhealthy_since: Optional[float] = None
            next_task_poll = 0.0
            last_task_updated_at = ""
            stale_since: Optional[float] = None
            while time.time() < end:
                now = time.time()
                if now >= next_health:
                    ok, msg = ping_system_info(base_url, timeout_sec=2.0)
                    if not ok:
                        if unhealthy_since is None:
                            unhealthy_since = now
                        # During heavy model load/generation, API can be briefly unresponsive.
                        # Fail only when this persists for a long period.
                        if now - unhealthy_since > 300:
                            raise RuntimeError(f"server health check failed for >300s while waiting task: {msg}")
                    else:
                        unhealthy_since = None
                    next_health = now + 2.0
                if now >= next_task_poll:
                    try:
                        resp = requests.get(f"{base_url}/api/tasks/{task_id}", timeout=3)
                        if resp.status_code == 200:
                            payload = resp.json()
                            status = str(payload.get("status") or "").lower()
                            updated_at = str(payload.get("updated_at") or "")
                            if status == "completed":
                                return f"status=completed progress={payload.get('progress')}"
                            if status == "error":
                                raise RuntimeError(f"task ended with error: {payload.get('error') or payload.get('message')}")
                            if updated_at and updated_at != last_task_updated_at:
                                last_task_updated_at = updated_at
                                stale_since = None
                            elif status in ("queued", "running"):
                                if stale_since is None:
                                    stale_since = now
                                elif now - stale_since > stale_sec:
                                    raise TimeoutError(f"task appears stale for >{stale_sec}s: id={task_id}")
                    except TimeoutError:
                        raise
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
                    next_task_poll = now + 3.0
                text = (node.text or "").strip()
                lower = text.lower()
                if ("status=完了" in text) or ("status=completed" in lower):
                    return text
                if ("status=エラー" in text) or ("status=error" in lower) or ("error=" in lower):
                    raise RuntimeError(f"task ended with error: {text}")
                time.sleep(1.0)
            raise TimeoutError(f"task timeout: {node.text}")

        click_tab(wait, "settings")
        outputs_dir = Path(driver.find_element(By.ID, "cfgOutputsDir").get_attribute("value"))
        models_dir = driver.find_element(By.ID, "cfgModelsDir").get_attribute("value")
        result["paths"] = {"models_dir": models_dir, "outputs_dir": str(outputs_dir)}
        runtime_text = driver.find_element(By.ID, "runtimeInfo").text
        result["runtime_info_initial"] = runtime_text

        def step_search_blank() -> dict[str, Any]:
            click_tab(wait, "models")
            Select(driver.find_element(By.ID, "searchTask")).select_by_value("text-to-image")
            Select(driver.find_element(By.ID, "searchSource")).select_by_value("all")
            set_value(driver, "searchQuery", "")
            set_value(driver, "searchLimit", "30")
            driver.find_element(By.CSS_SELECTOR, "#searchForm button[type='submit']").click()
            time.sleep(2.5)
            rows = driver.find_elements(By.CSS_SELECTOR, "#searchResults .row")
            thumbs = driver.find_elements(By.CSS_SELECTOR, "#searchResults img.model-preview")
            screenshot(driver, "01_search_blank")
            return {"rows": len(rows), "thumbnails": len(thumbs)}

        def ensure_download_via_gui(task: str, source: str, repo_id: str, shot: str) -> dict[str, Any]:
            click_tab(wait, "models")
            Select(driver.find_element(By.ID, "searchTask")).select_by_value(task)
            Select(driver.find_element(By.ID, "searchSource")).select_by_value(source)
            set_value(driver, "searchQuery", repo_id)
            set_value(driver, "searchLimit", "30")
            driver.find_element(By.CSS_SELECTOR, "#searchForm button[type='submit']").click()
            time.sleep(3.0)
            screenshot(driver, shot)
            buttons = driver.find_elements(By.CSS_SELECTOR, f"button.download-btn[data-repo='{repo_id}']")
            if not buttons:
                return {"repo": repo_id, "status": "already_downloaded_or_not_listed"}
            prev = read_task_id(driver)
            buttons[0].click()
            task_id = wait_new_task_id(driver, prev, timeout_sec=120)
            status_text = wait_task_done_monitored(task_id, timeout_sec=43200, stale_sec=1800)
            return {"repo": repo_id, "status": "downloaded", "task_id": task_id, "task_status": status_text}

        def step_ensure_models() -> dict[str, Any]:
            # Ensure recommended models + optional LoRA/VAE via GUI when needed.
            actions = [
                ensure_download_via_gui("text-to-image", "huggingface", "SG161222/RealVisXL_V5.0", "02_dl_realvis"),
                ensure_download_via_gui("text-to-image", "huggingface", "latent-consistency/lcm-lora-sdxl", "03_dl_lora"),
                ensure_download_via_gui("text-to-image", "huggingface", "madebyollin/sdxl-vae-fp16-fix", "04_dl_vae"),
                ensure_download_via_gui("text-to-video", "huggingface", "damo-vilab/text-to-video-ms-1.7b", "05_dl_t2v_damo"),
                ensure_download_via_gui("text-to-video", "huggingface", "ali-vilab/text-to-video-ms-1.7b", "06_dl_t2v_ali"),
                ensure_download_via_gui("image-to-video", "huggingface", "ali-vilab/i2vgen-xl", "06_dl_i2v"),
            ]
            return {"actions": actions}

        def step_apply_local_models() -> dict[str, Any]:
            click_tab(wait, "local-models")
            driver.find_element(By.ID, "refreshLocalModels").click()
            time.sleep(2.0)
            applied: dict[str, bool] = {}
            for task in ("text-to-image", "image-to-image", "text-to-video", "image-to-video"):
                btns = driver.find_elements(By.CSS_SELECTOR, f".local-apply-btn[data-task='{task}']")
                if btns:
                    btns[0].click()
                    time.sleep(0.5)
                    applied[task] = True
                else:
                    applied[task] = False
            screenshot(driver, "07_local_apply")
            return {"applied": applied}

        def step_settings_persistence() -> dict[str, Any]:
            click_tab(wait, "settings")
            before = driver.find_element(By.ID, "cfgGuidance").get_attribute("value")
            changed = "8.7" if before != "8.7" else "8.6"
            set_value(driver, "cfgGuidance", changed)
            driver.find_element(By.CSS_SELECTOR, "#settingsForm button[type='submit']").click()
            time.sleep(1.5)
            driver.get(base_url)
            click_tab(wait, "settings")
            after = driver.find_element(By.ID, "cfgGuidance").get_attribute("value")
            set_value(driver, "cfgGuidance", before)
            driver.find_element(By.CSS_SELECTOR, "#settingsForm button[type='submit']").click()
            screenshot(driver, "08_settings_persistence")
            return {"before": before, "after_reload": after, "persisted": after == changed}

        def run_t2i(use_lora_vae: bool) -> dict[str, Any]:
            click_tab(wait, "text-image")
            set_value(driver, "t2iPrompt", f"{PROMPT_BASE} photorealistic, crisp focus")
            set_value(driver, "t2iNegative", "low quality, blurry, watermark, text, artifact, deformed, cartoon, flat colors")
            selected_model = select_preferred_option(driver, "t2iModelSelect", RECOMMENDED_MODEL_KEYS["t2i"])
            if not selected_model:
                raise RuntimeError("t2i model not selectable from UI")
            set_value(driver, "t2iSteps", "24")
            set_value(driver, "t2iGuidance", "8.5")
            set_value(driver, "t2iWidth", "768")
            set_value(driver, "t2iHeight", "768")
            set_value(driver, "t2iSeed", "42")
            if use_lora_vae:
                lora = select_first_non_empty(driver, "t2iLoraSelect")
                if not lora:
                    raise RuntimeError("t2i LoRA not available in UI")
                select_multi(driver, "t2iLoraSelect", [lora])
                set_value(driver, "t2iLoraScale", "0.8")
                vae = select_first_non_empty(driver, "t2iVaeSelect")
                if not vae:
                    raise RuntimeError("t2i VAE not available in UI")
            else:
                select_multi(driver, "t2iLoraSelect", [])
                select_value_js(driver, "t2iVaeSelect", "")
            prev = read_task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#text2imageForm button[type='submit']").click()
            tid = wait_new_task_id(driver, prev, timeout_sec=120)
            status = wait_task_done_monitored(tid, timeout_sec=21600, stale_sec=900)
            src = driver.find_element(By.ID, "imagePreview").get_attribute("src") or ""
            file_name = parse_filename_from_url(src)
            file_path = outputs_dir / file_name
            screenshot(driver, f"09_t2i_{'on' if use_lora_vae else 'off'}")
            return {
                "task_id": tid,
                "status_text": status,
                "selected_model": selected_model,
                "file": file_name,
                "quality": image_quality(file_path) if file_name and file_path.exists() else {},
            }

        def run_i2i(use_lora_vae: bool) -> dict[str, Any]:
            click_tab(wait, "image-image")
            driver.find_element(By.ID, "i2iImage").send_keys(str(INPUT_IMAGE))
            set_value(driver, "i2iPrompt", f"{PROMPT_BASE} transform the subject to a flying whale, photorealistic")
            set_value(driver, "i2iNegative", "low quality, blurry, watermark, text, artifact, noisy, turtle, sticker")
            selected_model = select_preferred_option(driver, "i2iModelSelect", RECOMMENDED_MODEL_KEYS["i2i"])
            if not selected_model:
                raise RuntimeError("i2i model not selectable from UI")
            set_value(driver, "i2iSteps", "24")
            set_value(driver, "i2iGuidance", "8.0")
            set_value(driver, "i2iStrength", "0.85")
            set_value(driver, "i2iWidth", "768")
            set_value(driver, "i2iHeight", "768")
            set_value(driver, "i2iSeed", "42")
            if use_lora_vae:
                lora = select_first_non_empty(driver, "i2iLoraSelect")
                if not lora:
                    raise RuntimeError("i2i LoRA not available in UI")
                select_multi(driver, "i2iLoraSelect", [lora])
                set_value(driver, "i2iLoraScale", "0.8")
                vae = select_first_non_empty(driver, "i2iVaeSelect")
                if not vae:
                    raise RuntimeError("i2i VAE not available in UI")
            else:
                select_multi(driver, "i2iLoraSelect", [])
                select_value_js(driver, "i2iVaeSelect", "")
            prev = read_task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#image2imageForm button[type='submit']").click()
            tid = wait_new_task_id(driver, prev, timeout_sec=120)
            status = wait_task_done_monitored(tid, timeout_sec=21600, stale_sec=900)
            src = driver.find_element(By.ID, "imagePreview").get_attribute("src") or ""
            file_name = parse_filename_from_url(src)
            file_path = outputs_dir / file_name
            screenshot(driver, f"10_i2i_{'on' if use_lora_vae else 'off'}")
            return {
                "task_id": tid,
                "status_text": status,
                "selected_model": selected_model,
                "file": file_name,
                "quality": image_quality(file_path) if file_name and file_path.exists() else {},
            }

        def run_t2v() -> dict[str, Any]:
            click_tab(wait, "text")
            set_value(driver, "t2vPrompt", f"{PROMPT_BASE} smooth motion, cinematic aerial shot, photorealistic")
            set_value(driver, "t2vNegative", "low quality, blurry, flicker, artifact, watermark, cartoon")
            selected_model = select_preferred_option(driver, "t2vModelSelect", RECOMMENDED_MODEL_KEYS["t2v"])
            if not selected_model:
                raise RuntimeError("t2v model not selectable from UI")
            set_value(driver, "t2vSteps", "20")
            set_value(driver, "t2vFrames", str(VIDEO_TARGET_FRAMES))
            set_value(driver, "t2vGuidance", "9.0")
            set_value(driver, "t2vFps", str(VIDEO_TARGET_FPS))
            set_value(driver, "t2vSeed", "42")
            select_multi(driver, "t2vLoraSelect", [])
            prev = read_task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#text2videoForm button[type='submit']").click()
            tid = wait_new_task_id(driver, prev, timeout_sec=120)
            status = wait_task_done_monitored(tid, timeout_sec=21600, stale_sec=1800)
            src = driver.find_element(By.ID, "preview").get_attribute("src") or ""
            file_name = parse_filename_from_url(src)
            file_path = outputs_dir / file_name
            screenshot(driver, "11_t2v")
            return {
                "task_id": tid,
                "status_text": status,
                "selected_model": selected_model,
                "requested_duration_sec": round(VIDEO_TARGET_FRAMES / VIDEO_TARGET_FPS, 3),
                "file": file_name,
                "quality": video_quality(file_path) if file_name and file_path.exists() else {},
            }

        def run_i2v() -> dict[str, Any]:
            click_tab(wait, "image")
            selected_model = select_preferred_option(driver, "i2vModelSelect", RECOMMENDED_MODEL_KEYS["i2v"])
            if not selected_model:
                raise RuntimeError("i2v model not selectable from UI")
            driver.find_element(By.ID, "i2vImage").send_keys(str(INPUT_IMAGE))
            set_value(driver, "i2vPrompt", f"{PROMPT_BASE} animate with smooth camera movement and cloud motion")
            set_value(driver, "i2vNegative", "low quality, blurry, flicker, artifact, watermark, turtle, sticker")
            set_value(driver, "i2vSteps", "16")
            set_value(driver, "i2vFrames", str(VIDEO_TARGET_FRAMES))
            set_value(driver, "i2vGuidance", "9.0")
            set_value(driver, "i2vFps", str(VIDEO_TARGET_FPS))
            set_value(driver, "i2vWidth", "512")
            set_value(driver, "i2vHeight", "512")
            set_value(driver, "i2vSeed", "42")
            select_multi(driver, "i2vLoraSelect", [])
            prev = read_task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#image2videoForm button[type='submit']").click()
            tid = wait_new_task_id(driver, prev, timeout_sec=120)
            status = wait_task_done_monitored(tid, timeout_sec=21600, stale_sec=1800)
            src = driver.find_element(By.ID, "preview").get_attribute("src") or ""
            file_name = parse_filename_from_url(src)
            file_path = outputs_dir / file_name
            screenshot(driver, "12_i2v")
            return {
                "task_id": tid,
                "status_text": status,
                "selected_model": selected_model,
                "requested_duration_sec": round(VIDEO_TARGET_FRAMES / VIDEO_TARGET_FPS, 3),
                "file": file_name,
                "quality": video_quality(file_path) if file_name and file_path.exists() else {},
            }

        def step_outputs_view_only() -> dict[str, Any]:
            click_tab(wait, "outputs")
            driver.find_element(By.ID, "refreshOutputs").click()
            time.sleep(1.5)
            rows = driver.find_elements(By.CSS_SELECTOR, "#outputsList .row.model-row")
            screenshot(driver, "13_outputs")
            return {"count": len(rows)}

        def step_gpu_evidence() -> dict[str, Any]:
            runtime = driver.find_element(By.ID, "runtimeInfo").text
            server_text = "\n".join(server.lines[-3000:])
            return {
                "runtime_info": runtime,
                "runtime_has_cuda": "cuda=true" in runtime.lower(),
                "runtime_has_rocm": "rocm=true" in runtime.lower(),
                "server_has_pipeline_cuda": bool(re.search(r"pipeline load start.*device=cuda", server_text)),
            }

        step("01_blank_model_search", step_search_blank)
        step("02_ensure_models_via_gui_download", step_ensure_models)
        step("03_apply_local_models", step_apply_local_models)
        step("04_settings_persistence", step_settings_persistence)
        step("05_generate_t2i_lora_vae_off", lambda: run_t2i(False))
        step("06_generate_t2i_lora_vae_on", lambda: run_t2i(True))
        step("07_generate_i2i_lora_vae_off", lambda: run_i2i(False))
        step("08_generate_i2i_lora_vae_on", lambda: run_i2i(True))
        step("09_generate_t2v", run_t2v)
        step("10_generate_i2v", run_i2v)
        step("11_outputs_view_only", step_outputs_view_only)
        step("12_gpu_evidence", step_gpu_evidence)

        # Keep foreground browser visible shortly for manual observation.
        time.sleep(20)
    finally:
        if driver is not None:
            try:
                screenshot(driver, "99_final")
            except Exception:
                pass
            driver.quit()
        if monitor is not None:
            monitor.stop()
            result["server_monitor"] = monitor.summary()
        server.stop()
        coverage_report(coverage_file=coverage_file, out_dir=ART_DIR)

    result["ended_at"] = utc_now()
    (ART_DIR / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "# GUI QA Foreground Summary",
        "",
        f"- started_at: {result.get('started_at')}",
        f"- ended_at: {result.get('ended_at')}",
        f"- base_url: {result.get('base_url')}",
        f"- driver_mode: {result.get('driver_mode')}",
        "",
        "## Step Results",
    ]
    for s in result.get("steps", []):
        summary_lines.append(f"- {s.get('name')}: {s.get('status')} ({s.get('duration_sec'):.2f}s)")
        if s.get("details"):
            summary_lines.append(f"  - details: {json.dumps(s.get('details'), ensure_ascii=False)}")
        if s.get("error"):
            summary_lines.append("  - error captured in report.json")
    (ART_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    if LATEST_DIR.exists():
        for item in sorted(LATEST_DIR.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                try:
                    item.rmdir()
                except OSError:
                    pass
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for item in ART_DIR.rglob("*"):
        if item.is_file():
            dst = LATEST_DIR / item.relative_to(ART_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(item.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
