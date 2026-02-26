from __future__ import annotations

import json
import os
import re
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
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"
ART_ROOT = ROOT / "artifacts" / "e2e_full_ui_only"
ART_DIR = ART_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
LATEST_DIR = ART_ROOT / "latest"
INPUT_IMAGE = Path(r"C:\AI\VideoGen\TestAsset\sticker8377632396286038267.png")
PROMPT = "A whale is flying in the sky."
NEGATIVE = "low quality, blurry, jpeg artifacts, deformed, watermark, text, bad anatomy"
TASK_KEY = "videogen_last_task_id"
TASK_DONE_PATTERNS = ("status=completed", "status=完了")
TASK_ERR_PATTERNS = ("status=error", "status=エラー", "error=")


@dataclass
class Step:
    name: str
    status: str
    started_at: str
    ended_at: str
    duration_sec: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class Server:
    def __init__(self, port: int, log_path: Path) -> None:
        self.port = port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None
        self.lines: list[str] = []
        self.stop_event = threading.Event()
        self.reader: Optional[threading.Thread] = None

    def start(self) -> None:
        cmd = [str(PY), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(self.port)]
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
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as f:
            while not self.stop_event.is_set():
                line = self.proc.stdout.readline()
                if not line:
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                line = line.rstrip("\n")
                self.lines.append(line)
                f.write(line + "\n")
                f.flush()

    def wait_ready(self, timeout_sec: int = 120) -> None:
        url = f"http://127.0.0.1:{self.port}/api/system/info"
        end = time.time() + timeout_sec
        while time.time() < end:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"server exited: {self.proc.returncode}")
            try:
                r = requests.get(url, timeout=1.5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise TimeoutError("server not ready")

    def stop(self) -> None:
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.reader:
            self.reader.join(timeout=2)


def create_driver() -> tuple[Any, str]:
    opts = EdgeOptions()
    opts.add_argument("--window-size=1700,1100")
    try:
        return webdriver.Edge(options=opts), "edge-headful"
    except WebDriverException:
        opts_h = EdgeOptions()
        opts_h.add_argument("--headless=new")
        opts_h.add_argument("--window-size=1700,1100")
        return webdriver.Edge(options=opts_h), "edge-headless"


def ss(driver: Any, name: str) -> str:
    p = ART_DIR / "screenshots" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    driver.save_screenshot(str(p))
    return str(p)


def click_tab(wait: WebDriverWait, tab: str) -> None:
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"button.tab[data-tab='{tab}']"))).click()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"#panel-{tab}.active")))


def set_text(driver: Any, elem_id: str, value: str) -> None:
    node = driver.find_element(By.ID, elem_id)
    node.clear()
    node.send_keys(value)


def select_by_value_js(driver: Any, elem_id: str, value: str) -> bool:
    elem = driver.find_element(By.ID, elem_id)
    options = driver.execute_script("return Array.from(arguments[0].options).map(o=>o.value);", elem) or []
    if value not in options:
        return False
    driver.execute_script(
        "arguments[0].value=arguments[1]; arguments[0].dispatchEvent(new Event('change',{bubbles:true}));",
        elem,
        value,
    )
    return True


def select_first_non_empty(driver: Any, elem_id: str) -> Optional[str]:
    elem = driver.find_element(By.ID, elem_id)
    values = driver.execute_script("return Array.from(arguments[0].options).map(o=>o.value);", elem) or []
    for raw in values:
        v = str(raw or "").strip()
        if v:
            if select_by_value_js(driver, elem_id, v):
                return v
    return None


def set_multi_values(driver: Any, elem_id: str, values: list[str]) -> None:
    elem = driver.find_element(By.ID, elem_id)
    driver.execute_script(
        """
        const select = arguments[0];
        const wanted = new Set(arguments[1]);
        for (const o of select.options) o.selected = wanted.has(o.value);
        select.dispatchEvent(new Event("change", {bubbles:true}));
        """,
        elem,
        values,
    )


def task_id(driver: Any) -> str:
    return str(driver.execute_script(f"return localStorage.getItem('{TASK_KEY}') || '';") or "")


def wait_new_task(driver: Any, prev: str, timeout_sec: int = 60) -> str:
    end = time.time() + timeout_sec
    while time.time() < end:
        cur = task_id(driver)
        if cur and cur != prev:
            return cur
        time.sleep(0.3)
    raise TimeoutError("task id not updated")


def wait_task_done_ui(driver: Any, timeout_sec: int = 7200) -> dict[str, Any]:
    end = time.time() + timeout_sec
    status_el = driver.find_element(By.ID, "taskStatus")
    while time.time() < end:
        text = (status_el.text or "").strip()
        low = text.lower()
        if any(p in low for p in [x.lower() for x in TASK_DONE_PATTERNS]):
            return {"status": "completed", "task_status_text": text}
        if any(p in low for p in [x.lower() for x in TASK_ERR_PATTERNS]):
            raise RuntimeError(f"task error: {text}")
        time.sleep(0.5)
    raise TimeoutError(f"task not finished: {status_el.text}")


def parse_file_from_src(src: str) -> Optional[str]:
    if not src:
        return None
    path = urlparse(src).path
    name = Path(path).name
    return unquote(name) if name else None


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
    out = {"size_mb": round(path.stat().st_size / 1024 / 1024, 3)}
    try:
        import imageio.v3 as iio  # type: ignore

        f0 = iio.imread(path, index=0)
        out["first_frame_shape"] = [int(v) for v in f0.shape]
        out["first_frame_std"] = round(float(f0.std()), 3)
    except Exception as exc:
        out["probe_error"] = str(exc)
    return out


def run() -> int:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"started_at": now(), "steps": [], "prompt": PROMPT, "negative_prompt": NEGATIVE}
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = Server(port, ART_DIR / "server.log")
    driver = None
    outputs_dir = Path(ROOT / "outputs")

    def step(name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        st = now()
        t0 = time.perf_counter()
        try:
            details = fn() or {}
            report["steps"].append(Step(name, "passed", st, now(), time.perf_counter() - t0, details=details).__dict__)
        except Exception as exc:
            report["steps"].append(
                Step(
                    name,
                    "failed",
                    st,
                    now(),
                    time.perf_counter() - t0,
                    details={},
                    error=f"{exc}\n{traceback.format_exc()}",
                ).__dict__
            )

    try:
        server.start()
        server.wait_ready(timeout_sec=180)
        driver, mode = create_driver()
        report["driver_mode"] = mode
        wait = WebDriverWait(driver, 40)
        driver.get(base)
        ss(driver, "00_home")

        def step_search() -> dict[str, Any]:
            click_tab(wait, "models")
            Select(driver.find_element(By.ID, "searchTask")).select_by_value("text-to-image")
            Select(driver.find_element(By.ID, "searchSource")).select_by_value("all")
            set_text(driver, "searchQuery", "")
            set_text(driver, "searchLimit", "30")
            driver.find_element(By.CSS_SELECTOR, "#searchForm button[type='submit']").click()
            time.sleep(2)
            rows = driver.find_elements(By.CSS_SELECTOR, "#searchResults .row")
            thumbs = driver.find_elements(By.CSS_SELECTOR, "#searchResults img.model-preview")
            ss(driver, "search_all_blank")
            return {"rows": len(rows), "thumbs": len(thumbs)}

        def ensure_download(repo_id: str, task: str = "text-to-image") -> dict[str, Any]:
            click_tab(wait, "models")
            Select(driver.find_element(By.ID, "searchTask")).select_by_value(task)
            Select(driver.find_element(By.ID, "searchSource")).select_by_value("huggingface")
            set_text(driver, "searchQuery", repo_id)
            set_text(driver, "searchLimit", "30")
            driver.find_element(By.CSS_SELECTOR, "#searchForm button[type='submit']").click()
            time.sleep(2)
            button = driver.find_elements(By.XPATH, f"//button[contains(@class,'download-btn') and @data-repo=\"{repo_id}\"]")
            if not button:
                return {"repo": repo_id, "status": "already_downloaded_or_not_listed"}
            prev = task_id(driver)
            button[0].click()
            wait_new_task(driver, prev, timeout_sec=40)
            done = wait_task_done_ui(driver, timeout_sec=7200)
            return {"repo": repo_id, "status": done["status"], "task_status_text": done["task_status_text"]}

        def step_download_models() -> dict[str, Any]:
            res = []
            res.append(ensure_download("latent-consistency/lcm-lora-sdxl", "text-to-image"))
            res.append(ensure_download("madebyollin/sdxl-vae-fp16-fix", "text-to-image"))
            ss(driver, "downloads")
            return {"downloads": res}

        def step_settings() -> dict[str, Any]:
            nonlocal outputs_dir
            click_tab(wait, "settings")
            outputs_dir = Path(driver.find_element(By.ID, "cfgOutputsDir").get_attribute("value") or str(outputs_dir))
            before = driver.find_element(By.ID, "cfgGuidance").get_attribute("value")
            new_v = "8.7" if before != "8.7" else "8.6"
            set_text(driver, "cfgGuidance", new_v)
            driver.find_element(By.CSS_SELECTOR, "#settingsForm button[type='submit']").click()
            time.sleep(1.5)
            driver.get(base)
            click_tab(wait, "settings")
            after = driver.find_element(By.ID, "cfgGuidance").get_attribute("value")
            ok = after == new_v
            set_text(driver, "cfgGuidance", before)
            driver.find_element(By.CSS_SELECTOR, "#settingsForm button[type='submit']").click()
            runtime_info = driver.find_element(By.ID, "runtimeInfo").text
            ss(driver, "settings")
            return {"persisted": ok, "guidance_before": before, "guidance_after_reload": after, "runtime_info": runtime_info}

        def step_apply_local() -> dict[str, Any]:
            click_tab(wait, "local-models")
            driver.find_element(By.ID, "refreshLocalModels").click()
            time.sleep(1.5)
            applied: dict[str, bool] = {}
            for task_name in ("text-to-image", "image-to-image", "text-to-video", "image-to-video"):
                btns = driver.find_elements(By.CSS_SELECTOR, f".local-apply-btn[data-task='{task_name}']")
                if btns:
                    btns[0].click()
                    time.sleep(0.5)
                    applied[task_name] = True
                else:
                    applied[task_name] = False
            rows = driver.find_elements(By.CSS_SELECTOR, "#localModels .row.model-row")
            ss(driver, "local_models")
            return {"rows": len(rows), "applied": applied}

        def fill_common(prefix: str) -> None:
            set_text(driver, f"{prefix}Prompt", PROMPT)
            set_text(driver, f"{prefix}Negative", NEGATIVE)

        def step_t2i(use_lora_vae: bool) -> dict[str, Any]:
            click_tab(wait, "text-image")
            fill_common("t2i")
            select_first_non_empty(driver, "t2iModelSelect")
            set_text(driver, "t2iSteps", "8")
            set_text(driver, "t2iGuidance", "7.5")
            set_text(driver, "t2iWidth", "512")
            set_text(driver, "t2iHeight", "512")
            if use_lora_vae:
                l = select_first_non_empty(driver, "t2iLoraSelect")
                set_multi_values(driver, "t2iLoraSelect", [l] if l else [])
                set_text(driver, "t2iLoraScale", "0.8")
                select_first_non_empty(driver, "t2iVaeSelect")
            else:
                set_multi_values(driver, "t2iLoraSelect", [])
                select_by_value_js(driver, "t2iVaeSelect", "")
            prev = task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#text2imageForm button[type='submit']").click()
            tid = wait_new_task(driver, prev, timeout_sec=60)
            done = wait_task_done_ui(driver, timeout_sec=7200)
            src = driver.find_element(By.ID, "imagePreview").get_attribute("src") or ""
            fname = parse_file_from_src(src) or ""
            q = image_quality(outputs_dir / fname) if fname else {}
            ss(driver, f"t2i_{'on' if use_lora_vae else 'off'}")
            return {"task_id": tid, "status_text": done["task_status_text"], "image_file": fname, "quality": q}

        def step_i2i(use_lora_vae: bool) -> dict[str, Any]:
            if not INPUT_IMAGE.exists():
                raise RuntimeError(f"input image not found: {INPUT_IMAGE}")
            click_tab(wait, "image-image")
            driver.find_element(By.ID, "i2iImage").send_keys(str(INPUT_IMAGE))
            fill_common("i2i")
            select_first_non_empty(driver, "i2iModelSelect")
            set_text(driver, "i2iSteps", "8")
            set_text(driver, "i2iGuidance", "7.0")
            set_text(driver, "i2iStrength", "0.55")
            set_text(driver, "i2iWidth", "512")
            set_text(driver, "i2iHeight", "512")
            if use_lora_vae:
                l = select_first_non_empty(driver, "i2iLoraSelect")
                set_multi_values(driver, "i2iLoraSelect", [l] if l else [])
                set_text(driver, "i2iLoraScale", "0.8")
                select_first_non_empty(driver, "i2iVaeSelect")
            else:
                set_multi_values(driver, "i2iLoraSelect", [])
                select_by_value_js(driver, "i2iVaeSelect", "")
            prev = task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#image2imageForm button[type='submit']").click()
            tid = wait_new_task(driver, prev, timeout_sec=60)
            done = wait_task_done_ui(driver, timeout_sec=7200)
            src = driver.find_element(By.ID, "imagePreview").get_attribute("src") or ""
            fname = parse_file_from_src(src) or ""
            q = image_quality(outputs_dir / fname) if fname else {}
            ss(driver, f"i2i_{'on' if use_lora_vae else 'off'}")
            return {"task_id": tid, "status_text": done["task_status_text"], "image_file": fname, "quality": q}

        def step_t2v() -> dict[str, Any]:
            click_tab(wait, "text")
            fill_common("t2v")
            select_first_non_empty(driver, "t2vModelSelect")
            set_text(driver, "t2vSteps", "8")
            set_text(driver, "t2vFrames", "8")
            set_text(driver, "t2vGuidance", "8.0")
            set_text(driver, "t2vFps", "6")
            set_multi_values(driver, "t2vLoraSelect", [])
            prev = task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#text2videoForm button[type='submit']").click()
            tid = wait_new_task(driver, prev, timeout_sec=60)
            done = wait_task_done_ui(driver, timeout_sec=10800)
            src = driver.find_element(By.ID, "preview").get_attribute("src") or ""
            fname = parse_file_from_src(src) or ""
            q = video_quality(outputs_dir / fname) if fname else {}
            ss(driver, "t2v")
            return {"task_id": tid, "status_text": done["task_status_text"], "video_file": fname, "quality": q}

        def step_i2v() -> dict[str, Any]:
            click_tab(wait, "image")
            first = select_first_non_empty(driver, "i2vModelSelect")
            if not first:
                raise RuntimeError("No local image-to-video model available on UI")
            if not INPUT_IMAGE.exists():
                raise RuntimeError(f"input image not found: {INPUT_IMAGE}")
            driver.find_element(By.ID, "i2vImage").send_keys(str(INPUT_IMAGE))
            fill_common("i2v")
            set_text(driver, "i2vSteps", "8")
            set_text(driver, "i2vFrames", "8")
            set_text(driver, "i2vGuidance", "8.0")
            set_text(driver, "i2vFps", "6")
            set_text(driver, "i2vWidth", "512")
            set_text(driver, "i2vHeight", "512")
            set_multi_values(driver, "i2vLoraSelect", [])
            prev = task_id(driver)
            driver.find_element(By.CSS_SELECTOR, "#image2videoForm button[type='submit']").click()
            tid = wait_new_task(driver, prev, timeout_sec=60)
            done = wait_task_done_ui(driver, timeout_sec=10800)
            src = driver.find_element(By.ID, "preview").get_attribute("src") or ""
            fname = parse_file_from_src(src) or ""
            q = video_quality(outputs_dir / fname) if fname else {}
            ss(driver, "i2v")
            return {"task_id": tid, "status_text": done["task_status_text"], "video_file": fname, "quality": q}

        def step_outputs() -> dict[str, Any]:
            click_tab(wait, "outputs")
            driver.find_element(By.ID, "refreshOutputs").click()
            time.sleep(1.2)
            rows = driver.find_elements(By.CSS_SELECTOR, "#outputsList .row.model-row")
            before = len(rows)
            deleted = None
            if rows:
                deleted = rows[0].text.split("\n")[0].strip()
                btns = driver.find_elements(By.CSS_SELECTOR, "#outputsList .output-delete-btn")
                if btns:
                    btns[0].click()
                    try:
                        WebDriverWait(driver, 5).until(EC.alert_is_present())
                        driver.switch_to.alert.accept()
                    except TimeoutException:
                        pass
                    time.sleep(1.2)
            rows2 = driver.find_elements(By.CSS_SELECTOR, "#outputsList .row.model-row")
            ss(driver, "outputs")
            return {"count_before": before, "count_after": len(rows2), "deleted_candidate": deleted}

        def step_gpu() -> dict[str, Any]:
            runtime = driver.find_element(By.ID, "runtimeInfo").text
            log_joined = "\n".join(server.lines[-1000:])
            return {
                "runtime_info_text": runtime,
                "runtime_has_device_cuda": ("device=cuda" in runtime),
                "runtime_has_rocm_true": ("rocm=true" in runtime.lower()),
                "server_log_has_pipeline_cuda": bool(re.search(r"pipeline load start.*device=cuda", log_joined)),
            }

        step("Model Search (blank query)", step_search)
        step("Model Download via UI", step_download_models)
        step("Settings Persistence via UI", step_settings)
        step("Apply Local Models via UI", step_apply_local)
        step("Generate T2I (LoRA/VAE OFF)", lambda: step_t2i(False))
        step("Generate T2I (LoRA/VAE ON)", lambda: step_t2i(True))
        step("Generate I2I (LoRA/VAE OFF)", lambda: step_i2i(False))
        step("Generate I2I (LoRA/VAE ON)", lambda: step_i2i(True))
        step("Generate T2V", step_t2v)
        step("Generate I2V", step_i2v)
        step("Outputs View/Delete via UI", step_outputs)
        step("GPU Evidence", step_gpu)
    finally:
        if driver is not None:
            try:
                ss(driver, "99_final")
            except Exception:
                pass
            driver.quit()
        server.stop()

    report["ended_at"] = now()
    dump_json(ART_DIR / "report.json", report)
    _summary(report, ART_DIR)
    _run_cov(ART_DIR)
    _copy_latest(ART_DIR, LATEST_DIR)
    return 0


def _run_cov(out_dir: Path) -> None:
    xml_path = out_dir / "coverage.xml"
    run = subprocess.run(
        [
            str(PY),
            "-m",
            "pytest",
            "tests",
            "--maxfail=1",
            "--disable-warnings",
            "--cov=main",
            "--cov-report=term-missing",
            f"--cov-report=xml:{xml_path}",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    (out_dir / "coverage.txt").write_text((run.stdout or "") + "\n" + (run.stderr or ""), encoding="utf-8")


def _summary(report: dict[str, Any], out_dir: Path) -> None:
    lines = ["# UI Screen-Only Validation Summary", ""]
    lines.append(f"- started_at: {report.get('started_at')}")
    lines.append(f"- ended_at: {report.get('ended_at')}")
    lines.append(f"- driver_mode: {report.get('driver_mode')}")
    lines.append("")
    lines.append("## Step results")
    for s in report.get("steps", []):
        lines.append(f"- {s.get('name')}: {s.get('status')} ({s.get('duration_sec'):.2f}s)")
        if s.get("details"):
            lines.append(f"  - details: {json.dumps(s.get('details'), ensure_ascii=False)}")
        if s.get("error"):
            lines.append("  - error captured in report.json")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _copy_latest(src: Path, dst: Path) -> None:
    if dst.exists():
        for item in sorted(dst.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                try:
                    item.rmdir()
                except OSError:
                    pass
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_file():
            out = dst / item.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(item.read_bytes())


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(run())
