from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import requests
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "Scripts" / "python.exe"
INPUT_IMAGE = Path(r"C:\AI\VideoGen\TestAsset\sticker8377632396286038267.png")
PROMPT = "A whale is flying in the sky."
NEGATIVE = "low quality, blurry, jpeg artifacts, deformed, watermark, text, bad anatomy"
I2V_REPO = "ali-vilab/i2vgen-xl"
TASK_KEY = "videogen_last_task_id"
ART_DIR = ROOT / "artifacts" / "i2v_foreground" / datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class RunState:
    port: int
    base_url: str
    outputs_dir: Path


class Server:
    def __init__(self, port: int, log_path: Path) -> None:
        self.port = port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

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
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
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
                f.write(line)
                f.flush()

    def wait_ready(self, timeout_sec: int = 180) -> None:
        end = time.time() + timeout_sec
        url = f"http://127.0.0.1:{self.port}/api/system/info"
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
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=2)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def save_shot(driver: webdriver.Edge, name: str) -> None:
    path = ART_DIR / "screenshots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    driver.save_screenshot(str(path))


def click_tab(wait: WebDriverWait, tab: str) -> None:
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"button.tab[data-tab='{tab}']"))).click()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"#panel-{tab}.active")))


def set_text(driver: webdriver.Edge, elem_id: str, value: str) -> None:
    e = driver.find_element(By.ID, elem_id)
    e.clear()
    e.send_keys(value)


def select_by_value(driver: webdriver.Edge, elem_id: str, value: str) -> bool:
    elem = driver.find_element(By.ID, elem_id)
    options = driver.execute_script("return Array.from(arguments[0].options).map(o => o.value);", elem) or []
    if value not in options:
        return False
    driver.execute_script(
        "arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change',{bubbles:true}));",
        elem,
        value,
    )
    return True


def select_first_non_empty(driver: webdriver.Edge, elem_id: str) -> Optional[str]:
    elem = driver.find_element(By.ID, elem_id)
    values = driver.execute_script("return Array.from(arguments[0].options).map(o => o.value);", elem) or []
    for raw in values:
        value = str(raw or "").strip()
        if value and select_by_value(driver, elem_id, value):
            return value
    return None


def task_id(driver: webdriver.Edge) -> str:
    return str(driver.execute_script(f"return localStorage.getItem('{TASK_KEY}') || '';") or "")


def wait_new_task(driver: webdriver.Edge, prev: str, timeout_sec: int = 60) -> str:
    end = time.time() + timeout_sec
    while time.time() < end:
        cur = task_id(driver)
        if cur and cur != prev:
            return cur
        time.sleep(0.3)
    raise TimeoutError("task id not updated")


def wait_task_done(driver: webdriver.Edge, timeout_sec: int = 10800) -> str:
    status = driver.find_element(By.ID, "taskStatus")
    end = time.time() + timeout_sec
    while time.time() < end:
        text = (status.text or "").strip()
        low = text.lower()
        if "status=完了" in text or "status=completed" in low:
            return text
        if "status=エラー" in text or "status=error" in low or "error=" in low:
            raise RuntimeError(f"task error: {text}")
        time.sleep(1)
    raise TimeoutError(f"task timeout: {status.text}")


def parse_name_from_src(src: str) -> str:
    if not src:
        return ""
    name = Path(urlparse(src).path).name
    return unquote(name) if name else ""


def run() -> int:
    if not INPUT_IMAGE.exists():
        raise RuntimeError(f"input image missing: {INPUT_IMAGE}")
    ART_DIR.mkdir(parents=True, exist_ok=True)
    state_path = ART_DIR / "result.json"
    result = {"started_at": datetime.now().isoformat(), "repo": I2V_REPO}

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = Server(port, ART_DIR / "server.log")
    driver: Optional[webdriver.Edge] = None
    try:
        server.start()
        server.wait_ready()

        opts = EdgeOptions()
        opts.add_argument("--window-size=1700,1100")
        try:
            driver = webdriver.Edge(options=opts)
        except WebDriverException as exc:
            raise RuntimeError("Edge headful launch failed. foreground実行ができません。") from exc

        wait = WebDriverWait(driver, 50)
        driver.get(base)
        save_shot(driver, "00_home")

        settings = requests.get(f"{base}/api/settings", timeout=10).json()
        outputs_dir = Path(settings.get("paths", {}).get("outputs_dir", str(ROOT / "outputs")))
        result["outputs_dir"] = str(outputs_dir)

        click_tab(wait, "models")
        Select(driver.find_element(By.ID, "searchTask")).select_by_value("image-to-video")
        Select(driver.find_element(By.ID, "searchSource")).select_by_value("huggingface")
        set_text(driver, "searchQuery", I2V_REPO)
        set_text(driver, "searchLimit", "30")
        driver.find_element(By.CSS_SELECTOR, "#searchForm button[type='submit']").click()
        time.sleep(3)
        save_shot(driver, "01_search_i2v")

        btns = driver.find_elements(By.CSS_SELECTOR, f"button.download-btn[data-repo='{I2V_REPO}']")
        if btns:
            prev = task_id(driver)
            btns[0].click()
            wait_new_task(driver, prev, timeout_sec=80)
            dl_status = wait_task_done(driver, timeout_sec=28800)
            result["download_status"] = dl_status
            save_shot(driver, "02_download_done")
        else:
            result["download_status"] = "already_downloaded_or_not_found_on_search"

        click_tab(wait, "local-models")
        driver.find_element(By.ID, "refreshLocalModels").click()
        time.sleep(2)
        save_shot(driver, "03_local_models")
        i2v_apply = driver.find_elements(By.CSS_SELECTOR, ".local-apply-btn[data-task='image-to-video']")
        if not i2v_apply:
            raise RuntimeError("ローカルモデル一覧に image-to-video 適用ボタンがありません。")
        i2v_apply[0].click()
        time.sleep(1)
        save_shot(driver, "04_i2v_applied")

        click_tab(wait, "image")
        driver.find_element(By.ID, "i2vImage").send_keys(str(INPUT_IMAGE))
        set_text(driver, "i2vPrompt", PROMPT)
        set_text(driver, "i2vNegative", NEGATIVE)
        if not select_first_non_empty(driver, "i2vModelSelect"):
            raise RuntimeError("i2vModelSelectに選択可能なモデルがありません。")
        set_text(driver, "i2vSteps", "8")
        set_text(driver, "i2vFrames", "8")
        set_text(driver, "i2vGuidance", "8.0")
        set_text(driver, "i2vFps", "6")
        set_text(driver, "i2vWidth", "512")
        set_text(driver, "i2vHeight", "512")
        save_shot(driver, "05_before_generate_i2v")

        prev = task_id(driver)
        driver.find_element(By.CSS_SELECTOR, "#image2videoForm button[type='submit']").click()
        tid = wait_new_task(driver, prev, timeout_sec=80)
        status_text = wait_task_done(driver, timeout_sec=28800)
        result["task_id"] = tid
        result["task_status"] = status_text
        save_shot(driver, "06_after_generate_i2v")

        preview_src = driver.find_element(By.ID, "preview").get_attribute("src") or ""
        video_file = parse_name_from_src(preview_src)
        result["video_file"] = video_file
        video_path = outputs_dir / video_file if video_file else None
        if video_path and video_path.exists():
            result["video_size_mb"] = round(video_path.stat().st_size / 1024 / 1024, 3)
        else:
            result["video_size_mb"] = None

        click_tab(wait, "outputs")
        driver.find_element(By.ID, "refreshOutputs").click()
        time.sleep(1.5)
        save_shot(driver, "07_outputs")

        result["runtime_info"] = driver.find_element(By.ID, "runtimeInfo").text
        result["finished_at"] = datetime.now().isoformat()
        state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # Keep foreground browser visible for manual confirmation.
        time.sleep(120)
        return 0
    except Exception as exc:
        result["error"] = f"{exc}\n{traceback.format_exc()}"
        result["finished_at"] = datetime.now().isoformat()
        state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    finally:
        if driver is not None:
            driver.quit()
        server.stop()


if __name__ == "__main__":
    raise SystemExit(run())
