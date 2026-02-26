from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
START_BAT = ROOT / "start.bat"
ART_DIR = ROOT / "artifacts" / "t2i_only" / datetime.now().strftime("%Y%m%d_%H%M%S")
TASK_KEY = "videogen_last_task_id"
RECOMMENDED_MODEL = "SG161222/RealVisXL_V5.0"
RECOMMENDED_MODEL_DIR = "SG161222--RealVisXL_V5.0"
PROMPT = "A majestic humpback whale flying in a bright blue sky with soft clouds, photorealistic, cinematic, ultra detailed."
NEGATIVE = "low quality, blurry, watermark, text, artifact, deformed, cartoon, flat colors, logo"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class BatchServer:
    def __init__(self, env: dict[str, str], log_path: Path) -> None:
        self.env = env
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None
        self.port: Optional[int] = None
        self.stop_read = threading.Event()
        self.reader: Optional[threading.Thread] = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            ["cmd.exe", "/c", str(START_BAT)],
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
                m = re.search(r"Starting server at http://localhost:(\d+)", line)
                if m:
                    self.port = int(m.group(1))
                f.write(line + "\n")
                f.flush()
                if self.stop_read.is_set():
                    break

    def wait_ready(self, timeout_sec: int = 300) -> str:
        end = time.time() + timeout_sec
        while time.time() < end:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"start.bat terminated early: code={self.proc.returncode}")
            if self.port:
                base = f"http://127.0.0.1:{self.port}"
                try:
                    r = requests.get(f"{base}/api/system/info", timeout=2)
                    if r.status_code == 200:
                        return base
                except Exception:
                    pass
            time.sleep(0.5)
        raise TimeoutError("server not ready")

    def stop(self) -> None:
        self.stop_read.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.CTRL_C_EVENT)
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                    self.proc.wait(timeout=10)
                except Exception:
                    self.proc.terminate()
            if self.proc.poll() is None:
                self.proc.kill()
        if self.reader:
            self.reader.join(timeout=3)


def screenshot(driver: Any, name: str) -> str:
    p = ART_DIR / "screenshots" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    driver.save_screenshot(str(p))
    return str(p)


def click_tab(wait: WebDriverWait, tab: str) -> None:
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"button.tab[data-tab='{tab}']"))).click()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f"#panel-{tab}.active")))


def set_value(driver: Any, elem_id: str, value: str) -> None:
    node = driver.find_element(By.ID, elem_id)
    node.clear()
    node.send_keys(value)


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


def wait_task_done(driver: Any, task_id: str, base_url: str, timeout_sec: int = 7200) -> str:
    end = time.time() + timeout_sec
    node = driver.find_element(By.ID, "taskStatus")
    while time.time() < end:
        text = (node.text or "").strip()
        lower = text.lower()
        try:
            resp = requests.get(f"{base_url}/api/tasks/{task_id}", timeout=3)
            if resp.status_code == 200:
                payload = resp.json()
                status = str(payload.get("status") or "").lower()
                if status == "completed":
                    return text or "completed"
                if status == "error":
                    raise RuntimeError(str(payload.get("error") or payload.get("message") or "task error"))
        except RuntimeError:
            raise
        except Exception:
            pass
        if ("status=完了" in text) or ("status=completed" in lower):
            return text
        if ("status=エラー" in text) or ("status=error" in lower) or ("error=" in lower):
            raise RuntimeError(text)
        time.sleep(1.0)
    raise TimeoutError("task timeout")


def parse_filename_from_url(url: str) -> str:
    if not url:
        return ""
    name = Path(urlparse(url).path).name
    return unquote(name) if name else ""


def choose_model(driver: Any, select_id: str, token: str) -> bool:
    elem = driver.find_element(By.ID, select_id)
    options: list[dict[str, str]] = driver.execute_script(
        "return Array.from(arguments[0].options).map(o => ({value:o.value,text:o.textContent||''}));",
        elem,
    ) or []
    token_lower = token.lower()
    for opt in options:
        value = str(opt.get("value") or "")
        text = str(opt.get("text") or "")
        if token_lower in value.lower() or token_lower in text.lower():
            driver.execute_script(
                "arguments[0].value=arguments[1]; arguments[0].dispatchEvent(new Event('change',{bubbles:true}));",
                elem,
                value,
            )
            return True
    return False


def refresh_and_wait_model(
    driver: Any,
    refresh_button_id: str,
    select_id: str,
    tokens: list[str],
    timeout_sec: int = 180,
) -> bool:
    driver.find_element(By.ID, refresh_button_id).click()
    end = time.time() + timeout_sec
    while time.time() < end:
        for token in tokens:
            if choose_model(driver, select_id, token):
                return True
        time.sleep(1.0)
    return False


def local_model_exists(base_dir: str, dir_name: str) -> bool:
    p = Path(base_dir) / dir_name
    return p.exists() and p.is_dir()


def main() -> int:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"started_at": utc_now(), "recommended_model": RECOMMENDED_MODEL}

    env = os.environ.copy()
    env["AUTO_OPEN_BROWSER"] = "0"
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(free_port())

    server = BatchServer(env=env, log_path=ART_DIR / "server.log")
    driver = None
    try:
        server.start()
        base_url = server.wait_ready()
        result["base_url"] = base_url
        runtime = requests.get(f"{base_url}/api/system/info", timeout=10).json()
        result["runtime"] = runtime

        opts = EdgeOptions()
        opts.add_argument("--window-size=1700,1100")
        driver = webdriver.Edge(options=opts)
        wait = WebDriverWait(driver, 60)
        driver.get(base_url)
        screenshot(driver, "00_home")

        click_tab(wait, "settings")
        models_dir = driver.find_element(By.ID, "cfgModelsDir").get_attribute("value")
        result["models_dir"] = models_dir

        click_tab(wait, "models")
        Select(driver.find_element(By.ID, "searchTask")).select_by_value("text-to-image")
        Select(driver.find_element(By.ID, "searchSource")).select_by_value("huggingface")
        set_value(driver, "searchQuery", "RealVisXL_V5.0")
        set_value(driver, "searchLimit", "30")
        driver.find_element(By.CSS_SELECTOR, "#searchForm button[type='submit']").click()
        time.sleep(3.0)
        screenshot(driver, "01_search_recommended")

        buttons = driver.find_elements(By.CSS_SELECTOR, f"button.download-btn[data-repo='{RECOMMENDED_MODEL}']")
        if not buttons:
            buttons = driver.find_elements(By.CSS_SELECTOR, "button.download-btn")
            matched = []
            for b in buttons:
                repo = (b.get_attribute("data-repo") or "").strip()
                if repo.lower() == RECOMMENDED_MODEL.lower():
                    matched.append(b)
            buttons = matched

        if buttons:
            prev = read_task_id(driver)
            buttons[0].click()
            download_task_id = wait_new_task_id(driver, prev, timeout_sec=120)
            download_status = wait_task_done(driver, download_task_id, base_url, timeout_sec=43200)
            result["download"] = {"task_id": download_task_id, "status": download_status, "downloaded": True}
        else:
            already = local_model_exists(models_dir, RECOMMENDED_MODEL_DIR)
            result["download"] = {"downloaded": False, "reason": "already_local_or_not_listed", "already_local": already}
            if not already:
                raise RuntimeError(f"recommended model download button not found and model is not local: {RECOMMENDED_MODEL}")

        click_tab(wait, "text-image")
        set_value(driver, "t2iPrompt", PROMPT)
        set_value(driver, "t2iNegative", NEGATIVE)
        set_value(driver, "t2iSteps", "20")
        set_value(driver, "t2iGuidance", "8.5")
        set_value(driver, "t2iWidth", "640")
        set_value(driver, "t2iHeight", "640")
        set_value(driver, "t2iSeed", "42")
        set_value(driver, "t2iLoraScale", "1.0")

        # Ensure LoRA and VAE are off for clean baseline quality.
        driver.execute_script(
            """
            const lora = document.getElementById('t2iLoraSelect');
            if (lora) {
              for (const opt of lora.options) opt.selected = false;
              lora.dispatchEvent(new Event('change',{bubbles:true}));
            }
            const vae = document.getElementById('t2iVaeSelect');
            if (vae) {
              vae.value = '';
              vae.dispatchEvent(new Event('change',{bubbles:true}));
            }
            """
        )

        # Prefer recommended model and wait until model catalog refresh reflects recent download.
        selected = refresh_and_wait_model(
            driver,
            "refreshT2IModels",
            "t2iModelSelect",
            [RECOMMENDED_MODEL_DIR, "RealVisXL", RECOMMENDED_MODEL],
            timeout_sec=300,
        )
        if not selected:
            raise RuntimeError("recommended model was not selectable from t2iModelSelect")
        result["model_selected"] = selected
        result["model_value"] = driver.find_element(By.ID, "t2iModelSelect").get_attribute("value")
        screenshot(driver, "02_t2i_configured")

        prev = read_task_id(driver)
        driver.find_element(By.CSS_SELECTOR, "#text2imageForm button[type='submit']").click()
        task_id = wait_new_task_id(driver, prev, timeout_sec=120)
        status = wait_task_done(driver, task_id, base_url, timeout_sec=21600)
        src = driver.find_element(By.ID, "imagePreview").get_attribute("src") or ""
        file_name = parse_filename_from_url(src)
        screenshot(driver, "03_t2i_done")

        click_tab(wait, "settings")
        outputs_dir = driver.find_element(By.ID, "cfgOutputsDir").get_attribute("value")
        runtime_info = driver.find_element(By.ID, "runtimeInfo").text

        result["t2i"] = {
            "task_id": task_id,
            "status": status,
            "file_name": file_name,
            "outputs_dir": outputs_dir,
            "output_path": str(Path(outputs_dir) / file_name) if file_name else "",
            "runtime_info": runtime_info,
        }
    finally:
        if driver is not None:
            try:
                screenshot(driver, "99_final")
            except Exception:
                pass
            driver.quit()
        server.stop()

    result["ended_at"] = utc_now()
    (ART_DIR / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "# T2I Only Run",
        f"- started_at: {result.get('started_at')}",
        f"- ended_at: {result.get('ended_at')}",
        f"- base_url: {result.get('base_url')}",
        f"- recommended_model: {RECOMMENDED_MODEL}",
        f"- downloaded: {result.get('download', {}).get('downloaded')}",
        f"- t2i_task_id: {result.get('t2i', {}).get('task_id')}",
        f"- t2i_file: {result.get('t2i', {}).get('output_path')}",
        f"- runtime_info: {result.get('t2i', {}).get('runtime_info')}",
    ]
    (ART_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(str(ART_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
