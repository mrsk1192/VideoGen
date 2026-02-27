import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "tests" / "system"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(port: int) -> subprocess.Popen:
    cmd = [
        "python",
        "-m",
        "uvicorn",
        "tests.system.system_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_server_ready(base_url: str, timeout_sec: int = 30) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(base_url, timeout=1.0)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.3)
    raise RuntimeError("Server did not become ready")


def _create_driver():
    edge_opts = EdgeOptions()
    edge_opts.add_argument("--headless=new")
    edge_opts.add_argument("--disable-gpu")
    edge_opts.add_argument("--window-size=1600,1000")
    try:
        return webdriver.Edge(options=edge_opts)
    except WebDriverException:
        pass

    chrome_opts = ChromeOptions()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--window-size=1600,1000")
    try:
        return webdriver.Chrome(options=chrome_opts)
    except WebDriverException as exc:
        raise RuntimeError("No usable WebDriver (Edge/Chrome) was available") from exc


@pytest.mark.system
def test_local_models_filter_and_model_thumbnail():
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    server = _start_server(port)
    driver = None
    try:
        _wait_server_ready(base)
        driver = _create_driver()
        wait = WebDriverWait(driver, 25)
        driver.get(base)

        local_models_tab = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.tab[data-tab='local-models']")))
        local_models_tab.click()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#panel-local-models.active")))
        wait.until(EC.presence_of_element_located((By.ID, "localLineageFilter")))
        driver.save_screenshot(str(ARTIFACTS / "step1_loaded.png"))

        lineage_select = Select(driver.find_element(By.ID, "localLineageFilter"))
        wait.until(lambda d: len(lineage_select.options) >= 1)
        lineage_select.select_by_index(0)
        driver.save_screenshot(str(ARTIFACTS / "step2_lineage_filter.png"))

        text_tab = driver.find_element(By.CSS_SELECTOR, "button.tab[data-tab='text']")
        text_tab.click()
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#panel-text.active")))
        model_select = wait.until(EC.presence_of_element_located((By.ID, "t2vModelSelect")))
        wait.until(lambda d: len(Select(model_select).options) >= 2)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", model_select)
        Select(model_select).select_by_index(1)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#t2vModelPreview .model-picked-thumb")))
        thumb = driver.find_element(By.CSS_SELECTOR, "#t2vModelPreview .model-picked-thumb")
        src = thumb.get_attribute("src")
        assert "/api/models/preview" in src
        driver.save_screenshot(str(ARTIFACTS / "step3_model_thumbnail.png"))
    finally:
        if driver is not None:
            driver.quit()
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
