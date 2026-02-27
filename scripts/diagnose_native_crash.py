import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ProbeCase:
    name: str
    prefer_safetensors: bool
    force_bin_weights: bool
    verify_safetensors_on_load: bool
    force_full_vram_load: bool
    disable_cpu_offload: bool
    disable_vae_tiling: bool


def build_child_code() -> str:
    return r"""
import json
import os
import sys

import main

payload = json.loads(sys.argv[1])
settings = main.settings_store.get()
server = settings.setdefault("server", {})
for key, value in payload.items():
    if key.startswith("server__"):
        server[key.replace("server__", "")] = value

result = {"ok": False}
runtime = main.detect_runtime()
result["runtime"] = {
    "python_version": runtime.get("python_version"),
    "torch_version": runtime.get("torch_version"),
    "torch_hip_version": runtime.get("torch_hip_version"),
    "cuda_available": runtime.get("cuda_available"),
    "gpu_name": runtime.get("gpu_name"),
    "total_vram_gb": runtime.get("total_vram_gb"),
    "free_vram_gb": runtime.get("free_vram_gb"),
    "safetensors_version": runtime.get("safetensors_version"),
    "diffusers_version": runtime.get("diffusers_version"),
    "accelerate_version": runtime.get("accelerate_version"),
    "selected_dtype": runtime.get("selected_dtype"),
    "vram_profile": runtime.get("vram_profile"),
    "pytorch_hip_alloc_conf": runtime.get("pytorch_hip_alloc_conf"),
}
try:
    pipe = main.get_pipeline("text-to-video", payload["model_ref"], settings)
    policy = getattr(pipe, "_videogen_load_policy", {}) or {}
    result["policy"] = policy
    result["meta_tensor_count"] = len(main.pipeline_meta_tensor_names(pipe, limit=64))
    result["ok"] = True
except Exception as exc:
    result["error"] = f"{exc.__class__.__name__}: {exc}"
print(json.dumps(result, ensure_ascii=False))
"""


def run_case(case: ProbeCase, model_ref: str, timeout_sec: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_ref": model_ref,
        "server__prefer_safetensors": case.prefer_safetensors,
        "server__force_bin_weights": case.force_bin_weights,
        "server__verify_safetensors_on_load": case.verify_safetensors_on_load,
        "server__force_full_vram_load": case.force_full_vram_load,
        "server__disable_cpu_offload": case.disable_cpu_offload,
        "server__disable_vae_tiling": case.disable_vae_tiling,
    }
    command = [sys.executable, "-c", build_child_code(), json.dumps(payload)]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    output: Dict[str, Any] = {
        "case": case.name,
        "returncode": int(completed.returncode),
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
    }
    if completed.returncode == 0:
        with_value = (completed.stdout or "").strip().splitlines()
        if with_value:
            with_value = with_value[-1]
            try:
                output["result"] = json.loads(with_value)
            except Exception:
                output["result_parse_error"] = with_value
    return output


def default_cases() -> List[ProbeCase]:
    return [
        ProbeCase(
            name="prefer_safetensors",
            prefer_safetensors=True,
            force_bin_weights=False,
            verify_safetensors_on_load=True,
            force_full_vram_load=True,
            disable_cpu_offload=True,
            disable_vae_tiling=True,
        ),
        ProbeCase(
            name="force_bin_weights",
            prefer_safetensors=False,
            force_bin_weights=True,
            verify_safetensors_on_load=False,
            force_full_vram_load=True,
            disable_cpu_offload=True,
            disable_vae_tiling=True,
        ),
        ProbeCase(
            name="auto_weight_format",
            prefer_safetensors=False,
            force_bin_weights=False,
            verify_safetensors_on_load=False,
            force_full_vram_load=True,
            disable_cpu_offload=True,
            disable_vae_tiling=True,
        ),
    ]


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Native crash probe for pipeline loading.")
    parser.add_argument("--model-ref", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    reports: List[Dict[str, Any]] = []
    for case in default_cases():
        report = run_case(case, model_ref=args.model_ref, timeout_sec=max(60, int(args.timeout_sec)))
        report["settings"] = asdict(case)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    has_crash = any(int(item.get("returncode", 0)) not in (0, 1) for item in reports)
    return 2 if has_crash else 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
