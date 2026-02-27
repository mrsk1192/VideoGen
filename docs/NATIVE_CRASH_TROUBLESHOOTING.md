# Native Crash Troubleshooting (Windows + ROCm)

## Scope
- Symptom: process exits with `0xC0000005` (Access Violation), often during `Loading pipeline components...`
- Typical stack hints: `torch_cpu.dll`, `_local_scalar_dense_cpu`, `Tensor.item()`, `safetensors/_safetensors_rust.pyd`

## Quick triage checklist
1. Confirm runtime diagnostics:
   - `GET /api/runtime`
   - check `torch_version`, `torch_hip_version`, `gpu_name`, `total_vram_gb`, `safetensors_version`
2. Confirm allocator:
   - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` is set before `torch` import
3. Confirm load strategy:
   - `vram_profile=96gb_class` uses full VRAM path
   - `device_map_used` is `null` (no `auto` / no dict map)
4. Run crash probe:
   - `python scripts/diagnose_native_crash.py --model-ref Wan-AI/Wan2.2-TI2V-5B-Diffusers`

## Clean venv rebuild procedure
```powershell
cd C:\AI\VideoGen
py -3.12 -m venv .venv_clean
.\.venv_clean\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -c constraints.windows-rocm.txt
python -m pip check
python -m compileall .
python -m ruff check .
```

## Version-difference probe for safetensors
Use an isolated venv with system packages visible to quickly test only safetensors delta:
```powershell
cd C:\AI\VideoGen
py -3.12 -m venv .venv_diag --system-site-packages
.\.venv_diag\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install safetensors==0.6.2
python scripts/diagnose_native_crash.py --model-ref Wan-AI/Wan2.2-TI2V-5B-Diffusers

python -m pip install --force-reinstall safetensors==0.5.3
python scripts/diagnose_native_crash.py --model-ref Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

## Runtime maintenance switches (`settings.server`)
- `prefer_safetensors` (default: `true`)
- `force_bin_weights` (default: `false`)
- `verify_safetensors_on_load` (default: `true`)
- `force_full_vram_load` (default: `false`)
- `disable_cpu_offload` (default: `false`)
- `disable_vae_tiling` (default: `true`)

## Recommended fallback order
1. `prefer_safetensors=true`, `force_bin_weights=false`
2. If crash persists, switch to `force_bin_weights=true`
3. If only one local model fails, re-download and validate model files
4. Pin to known-good package set in `constraints.windows-rocm.txt`
