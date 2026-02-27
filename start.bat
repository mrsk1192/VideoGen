@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" (
  for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$p='data/settings.json'; if(Test-Path $p){ try { $j=Get-Content $p -Raw | ConvertFrom-Json; $v=[int]($j.server.listen_port); if($v -ge 1 -and $v -le 65535){ Write-Output $v } } catch {} }"`) do set "PORT=%%P"
  if "%PORT%"=="" set "PORT=8000"
)
set "ROCM_AOTRITON_FROM_SETTINGS="
for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "$p='data/settings.json'; if(Test-Path $p){ try { $j=Get-Content $p -Raw | ConvertFrom-Json; $v=$j.server.rocm_aotriton_experimental; if($null -ne $v){ if($v -is [bool]){ if($v){ '1' } else { '0' } } elseif($v -is [string]){ $n=$v.Trim().ToLower(); if(@('1','true','yes','on') -contains $n){ '1' } elseif(@('0','false','no','off') -contains $n){ '0' } } else { if([bool]$v){ '1' } else { '0' } } } } catch {} }"`) do set "ROCM_AOTRITON_FROM_SETTINGS=%%V"
if not "%ROCM_AOTRITON_FROM_SETTINGS%"=="" (
  set "ROCM_AOTRITON_EXPERIMENTAL=%ROCM_AOTRITON_FROM_SETTINGS%"
) else (
  if "%ROCM_AOTRITON_EXPERIMENTAL%"=="" set "ROCM_AOTRITON_EXPERIMENTAL=1"
)
if "%AUTO_OPEN_BROWSER%"=="" set "AUTO_OPEN_BROWSER=1"
set "PYTHON_EXE="
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if exist "venv\Scripts\python.exe" (
  set "PYTHON_EXE=venv\Scripts\python.exe"
  goto :ensure_deps
)

if exist "%VENV_PY%" (
  set "PYTHON_EXE=%VENV_PY%"
  goto :ensure_deps
)

echo [INFO] No venv found. Creating .venv...
python -m venv "%VENV_DIR%"
if errorlevel 1 goto :error
set "PYTHON_EXE=%VENV_PY%"

:ensure_deps
"%PYTHON_EXE%" -m uvicorn --version >nul 2>&1
if errorlevel 1 (
  echo [INFO] Installing/updating dependencies...
  "%PYTHON_EXE%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%PYTHON_EXE%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)
set "TRANSFORMERS_OK="
for /f "usebackq delims=" %%V in (`"%PYTHON_EXE%" -c "import transformers as t; v=getattr(t,'__version__','0'); p=v.split('.'); major=int(p[0]) if p and p[0].isdigit() else 999; print('1' if ((major < 5) and hasattr(t,'MT5Tokenizer')) else '0')" 2^>nul`) do set "TRANSFORMERS_OK=%%V"
if not "%TRANSFORMERS_OK%"=="1" (
  echo [WARN] transformers compatibility check failed. Current environment may be partially incompatible.
  if "%REPAIR_TRANSFORMERS_ON_START%"=="1" (
    echo [INFO] REPAIR_TRANSFORMERS_ON_START=1 so dependencies will be re-installed.
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
  )
)

:pick_port
if not defined REQUESTED_PORT set "REQUESTED_PORT=%PORT%"
set "PORT_IN_USE="
for /f "tokens=1" %%A in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do set "PORT_IN_USE=1"
if defined PORT_IN_USE (
  set /a PORT+=1
  set "PORT_IN_USE="
  goto :pick_port
)
if not "%PORT%"=="%REQUESTED_PORT%" echo [WARN] Port %REQUESTED_PORT% is in use. Using port %PORT% instead.

set "HIP_VER="
for /f "usebackq delims=" %%H in (`"%PYTHON_EXE%" -c "import torch; print(getattr(torch.version,'hip',None) or '')" 2^>nul`) do set "HIP_VER=%%H"
set "CUDA_AVAIL="
for /f "usebackq delims=" %%C in (`"%PYTHON_EXE%" -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2^>nul`) do set "CUDA_AVAIL=%%C"
if not "%HIP_VER%"=="" (
  echo [INFO] ROCm HIP version: %HIP_VER%
) else (
  if "%CUDA_AVAIL%"=="1" (
    echo [INFO] GPU backend detected via torch.cuda ^(HIP version string unavailable in this environment^).
  ) else (
    echo [WARN] ROCm/GPU backend was not detected in this Python environment.
  )
)

:run
echo [INFO] Starting server at http://localhost:%PORT%
set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=%ROCM_AOTRITON_EXPERIMENTAL%"
echo [INFO] TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=%TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL%
if "%PYTORCH_ALLOC_CONF%"=="" if "%PYTORCH_CUDA_ALLOC_CONF%"=="" set "PYTORCH_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
if "%PYTORCH_ALLOC_CONF%"=="" if not "%PYTORCH_CUDA_ALLOC_CONF%"=="" set "PYTORCH_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%"
if "%PYTORCH_CUDA_ALLOC_CONF%"=="" if not "%PYTORCH_ALLOC_CONF%"=="" set "PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_ALLOC_CONF%"
if not "%AUTO_OPEN_BROWSER%"=="0" (
  start "" /B "%PYTHON_EXE%" "scripts\open_browser_when_ready.py" --url "http://localhost:%PORT%" --timeout 120 --interval 0.5
)
if "%QA_COVERAGE%"=="1" (
  echo [INFO] QA coverage mode enabled.
  "%PYTHON_EXE%" -m coverage run --parallel-mode -m uvicorn main:app --host %HOST% --port %PORT%
) else (
  "%PYTHON_EXE%" -m uvicorn main:app --host %HOST% --port %PORT%
)
if errorlevel 1 goto :error

goto :eof

:error
echo [ERROR] Failed to start application.
pause
exit /b 1
