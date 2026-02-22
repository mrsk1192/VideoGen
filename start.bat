@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"
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
if "%HIP_VER%"=="" (
  echo [WARN] ROCm was not detected in this Python environment.
) else (
  echo [INFO] ROCm HIP version: %HIP_VER%
)

:run
echo [INFO] Starting server at http://localhost:%PORT%
if not "%AUTO_OPEN_BROWSER%"=="0" (
  start "" powershell -NoProfile -Command ^
    "$url='http://localhost:%PORT%'; for($i=0;$i -lt 240;$i++){ try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1 ^| Out-Null; Start-Process $url; break } catch {}; Start-Sleep -Milliseconds 500 }"
)
"%PYTHON_EXE%" -m uvicorn main:app --host %HOST% --port %PORT%
if errorlevel 1 goto :error

goto :eof

:error
echo [ERROR] Failed to start application.
pause
exit /b 1
