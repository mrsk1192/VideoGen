@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [INFO] Creating venv...
  py -3.12 -m venv "%VENV_DIR%"
  if errorlevel 1 goto :error
)

echo [INFO] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [INFO] Installing ROCm 7.2 SDK wheels...
"%PYTHON_EXE%" -m pip install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
if errorlevel 1 goto :error

echo [INFO] Installing ROCm 7.2 PyTorch wheels...
"%PYTHON_EXE%" -m pip install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%%2Brocmsdk20260116-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
if errorlevel 1 goto :error

echo [INFO] Installing project requirements...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [INFO] Done. You can now run: start.bat
exit /b 0

:error
echo [ERROR] ROCm 7.2 setup failed.
pause
exit /b 1
