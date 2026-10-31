@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Hydragen One-Click Setup Script (Windows CMD)
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ==========================================================
echo          Hydragen Engine Automated Setup (Windows)
echo ==========================================================

REM 1. Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo [-] Error: Python was not found in PATH.
    echo     Please install Python 3.10+ from python.org and add to PATH.
    exit /b 1
)

for /f "tokens=2" %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
echo [+] Detected Python %PY_VER%

REM 2. Check NVIDIA GPU
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader') do (
        echo [+] Detected GPU: %%g
        goto :gpu_done
    )
) else (
    echo [!] Warning: nvidia-smi not found. CUDA GPU is required for acceleration.
)
:gpu_done

REM 3. Virtual Environment Setup
set "VENV_DIR=%SCRIPT_DIR%.venv"
if not defined VIRTUAL_ENV (
    if not exist "%VENV_DIR%" (
        echo [+] Creating virtual environment in %VENV_DIR%...
        python -m venv "%VENV_DIR%"
    )
    echo [+] Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo [+] Using active virtual environment: %VIRTUAL_ENV%
)

REM 4. Check & Install PyTorch with CUDA (if not already installed)
python -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr /i "True" >nul
if errorlevel 1 (
    echo [+] Installing PyTorch with CUDA 12.4 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)

REM 5. Install Hydragen package in editable mode
echo [+] Installing Hydragen and dependencies...
python -m pip install --upgrade pip
python -m pip install -e .

REM 6. Ensure bin directory exists
if not exist "%SCRIPT_DIR%bin" mkdir "%SCRIPT_DIR%bin"

REM 7. Generate default config if missing
set "CONFIG_FILE=%SCRIPT_DIR%hydragen.json"
if not exist "%CONFIG_FILE%" (
    echo [+] Generating default hydragen.json template...
    (
        echo {
        echo   "exe": "bin\\strata.exe",
        echo   "args": [
        echo     "--expert-cache", "auto",
        echo     "--prefill", "auto:16384",
        echo     "--spec", "4",
        echo     "--max-context", "131072",
        echo     "--kv", "int8",
        echo     "--vram-reserve-mib", "1152",
        echo     "--pcie-frac", "0.20",
        echo     "--spec-min-p", "0.45",
        echo     "--adapt-async", "1",
        echo     "--kv-resident", "32768",
        echo     "--batch", "2",
        echo     "--batch-mtp"
        echo   ],
        echo   "batch": 2,
        echo   "max_context": 131072,
        echo   "log": "hydragen.log"
        echo }
    ) > "%CONFIG_FILE%"
)

echo.
echo ==========================================================
echo          Hydragen Windows Setup Complete!
echo ==========================================================
echo To start the server:
echo   1. call .venv\Scripts\activate.bat
echo   2. hydragen --config hydragen.json --port 8080 --api-monitor
echo ==========================================================
