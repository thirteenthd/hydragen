@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Hydragen Automated Setup & Updater (Windows CMD)
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "%SCRIPT_DIR%setup.py" del /f /q "%SCRIPT_DIR%setup.py"

echo ==========================================================
echo      Hydragen Engine Automated Setup ^& Updater (Windows)
echo ==========================================================

if "%~1"=="--update" goto :do_update
if "%~1"=="-u" goto :do_update
goto :skip_update

:do_update
echo [+] Updating Hydragen from git repository...
if exist ".git" git pull
:skip_update

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
python -c "import importlib.util as u; s = u.find_spec('torch'); m = __import__('torch') if s else None; print(bool(m and m.cuda.is_available()))" 2>nul | findstr /i "True" >nul
if errorlevel 1 (
    echo [+] Installing PyTorch with CUDA 12.4 support...
    pip install torch --index-url https://download.pytorch.org/whl/cu124
)

REM 5. Ensure bin directory & precompiled worker binary
if not exist "%SCRIPT_DIR%bin" mkdir "%SCRIPT_DIR%bin"
set "STRATA_EXE=%SCRIPT_DIR%bin\strata.exe"
if not exist "%STRATA_EXE%" (
    if defined STRATA_DIR if exist "%STRATA_DIR%\engine\strata.exe" (
        echo [+] Copying Strata binary from STRATA_DIR...
        copy /Y "%STRATA_DIR%\engine\strata.exe" "%STRATA_EXE%" >nul
    ) else (
        echo [+] Downloading precompiled Strata worker binary (v0.2.9 / engine 0.1.55)...
        powershell -Command "try { Invoke-WebRequest -Uri 'https://github.com/Niko1221/Strata/releases/download/v0.2.9/strata-engine-0.1.55-windows-x86_64-cuda-cuda12.tar.gz' -OutFile '%SCRIPT_DIR%bin\strata-engine.tar.gz' -UseBasicParsing; tar -xzf '%SCRIPT_DIR%bin\strata-engine.tar.gz' -C '%SCRIPT_DIR%bin'; Remove-Item '%SCRIPT_DIR%bin\strata-engine.tar.gz' -Force; Write-Host '[+] Extracted Strata worker into bin\' -ForegroundColor Green } catch { Write-Warning 'Auto-download failed; please copy strata.exe into bin\' }"
    )
)

REM 6. Ensure strata_control and tools modules are available
if not exist "%SCRIPT_DIR%strata_control" (
    echo [+] Downloading Strata control modules (v0.2.9 / 0.1.55)...
    powershell -Command "try { Invoke-WebRequest -Uri 'https://github.com/Niko1221/Strata/releases/download/v0.2.9/strata-control-0.1.55-windows-x86_64.zip' -OutFile '%SCRIPT_DIR%bin\control.zip' -UseBasicParsing; Expand-Archive -Path '%SCRIPT_DIR%bin\control.zip' -DestinationPath '%SCRIPT_DIR%' -Force; Remove-Item '%SCRIPT_DIR%bin\control.zip' -Force; Write-Host '[+] Extracted strata_control and tools' -ForegroundColor Green } catch { Write-Warning 'Control modules download failed' }"
)
if exist "%SCRIPT_DIR%setup.py" del /f /q "%SCRIPT_DIR%setup.py"

REM 7. Install Hydragen package in editable mode
echo [+] Installing Hydragen and dependencies...
python -m pip install --upgrade pip
python -m pip install -e .

REM 8. Generate default config if missing
set "CONFIG_FILE=%SCRIPT_DIR%hydragen.json"
if not exist "%CONFIG_FILE%" (
    echo [+] Generating default hydragen.json template...
    (
        echo {
        echo   "exe": "bin\\strata.exe",
        echo   "args": [
        echo     "--expert-cache", "2200",
        echo     "--prefill", "auto:8192",
        echo     "--spec", "4",
        echo     "--max-context", "131072",
        echo     "--kv", "int8",
        echo     "--vram-reserve-mib", "384",
        echo     "--pcie-frac", "0.20",
        echo     "--spec-min-p", "0.45",
        echo     "--adapt-async", "1",
        echo     "--kv-resident", "20480",
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
echo   run.bat
echo   (or: hydragen --config hydragen.json --port 8080 --api-monitor)
echo.
echo To update in the future:
echo   setup.bat -u
echo ==========================================================
