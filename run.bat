@echo off
REM ============================================================================
REM Hydragen Server Launcher (Windows CMD)
REM ============================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

set "CONFIG_FILE=hydragen.json"
if not exist "%CONFIG_FILE%" (
    if exist "strata-iq3_xxs.json" set "CONFIG_FILE=strata-iq3_xxs.json"
)

echo [hydragen] Starting server using config: %CONFIG_FILE%
hydragen --config "%CONFIG_FILE%" --port 8080 --api-monitor %*
