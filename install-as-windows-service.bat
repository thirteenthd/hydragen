@echo off
title Install Hydragen as Windows Service (services.msc)
cd /d "%~dp0"

:: Check for Administrator elevation
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [Hydragen] Requesting Administrator privileges to register Windows Service...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo ======================================================================
echo   Replacing Legacy Strata Service with Hydragen AI Local Server
echo ======================================================================
echo.

set "HYDRAGEN_ROOT=C:\Hydragen"
set "NSSM=%HYDRAGEN_ROOT%\tools\nssm.exe"
set "PYTHON=%HYDRAGEN_ROOT%\.venv\Scripts\python.exe"
set "LOG=%HYDRAGEN_ROOT%\hydragen-service.log"

echo [1/5] Stopping and removing legacy Strata service (if present)...
net stop Strata >nul 2>&1
"%NSSM%" stop Strata >nul 2>&1
"%NSSM%" remove Strata confirm >nul 2>&1

echo [2/5] Stopping existing Hydragen service (if present)...
net stop Hydragen >nul 2>&1
"%NSSM%" stop Hydragen >nul 2>&1
"%NSSM%" remove Hydragen confirm >nul 2>&1

echo [3/5] Terminating any stale processes holding port 8080...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Get-Process strata, strata-vision -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue" >nul 2>&1

echo [4/5] Registering "Hydragen AI Local Server" in services.msc...
"%NSSM%" install Hydragen "%PYTHON%" "-m hydragen.cli --engine strata --config hydragen.json --port 8080 --api-monitor --no-api-key"
"%NSSM%" set Hydragen AppDirectory "%HYDRAGEN_ROOT%"
"%NSSM%" set Hydragen DisplayName "Hydragen AI Local Server"
"%NSSM%" set Hydragen Description "Hydragen AI Local Inference Server with CUDA acceleration and resident MoE expert tuning"
"%NSSM%" set Hydragen Start SERVICE_AUTO_START
"%NSSM%" set Hydragen AppStdout "%LOG%"
"%NSSM%" set Hydragen AppStderr "%LOG%"
"%NSSM%" set Hydragen AppStopMethodSkip 0
"%NSSM%" set Hydragen AppStopMethodConsole 2000
"%NSSM%" set Hydragen AppStopMethodWindow 2000
"%NSSM%" set Hydragen AppStopMethodThreads 2000

echo [5/5] Starting Hydragen service...
schtasks /run /tn "Strata-GPU-Power-Clamp" >nul 2>&1
net start Hydragen

echo.
echo ======================================================================
echo [SUCCESS] "Hydragen AI Local Server" is now running as a Windows Service!
echo ======================================================================
echo.
echo Service Name  : Hydragen
echo Display Name  : Hydragen AI Local Server
echo Startup Type  : Automatic (SERVICE_AUTO_START)
echo Port          : 8080
echo Log File      : %LOG%
echo.
echo You can manage it with:
echo   - services.msc ("Hydragen AI Local Server")
echo   - net start Hydragen / net stop Hydragen
echo   - Start-Service Hydragen / Stop-Service Hydragen
echo   - .\restart-service.bat / .\stop-service.bat
echo.
pause
