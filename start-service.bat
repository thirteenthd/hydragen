@echo off
title Start Hydragen Windows Service
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges to start Windows Service...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo [Hydragen] Applying AI GPU Power Clamp (260W)...
schtasks /run /tn "Strata-GPU-Power-Clamp" >nul 2>&1

echo [Hydragen] Starting service...
net start Hydragen

echo.
echo [Hydragen] Service started on port 8080!
pause
