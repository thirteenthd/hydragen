@echo off
title Restart Hydragen Windows Service
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges to restart Windows Service...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo [Hydragen] Stopping Windows Service and releasing port 8080...
net stop Hydragen >nul 2>&1
powershell -Command "Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Get-Process strata, strata-vision -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue" >nul 2>&1
ping 127.0.0.1 -n 3 >nul

echo [Hydragen] Applying AI GPU Power Clamp (260W)...
schtasks /run /tn "Strata-GPU-Power-Clamp" >nul 2>&1

echo [Hydragen] Starting Hydragen Windows Service...
net start Hydragen

echo.
echo ==============================================================
echo [SUCCESS] Hydragen service restarted successfully!
echo ==============================================================
echo.
pause
