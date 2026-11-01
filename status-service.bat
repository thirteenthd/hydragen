@echo off
title Hydragen Windows Service Status
cd /d "%~dp0"

echo ==============================================================
echo   Hydragen Windows Service Status
echo ==============================================================
echo.
sc.exe query Hydragen 2>nul
if %errorlevel% neq 0 (
    echo [Status] "Hydragen" service is not installed in services.msc.
    echo Checking legacy "Strata" service:
    sc.exe query Strata 2>nul
)

echo.
echo Active listeners on port 8080:
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Format-Table -AutoSize"

echo.
pause
