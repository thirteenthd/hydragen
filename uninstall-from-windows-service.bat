@echo off
title Uninstall Hydragen Windows Service (services.msc)
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [Hydragen] Requesting Administrator privileges to remove Windows Service...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

set "HYDRAGEN_ROOT=C:\Hydragen"
set "NSSM=%HYDRAGEN_ROOT%\tools\nssm.exe"

echo Stopping Hydragen service if running...
net stop Hydragen >nul 2>&1
"%NSSM%" stop Hydragen >nul 2>&1

echo Removing Hydragen service from services.msc...
"%NSSM%" remove Hydragen confirm

echo.
echo [SUCCESS] Hydragen service has been unregistered from services.msc.
echo.
pause
