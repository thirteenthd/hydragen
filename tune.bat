@echo off
title Hydragen Engine Performance and Concurrency Tuner
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    set "PYTHON=python"
)

"%PYTHON%" tune_hydragen.py %*
pause
