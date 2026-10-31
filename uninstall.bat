@echo off
set "TARGET_DIR=%~dp0"
echo ==========================================================
echo   Reverting Strata to Original Stock Installation
echo ==========================================================

if exist "%TARGET_DIR%serve\server.py.orig_backup" (
    echo [1/2] Restoring original server.py from safety backup...
    copy /Y "%TARGET_DIR%serve\server.py.orig_backup" "%TARGET_DIR%serve\server.py" >nul
    del /F /Q "%TARGET_DIR%serve\server.py.orig_backup" >nul
) else (
    echo [!] Notice: server.py.orig_backup not found.
)

echo [2/2] Removing vLLM drop-in files...
del /F /Q "%TARGET_DIR%serve\vllm_engine.py" 2>nul
del /F /Q "%TARGET_DIR%serve\strata_gguf.py" 2>nul
del /F /Q "%TARGET_DIR%serve\libstrata_iq_kernels.so" 2>nul
del /F /Q "%TARGET_DIR%run_vllm.bat" 2>nul

echo.
echo ==========================================================
echo   Revert Complete! Strata is back to its original state.
echo ==========================================================
