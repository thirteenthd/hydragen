@echo off
setlocal enabledelayedexpansion

echo ==========================================================
echo   Installing Strata vLLM PagedMoE Engine Drop-in (Windows)
echo ==========================================================

set "SCRIPT_DIR=%~dp0"
set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" set "TARGET_DIR=."

if not exist "%TARGET_DIR%\serve" (
    echo [-] Error: "%TARGET_DIR%" does not look like a Strata repository.
    echo     Usage: install.bat C:\path\to\Strata
    exit /b 1
)

echo [1/4] Copying engine bridge files to %TARGET_DIR%\serve\...
copy /Y "%SCRIPT_DIR%serve\vllm_engine.py" "%TARGET_DIR%\serve\" >nul
copy /Y "%SCRIPT_DIR%serve\strata_gguf.py" "%TARGET_DIR%\serve\" >nul
copy /Y "%SCRIPT_DIR%serve\libstrata_iq_kernels.so" "%TARGET_DIR%\serve\" >nul

echo [2/4] Backing up and deploying server.py...
if not exist "%TARGET_DIR%\serve\server.py.orig_backup" (
    echo   -^> Creating backup: %TARGET_DIR%\serve\server.py.orig_backup
    copy /Y "%TARGET_DIR%\serve\server.py" "%TARGET_DIR%\serve\server.py.orig_backup" >nul
)
copy /Y "%SCRIPT_DIR%serve\server.py" "%TARGET_DIR%\serve\" >nul

echo [3/4] Deploying launch and uninstall scripts...
copy /Y "%SCRIPT_DIR%run_vllm.bat" "%TARGET_DIR%\" >nul
copy /Y "%SCRIPT_DIR%uninstall.bat" "%TARGET_DIR%\" >nul

echo.
echo ==========================================================
echo   Installation Successful!
echo ==========================================================
echo NOTE: Your original Strata setup is 100%% preserved!
echo   - To run with vLLM engine:     run_vllm.bat (or --engine vllm)
echo   - To run stock Strata:         RUN.bat (or --engine strata)
echo   - To completely revert/remove: uninstall.bat
echo ==========================================================
