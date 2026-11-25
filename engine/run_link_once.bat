@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\DevOps\HydraGen\engine
ninja -C build\cuda-win hydragen.exe
echo NINJA_EXIT=%errorlevel%
build\cuda-win\hydragen.exe --version 2>nul
