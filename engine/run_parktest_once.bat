@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\DevOps\HydraGen\engine
set "PATH=C:\DevOps\toolchains\mingw64\bin;%PATH%"
cmake --build build\parkv2-tests --target park_disk_boot_test -j 8 || exit /b 1
build\parkv2-tests\park_disk_boot_test.exe
