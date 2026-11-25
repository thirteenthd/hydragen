@echo off
rem M-C11-win: build the CPU-only conversation tests (the park-disk gate) on Windows.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\DevOps\HydraGen\engine
set "PATH=C:\DevOps\toolchains\mingw64\bin;%PATH%"
cmake -G Ninja -S . -B build\parkv2-tests -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF ^
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl || exit /b 1
cmake --build build\parkv2-tests --target park_disk_boot_test conversation_file_test session_save_reclaim_test conversation_memory_test conversation_cache_test conversation_split_failure_test -j 8 || exit /b 1
ctest --test-dir build\parkv2-tests --output-on-failure
