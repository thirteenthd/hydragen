@echo off
rem M-C11-WIN: local CUDA build of strata.exe from this patched tree (the E2E park-cycle engine).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\DevOps\HydraGen\engine
cmake -G Ninja -S . -B build\cuda-win -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON "-DCMAKE_CUDA_ARCHITECTURES=86-real;86-virtual" || exit /b 1
cmake --build build\cuda-win --target strata -j 8 || exit /b 1
build\cuda-win\strata.exe --version
