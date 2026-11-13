# HydraGen installer (Windows 10/11, NVIDIA).
# Builds the vendored engine FROM SOURCE - prebuilt Strata release binaries do not contain
# HydraGen's CoW chain (expert-slot refcounting + admission fork, host-tier parked-session
# dedup; see docs/FINAL_REPORT.md) - and sets up the Python venv for the serving front.
#
# Needs: an NVIDIA driver, a CUDA toolkit (nvcc), an MSVC C++ toolchain, Python 3.10+.
#   CUDA toolkit:  https://developer.nvidia.com/cuda-downloads
#   MSVC:          winget install Microsoft.VisualStudio.2022.BuildTools --override `
#                    "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
#   cmake + ninja come from pip (engine/requirements-build.txt); nothing else to install.
#
#   .\install.ps1                                  # sm_86 (RTX 3060 Ti / 3080 Ti / 3090)
#   $env:HYDRAGEN_CUDA_ARCH="89-real"; .\install.ps1     # RTX 40-series
#   $env:HYDRAGEN_CUDA_ARCH="86-real;89-real"; .\install.ps1

param()
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ARCH = if ($env:HYDRAGEN_CUDA_ARCH) { $env:HYDRAGEN_CUDA_ARCH } else { "86-real" }

# ------------------------------------------------------------------ CUDA
$CUDA = $null
$cands = @()
if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) { $cands += $env:CUDA_PATH }
$pf = ${env:ProgramFiles}
if ($pf) {
  $cudaRoot = Join-Path $pf "NVIDIA GPU Computing Toolkit\CUDA"
  if (Test-Path $cudaRoot) {
    $cands += (Get-ChildItem $cudaRoot -Directory | Where-Object { Test-Path (Join-Path $_.FullName "bin\nvcc.exe") } | Sort-Object Name -Descending | ForEach-Object { $_.FullName })
  }
}
if (-not $cands) {
  $nvccInPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  if ($nvccInPath) { $cands += (Split-Path (Split-Path $nvccInPath.Source)) }
}
if ($cands) { $CUDA = $cands[0] }
if (-not $CUDA) {
  Write-Error @"
install.ps1: no nvcc found. Install a CUDA toolkit (https://developer.nvidia.com/cuda-downloads)
or set CUDA_PATH to one. cmake/ninja are handled for you.
"@
}
Write-Host "CUDA: $CUDA"

# ------------------------------------------------------------------ MSVC
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = $null
if (Test-Path $vswhere) {
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vs -and -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  Write-Error @"
install.ps1: no MSVC C++ toolchain found. Install VS Build Tools with the VC workload:
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
"@
}
if ($vs) { Write-Host "MSVC: $vs" }

# ------------------------------------------------------------------ Python + venv
$PY = $null
foreach ($c in @("py", "python")) {
  $cmd = Get-Command $c -ErrorAction SilentlyContinue
  if (-not $cmd) { continue }
  $args = if ($c -eq "py") { @("-3","-c","import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)") } else { @("-c","import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)") }
  & $cmd.Source @args 2>$null
  if ($LASTEXITCODE -eq 0) { $PY = "$c"; break }
}
if (-not $PY) {
  Write-Error "install.ps1: Python 3.10+ with venv+pip is required (https://python.org - check 'Add to PATH')."
}
if ($PY -eq "py") { $pyv = "py"; $pya = @("-3") } else { $pyv = "python"; $pya = @() }
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $pyv @pya -m venv .venv }
& .venv\Scripts\python.exe -m pip -q install -U pip
Write-Host "Python: installing runtime and build packages into .venv ..."
& .venv\Scripts\python.exe -m pip -q install -r engine/requirements.txt -r engine/requirements-build.txt
if ($LASTEXITCODE -ne 0) { Write-Error "install.ps1: pip install failed" }

# ------------------------------------------------------------------ engine build
$CMAKE = ".venv\Scripts\cmake.exe"
$NINJA = (Resolve-Path ".venv\Scripts\ninja.exe").Path
Write-Host "Building the HydraGen engine (CUDA arch: $ARCH) ..."
& $CMAKE -S engine -B engine/build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DSTRATA_ENABLE_CUDA=ON `
  -DSTRATA_BUILD_TESTS=OFF `
  -DSTRATA_NATIVE_EXPERTS=ON `
  -DSTRATA_PORTABLE=ON `
  -DSTRATA_Q6K_EXPERTS=ON `
  -DSTRATA_MMQ_KQUANTS=ON `
  -DCMAKE_CUDA_ARCHITECTURES=$ARCH `
  -DCMAKE_CUDA_COMPILER="$CUDA\bin\nvcc.exe" `
  -DCMAKE_MAKE_PROGRAM=$NINJA
if ($LASTEXITCODE -ne 0) { Write-Error "install.ps1: cmake configure failed" }
& $CMAKE --build engine/build --parallel ([Environment]::ProcessorCount)
if ($LASTEXITCODE -ne 0) { Write-Error "install.ps1: build failed" }
$BIN = Join-Path $PWD "engine\build\strata.exe"
if (-not (Test-Path $BIN)) { Write-Error "install.ps1: build finished but engine\build\strata.exe is missing" }
Write-Host "engine OK: $BIN"

# ------------------------------------------------------------------ local config
if (-not (Test-Path hydragen-local.json)) {
  $cfg = Get-Content configs/strata-iq3xxs-m25-0213-ctx131k.json -Raw | ConvertFrom-Json
  $cfg.exe = (Join-Path $PWD "engine\build\strata.exe") -replace '\\','/'
  $cfg.log = (Join-Path $PWD "logs\strata.log") -replace '\\','/'
  ($cfg | ConvertTo-Json -Depth 6) | Set-Content hydragen-local.json
  Write-Host "created hydragen-local.json - EDIT the model/pack paths (they default to the developer's Linux layout) before serving."
} else {
  Write-Host "hydragen-local.json already exists; leaving it alone."
}

Write-Host ""
Write-Host "Install complete. Next:"
Write-Host "  1. edit hydragen-local.json: point --native/--ple-gguf/--pack at your model files"
Write-Host "  2. .venv\Scripts\python.exe engine\serve\server.py hydragen-local.json"
Write-Host "  3. dashboard: http://127.0.0.1:8095/   (first page load sets the API key)"
