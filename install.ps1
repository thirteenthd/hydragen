# HydraGen installer (Windows 10/11, NVIDIA).
# Prefers a PREBUILT CUDA engine (this repo's GitHub Releases: RTX 30/40/50 fat
# sm_86/89/120 binary + the CUDA runtime DLLs it needs; SHA-256-verified against
# the release manifest, download marked-and-unblocked by this script) and falls
# back to building the vendored engine FROM SOURCE when the download or hash
# check fails - or on request with -BuildFromSource. The prebuilt engine IS the
# canonical cow chain (expert-slot refcounting + admission fork, host-tier
# parked-session dedup; see docs/FINAL_REPORT.md) - same flags the source build uses.
#
# The prebuilt binary is UNSIGNED, like every community CUDA build: if you grab
# the .zip yourself in a browser, Windows SmartScreen may show "Windows protected
# your PC" - click More info -> Run anyway. This installer's own download carries
# no browser mark and every extracted file is unblocked here, so you will not see
# that warning when installing this way. Unsure about running a downloaded
# binary? Use -BuildFromSource: same engine compiled on your machine, just more
# time (and the toolchain requirements below).
#
# Source-build path needs: an NVIDIA driver, a CUDA toolkit (nvcc), an MSVC C++ toolchain, Python 3.10+.
#   CUDA toolkit:  https://developer.nvidia.com/cuda-downloads
#   MSVC:          winget install Microsoft.VisualStudio.2022.BuildTools --override `
#                    "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
#   cmake + ninja come from pip (engine/requirements-build.txt); nothing else to install.
#
#   .\install.ps1                                  # prebuilt if available, else source (sm_86)
#   .\install.ps1 -BuildFromSource                 # never run a downloaded binary; compile locally
#   $env:HYDRAGEN_CUDA_ARCH="89-real"; .\install.ps1     # RTX 40-series
#   $env:HYDRAGEN_CUDA_ARCH="86-real;89-real"; .\install.ps1

param([string]$ModelDir, [switch]$BuildFromSource)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ARCH = if ($env:HYDRAGEN_CUDA_ARCH) { $env:HYDRAGEN_CUDA_ARCH } else { "86-real" }
$RELEASE_TAG = "prebuilt-1"

# ------------------------------------------------- environment sizing (auto-detected; edit hydragen-local.json to override)
$DetMib = 65536
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) { $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue }
if ($smi) {
  $gpu = & $smi.Source --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>$null |
    Sort-Object { [int](($_ -split ',\s*')[1]) } -Descending | Select-Object -First 1
  if ($gpu) {
    $p = $gpu -split ',\s*'
    Write-Host "install.ps1: GPU detected: $gpu"
    if (([int]$p[1]) -lt 12000) { Write-Warning "largest GPU is $($p[1]) MiB - the shipped config targets 12 GB+ VRAM" }
    if ($p[2] -notmatch '^(8|9|1[0-9])\.') { Write-Warning "compute capability $($p[2]) - this engine build targets Ampere (sm_80) or newer" }
  }
} else { Write-Warning "install.ps1: nvidia-smi not found - install the NVIDIA driver before serving (skipping GPU sizing)" }
$ramMib = [int]((Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue).TotalPhysicalMemory / 1MB)
if ($ramMib -gt 0) {
  $DetMib = [Math]::Min($ramMib, 131072); if ($DetMib -lt 16384) { $DetMib = 16384 }
  if ($ramMib -lt 49152) { Write-Warning "$ramMib MiB RAM detected - the documented floor is 64 GB; expect parked sessions to be re-prefilled more often" }
  Write-Host "install.ps1: RAM detected: $ramMib MiB -> park-disk tier budget $DetMib MiB (a tier pinned above the RAM class measurably decays steady-state throughput - docs/HOW_IT_WORKS.md)"
}

# ------------------------------------------------------------------ prebuilt engine (preferred)
$BIN = $null
if (-not $BuildFromSource) {
  $base = "https://github.com/thirteenthd/hydragen/releases/download/$RELEASE_TAG"
  $zip = Join-Path $env:TEMP "hydragen-windows-cuda-x86_64.zip"
  try {
    Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS.txt" -OutFile "$zip.sha256sums" -TimeoutSec 30 | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri "$base/hydragen-windows-cuda-x86_64.zip" -OutFile $zip -TimeoutSec 900 | Out-Null
    $want = ((Get-Content "$zip.sha256sums" | Where-Object { $_ -match "windows-cuda" } | Select-Object -First 1) -split '\s+')[0]
    $have = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($want -and $have -eq $want.ToLower()) {
      New-Item -ItemType Directory -Force -Path "runtime" | Out-Null
      Expand-Archive -Force -Path $zip -DestinationPath "runtime"
      Get-ChildItem "runtime" -Recurse | Unblock-File
      $BIN = Join-Path $PWD "runtime\hydragen.exe"
      if (-not (Test-Path $BIN)) { $BIN = $null; Write-Warning "prebuilt zip has no hydragen.exe - falling back to a source build" }
      else {
        Write-Host "install.ps1: prebuilt engine installed (runtime\hydragen.exe; SHA-256 verified; download unblocked):"
        Get-Content "runtime\BUILD_INFO.txt" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
        Write-Host "install.ps1: prefer compiling it yourself? rerun: .\install.ps1 -BuildFromSource (same engine,"
        Write-Host "            nothing downloaded, just more time)."
      }
    } else { Write-Warning "prebuilt hash check failed (manifest $want, got $have) - falling back to a source build" }
  } catch { Write-Warning "prebuilt download failed ($($_.Exception.Message)) - falling back to a source build" }
  Remove-Item $zip, "$zip.sha256sums" -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------------ CUDA (source-build path)
$CUDA = $null
if (-not $BIN) {
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
}

# ------------------------------------------------------------------ MSVC
if (-not $BIN) {
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
$vcvars = if ($vs) { Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
if ($vcvars -and (Test-Path $vcvars)) {
  cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
  }
  Write-Host "MSVC environment activated (vcvars64)"
}
}

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

# ------------------------------------------------------------------ engine build (fallback path)
if (-not $BIN) {
$CMAKE = ".venv\Scripts\cmake.exe"
$NINJA = (Resolve-Path ".venv\Scripts\ninja.exe").Path
Write-Host "Building the HydraGen engine from source (CUDA arch: $ARCH) ..."
& $CMAKE -S engine -B engine/build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DSTRATA_ENABLE_CUDA=ON `
  -DSTRATA_BUILD_TESTS=OFF `
  -DSTRATA_NATIVE_EXPERTS=ON `
  -DSTRATA_PORTABLE=ON `
  -DSTRATA_Q6K_EXPERTS=ON `
  -DSTRATA_MMQ_KQUANTS=ON `
  "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" `
  "-DCMAKE_CUDA_COMPILER=$CUDA\bin\nvcc.exe" `
  "-DCMAKE_MAKE_PROGRAM=$NINJA"
if ($LASTEXITCODE -ne 0) { Write-Error "install.ps1: cmake configure failed" }
& $CMAKE --build engine/build --target strata --parallel ([Environment]::ProcessorCount)
if ($LASTEXITCODE -ne 0) { Write-Error "install.ps1: build failed" }
$BIN = Join-Path $PWD "engine\build\hydragen.exe"
if (-not (Test-Path $BIN)) { Write-Error "install.ps1: build finished but engine\build\hydragen.exe is missing" }
Write-Host "engine OK: $BIN"
}

# ------------------------------------------------------------------ local config
if (-not $ModelDir) { $ModelDir = Join-Path $PWD "models" }
if (-not [System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir = Join-Path $PWD $ModelDir }
foreach ($d in @("packs", "models", "mtp")) { New-Item -ItemType Directory -Force -Path (Join-Path $ModelDir $d) | Out-Null }
try { $freeMib = [int]((New-Object System.IO.DriveInfo($ModelDir.Substring(0, 2))).AvailableFreeSpace / 1MB) } catch { $freeMib = -1 }
if ($freeMib -ge 0 -and $freeMib -lt ($DetMib + 174080)) {
  Write-Warning "install.ps1: $freeMib MiB free on the $ModelDir drive - model blobs need ~160 GiB plus the $DetMib MiB tier budget (the tier fills lazily; now-check, not a hard gate)"
}
$mdFwd = $ModelDir -replace '\\','/'
$ggufDir = "$mdFwd/models/IQ3_XXS/IQ3_XXS"
$blobTargets = @{
  "--pack" = "$mdFwd/packs/iq3_xxs";
  "--native" = "$ggufDir/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
  "--ple-gguf" = "$ggufDir/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf";
  "--expert-profile" = ((Join-Path $PWD "engine\data\expert-profile.bin") -replace '\\','/');
  "--mtp" = "$mdFwd/mtp/rt"
}
# per-artifact learned-profile loop (mirrors install.sh): --expert-profile-save persists the
# adaptive tier's live ranking; the file is keyed by the model blob name so a stale ranking
# from a different artifact can never abort the next boot.
$_pkey = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"))).Replace("-","").Substring(0,10).ToLower()
$learned = (Join-Path $PWD "expert-profile-learned-$_pkey.bin") -replace '\\','/'
if (Test-Path $learned) { $blobTargets["--expert-profile"] = $learned }
if (-not (Test-Path hydragen-local.json)) {
  $cfg = Get-Content configs/strata-iq3xxs-m25-0213-ctx131k.json -Raw | ConvertFrom-Json
  $cfg.exe = ($BIN -replace '\\','/')
  $cfg.log = (Join-Path $PWD "logs\strata.log") -replace '\\','/'
  foreach ($d in @("logs", "park-disk")) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
  $cfg.env | Add-Member -Force -NotePropertyMembers @{
    HG_PARK_COW = "1"; HG_PARK_DISK_V2 = "1"; HG_PARK_DISK_MIB = "$DetMib";
    STRATA_ADMIT_SHORT_FIRST = "8192";
    HG_RESUME_DEEPEST = "1";
    HG_PARK_DISK = (Join-Path $PWD "park-disk") -replace '\\','/'
  }
  for ($i = 0; $i -lt $cfg.args.Count; $i++) {
    if ($blobTargets.ContainsKey($cfg.args[$i]) -and $i + 1 -lt $cfg.args.Count) {
      $cfg.args[$i + 1] = $blobTargets[$cfg.args[$i]]
    }
  }
  $cfg.cwd = ($PWD -replace '\\','/')
  $cfg.tokenizer = "$mdFwd/packs/iq3_xxs/tokenizer"
  if ($cfg.args -notcontains "--expert-profile-save") { $cfg.args += @("--expert-profile-save", $learned) }
  if ($cfg.args -notcontains "--conversation-cache-min-free-mib") { $cfg.args += @("--conversation-cache-min-free-mib", "512") }
  if ($cfg.args -notcontains "--prompt-cache-tail") { $cfg.args += "--prompt-cache-tail" }
  ($cfg | ConvertTo-Json -Depth 6) | Set-Content hydragen-local.json
  Write-Host "created hydragen-local.json (canonical cow stack wired) - model paths point under $ModelDir (override: .\install.ps1 -ModelDir D:\models)."
  Write-Host "  the expert-routing profile is already in this repo (engine\data\expert-profile.bin); the GGUFs/pack/MTP come next."
} else {
  Write-Host "hydragen-local.json already exists; leaving it alone."
}

Write-Host ""
Write-Host "Install complete. Next:"
Write-Host "  1. model blobs (~76 GB IQ3_XXS set), sources and layout: docs\MODELS.md. Easiest:"
Write-Host "       cmd /c engine\SETUP.bat --model IQ3_XXS --data-dir `"$ModelDir`""
Write-Host "     (Strata's own downloader: resumable, SHA-256-checked; builds the pack and fetches MTP)."
Write-Host "     Or: uvx hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include `"IQ3_XXS/*`" --local-dir DIR"
Write-Host "         then engine\SETUP.bat --gguf-dir DIR"
Write-Host "  2. .venv\Scripts\python.exe engine\serve\server.py --engine hydragen --config hydragen-local.json --port 8181"
Write-Host "  3. dashboard + Monitor: http://127.0.0.1:8181/"
Write-Host "  4. the canonical stack is pre-wired: STRATA_PARK_COW=1, park-disk v2 tier (park-disk\),"
Write-Host "     STRATA_RESUME_DEEPEST=1, --prompt-cache-tail (tier budget: STRATA_PARK_DISK_MIB, default 64 GiB -"
Write-Host "     a pinned bigger cap measurably decays throughput on a 64 GB-RAM host), STRATA_ADMIT_SHORT_FIRST=8192"
