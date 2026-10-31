# ============================================================================
# Hydragen One-Click Setup Script (Windows PowerShell)
# ============================================================================
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Hydragen Engine Automated Setup & Updater (PowerShell) " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if update requested
if ($args -contains "--update" -or $args -contains "-u") {
    Write-Host "[+] Updating Hydragen from git repository..." -ForegroundColor Yellow
    if (Test-Path ".git") {
        & git pull
    }
}

# 1. Check Python
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    Write-Error "[-] Python not found in PATH. Please install Python 3.10+ from python.org and add to PATH."
}

$pyVer = & python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
Write-Host "[+] Detected Python $pyVer" -ForegroundColor Green

# 2. Check GPU
$gpuCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($gpuCmd) {
    $gpuName = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | Select-Object -First 1
    Write-Host "[+] Detected GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Warning "[!] nvidia-smi not found. CUDA GPU is required for acceleration."
}

# 3. Virtual Environment
$venvDir = Join-Path $scriptDir ".venv"
if (-not $env:VIRTUAL_ENV) {
    if (-not (Test-Path $venvDir)) {
        Write-Host "[+] Creating virtual environment in $venvDir..." -ForegroundColor Yellow
        & python -m venv $venvDir
    }
    Write-Host "[+] Activating virtual environment..." -ForegroundColor Yellow
    $activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
    & $activateScript
} else {
    Write-Host "[+] Using active virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
}

# 4. Check & Install PyTorch with CUDA
$cudaAvailable = & python -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaAvailable -ne "True") {
    Write-Host "[+] Installing PyTorch with CUDA 12.4 support..." -ForegroundColor Yellow
    & pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

# 5. Install package in editable mode
Write-Host "[+] Installing Hydragen and dependencies..." -ForegroundColor Yellow
& python -m pip install --upgrade pip
& python -m pip install -e .

# 6. Ensure bin directory & precompiled worker binary
$binDir = Join-Path $scriptDir "bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}
$strataExe = Join-Path $binDir "strata.exe"
if (-not (Test-Path $strataExe)) {
    if ($env:STRATA_DIR -and (Test-Path (Join-Path $env:STRATA_DIR "engine\strata.exe"))) {
        Write-Host "[+] Copying Strata binary from `$env:STRATA_DIR..." -ForegroundColor Green
        Copy-Item (Join-Path $env:STRATA_DIR "engine\strata.exe") $strataExe
    } elseif (Test-Path "..\Strata\engine\strata.exe") {
        Write-Host "[+] Copying local Strata binary..." -ForegroundColor Green
        Copy-Item "..\Strata\engine\strata.exe" $strataExe
    } else {
        Write-Host "[+] Downloading precompiled Strata worker binary (v0.2.9 / engine 0.1.55)..." -ForegroundColor Yellow
        $cudaVer = "cuda12"
        if (Get-Command nvcc -ErrorAction SilentlyContinue) {
            $nvccOut = nvcc --version | Out-String
            if ($nvccOut -match "release 13") {
                $cudaVer = "cuda13"
            }
        }
        Write-Host "[+] Selected CUDA target: $cudaVer" -ForegroundColor Cyan
        $winUrl = "https://github.com/Niko1221/Strata/releases/download/v0.2.9/strata-engine-0.1.55-windows-x86_64-cuda-${cudaVer}.tar.gz"
        $tarPath = Join-Path $binDir "strata-engine.tar.gz"
        try {
            Invoke-WebRequest -Uri $winUrl -OutFile $tarPath -UseBasicParsing
            tar -xzf $tarPath -C $binDir
            Remove-Item $tarPath -Force
            Write-Host "[+] Successfully extracted Strata v0.2.9 worker to $binDir" -ForegroundColor Green
        } catch {
            Write-Warning "[!] Auto-download failed. You can copy your compiled 'strata.exe' into $binDir\strata.exe."
        }
    }
}

# 7. Ensure strata_control and tools modules are available
$ctrlDir = Join-Path $scriptDir "strata_control"
if ((-not (Test-Path $ctrlDir)) -and (-not (Test-Path "..\Strata\strata_control"))) {
    Write-Host "[+] Downloading Strata control modules (v0.2.9 / 0.1.55)..." -ForegroundColor Yellow
    $ctrlUrl = "https://github.com/Niko1221/Strata/releases/download/v0.2.9/strata-control-0.1.55-windows-x86_64.zip"
    $zipPath = Join-Path $binDir "control.zip"
    try {
        Invoke-WebRequest -Uri $ctrlUrl -OutFile $zipPath -UseBasicParsing
        Expand-Archive -Path $zipPath -DestinationPath $scriptDir -Force
        Remove-Item $zipPath -Force
        Write-Host "[+] Extracted strata_control and tools" -ForegroundColor Green
    } catch {
        Write-Warning "[!] Control modules auto-download failed."
    }
}

# 8. Generate default hydragen.json
$cfgPath = Join-Path $scriptDir "hydragen.json"
if (-not (Test-Path $cfgPath)) {
    Write-Host "[+] Generating default hydragen.json template..." -ForegroundColor Yellow
    $defaultCfg = @"
{
  "exe": "bin\\strata.exe",
  "args": [
    "--expert-cache", "auto",
    "--prefill", "auto:16384",
    "--spec", "4",
    "--max-context", "131072",
    "--kv", "int8",
    "--vram-reserve-mib", "1152",
    "--pcie-frac", "0.20",
    "--spec-min-p", "0.45",
    "--adapt-async", "1",
    "--kv-resident", "32768",
    "--batch", "2",
    "--batch-mtp"
  ],
  "batch": 2,
  "max_context": 131072,
  "log": "hydragen.log"
}
"@
    Set-Content -Path $cfgPath -Value $defaultCfg -Encoding UTF8
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         Hydragen Windows Setup Complete!                 " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "To start the server:"
Write-Host "  .\run.ps1"
Write-Host "  (or: hydragen --config hydragen.json --port 8080 --api-monitor)"
Write-Host ""
Write-Host "To update in the future:"
Write-Host "  .\setup.ps1 -u"
Write-Host "==========================================================" -ForegroundColor Cyan
