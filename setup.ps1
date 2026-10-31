# ============================================================================
# Hydragen One-Click Setup Script (Windows PowerShell)
# ============================================================================
$ProgressPreference = "SilentlyContinue"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Clean up any legacy or unpacked Strata installer CLI (setup.py) so setuptools
# builds strictly from pyproject.toml without mistaking it for a package setup script.
$straySetupPy = Join-Path $scriptDir "setup.py"
if (Test-Path $straySetupPy) {
    Remove-Item $straySetupPy -Force -ErrorAction SilentlyContinue
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Hydragen Engine Automated Setup & Updater (PowerShell) " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Parse arguments
$modelDir = ""
for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = $args[$i]
    if ($arg -eq "--update" -or $arg -eq "-u") {
        Write-Host "[+] Updating Hydragen from git repository..." -ForegroundColor Yellow
        if (Test-Path ".git") {
            & git pull
        }
    } elseif ($arg -like "--model-dir=*") {
        $modelDir = $arg.Substring(12)
    } elseif ($arg -eq "--model-dir" -or $arg -eq "-m") {
        if ($i + 1 -lt $args.Count) {
            $modelDir = $args[$i + 1]
            $i++
        }
    }
}

# 1. Check Python (PyTorch/Triton require Python >=3.10 and <=3.13)
$pyCmd = $null
$candidates = @("python", "python3.12", "python3.11", "python3.10", "py")
foreach ($c in $candidates) {
    $check = Get-Command $c -ErrorAction SilentlyContinue
    if ($check) {
        $verOut = ""
        try {
            $verOut = & $c -c "import sys; print('%s.%s' % sys.version_info[:2])"
        } catch {}
        if ($verOut -and $verOut -match "^\s*3\.(1[0-3])") {
            $pyCmd = $c
            Write-Host "[+] Detected Python $verOut ($c)" -ForegroundColor Green
            break
        }
    }
}

if (-not $pyCmd) {
    $genCheck = Get-Command python -ErrorAction SilentlyContinue
    if ($genCheck) {
        $pyCmd = "python"
        $verOut = & python -c "import sys; print('%s.%s' % sys.version_info[:2])"
        Write-Host "[+] Detected Python $verOut (python)" -ForegroundColor Green
    } else {
        Write-Error "[-] Python not found in PATH. Please install Python 3.10-3.13 from python.org and add to PATH."
        exit 1
    }
}

# 2. Check GPU
$gpuCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($gpuCmd) {
    $gpuName = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | Select-Object -First 1
    Write-Host "[+] Detected GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Warning "[!] nvidia-smi not found. NVIDIA CUDA GPU is required for acceleration."
}

# 3. Virtual Environment Setup
$venvDir = Join-Path $scriptDir ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[+] Creating virtual environment in $venvDir..." -ForegroundColor Yellow
    & $pyCmd -m venv $venvDir
}

# Prepend virtual environment Scripts to PATH
$venvScripts = Join-Path $venvDir "Scripts"
$env:PATH = "$venvScripts;$env:PATH"
$env:VIRTUAL_ENV = $venvDir

# Attempt activation script if allowed by execution policy
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    try {
        & $activateScript
    } catch {
        # Execution policy won't block python/pip execution directly
    }
}
Write-Host "[+] Virtual environment ready: $venvDir" -ForegroundColor Green

# 4. Check & Install PyTorch with CUDA
Write-Host "[+] Checking PyTorch CUDA support..." -ForegroundColor Yellow
$cudaAvailable = $false
try {
    $probe = & $venvPy -c "import importlib.util as u; s = u.find_spec('torch'); m = __import__('torch') if s else None; print(bool(m and m.cuda.is_available()))"
    if ($probe -and $probe.Trim() -eq "True") {
        $cudaAvailable = $true
    }
} catch {
    $cudaAvailable = $false
}

if (-not $cudaAvailable) {
    Write-Host "[+] Installing PyTorch with CUDA 12.4 support..." -ForegroundColor Yellow
    & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "[+] PyTorch with CUDA is already installed." -ForegroundColor Green
}

# 5. Ensure bin directory & precompiled worker binary
$binDir = Join-Path $scriptDir "bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}
$strataExe = Join-Path $binDir "strata.exe"
if (-not (Test-Path $strataExe)) {
    if ($env:STRATA_DIR -and (Test-Path (Join-Path $env:STRATA_DIR "engine\strata.exe"))) {
        Write-Host "[+] Copying Strata binary from `$env:STRATA_DIR..." -ForegroundColor Green
        Copy-Item (Join-Path $env:STRATA_DIR "engine\strata.exe") $strataExe
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
            if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
                & curl.exe -sL --fail $winUrl -o $tarPath
            } else {
                Invoke-WebRequest -Uri $winUrl -OutFile $tarPath -UseBasicParsing
            }
            & tar -xzf $tarPath -C $binDir
            Remove-Item $tarPath -Force -ErrorAction SilentlyContinue
            Write-Host "[+] Successfully extracted Strata v0.2.9 worker to $binDir" -ForegroundColor Green
        } catch {
            Write-Warning "[!] Auto-download failed: $_"
            Write-Warning "[!] You can manually copy 'strata.exe' into $binDir\strata.exe."
        }
    }
}

# 6. Ensure strata_control, tools, and data modules are available
$ctrlDir = Join-Path $scriptDir "strata_control"
if (-not (Test-Path $ctrlDir)) {
    Write-Host "[+] Downloading Strata control & data modules (v0.2.9 / 0.1.55)..." -ForegroundColor Yellow
    $ctrlUrl = "https://github.com/Niko1221/Strata/releases/download/v0.2.9/strata-control-0.1.55-windows-x86_64.zip"
    $zipPath = Join-Path $binDir "control.zip"
    try {
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            & curl.exe -sL --fail $ctrlUrl -o $zipPath
        } else {
            Invoke-WebRequest -Uri $ctrlUrl -OutFile $zipPath -UseBasicParsing
        }
        Expand-Archive -Path $zipPath -DestinationPath $scriptDir -Force
        Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
        Write-Host "[+] Extracted strata_control and tools" -ForegroundColor Green
    } catch {
        Write-Warning "[!] Control modules auto-download failed: $_"
    }
}

# Clean up any unpacked Strata installer CLI (setup.py) so setuptools does not mistake it for a package config
if (Test-Path $straySetupPy) {
    Remove-Item $straySetupPy -Force -ErrorAction SilentlyContinue
}

# 7. Install Hydragen package in editable mode
Write-Host "[+] Installing Hydragen and dependencies..." -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Error "[-] Failed to install Hydragen package. Please review the pip output above."
    exit $LASTEXITCODE
}

# 8. Model Storage Configuration
if ($modelDir -and (Test-Path $modelDir)) {
    $modelsTarget = Join-Path $scriptDir "models"
    if (-not (Test-Path $modelsTarget)) {
        Write-Host "[+] Linking model directory: $modelDir -> $modelsTarget..." -ForegroundColor Green
        New-Item -ItemType Junction -Path $modelsTarget -Target $modelDir | Out-Null
    }
}

if (-not (Test-Path (Join-Path $scriptDir "models"))) {
    Write-Host "[i] Note: Model weights not found in models folder." -ForegroundColor Yellow
    Write-Host "    Link your models directory with:"
    Write-Host "      .\setup.ps1 --model-dir=D:\path\to\models"
}

# 9. Generate default hydragen.json template if missing
$cfgPath = Join-Path $scriptDir "hydragen.json"
if (-not (Test-Path $cfgPath)) {
    Write-Host "[+] Generating default hydragen.json template..." -ForegroundColor Yellow
    $defaultCfg = @"
{
  "exe": "bin\\strata.exe",
  "args": [
    "--pack", "models/packs/iq3_xxs",
    "--native", "models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "--ple-gguf", "models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto:16384",
    "--spec", "4",
    "--mtp", "models/mtp/rt",
    "--max-context", "131072",
    "--kv", "int8",
    "--vision",
    "--vram-reserve-mib", "1152",
    "--pcie-frac", "0.20",
    "--spec-min-p", "0.45",
    "--coupled-draft",
    "--adapt-every", "1",
    "--adapt-swaps", "160",
    "--adapt-decay", "0.97",
    "--adapt-async", "1",
    "--kv-resident", "32768",
    "--conversation-cache-mib", "6144",
    "--conversation-cache-slots", "8",
    "--prompt-cache-tail",
    "--batch", "2",
    "--batch-mtp"
  ],
  "cwd": ".",
  "tokenizer": "models/packs/iq3_xxs/tokenizer",
  "model_name": "qwen3.8-flash-next-iq3_xxs",
  "log": "hydragen.log",
  "port": 8080,
  "gpu": 0,
  "gpus_asked": true,
  "host": "0.0.0.0",
  "open_browser": false,
  "vision": {
    "exe": "bin\\strata-vision.exe",
    "mmproj": "models/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    "model": "models/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "gpu": false,
    "max_tokens": 300,
    "threads": 12
  },
  "allowed_hosts": [
    "*"
  ],
  "env": {
    "STRATA_PREFILL_CPU_SHARE": "auto",
    "STRATA_PF_FUSED": "1",
    "STRATA_PREFILL_EQUAL": "1",
    "STRATA_ROUTE_PRIOR": "0.8"
  },
  "parallel": 2,
  "reasoning_loop_recovery": "recover",
  "reasoning_close_retry": true,
  "repeat_stop_tokens": 64
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
