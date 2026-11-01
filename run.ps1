# ============================================================================
# Hydragen Server Launcher (Windows PowerShell)
# ============================================================================
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Prepend virtual environment Scripts and bin to PATH
$venvScripts = Join-Path $scriptDir ".venv\Scripts"
if (Test-Path $venvScripts) {
    $env:PATH = "$venvScripts;$env:PATH"
}

$binDir = Join-Path $scriptDir "bin"
if (Test-Path $binDir) {
    $env:PATH = "$binDir;$env:PATH"
}

# Optional activation if execution policy allows
$activateScript = Join-Path $venvScripts "Activate.ps1"
if (Test-Path $activateScript) {
    try {
        & $activateScript
    } catch {}
}

$configFile = "hydragen.json"
if (-not (Test-Path $configFile)) {
    if (Test-Path "strata-iq3_xxs.json") {
        $configFile = "strata-iq3_xxs.json"
    }
}

$portArg = @()
if (-not ($args -contains "--port")) {
    $portArg = @("--port", "8080")
}

Write-Host "[hydragen] Starting server using config: $configFile" -ForegroundColor Green

# Invoke hydragen CLI directly from venv
$hydragenExe = Join-Path $venvScripts "hydragen.exe"
if (Test-Path $hydragenExe) {
    & $hydragenExe --config $configFile --api-monitor @portArg @args
} else {
    & python -m hydragen.cli --config $configFile --api-monitor @portArg @args
}
