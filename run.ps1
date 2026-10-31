# ============================================================================
# Hydragen Server Launcher (Windows PowerShell)
# ============================================================================
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$activateScript = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
}

$binDir = Join-Path $scriptDir "bin"
if (Test-Path $binDir) {
    $env:PATH = "$binDir;$env:PATH"
}

$configFile = "hydragen.json"
if (-not (Test-Path $configFile)) {
    if (Test-Path "strata-iq3_xxs.json") {
        $configFile = "strata-iq3_xxs.json"
    }
}

Write-Host "[hydragen] Starting server using config: $configFile" -ForegroundColor Green
& hydragen --config $configFile --port 8080 --api-monitor @args
