# ==============================================================================
# Hydragen - Performance & Concurrency Auto-Tuning Wizard (PowerShell)
# ==============================================================================
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$python = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python (Join-Path $scriptDir "tune_hydragen.py") @args
