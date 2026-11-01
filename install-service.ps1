# ==============================================================================
# Hydragen - Register as Windows Service (services.msc)
# ==============================================================================
$ErrorActionPreference = "Stop"

# Self-elevate if not running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[Hydragen] Requesting Administrator elevation..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

$hydragenRoot = "C:\Hydragen"
$nssm = Join-Path $hydragenRoot "tools\nssm.exe"
$python = Join-Path $hydragenRoot ".venv\Scripts\python.exe"
$logFile = Join-Path $hydragenRoot "hydragen-service.log"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   Replacing Legacy Strata Service with Hydragen AI Local Server" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Stop and remove old Strata service
Write-Host "[1/5] Removing legacy Strata service (if present)..." -ForegroundColor Yellow
try { Stop-Service Strata -Force -ErrorAction SilentlyContinue } catch {}
& $nssm stop Strata *>$null
& $nssm remove Strata confirm *>$null

# 2. Stop and remove existing Hydragen service if reinstalling
Write-Host "[2/5] Stopping existing Hydragen service (if present)..." -ForegroundColor Yellow
try { Stop-Service Hydragen -Force -ErrorAction SilentlyContinue } catch {}
& $nssm stop Hydragen *>$null
& $nssm remove Hydragen confirm *>$null

# 3. Kill any lingering process holding port 8080
Write-Host "[3/5] Cleaning up port 8080 and engine processes..." -ForegroundColor Yellow
Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
}
Get-Process strata, strata-vision -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } catch {}
}

# 4. Install Hydragen service with NSSM
Write-Host "[4/5] Registering 'Hydragen AI Local Server' with NSSM..." -ForegroundColor Green
$appArgs = "-m hydragen.cli --engine strata --config hydragen.json --port 8080 --api-monitor --no-api-key"
& $nssm install Hydragen "$python" "$appArgs"
& $nssm set Hydragen AppDirectory "$hydragenRoot"
& $nssm set Hydragen DisplayName "Hydragen AI Local Server"
& $nssm set Hydragen Description "Hydragen AI Local Inference Server with CUDA acceleration and resident MoE expert tuning"
& $nssm set Hydragen Start SERVICE_AUTO_START
& $nssm set Hydragen AppStdout "$logFile"
& $nssm set Hydragen AppStderr "$logFile"
& $nssm set Hydragen AppStopMethodSkip 0
& $nssm set Hydragen AppStopMethodConsole 2000
& $nssm set Hydragen AppStopMethodWindow 2000
& $nssm set Hydragen AppStopMethodThreads 2000

# 5. Start the service
Write-Host "[5/5] Starting Hydragen service..." -ForegroundColor Green
try { schtasks /run /tn "Strata-GPU-Power-Clamp" *>$null } catch {}
Start-Sleep -Seconds 1
Start-Service Hydragen

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Hydragen AI Local Server is now running as a Windows Service!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  Service Name : Hydragen"
Write-Host "  Display Name : Hydragen AI Local Server"
Write-Host "  Startup Type : Automatic"
Write-Host "  Port         : 8080"
Write-Host "  Log File     : $logFile"
Write-Host ""
Write-Host "Press any key to close..."
[void][System.Console]::ReadKey()
