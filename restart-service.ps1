# ==============================================================================
# Hydragen - Restart Windows Service
# ==============================================================================
$ErrorActionPreference = "SilentlyContinue"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

Write-Host "[Hydragen] Stopping service and clearing port 8080..." -ForegroundColor Yellow
Stop-Service Hydragen -Force
Get-NetTCPConnection -LocalPort 8080 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Get-Process strata, strata-vision | ForEach-Object { Stop-Process -Id $_.Id -Force }
Start-Sleep -Seconds 2

try { schtasks /run /tn "Strata-GPU-Power-Clamp" } catch {}
Write-Host "[Hydragen] Starting Hydragen service..." -ForegroundColor Green
Start-Service Hydragen

Write-Host "[Hydragen] Service restarted on port 8080!" -ForegroundColor Green
Start-Sleep -Seconds 2
