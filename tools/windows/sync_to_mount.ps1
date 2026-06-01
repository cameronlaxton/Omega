# tools/windows/sync_to_mount.ps1
#
# One-way archival sync at the end of a Cowork session. Runs on the Windows
# host AFTER the Cowork CLI / sandbox has exited.
#
# Direction is always: local workspace -> CIFS mount. The mount is an
# append-only historical ledger (robocopy /E, never /MIR). The runtime
# var/omega_traces.db NEVER overwrites the mount copy in place; instead a
# timestamped snapshot is dropped under <mount>\backups\omega_traces\.
#
# Code is pushed to the `backup` remote via `git push backup main --tags`.
# Trace JSON, session sidecars, and reports are mirrored additively via
# robocopy /E with explicit excludes for DB sidecars and noise.
#
# Pre-sync gate: cowork_preflight.py --direct-only and pytest must pass from
# the local workspace. If either fails, sync is aborted and the failure log
# is dropped under sync_failures\.
#
# Usage:
#   .\scripts\sync_to_mount.ps1
#   .\scripts\sync_to_mount.ps1 -MountRoot \\share\Omega
#   .\scripts\sync_to_mount.ps1 -WhatIf            # dry-run

[CmdletBinding()]
param(
    [string]$Workspace = (Join-Path $env:USERPROFILE ".omega\workspace\Omega"),
    [string]$MountRoot = "\\share\Omega",
    [string]$Branch = "main",
    [string]$BackupRemote = "backup",
    [switch]$SkipTests,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3

function Write-Step($msg) { Write-Host "[sync] $msg" -ForegroundColor Cyan }
function Write-Note($msg) { Write-Host "[sync] $msg" -ForegroundColor DarkGray }
function Write-Err($msg)  { Write-Host "[sync] $msg" -ForegroundColor Red }

function Log-Failure($section, $body) {
    $logRoot = Join-Path $env:USERPROFILE ".omega\workspace\sync_failures"
    if (-not (Test-Path $logRoot)) { New-Item -ItemType Directory -Path $logRoot -Force | Out-Null }
    $stamp = (Get-Date).ToString("yyyyMMdd-HHmm")
    $logFile = Join-Path $logRoot "$stamp-$section.log"
    Set-Content -Path $logFile -Value $body -Encoding UTF8
    Write-Err "Sync failure logged to $logFile"
}

if (-not (Test-Path $Workspace)) {
    Write-Err "Workspace not found: $Workspace. Run tools/windows/cowork_bootstrap.ps1 first."
    exit 1
}

Write-Step "Workspace: $Workspace"
Write-Step "Mount root: $MountRoot"
if ($WhatIf) { Write-Note "DRY RUN: no destructive side effects." }

# 1. Preflight gate. Fail closed.
if (-not $SkipTests) {
    Write-Step "Running omega-preflight --direct-only"
    if (-not $WhatIf) {
        $preflight = & omega-preflight --direct-only 2>&1
        if ($LASTEXITCODE -ne 0) {
            Log-Failure "preflight" ($preflight -join "`n")
            exit 1
        }
    }

    Write-Step "Running pytest -q --maxfail=3"
    if (-not $WhatIf) {
        Push-Location $Workspace
        try {
            $pytest = & python -m pytest tests/ -q --maxfail=3 2>&1
            if ($LASTEXITCODE -ne 0) {
                Log-Failure "pytest" ($pytest -join "`n")
                Pop-Location
                exit 1
            }
        } finally {
            if ((Get-Location).Path -eq $Workspace) { Pop-Location }
        }
    }
}

# 2. Push code to the `backup` bare repo on the CIFS share.
Write-Step "Pushing $Branch + tags to $BackupRemote"
if ($WhatIf) {
    Write-Note "DRYRUN git push $BackupRemote $Branch --tags"
} else {
    $push = & git -C $Workspace push $BackupRemote $Branch --tags 2>&1
    if ($LASTEXITCODE -ne 0) {
        Log-Failure "git_push" ($push -join "`n")
        Write-Err "git push backup failed; aborting artifact sync to avoid drift."
        exit 1
    }
    Write-Note ($push -join "`n")
}

# 3. Mirror artifact directories additively (robocopy /E, NOT /MIR). The mount
# is append-only — files removed locally must persist in the historical
# ledger.
$artifactPairs = @(
    @{ Src = (Join-Path $Workspace "inbox");   Dst = (Join-Path $MountRoot "inbox");   ExtraExclude = @("failed") },
    @{ Src = (Join-Path $Workspace "reports"); Dst = (Join-Path $MountRoot "reports"); ExtraExclude = @() }
)

foreach ($pair in $artifactPairs) {
    if (-not (Test-Path $pair.Src)) {
        Write-Note "Skipping missing source $($pair.Src)"
        continue
    }
    if (-not (Test-Path $pair.Dst)) {
        if ($WhatIf) {
            Write-Note "DRYRUN mkdir $($pair.Dst)"
        } else {
            New-Item -ItemType Directory -Path $pair.Dst -Force | Out-Null
        }
    }

    $robocopyArgs = @($pair.Src, $pair.Dst, "/E",
        "/XF", "*.db", "*.db-wal", "*.db-shm", "*.db-journal",
        "/XD") + $pair.ExtraExclude + @("__pycache__", ".pytest_cache", ".venv")

    Write-Step "robocopy $($pair.Src) -> $($pair.Dst)"
    if ($WhatIf) {
        $robocopyArgs = @("/L") + $robocopyArgs
    }
    & robocopy @robocopyArgs | Out-Null
    # robocopy exit codes: 0–7 are success/info, >=8 indicate real failure.
    if ($LASTEXITCODE -ge 8) {
        Log-Failure ("robocopy-" + [IO.Path]::GetFileName($pair.Src)) "robocopy exit=$LASTEXITCODE"
        exit 1
    }
}

# 4. Timestamped var/omega_traces.db snapshot on the mount. Write-once; the live
# DB never crosses the boundary as a moving target.
$liveDb = Join-Path $Workspace "var\var/omega_traces.db"
if (Test-Path $liveDb) {
    $backupDir = Join-Path $MountRoot "backups\omega_traces"
    if (-not (Test-Path $backupDir)) {
        if ($WhatIf) {
            Write-Note "DRYRUN mkdir $backupDir"
        } else {
            New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        }
    }
    $stamp = (Get-Date).ToString("yyyyMMdd-HHmm")
    $snapshot = Join-Path $backupDir "$stamp.db"
    Write-Step "Snapshot DB -> $snapshot"
    if ($WhatIf) {
        Write-Note "DRYRUN copy $liveDb $snapshot"
    } else {
        Copy-Item -Path $liveDb -Destination $snapshot -ErrorAction Stop
    }
} else {
    Write-Note "No local var/omega_traces.db found at $liveDb; skipping DB snapshot."
}

Write-Host ""
Write-Host "sync_to_mount_ready" -ForegroundColor Green
