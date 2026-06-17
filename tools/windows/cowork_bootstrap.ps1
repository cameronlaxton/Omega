# tools/windows/cowork_bootstrap.ps1
#
# Cowork local-workspace bootstrap. Runs on the Windows host BEFORE the Cowork
# CLI / sandbox is launched. Moves Omega execution off the network/FUSE mount
# (root cause of BUG-FUSE-2 and BUG-PREFLIGHT-3) and into a per-user local
# clone under %USERPROFILE%\.omega\workspace\Omega.
#
# Idempotent: safe to re-run. If the local clone already exists, this script
# fetches origin, fast-forwards main, and re-verifies the `backup` remote.
#
# Usage:
#   .\scripts\cowork_bootstrap.ps1                    # default origin/backup
#   .\scripts\cowork_bootstrap.ps1 -OriginUrl <url>   # override clone source
#   .\scripts\cowork_bootstrap.ps1 -BackupRepo <path> # override backup remote
#   .\scripts\cowork_bootstrap.ps1 -DryRun            # no side effects
#
# After this script runs, launch the Cowork CLI from the printed workspace
# path. Sync archival back to the mount with tools/windows/sync_to_mount.ps1 at
# session close.

[CmdletBinding()]
param(
    [string]$OriginUrl = "https://github.com/anthropic-cl/Omega.git",
    [string]$BackupRepo = "\\share\Omega.git",
    [string]$WorkspaceRoot = (Join-Path $env:USERPROFILE ".omega\workspace"),
    [string]$Branch = "main",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3

function Write-Step($msg) { Write-Host "[bootstrap] $msg" -ForegroundColor Cyan }
function Write-Note($msg) { Write-Host "[bootstrap] $msg" -ForegroundColor DarkGray }
function Write-Err($msg)  { Write-Host "[bootstrap] $msg" -ForegroundColor Red }

function Invoke-Git {
    param([string[]]$GitArgs, [string]$WorkDir)
    if ($DryRun) {
        Write-Note ("DRYRUN git " + ($GitArgs -join ' ') + "  (cwd=$WorkDir)")
        return ""
    }
    $output = & git -C $WorkDir @GitArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Err "git $($GitArgs -join ' ') failed: $output"
        throw "git command failed"
    }
    return ($output -join "`n")
}

# 1. Ensure local workspace root exists.
$workspaceOmega = Join-Path $WorkspaceRoot "Omega"
Write-Step "Workspace target: $workspaceOmega"
if (-not (Test-Path $WorkspaceRoot)) {
    if ($DryRun) {
        Write-Note "DRYRUN mkdir $WorkspaceRoot"
    } else {
        New-Item -ItemType Directory -Path $WorkspaceRoot -Force | Out-Null
    }
}

# 2. Clone or refresh the local copy. We deliberately prefer the upstream
# remote over the FUSE-mounted source to avoid pulling Pattern C corruption
# into the local clone.
if (-not (Test-Path (Join-Path $workspaceOmega ".git"))) {
    Write-Step "Cloning $OriginUrl into $workspaceOmega"
    if ($DryRun) {
        Write-Note "DRYRUN git clone $OriginUrl $workspaceOmega"
    } else {
        & git clone --branch $Branch $OriginUrl $workspaceOmega
        if ($LASTEXITCODE -ne 0) {
            Write-Err "git clone failed (exit=$LASTEXITCODE). Aborting."
            exit 1
        }
    }
} else {
    Write-Step "Local clone exists. Fetching origin and fast-forwarding $Branch."
    Invoke-Git -GitArgs @("fetch", "origin") -WorkDir $workspaceOmega
    Invoke-Git -GitArgs @("checkout", $Branch) -WorkDir $workspaceOmega
    Invoke-Git -GitArgs @("pull", "--ff-only", "origin", $Branch) -WorkDir $workspaceOmega
}

# 3. Configure the `backup` remote to the CIFS bare repo (idempotent). If the
# bare repo does not exist yet, the caller must create it once:
#     git init --bare \\share\Omega.git
$existingBackup = ""
try {
    $existingBackup = Invoke-Git -GitArgs @("remote", "get-url", "backup") -WorkDir $workspaceOmega
} catch {
    $existingBackup = ""
}

if (-not $existingBackup) {
    Write-Step "Adding 'backup' remote -> $BackupRepo"
    Invoke-Git -GitArgs @("remote", "add", "backup", $BackupRepo) -WorkDir $workspaceOmega
} elseif ($existingBackup.Trim() -ne $BackupRepo) {
    Write-Step "Updating 'backup' remote: $($existingBackup.Trim()) -> $BackupRepo"
    Invoke-Git -GitArgs @("remote", "set-url", "backup", $BackupRepo) -WorkDir $workspaceOmega
} else {
    Write-Note "'backup' remote already points at $BackupRepo"
}

# 4. Export the workspace path for the caller to consume. The Cowork CLI
# should `Set-Location $workspaceOmega` before launching.
$env:COWORK_LOCAL_WORKSPACE = "1"
$env:OMEGA_LOCAL_WORKSPACE = $workspaceOmega

Write-Host ""
Write-Host "cowork_bootstrap_ready" -ForegroundColor Green
Write-Host "  workspace      = $workspaceOmega"
Write-Host "  branch         = $Branch"
Write-Host "  backup_remote  = $BackupRepo"
Write-Host ""
Write-Host "Next step:  Set-Location $workspaceOmega ; <launch Cowork CLI>"
