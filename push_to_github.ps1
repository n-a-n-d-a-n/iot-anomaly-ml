# push_to_github.ps1 - Push project to GitHub
# Usage: .\push_to_github.ps1

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/n-a-n-d-a-n/iot-anomaly-ml.git"
$Branch = "main"

Write-Host ""
Write-Host "=== Industrial IoT Anomaly Detection - GitHub Push ===" -ForegroundColor Cyan
Write-Host "Target: $RepoUrl" -ForegroundColor Cyan

# 1. Initialize git if needed
if (-not (Test-Path ".git")) {
    Write-Host "[1/6] Initialising git repository..." -ForegroundColor Yellow
    git init
    git checkout -b $Branch
} else {
    Write-Host "[1/6] Git repository already initialised." -ForegroundColor Green
    git checkout -B $Branch
}

# 2. Set remote
Write-Host "[2/6] Configuring remote origin..." -ForegroundColor Yellow
$remotes = git remote
if ($remotes -contains "origin") {
    git remote set-url origin $RepoUrl
    Write-Host "  Remote 'origin' updated." -ForegroundColor Green
} else {
    git remote add origin $RepoUrl
    Write-Host "  Remote 'origin' added." -ForegroundColor Green
}

# 3. Stage all files
Write-Host "[3/6] Staging all project files..." -ForegroundColor Yellow
git add .
Write-Host "  Files staged." -ForegroundColor Green

# 4. Show staged files
Write-Host "[4/6] Files to be committed:" -ForegroundColor Yellow
git status --short

# 5. Commit
Write-Host "[5/6] Creating commit..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$msg = "feat: complete Industrial IoT Anomaly Intelligence Platform ($timestamp)"
$changes = git status --porcelain
if ($changes) {
    git commit -m $msg
    Write-Host "  Committed: $msg" -ForegroundColor Green
} else {
    Write-Host "  Nothing new to commit - working tree clean." -ForegroundColor Green
}

# 6. Push
Write-Host "[6/6] Pushing to GitHub ($Branch)..." -ForegroundColor Yellow
git push -u origin $Branch --force

Write-Host ""
Write-Host "=== DONE ===================================================" -ForegroundColor Cyan
Write-Host "Live at: https://github.com/n-a-n-d-a-n/iot-anomaly-ml" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
