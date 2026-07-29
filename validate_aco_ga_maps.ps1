$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$matlab = Get-Command matlab.exe -ErrorAction SilentlyContinue
if (-not $matlab) {
    $candidates = @(
        "E:\Program Files\MATLAB\R2025b\bin\matlab.exe",
        "C:\Program Files\MATLAB\R2025b\bin\matlab.exe",
        "E:\Program Files\MATLAB\R2025a\bin\matlab.exe",
        "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
    )
    $matlabPath = $candidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if (-not $matlabPath) {
        throw "MATLAB R2025a/R2025b was not found. Add MATLAB\bin to PATH."
    }
    $env:Path = "$(Split-Path -Parent $matlabPath);$env:Path"
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python was not found. Create .venv and install the package first."
    }
    $python = $pythonCommand.Source
}

& $python "experiments\run_benchmark.py" `
    --config "configs/aco_ga_validation.yaml"

if ($LASTEXITCODE -ne 0) {
    throw "ACO-GA project-map validation failed with exit code $LASTEXITCODE."
}

$summary = Join-Path $projectRoot `
    "results\verification\aco_ga_maps\summary.csv"
Write-Host ""
Write-Host "Validation finished."
Write-Host "Summary: $summary"
