$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$matlabDirectory = Join-Path `
    $projectRoot `
    "third_party\aco_ga_paper_reproduction\matlab"
$outputDirectory = Join-Path `
    $projectRoot `
    "results\verification\aco_ga_paper_selftest"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$matlabCommand = Get-Command matlab.exe -ErrorAction SilentlyContinue
if ($matlabCommand) {
    $matlabPath = $matlabCommand.Source
}
else {
    $candidates = @(
        "E:\Program Files\MATLAB\R2025b\bin\matlab.exe",
        "C:\Program Files\MATLAB\R2025b\bin\matlab.exe",
        "E:\Program Files\MATLAB\R2025a\bin\matlab.exe",
        "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
    )
    $matlabPath = $candidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if (-not $matlabPath) {
    throw "MATLAB R2025a/R2025b was not found. Add MATLAB\bin to PATH."
}

$escapedMatlabDirectory = $matlabDirectory.Replace("'", "''")
$escapedOutputDirectory = $outputDirectory.Replace("'", "''")
$expression = (
    "addpath('$escapedMatlabDirectory'); " +
    "run_paper_reproduction_selftest('$escapedOutputDirectory');"
)
& $matlabPath -batch $expression
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB self-test failed with exit code $LASTEXITCODE."
}

$report = Join-Path $outputDirectory "paper_reproduction_selftest.txt"
if (-not (Test-Path -LiteralPath $report)) {
    throw "MATLAB completed without producing the expected report."
}
Get-Content -LiteralPath $report
