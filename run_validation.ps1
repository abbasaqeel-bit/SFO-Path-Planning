$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Create .venv and install .[dev] first."
}

$matlabCandidates = @(
    (Get-Command matlab.exe -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    "E:\Program Files\MATLAB\R2025b\bin\matlab.exe",
    "C:\Program Files\MATLAB\R2025b\bin\matlab.exe",
    "E:\Program Files\MATLAB\R2025a\bin\matlab.exe",
    "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$matlabPath = $matlabCandidates | Select-Object -First 1
if (-not $matlabPath) {
    throw "MATLAB R2025a/R2025b was not found."
}

Push-Location $projectRoot
try {
    $env:SFO_TEST_MATLAB_EXECUTABLE = $matlabPath

    & $python -m pytest
    if ($LASTEXITCODE -ne 0) {
        throw "The Python/MATLAB test suite failed."
    }

    powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File ".\validate_aco_source.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "The ACO native verification failed."
    }

    powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File ".\validate_aco_ga_reproduction.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "The ACO-GA paper self-test failed."
    }

    powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File ".\validate_aco_ga_maps.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "The ACO-GA project-map validation failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Validation completed successfully."
