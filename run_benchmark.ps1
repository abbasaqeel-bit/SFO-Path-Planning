$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Create .venv and install the package."
}

Push-Location $projectRoot
try {
    & $python ".\experiments\run_benchmark.py" `
        --config "configs/benchmark.yaml"
    if ($LASTEXITCODE -ne 0) {
        throw "The benchmark failed."
    }
}
finally {
    Pop-Location
}
