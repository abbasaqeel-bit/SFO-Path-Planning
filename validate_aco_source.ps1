$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python virtual environment not found. Create .venv and install the package first."
}

Push-Location $ProjectRoot
try {
    & $Python ".\verification\verify_haghrah_aco.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Haghrah ACO verification failed."
    }
}
finally {
    Pop-Location
}
