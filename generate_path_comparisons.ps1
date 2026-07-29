$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultResults = Join-Path `
    $projectRoot `
    "results\experiments\static_benchmark.zip"

$results = if ($args.Count -ge 1) { $args[0] } else { $defaultResults }
$output = if ($args.Count -ge 2) {
    $args[1]
} else {
    Join-Path $projectRoot "results\path_comparisons"
}

Push-Location $projectRoot
try {
    python ".\experiments\generate_path_comparison.py" `
        --results $results `
        --maps ".\configs\maps.yaml" `
        --output $output `
        --dpi 600
    if ($LASTEXITCODE -ne 0) {
        throw "Path-comparison generation failed."
    }
} finally {
    Pop-Location
}

Write-Host "Figures written to: $output"
