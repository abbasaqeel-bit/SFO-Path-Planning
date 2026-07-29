$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultResults = Join-Path `
    $projectRoot `
    "results\experiments\static_benchmark.zip"

$results = if ($args.Count -ge 1) { $args[0] } else { $defaultResults }
$output = if ($args.Count -ge 2) {
    $args[1]
} else {
    Join-Path $projectRoot "results\quality_figures"
}

Push-Location $projectRoot
try {
    python ".\experiments\generate_pareto_quality_figure.py" `
        --results $results `
        --output $output `
        --dpi 600
    if ($LASTEXITCODE -ne 0) {
        throw "Pareto quality-figure generation failed."
    }
} finally {
    Pop-Location
}

Write-Host "Figures written to: $output"
