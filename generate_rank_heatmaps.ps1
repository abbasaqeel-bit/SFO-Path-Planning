param(
    [Parameter(Mandatory = $true)]
    [string]$Results,

    [string]$Output = "results/rank_heatmaps",

    [int]$Dpi = 600
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

& $python "$projectRoot\experiments\generate_rank_heatmaps.py" `
    --results $Results `
    --output $Output `
    --dpi $Dpi

if ($LASTEXITCODE -ne 0) {
    throw "Rank-heatmap generation failed."
}
