# Download vegetable dataset from Kaggle
# Usage: powershell -ExecutionPolicy Bypass -File download_dataset.ps1 [output_dir]

param(
    [string]$OutputDir = "."
)

$OutputFile = Join-Path $OutputDir "dataset.zip"
$Url = "https://www.kaggle.com/api/v1/datasets/download/ayyuce/vegetables"

Write-Host "Downloading vegetable dataset from Kaggle..."

try {
    Invoke-WebRequest -Uri $Url -OutFile $OutputFile -UseBasicParsing
    Write-Host "Download complete: $OutputFile"
} catch {
    Write-Host "Download failed: $_"
    exit 1
}
