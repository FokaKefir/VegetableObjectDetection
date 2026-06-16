# Download the trained vegetable detector model from Dropbox
# Usage: powershell -ExecutionPolicy Bypass -File download_best_model.ps1 [output_dir]

param(
    [string]$OutputDir = "."
)

$OutputFile = Join-Path $OutputDir "best_model.pth"
$Url = "https://www.dropbox.com/scl/fi/xeqdf23h0h7oetw426vrw/best_model.pth?rlkey=o7h2bfsew8wslv3bfl2dams55&st=hqnifqah&dl=1"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Downloading best model checkpoint..."

try {
    Invoke-WebRequest -Uri $Url -OutFile $OutputFile
    Write-Host "Download complete: $OutputFile"
} catch {
    Write-Host "Download failed: $_"
    exit 1
}