# Download the custom vegetable dataset from Dropbox
# Usage: powershell -ExecutionPolicy Bypass -File download_custom_dataset.ps1 [output_dir]

param(
    [string]$OutputDir = "."
)

$OutputFile = Join-Path $OutputDir "custom_dataset.zip"
$Url = "https://www.dropbox.com/scl/fi/ni48b5elv4l2esqma3euh/custom_dataset.zip?rlkey=u8blw19ywsfpc7jhee6y9yh3z&st=p419tqzd&dl=1"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Downloading custom vegetable dataset..."

try {
    Invoke-WebRequest -Uri $Url -OutFile $OutputFile
    Write-Host "Download complete: $OutputFile"
} catch {
    Write-Host "Download failed: $_"
    exit 1
}