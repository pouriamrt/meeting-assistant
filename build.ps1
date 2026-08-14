# Build a distributable folder. Run from the repo root:  .\build.ps1
#
# Output: dist\meeting-assistant\  and  dist\meeting-assistant.zip

$ErrorActionPreference = "Stop"

Write-Host "Cleaning previous build..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

Write-Host "Running PyInstaller..."
uv run pyinstaller assistant.spec --noconfirm --clean

$app = "dist\meeting-assistant"
if (-not (Test-Path "$app\meeting-assistant.exe")) {
    throw "Build produced no exe. Check the PyInstaller output above."
}

# Ship the scope note with the app. It is the part a recipient most needs and
# is least likely to go looking for.
Copy-Item "DISTRIBUTING.md" "$app\READ ME FIRST.txt"

$size = "{0:N0} MB" -f ((Get-ChildItem $app -Recurse | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "Folder built: $app  ($size)"

Write-Host "Zipping..."
Compress-Archive -Path $app -DestinationPath "dist\meeting-assistant.zip" -Force
$zip = "{0:N0} MB" -f ((Get-Item "dist\meeting-assistant.zip").Length / 1MB)
Write-Host "Done: dist\meeting-assistant.zip  ($zip)"
Write-Host ""
Write-Host "Smoke-test it before sending:  $app\meeting-assistant.exe"
Write-Host "It is unsigned, so SmartScreen will warn once on a new machine."
