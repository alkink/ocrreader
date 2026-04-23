param(
    [string]$PythonVersion = "3.11",
    [string]$VenvDir = ".venv",
    [switch]$WithGlmocr,
    [switch]$SkipVlRuntime,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-PythonExecutable {
    param([string]$Version)

    $compactVersion = $Version.Replace(".", "")

    if (Test-CommandExists "py") {
        try {
            $resolved = & py -$Version -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                return $resolved.Trim()
            }
        } catch {
        }
    }

    $candidates = @(
        (Join-Path $env:LocalAppData "Programs\Python\Python$compactVersion\python.exe"),
        (Join-Path $env:ProgramFiles "Python$compactVersion\python.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Python$compactVersion\python.exe")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Install-Python {
    param([string]$Version)

    if (-not (Test-CommandExists "winget")) {
        throw "Python $Version is not installed and winget is not available. Install Python $Version manually, then rerun this script."
    }

    Write-Host "[setup] Installing Python $Version with winget..."
    & winget install -e --id "Python.Python.$Version" --scope user --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install Python $Version."
    }
}

$pythonExe = Get-PythonExecutable -Version $PythonVersion
if (-not $pythonExe) {
    Install-Python -Version $PythonVersion
    $pythonExe = Get-PythonExecutable -Version $PythonVersion
}

if (-not $pythonExe) {
    throw "Python $PythonVersion appears to be installed, but its executable could not be located. Open a new shell and rerun this script."
}

$venvPath = $VenvDir
if (-not [System.IO.Path]::IsPathRooted($venvPath)) {
    $venvPath = Join-Path $RepoRoot $venvPath
}
$venvPython = Join-Path $venvPath "Scripts\python.exe"

Write-Host "[setup] Using Python: $pythonExe"
Write-Host "[setup] Virtual environment: $venvPath"

if ($DryRun) {
    Write-Host "[dry-run] $pythonExe -m venv `"$venvPath`""
    Write-Host "[dry-run] `"$venvPython`" -m pip install --upgrade pip"
    $preview = @("scripts\bootstrap_runtime.py")
    if ($WithGlmocr) {
        $preview += "--with-glmocr"
    }
    if ($SkipVlRuntime) {
        $preview += "--skip-vl-runtime"
    }
    $preview += "--dry-run"
    Write-Host ("[dry-run] `"{0}`" {1}" -f $venvPython, ($preview -join " "))
    exit 0
}

& $pythonExe -m venv $venvPath
& $venvPython -m pip install --upgrade pip

$bootstrapArgs = @((Join-Path $RepoRoot "scripts\bootstrap_runtime.py"))
if ($WithGlmocr) {
    $bootstrapArgs += "--with-glmocr"
}
if ($SkipVlRuntime) {
    $bootstrapArgs += "--skip-vl-runtime"
}

Write-Host "[setup] Running repo bootstrap..."
& $venvPython @bootstrapArgs

Write-Host ""
Write-Host "Bootstrap completed."
Write-Host "Next commands:"
Write-Host "  $venvPath\Scripts\Activate.ps1"
Write-Host "  python -m ocrreader.cli --image `"testdata/WhatsApp Image 2026-03-03 at 18.31.01.jpeg`" --config `"config/ruhsat_schema_paddle_v29.yaml`" --output `"output/result.json`""
