# Environment for the MATSim Within-Day Python Bridge (Windows PowerShell).
#
# Usage (in every new PowerShell window):
#     . .\env.ps1
# (note the leading dot and space: the file must be dot-sourced, not run,
#  otherwise the variables are lost when the script ends)
#
# Edit the "USER SETTINGS" block below, then leave the rest alone.
#
# If PowerShell blocks the script, run this once:
#     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# ===========================================================================
# USER SETTINGS - edit these
# ===========================================================================

# Path to your JDK 25. Leave empty to auto-detect from `javac` on your PATH.
# Example: "C:\Program Files\Eclipse Adoptium\jdk-25.0.0.0-hotspot"
$JavaHomeOverride = ""

# Scenario folder name under scenarios\   (run `dir scenarios` to see them)
$Scenario = "<scenario>"

# Python bridge service Java should start:
#   <project_name>.python.networking.bridge_service.<Service>
# (project_name = a folder under src\projects)
$ServiceClassValue = "<project_name>.python.networking.bridge_service.<Service>"

# ===========================================================================
# DO NOT EDIT BELOW (unless you know what you are doing)
# ===========================================================================

# Repo root = folder containing this file
$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }

# --- Python virtual environment -------------------------------------------
$Activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $Activate) {
    . $Activate
} else {
    Write-Warning "$RepoRoot\.venv not found. Create it first (see README, section 'Install Python dependencies')."
}

# --- Java -------------------------------------------------------------------
if ($JavaHomeOverride) {
    $env:JAVA_HOME = $JavaHomeOverride
} elseif (-not $env:JAVA_HOME) {
    $javac = Get-Command javac -ErrorAction SilentlyContinue
    if ($javac) {
        # ...\bin\javac.exe -> JDK root
        $env:JAVA_HOME = Split-Path (Split-Path $javac.Source -Parent) -Parent
    }
}

if ($env:JAVA_HOME) {
    $env:PATH = (Join-Path $env:JAVA_HOME "bin") + [IO.Path]::PathSeparator + $env:PATH
} else {
    Write-Warning "JAVA_HOME is not set and javac was not found. Install a JDK 25 or set `$JavaHomeOverride."
}

# --- Project variables --------------------------------------------------------
$sep = [IO.Path]::PathSeparator
$env:PYTHONPATH = (Join-Path $RepoRoot "src\main\python") + $sep + (Join-Path $RepoRoot "src\projects")
$env:MATSIM_OUTPUT_BASE = (Join-Path $RepoRoot "scenarios\$Scenario\output") + "\"
$env:SERVICE_CLASS = $ServiceClassValue

# --- Sanity checks ------------------------------------------------------------
if (("$Scenario$ServiceClassValue") -match "<") {
    Write-Warning "Placeholders like <scenario> are still in env.ps1 - edit the USER SETTINGS block."
}

Write-Host "Environment ready:"
Write-Host "  JAVA_HOME          = $env:JAVA_HOME"
Write-Host "  MATSIM_OUTPUT_BASE = $env:MATSIM_OUTPUT_BASE"
Write-Host "  SERVICE_CLASS      = $env:SERVICE_CLASS"