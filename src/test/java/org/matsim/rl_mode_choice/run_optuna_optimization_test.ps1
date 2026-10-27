# ==============================================================================
# Local (Windows / PowerShell + WSL Docker) version of the Optuna job script.
#
#   - 3 Optuna workers run in parallel, each in its own container via WSL Docker.
#   - Creates a unique timestamped run folder + individual worker/trial subfolders.
#   - Each worker's console output goes to shared_storage\worker_<n>.log.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. VARIABLES
# ------------------------------------------------------------------------------
$ProjectRoot           = $PSScriptRoot            # folder that contains scenarios\ and the Dockerfile
$ScenarioName          = "sioux-falls"
$Objective             = "single_agent"           # triggers the optimization flow
$NumWorkers            = 3                        # parallel Optuna workers (= Slurm array size)
$NumAgentsPerIteration = "1"
$AgentId               = "10047_1"
$SamplingPercentage    = "0.01"

$MatsimIteration       = "30"                     # MATSim iterations per run
$MaxTrainingIteration  = "25"                     # training cut-off, must be below $MatsimIteration
$NumThreads            = "2"
$Memory                = "6g"

$RunScript       = "/app/src/projects/modechoice/rl/python/optimization/run_rl_optimization.py"
$ObserverClass   = "CustomRLObserver"
$ReplannerClass  = "CustomRLReplanner"
$RunFlag         = "tuning_test_v1"               # same flag = same Optuna study
$StaggerSeconds  = 5                              # delay between worker starts

# Base Windows workspace directory 
$CurrentDir = $PSScriptRoot
while ($CurrentDir -and !(Test-Path (Join-Path $CurrentDir "pom.xml"))) {
    $Parent = Split-Path -Parent $CurrentDir
    if ($Parent -eq $CurrentDir) { break }
    $CurrentDir = $Parent
}

$BASE_WORKSPACE = $CurrentDir
Write-Host "Using Base Workspace: $BASE_WORKSPACE" -ForegroundColor Cyan

# Ensure Docker service is running inside WSL silently as root
Write-Host "Ensuring Docker daemon is active in WSL..." -ForegroundColor Yellow
wsl -u root service docker start | Out-Null

$ImageName = "matsim-rl:development"
$ImageTar  = Join-Path $BASE_WORKSPACE "matsim_rl_image.tar"
$UseNamedVolumeForShared = $false

# ------------------------------------------------------------------------------
# 2. PATHS & UNIQUE RUN DIRECTORY GENERATION
# ------------------------------------------------------------------------------
$ScenarioDirectory = Join-Path $BASE_WORKSPACE "scenarios\$ScenarioName"
$InputDirectory    = Join-Path $ScenarioDirectory "input"
$SharedDirectory   = Join-Path $ScenarioDirectory "shared_storage"

# Generate a unique timestamped run ID for this overall execution session
$RunId             = Get-Date -Format "yyyyMMdd_HHmmss"
$BaseOutputDir     = Join-Path $ScenarioDirectory "output\project\reinforcement_learning\optimization"
$ActiveOutputDir   = Join-Path $BaseOutputDir "run_$RunId"


# Create shared storage locally on the host
if (!(Test-Path $SharedDirectory)) {
    New-Item -ItemType Directory -Force -Path $SharedDirectory | Out-Null
}
Write-Host "Created base run output directory root: $ActiveOutputDir" -ForegroundColor Green

if (-not (Test-Path $InputDirectory)) {
    Write-Error "Input directory not found: $InputDirectory"
    exit 1
}

$ScenarioDirectory = Join-Path $BASE_WORKSPACE "scenarios\$ScenarioName"
$InputDirectory    = Join-Path $ScenarioDirectory "input"
$SharedDirectory   = Join-Path $ScenarioDirectory "shared_storage"

# Create shared storage directory locally on Windows host
if (!(Test-Path $SharedDirectory)) {
    New-Item -ItemType Directory -Force -Path $SharedDirectory | Out-Null
}

if (-not (Test-Path $InputDirectory)) {
    Write-Error "Input directory not found: $InputDirectory"
    exit 1
}

# Clean backslashes for WSL compatibility
$DockerShared = $SharedDirectory -replace '\\', '/'
$DockerInput  = $InputDirectory -replace '\\', '/'

Write-Host "Converting Windows paths to WSL paths..." -ForegroundColor DarkGray
$WSL_SHARED   = (wsl wslpath -a -u "$DockerShared").Trim()
$WSL_INPUT    = (wsl wslpath -a -u "$DockerInput").Trim()

# ------------------------------------------------------------------------------
# 3. UPDATE config.xml (agent filter list and sampling percentage)
# ------------------------------------------------------------------------------
function Set-ConfigParam {
    param([string]$Text, [string]$Name, [string]$Value)
    $pattern = '(?m)^(\s*<param name="' + $Name + '" value=")[^"]*("\s*/>)'
    if ($Text -notmatch $pattern) {
        Write-Warning "config.xml: parameter '$Name' not found, nothing changed for it."
        return $Text
    }
    return $Text -replace $pattern, ('${1}' + $Value + '${2}')
}

$ConfigPath = Join-Path $InputDirectory "config.xml"
if (Test-Path $ConfigPath) {
    if (-not (Test-Path "$ConfigPath.bak")) {
        Copy-Item $ConfigPath "$ConfigPath.bak"
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $configText = [System.IO.File]::ReadAllText($ConfigPath)
    Write-Host "### Updating trainAgentIDList to: $AgentId ###"
    $configText = Set-ConfigParam -Text $configText -Name "trainAgentIDList" -Value $AgentId
    Write-Host "### Updating samplingPercentage to: $SamplingPercentage ###"
    $configText = Set-ConfigParam -Text $configText -Name "samplingPercentage" -Value $SamplingPercentage
    [System.IO.File]::WriteAllText($ConfigPath, $configText, $utf8NoBom)
} else {
    Write-Warning "config.xml not found at $ConfigPath, skipping config update."
}

# ------------------------------------------------------------------------------
# 4. WSL DOCKER IMAGE CHECK
# ------------------------------------------------------------------------------
wsl docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "WSL Docker daemon is not running. Please check your WSL Docker service."
    exit 1
}

wsl docker image inspect $ImageName 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $ImageTar) {
        Write-Host "### Loading image from $ImageTar ... ###"
        $loadOutput = wsl docker load -i $ImageTar | Out-String
        Write-Host $loadOutput
        if ($loadOutput -match 'Loaded image: (\S+)') {
            wsl docker tag $Matches[1] $ImageName
        }
    } else {
        $DockerFile = @("Dockerfile", "Dockerfile.txt") |
            ForEach-Object { Join-Path $ProjectRoot $_ } |
            Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $DockerFile) {
            Write-Error "No image '$ImageName', no tar file and no Dockerfile found in $ProjectRoot."
            exit 1
        }
        Write-Host "### Building image from $DockerFile ... ###"
        $DockerWorkspace = $ProjectRoot -replace '\\', '/'
        $WSL_WORKSPACE = (wsl wslpath -a -u "$DockerWorkspace").Trim()
        wsl docker build -f "$WSL_WORKSPACE/$(Split-Path $DockerFile -Leaf)" -t $ImageName $WSL_WORKSPACE
        if ($LASTEXITCODE -ne 0) { Write-Error "Image build failed."; exit 1 }
    }
    wsl docker image inspect $ImageName 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Error "Image '$ImageName' is still missing after load/build."; exit 1 }
} else {
    Write-Host "### Image $ImageName already exists. Skipping build. ###"
}

# ------------------------------------------------------------------------------
# 5. RUN THE OPTUNA WORKERS (Isolated trial subdirectories per worker)
# ------------------------------------------------------------------------------
function Get-DockerArguments {
    param([int]$WorkerId, [string]$WorkerOutputDir, [string]$WSL_WorkerOutput)
    return @(
        'run', '--rm', '--name', "matsim-optuna-$WorkerId",
        '--env', "OBJECTIVE=$Objective",
        '--env', "AGENTS_PER_ITERATION=$NumAgentsPerIteration",
        '--env', "MATSIM_ITERATION=$MatsimIteration",
        '--env', "MAX_TRAINING_ITERATION=$MaxTrainingIteration",
        '--env', "NUM_THREADS=$NumThreads",
        '--env', "JAVA_HEAP=$Memory",
        '--env', "SCENARIO=$ScenarioName",
        '--env', "REPLANNER_CLASS=$ReplannerClass",
        '--env', "OBSERVER_CLASS=$ObserverClass",
        '--env', "RUN_SCRIPT=$RunScript",
        '--env', "SLURM_ARRAY_TASK_ID=$WorkerId",
        '--env', "RUN_FLAG=$RunFlag",
        # Pass the isolated container path target for this specific worker trial
        '-v', "${WSL_INPUT}:/app/scenarios/$ScenarioName/input",
        '-v', "${WSL_WorkerOutput}:/app/scenarios/$ScenarioName/output",
        '-v', "${WSL_SHARED}:/app/scenarios/$ScenarioName/shared_storage",
        $ImageName
    )
}

Write-Host "### Launching $NumWorkers Optuna workers (logs in $SharedDirectory) ###"
$jobs = @()
try {
    foreach ($workerId in 1..$NumWorkers) {
        # Create an individual trial subfolder for each worker on the host
        $WorkerOutputDir   = Join-Path $ActiveOutputDir "trial_$workerId"
        New-Item -ItemType Directory -Force -Path $WorkerOutputDir | Out-Null
        
        $DockerWorkerOutput = $WorkerOutputDir -replace '\\', '/'
        $WSL_WorkerOutput   = (wsl wslpath -a -u "$DockerWorkerOutput").Trim()

        $dockerArgs = Get-DockerArguments -WorkerId $workerId -WorkerOutputDir $WorkerOutputDir -WSL_WorkerOutput $WSL_WorkerOutput

        # Spawn background job without file redirection
        $jobs += Start-Job -Name "optuna_worker_$workerId" -ArgumentList (, $dockerArgs) -ScriptBlock {
            param($dockerArgs)
            & wsl docker @dockerArgs
            return $LASTEXITCODE
        }
        Write-Host "  -> Worker $workerId started -> trial folder: trial_$workerId"

        if ($workerId -lt $NumWorkers) { 
            Write-Host "     Waiting $StaggerSeconds seconds before starting next worker..." -ForegroundColor DarkGray
            Start-Sleep -Seconds $StaggerSeconds 
        }
    }

    Write-Host "### Waiting for all local Optuna workers to complete (Ctrl+C stops them) ###" -ForegroundColor Yellow
    Wait-Job -Job $jobs | Out-Null

    $failed = 0
    foreach ($job in $jobs) {
        $exitCode = [int](@(Receive-Job -Job $job)[-1])
        if ($exitCode -eq 0) {
            Write-Host "### SUCCESS: $($job.Name) complete ###" -ForegroundColor Green
        } else {
            Write-Host "### FAILURE: $($job.Name) exited with code $exitCode ###" -ForegroundColor Red
            $failed++
        }
    }
}
finally {
    foreach ($workerId in 1..$NumWorkers) {
        wsl docker stop "matsim-optuna-$workerId" 2>$null | Out-Null
    }
    if ($jobs.Count -gt 0) { Remove-Job -Job $jobs -Force -ErrorAction SilentlyContinue }
}

Write-Host "### JOB END: $(Get-Date) ###" -ForegroundColor Cyan
if ($failed -gt 0) { exit 1 }