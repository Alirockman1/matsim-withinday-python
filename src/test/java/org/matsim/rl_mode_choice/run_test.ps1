# --- PARAMETERS (Change these as needed to test your setup) ---
$RUN_SCRIPT = "/app/src/projects/modechoice/rl/python/core/run_rl_pipeline.py"
$OBSERVER_CLASS = "CustomRLObserver"
$REPLANNER_CLASS = "CustomRLReplanner"
$SCENARIO_NAME = "sioux-falls"
$OBJECTIVE = "single" 

$MATSIM_ITERATION = "10"
$MAX_TRAINING_ITERATION = "9"
$NUM_THREADS = "4"
$MEMORY = "12g"
$PARAMS = "alpha=0.367435 gamma=0.892101 epsilon_minimum=0.001044 epsilon_decay=0.966574 w_1=2.59894 w_2=3.405791"
#$PARAMS = "alpha=0.367435 gamma=0.892101 epsilon_minimum=0.001044 epsilon_decay=0.966574 w_1=1.0 w_2=1.0"

# Development Pipeline Toggles
$AGENT_ID = "10047_1"
$UPDATE_JAR = $true
$REBUILT_DOCKER = $true

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

# Define your input/output directories dynamically
$SCENARIO_INPUT_DIR = "$BASE_WORKSPACE\scenarios\$SCENARIO_NAME\input"
$SCENARIO_OUTPUT_DIR = "$BASE_WORKSPACE\scenarios\$SCENARIO_NAME\output\reinforcement_learning"
$SHARED_STORAGE_DIR = "$BASE_WORKSPACE\scenarios\$SCENARIO_NAME\shared_storage"

# File pattern for the target jar file to delete
$JAR_FILE = "$BASE_WORKSPACE\target\*.jar"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting MATSim-RL Automated Local Docker Pipeline"
Write-Host "Mode Configuration: $OBJECTIVE"
Write-Host "Scenario:           $SCENARIO_NAME"
Write-Host "=========================================" -ForegroundColor Cyan

# Ensure runtime folders exist locally
if (!(Test-Path $SHARED_STORAGE_DIR)) { New-Item -ItemType Directory -Path $SHARED_STORAGE_DIR | Out-Null }
if (!(Test-Path $SCENARIO_OUTPUT_DIR)) { New-Item -ItemType Directory -Path $SCENARIO_OUTPUT_DIR | Out-Null }

# 1. Dynamically update the config.xml agent list using regex
$CONFIG_PATH = "$SCENARIO_INPUT_DIR\config.xml"
if (Test-Path $CONFIG_PATH) {
    Write-Host "Updating agent filter list in config.xml to: $AGENT_ID..." -ForegroundColor Yellow
    $content = Get-Content $CONFIG_PATH
    $content = $content -replace '(<param name="agentFilterList" value=")[^"]*("\/>)', "`$1${AGENT_ID}`$2"
    Set-Content $CONFIG_PATH $content
} else {
    Write-Host "Warning: config.xml not found at $CONFIG_PATH, skipping config update." -ForegroundColor Red
}

# 2. Move execution context to base workspace
Set-Location $BASE_WORKSPACE

# 3. Clean old build artifacts safely
Write-Host "Cleaning old build artifacts..." -ForegroundColor Yellow
Remove-Item -Path $JAR_FILE -Force -ErrorAction SilentlyContinue

# 🌟 Safety Check: Only clean output tracking files if doing a completely fresh single instance run
if ($OBJECTIVE -eq "single") {
    Write-Host "Objective is 'single' -> Wiping output folder for fresh simulation cycle..." -ForegroundColor Yellow
    if (Test-Path $SCENARIO_OUTPUT_DIR) {
        Get-ChildItem -Path $SCENARIO_OUTPUT_DIR -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Objective is 'optimization' -> Preserving directory content for tracking sync..." -ForegroundColor Magenta
}

# 4. Compile Java Maven Project
if ($UPDATE_JAR){
    Write-Host "Compiling Maven project..." -ForegroundColor Yellow
    & .\mvnw.cmd clean install -DskipTests
}

# 5. Convert Windows paths to linux path for wsl container mapping compatibility
$DOCKER_WORKSPACE = $BASE_WORKSPACE -replace '\\', '/' 
$DOCKER_INPUT = $SCENARIO_INPUT_DIR -replace '\\', '/'
$DOCKER_OUTPUT = $SCENARIO_OUTPUT_DIR -replace '\\', '/'
$DOCKER_SHARED = $SHARED_STORAGE_DIR -replace '\\', '/'

$WSL_WORKSPACE = (wsl wslpath -a -u "$DOCKER_WORKSPACE").Trim()
$WSL_INPUT     = (wsl wslpath -a -u "$DOCKER_INPUT").Trim()
$WSL_OUTPUT    = (wsl wslpath -a -u "$DOCKER_OUTPUT").Trim()
$WSL_SHARED    = (wsl wslpath -a -u "$DOCKER_SHARED").Trim()

# 6. Remove old Docker image & build fresh Docker image
if($REBUILT_DOCKER){
    # Create a dynamic tag name based on the current objective
    $IMAGE_TAG = "matsim-rl:$OBJECTIVE"

    Write-Host "Removing old Docker image [$IMAGE_TAG]..." -ForegroundColor Yellow
    wsl docker rmi $IMAGE_TAG -f 2>$null

    Write-Host "Building new Docker image [$IMAGE_TAG] using Dockerfile_compressed.txt..." -ForegroundColor Yellow
    # 🌟 -f points to your custom layer file, and -t applies the dynamic tag
    wsl docker build -f "$WSL_WORKSPACE/Dockerfile_compressed.txt" -t $IMAGE_TAG $WSL_WORKSPACE
    #docker build -f .\Dockerfile -t $IMAGE_TAG .
}

# 6. Convert Windows paths to forward slashes for Docker container mapping compatibility


# 7. Execute Docker Run Container with Complete Environment Matrix
Write-Host "Launching containerized simulation environment..." -ForegroundColor Green
wsl docker run --rm -it `
    -e OBJECTIVE="$OBJECTIVE" `
    -e MATSIM_OUTPUT_BASE="/app/scenarios/${SCENARIO_NAME}/output" `
    -e MATSIM_ITERATION="$MATSIM_ITERATION" `
    -e MAX_TRAINING_ITERATION="$MAX_TRAINING_ITERATION" `
    -e NUM_THREADS="$NUM_THREADS" `
    -e JAVA_HEAP="$MEMORY" `
    -e PARAMS="$PARAMS" `
    -e SCENARIO="$SCENARIO_NAME" `
    -e REPLANNER_CLASS="$REPLANNER_CLASS" `
    -e OBSERVER_CLASS="$OBSERVER_CLASS" `
    -e RUN_SCRIPT="$RUN_SCRIPT" `
    -v "${WSL_INPUT}:/app/scenarios/${SCENARIO_NAME}/input" `
    -v "${WSL_OUTPUT}:/app/scenarios/${SCENARIO_NAME}/output" `
    -v "${WSL_SHARED}:/app/shared_storage" `
    $IMAGE_TAG

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Pipeline execution steps finished."
Write-Host "=========================================" -ForegroundColor Cyan