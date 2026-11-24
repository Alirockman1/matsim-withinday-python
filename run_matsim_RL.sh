#!/bin/bash

echo "### JOB STARTED ###"

# ==============================================================================
# 1. VARIABLES
# ==============================================================================
# Simulation Parameters
SIMULATION_TYPE="global_policy"
SCENARIO_NAME="sioux_falls"
CONFIG_NAME="synthetic/config.xml"
OBJECTIVE="multi"
NUM_AGENTS_PER_ITERATION="1"

MATSIM_ITERATION="3001"
MAX_TRAINING_ITERATION="3000"
NUM_THREADS="4"
MEMORY="12g"
SEED_NUMBER="999"
PARAMS="alpha=0.367435 gamma=0.892101 epsilon_minimum=0.001044 epsilon_decay=0.966574 w_1=1.0 w_2=1.0 filter_zone_count=5 train_percentage=0.2"

# Custom Project Parameters
RUN_SCRIPT="/app/src/projects/modechoice/rl/python/core/run_rl_pipeline.py"
OBSERVER_CLASS="CustomRLObserver"
REPLANNER_CLASS="CustomRLReplanner"
DOCKER_IMAGE_NAME="matsim-rl:development"

# Local directories based on current working directory ($PWD)
BASE_WORKSPACE="$PWD"
INPUT_DIRECTORY="${BASE_WORKSPACE}/scenarios/${SCENARIO_NAME}/input"
OUTPUT_DIRECTORY="${BASE_WORKSPACE}/scenarios/${SCENARIO_NAME}/output/project/reinforcement_learning"
SHARED_DIRECTORY="${BASE_WORKSPACE}/scenarios/${SCENARIO_NAME}/shared_storage"

# Ensure local directories exist
mkdir -p "$OUTPUT_DIRECTORY"
mkdir -p "$SHARED_DIRECTORY"

# ==============================================================================
# 2. PULL SIMULATION JAR FROM TU DRESDEN DATASHARE (IF MISSING OR OUTDATED)
# ==============================================================================
DATASHARE_JAR_URL="https://datashare.tu-dresden.de/s/LKPw4fnWzPHMBxx"
LOCAL_JAR_PATH="./a7-rl-testbed-0.0.1-SNAPSHOT.jar"

echo "### Checking simulation.jar status... ###"

if [ ! -f "$LOCAL_JAR_PATH" ]; then
    echo "### Local jar not found. Downloading from Datashare... ###"
    curl -L -o "$LOCAL_JAR_PATH" "$DATASHARE_JAR_URL"
else
    echo "### Local jar exists. Checking for updates... ###"
    
    TEMP_JAR="./simulation_temp.jar"
    curl -L -o "$TEMP_JAR" "$DATASHARE_JAR_URL"
    
    if [ ! -f "$TEMP_JAR" ] || [ ! -s "$TEMP_JAR" ]; then
        echo "### Warning: Download failed or empty file received. Keeping existing jar. ###"
        rm -f "$TEMP_JAR"
    elif [ "$(stat -c%s "$TEMP_JAR")" != "$(stat -c%s "$LOCAL_JAR_PATH")" ]; then
        echo "### New version detected on Datashare! Updating local jar... ###"
        mv "$TEMP_JAR" "$LOCAL_JAR_PATH"
    else
        echo "### Local jar is already up to date. ###"
        rm -f "$TEMP_JAR"
    fi
fi

# ==============================================================================
# 3. BUILD DOCKER IMAGE LOCALLY
# ==============================================================================
echo "### Building Docker image locally from Dockerfile.txt... ###"
docker build -f Dockerfile.txt -t "$DOCKER_IMAGE_NAME" . || { echo "### Docker build failed ###"; exit 1; }

# ==============================================================================
# 4. RUN MATSim IN LOCAL DOCKER CONTAINER
# ==============================================================================
echo "### Launching MATSim in the local Docker container (seed ${SEED_NUMBER}) ###"

docker run --rm \
    --env CONTAINER_APP_ROOT="/app" \
    --env CONFIG="$CONFIG_NAME" \
    --env OBJECTIVE="$OBJECTIVE" \
    --env AGENTS_PER_ITERATION="$NUM_AGENTS_PER_ITERATION" \
    --env MATSIM_OUTPUT_BASE="/app/scenarios/${SCENARIO_NAME}/output/run_local" \
    --env MATSIM_ITERATION="$MATSIM_ITERATION" \
    --env MAX_TRAINING_ITERATION="$MAX_TRAINING_ITERATION" \
    --env NUM_THREADS="$NUM_THREADS" \
    --env JAVA_HEAP="$MEMORY" \
    --env PARAMS="$PARAMS" \
    --env SCENARIO="$SCENARIO_NAME" \
    --env REPLANNER_CLASS="$REPLANNER_CLASS" \
    --env OBSERVER_CLASS="$OBSERVER_CLASS" \
    --env RUN_SCRIPT="$RUN_SCRIPT" \
    --env SEED="$SEED_NUMBER" \
    -v "$INPUT_DIRECTORY:/app/scenarios/${SCENARIO_NAME}/input" \
    -v "$OUTPUT_DIRECTORY:/app/scenarios/${SCENARIO_NAME}/output" \
    -v "$SHARED_DIRECTORY:/app/shared_storage" \
    "$DOCKER_IMAGE_NAME"

DOCKER_EXIT=$?

if [ $DOCKER_EXIT -eq 0 ]; then
    echo "### SUCCESS: Local simulation finished successfully ###"
else
    echo "### FAILURE: Local container exited with code ${DOCKER_EXIT} ###"
fi

# ==============================================================================
# 5. STATUS
# ==============================================================================
echo "### LOCAL PIPELINE END: $(date) ###"
exit $DOCKER_EXIT