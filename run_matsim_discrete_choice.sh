#!/bin/bash

echo "### JOB START - Discrete Choice Model ###"

# ==============================================================================
# 1. VARIABLES
# ==============================================================================
# Simulation Parameters
SIMULATION_TYPE="single_agent"
SCENARIO_NAME="sioux_falls"
OBJECTIVE="single"
CONFIG_NAME="config.xml"
NUM_AGENTS_PER_ITERATION="1"

MATSIM_ITERATION="800"
NUM_THREADS="4"
MEMORY="12g"
SEED_NUMBER="999"

# Custom Project Parameters
RUN_SCRIPT="/app/src/projects/modechoice/discrete_choice/python/core/run_discrete_mode_choice.py"
OBSERVER_CLASS="DiscreteModeChoiceObserver"
REPLANNER_CLASS="DiscreteModeChoiceReplanner"
DOCKER_IMAGE_NAME="matsim-rl:development"

# Local directories based on current working directory ($PWD)
BASE_WORKSPACE="$PWD"
INPUT_DIRECTORY="${BASE_WORKSPACE}/scenarios/${SCENARIO_NAME}/input"
OUTPUT_DIRECTORY="${BASE_WORKSPACE}/scenarios/${SCENARIO_NAME}/output/project/discrete_choice"
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
# 4. RUN MATSIM
# ==============================================================================
echo "### Launching MATSim in the container ###"

docker run --rm \
    --env CONTAINER_APP_ROOT="/app" \
    --env CONFIG="$CONFIG_NAME" \
    --env OBJECTIVE="$OBJECTIVE" \
    --env AGENTS_PER_ITERATION="$NUM_AGENTS_PER_ITERATION" \
    --env MATSIM_OUTPUT_BASE="/app/scenarios/$SCENARIO_NAME/output/run_local" \
    --env MATSIM_ITERATION="$MATSIM_ITERATION" \
    --env NUM_THREADS="$NUM_THREADS" \
    --env JAVA_HEAP="$MEMORY" \
    --env SCENARIO="$SCENARIO_NAME" \
    --env REPLANNER_CLASS="$REPLANNER_CLASS" \
    --env OBSERVER_CLASS="$OBSERVER_CLASS" \
    --env RUN_SCRIPT="$RUN_SCRIPT" \
    --bind "$INPUT_DIRECTORY:/app/scenarios/$SCENARIO_NAME/input" \
    --bind "$OUTPUT_DIRECTORY:/app/scenarios/$SCENARIO_NAME/output" \
    --bind "$SHARED_DIRECTORY:/app/shared_storage" \
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