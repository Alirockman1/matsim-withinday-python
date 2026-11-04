#!/bin/bash
# Linux/macOS counterpart of run_test_berlin.ps1: runs the within-day RL mode choice on the
# mini Open Berlin scenario in the container. Docker runs natively here, so there is no WSL
# service start and no wslpath translation; everything else follows the PowerShell script.
#
# Usage (from anywhere):
#     ./src/test/java/org/matsim/rl_mode_choice/run_test_berlin.sh
#
# Every parameter below can be overridden from the environment, e.g.
#     MATSIM_ITERATION=20 UPDATE_JAR=false ./src/test/java/org/matsim/rl_mode_choice/run_test_berlin.sh

set -euo pipefail

# ==============================================================================
# 1. PARAMETERS (Change these as needed to test your setup)
# ==============================================================================
RUN_SCRIPT="${RUN_SCRIPT:-/app/src/projects/modechoice/rl/python/core/run_berlin_rl_pipeline.py}"
OBSERVER_CLASS="${OBSERVER_CLASS:-CustomRLObserver}"
REPLANNER_CLASS="${REPLANNER_CLASS:-CustomRLReplanner}"
SCENARIO_NAME="${SCENARIO_NAME:-berlin-v7.1.1}"
CONFIG_FILE="${CONFIG_FILE:-berlin-mini.config.xml}"
OBJECTIVE="${OBJECTIVE:-single}"

MATSIM_ITERATION="${MATSIM_ITERATION:-5}"
MAX_TRAINING_ITERATION="${MAX_TRAINING_ITERATION:-4}"
NUM_THREADS="${NUM_THREADS:-8}"
MEMORY="${MEMORY:-16g}"
PARAMS="${PARAMS:-alpha=0.367435 gamma=0.892101 epsilon_minimum=0.001044 epsilon_decay=0.966574 w_1=1.0 w_2=1.0}"

# Development Pipeline Toggles
NUM_AGENTS_PER_ITERATION="${NUM_AGENTS_PER_ITERATION:-1}"
UPDATE_JAR="${UPDATE_JAR:-true}"
REBUILT_DOCKER="${REBUILT_DOCKER:-true}"
IMAGE_TAG="${IMAGE_TAG:-matsim-project:development}"

# ==============================================================================
# 2. LOCATE THE REPOSITORY ROOT
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_WORKSPACE="$SCRIPT_DIR"
while [ "$BASE_WORKSPACE" != "/" ] && [ ! -f "$BASE_WORKSPACE/pom.xml" ]; do
    BASE_WORKSPACE="$(dirname "$BASE_WORKSPACE")"
done

if [ ! -f "$BASE_WORKSPACE/pom.xml" ]; then
    echo "### ERROR: no pom.xml found above $SCRIPT_DIR ###" >&2
    exit 1
fi

echo "### Using Base Workspace: $BASE_WORKSPACE ###"
cd "$BASE_WORKSPACE"

SCENARIO_INPUT_DIR="$BASE_WORKSPACE/scenarios/$SCENARIO_NAME/input"
SCENARIO_OUTPUT_DIR="$BASE_WORKSPACE/scenarios/$SCENARIO_NAME/output/project/rl_choice"
SHARED_STORAGE_DIR="$BASE_WORKSPACE/scenarios/$SCENARIO_NAME/shared_storage"

echo "========================================="
echo "Starting MATSim-RL Automated Local Docker Pipeline"
echo "Mode Configuration: $OBJECTIVE"
echo "Scenario:           $SCENARIO_NAME"
echo "Config:             $CONFIG_FILE"
echo "========================================="

mkdir -p "$SCENARIO_OUTPUT_DIR" "$SHARED_STORAGE_DIR"

# ==============================================================================
# 3. CLEAN OLD ARTIFACTS
# ==============================================================================
echo "### Cleaning old build artifacts... ###"
rm -f "$BASE_WORKSPACE"/target/*.jar

# Only wipe the output folder for a fresh single run; an optimization run keeps its tracking state.
if [ "$OBJECTIVE" = "single" ]; then
    echo "### Objective is 'single' -> Wiping output folder for fresh simulation cycle... ###"
    rm -rf "${SCENARIO_OUTPUT_DIR:?}"/* "${SCENARIO_OUTPUT_DIR:?}"/.[!.]* 2>/dev/null || true
else
    echo "### Objective is '$OBJECTIVE' -> Preserving directory content for tracking sync... ###"
fi

# ==============================================================================
# 4. COMPILE THE JAVA MAVEN PROJECT
# ==============================================================================
if [ "$UPDATE_JAR" = "true" ]; then
    echo "### Compiling Maven project... ###"
    ./mvnw clean install -DskipTests
fi

# ==============================================================================
# 5. BUILD THE DOCKER IMAGE
# ==============================================================================
if [ "$REBUILT_DOCKER" = "true" ]; then
    echo "### Removing old Docker image [$IMAGE_TAG]... ###"
    docker rmi -f "$IMAGE_TAG" 2>/dev/null || true

    echo "### Building new Docker image [$IMAGE_TAG] using Dockerfile.txt... ###"
    docker build -f "$BASE_WORKSPACE/Dockerfile.txt" -t "$IMAGE_TAG" "$BASE_WORKSPACE"
fi

# ==============================================================================
# 6. EXECUTE THE CONTAINER
# ==============================================================================
# -it only when a terminal is attached, so the script also works from CI or a pipe.
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS=(-it)
fi

echo "### Launching containerized simulation environment... ###"
docker run --rm "${TTY_FLAGS[@]}" \
    -e OBJECTIVE="$OBJECTIVE" \
    -e AGENTS_PER_ITERATION="$NUM_AGENTS_PER_ITERATION" \
    -e MATSIM_OUTPUT_BASE="/app/scenarios/${SCENARIO_NAME}/output" \
    -e MATSIM_ITERATION="$MATSIM_ITERATION" \
    -e MAX_TRAINING_ITERATION="$MAX_TRAINING_ITERATION" \
    -e NUM_THREADS="$NUM_THREADS" \
    -e JAVA_HEAP="$MEMORY" \
    -e PARAMS="$PARAMS" \
    -e SCENARIO="$SCENARIO_NAME" \
    -e CONFIG_FILE="$CONFIG_FILE" \
    -e REPLANNER_CLASS="$REPLANNER_CLASS" \
    -e OBSERVER_CLASS="$OBSERVER_CLASS" \
    -e RUN_SCRIPT="$RUN_SCRIPT" \
    -v "${SCENARIO_INPUT_DIR}:/app/scenarios/${SCENARIO_NAME}/input" \
    -v "${SCENARIO_OUTPUT_DIR}:/app/scenarios/${SCENARIO_NAME}/output" \
    -v "${SHARED_STORAGE_DIR}:/app/shared_storage" \
    "$IMAGE_TAG"

# ==============================================================================
# 7. HAND THE OUTPUT BACK TO THE CALLING USER
# ==============================================================================
# The container runs as root, so everything it wrote into the bind mounts is owned by root.
# A throwaway container does the chown, which needs no sudo on the host.
echo "### Returning ownership of the output to $(id -un)... ###"
docker run --rm \
    -v "${SCENARIO_OUTPUT_DIR}:/out" \
    -v "${SHARED_STORAGE_DIR}:/shared" \
    --entrypoint sh "$IMAGE_TAG" \
    -c "chown -R $(id -u):$(id -g) /out /shared"

echo "========================================="
echo "Pipeline execution steps finished."
echo "Output: $SCENARIO_OUTPUT_DIR"
echo "========================================="
