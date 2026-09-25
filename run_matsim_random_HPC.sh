#!/bin/bash
#SBATCH --job-name="MATSIM_RL"                 
#SBATCH --nodes=1                      
#SBATCH --ntasks-per-node=1            
#SBATCH --cpus-per-task=4             
#SBATCH --mem=128G                      
#SBATCH --time=10:00:00
#SBATCH --output=/home/h2/alka383h/logs/matsim/Random_Mode_Choice/Decenteralize/single_agent/single_policy/output/matsim_out_%j.log
#SBATCH --error=/home/h2/alka383h/logs/matsim/Random_Mode_Choice/Decenteralize/single_agent/single_policy/error/matsim_err_%j.err

echo "### JOB START: $(date) ###"

# ==============================================================================
# 1. VARIABLES
# ==============================================================================
# Simulation Parameters
SIMULATION_TYPE="single_agent"
DATA_FOLDER="/home/h2/alka383h/researchDataFolder/HPC/Withinday_Mode_Choice/decenteralize_training"
WS_NAME="matsim"
SCENARIO_NAME="sioux_falls"
OBJECTIVE="single"

AGENT_LIST="10047_1"
MATSIM_ITERATION="700"
NUM_THREADS="4"
MEMORY="12g"

# Custom Project Parameters
RUN_SCRIPT="/app/src/projects/modechoice/random/python/core/run_random_mode_choice.py"
OBSERVER_CLASS="RandomModeChoiceObserver"
REPLANNER_CLASS="RandomModeChoiceReplanner"

# ==============================================================================
# 2. CREATE / VERIFY TU DRESDEN ZIH WORKSPACE
# ==============================================================================
# Check if workspace already exists
WS_PATH=$(ws_find $WS_NAME)

if [ $? -ne 0 ]; then
    echo "### Workspace not found. Requesting new workspace. Requesting new workspace on 'horse'... ###"
    ws_allocate -F horse $WS_NAME --duration 100 --reminder 7 --mailaddress ali_abbas.kapadia@tu-dresden.de
    WS_PATH=$(ws_find $WS_NAME)
    ws_register "/home/h2/alka383h/workspaces/"

    mkdir -p "$WS_PATH/RL_Mode_Choice/decenteralize_training/$SIMULATION_TYPE"
    
    echo "### Workspace created at: $WS_PATH ###"
    echo "### Copying data from Research Storage... ###"
    cp -r "$DATA_FOLDER"/* "$WS_PATH/RL_Mode_Choice/decenteralize_training/$SIMULATION_TYPE"
else
    echo "### Workspace found at: $WS_PATH ###"

    # Check if the folder exists and has content
    if [ ! -d "$WS_PATH/RL_Mode_Choice/decenteralize_training/$SIMULATION_TYPE" ] || [ -z "$(ls -A "$WS_PATH/RL_Mode_Choice/decenteralize_training/$SIMULATION_TYPE")" ]; then
        echo "### ERROR: Folder missing or empty in existing workspace! ###"
        mkdir -p "$WS_PATH/RL_Mode_Choice/decenteralize_training/$SIMULATION_TYPE"
        cp -r "$DATA_FOLDER"/* "$WS_PATH/RL_Mode_Choice/decenteralize_training/$SIMULATION_TYPE/"
    else
        echo "### RL_Mode_Choice folder verified with content. ###"
    fi
fi

# ==============================================================================
# 3. SET RUNTIME PATHS BASED ON WORKSPACE TARGET
# ==============================================================================
MATSIM_ROOT="$WS_PATH/RL_Mode_Choice/decenteralize_training/${SIMULATION_TYPE}"

INPUT_DIRECTORY="${MATSIM_ROOT}/scenarios/${SCENARIO_NAME}/input"
OUTPUT_DIRECTORY="${MATSIM_ROOT}/scenarios/${SCENARIO_NAME}/output/project/random_choice"
SIF_FILE="${MATSIM_ROOT}/matsim_random_choice_v1.sif"
DOCKER_IMAGE_PATH="${MATSIM_ROOT}/matsim_random_image.tar"
SHARED_DIRECTORY="${MATSIM_ROOT}/scenarios/${SCENARIO_NAME}/shared_storage"

# Create log and database directories inside the high-performance workspace storage
mkdir -p "$OUTPUT_DIRECTORY"
mkdir -p "$SHARED_DIRECTORY"

# Dynamically update the config.xml agent filter list
CONFIG_PATH="${INPUT_DIRECTORY}/config.xml"
if [ -f "$CONFIG_PATH" ]; then
    echo "### Updating agent filter list in config.xml to: ${AGENT_ID} ###"
    sed -i.bak -E 's/(<param name="agentFilterList" value=")[^"]*("\/>)/\1'"${AGENT_ID}"'\2/' "$CONFIG_PATH"
else
    echo "### Warning: config.xml not found at ${CONFIG_PATH}, skipping config update. ###"
fi

echo "### Workspace ready ###"

# ==============================================================================
# 5. Install dependencies
# ==============================================================================
cd "$MATSIM_ROOT" || exit 1

module purge
module load release/24.10
module load container/all

# ==============================================================================
# 6. BUILD THE IMAGE (Only if SIF doesn't exist)
# ==============================================================================
if [ ! -f "$SIF_FILE" ]; then
    echo "### Building SIF from Tarball... ###"
    
    mkdir -p "$WS_PATH/tmp"
    export SINGULARITY_TMPDIR="$WS_PATH/tmp"
    export APPTAINER_TMPDIR="$WS_PATH/tmp"

    singularity build "$SIF_FILE" docker-archive:"$DOCKER_IMAGE_PATH"
    
    # Optional: Clean up the temp files after build
    rm -rf "$WS_PATH/tmp"
else
    echo "### SIF image already exists. Skipping build. ###"
fi

# ==============================================================================
# 7. RUN MATSIM
# ==============================================================================
echo "### Launching MATSim in the container ###"

srun singularity run --containall --writable-tmpfs --cleanenv \
    --env OBJECTIVE="$OBJECTIVE" \
    --env MATSIM_OUTPUT_BASE="/app/scenarios/$SCENARIO_NAME/output" \
    --env MATSIM_ITERATION="$MATSIM_ITERATION" \
    --env MAX_TRAINING_ITERATION="$MAX_TRAINING_ITERATION" \
    --env NUM_THREADS="$NUM_THREADS" \
    --env JAVA_HEAP="$MEMORY" \
    --env PARAMS="$PARAMS" \
    --env SCENARIO="$SCENARIO_NAME" \
    --env REPLANNER_CLASS="$REPLANNER_CLASS" \
    --env OBSERVER_CLASS="$OBSERVER_CLASS" \
    --env RUN_SCRIPT="$RUN_SCRIPT" \
    --bind "$INPUT_DIRECTORY:/app/scenarios/$SCENARIO_NAME/input" \
    --bind "$OUTPUT_DIRECTORY:/app/scenarios/$SCENARIO_NAME/output" \
    --bind "$SHARED_DIRECTORY:/app/shared_storage" \
    "$SIF_FILE"

# ==============================================================================
# 8. STATUS
# ==============================================================================
if [ $? -eq 0 ]; then
    echo "### SUCCESS: Array Node Optimization Core Sequence Complete ###"
else
    echo "### FAILURE: Worker process exited with an error code. Check error logs. ###"
    exit 1
fi

echo "### JOB END: $(date) ###"fi