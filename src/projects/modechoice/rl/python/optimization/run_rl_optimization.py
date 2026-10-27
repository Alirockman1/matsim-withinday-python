import os
import sys
import pandas as pd

from modechoice.rl.python.optimization.optimizer import HyperParameterOptimizer

def main():
    # Read the Slurm Array Task ID
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if not task_id:
        print("Error: SLURM_ARRAY_TASK_ID not found. Run this via a Slurm Array job.")
        sys.exit(1)

    os.environ["SERVICE_CLASS"] = "modechoice.rl.python.networking.bridge_service.ReinforcementLearningBridgeService"

    # define the shared folder to write the logs in
    scenario = os.environ.get("SCENARIO", "sioux-falls")
    objective = os.environ.get("OBJECTIVE", "single")
    
    ## set run-flag as run_YYYY_MM_DD
    run_flag = os.environ.get("RUN_FLAG")

    optimizer = HyperParameterOptimizer("single_agent_policy", scenario, objective, run_flag)

    storage_engine, trial_number = optimizer.initialize_shared_database()
    optimizer.run_optuna(storage_engine, trial_number) 

if __name__ == "__main__":
    main()