import os
import csv
import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.storages import JournalStorage, JournalFileStorage
import numpy as np
from objectives import *

# ==========================================
# 1. DEFINE PATHS (Update these for your machine)
# ==========================================
# Path to your original database containing the hyperparameter values
LOG_DIR_PATH = r"C:\\ResearchWork\\Matsim_integration\\HPC\\RL_Mode_Choice\\decenteralize_training\\scenarios\\sioux-falls\\copied_output\\run_13June\\optimization_logs"
CSV_DIR_PATH = r"C:\\ResearchWork\\Matsim_integration\\HPC\\RL_Mode_Choice\\decenteralize_training\\scenarios\\sioux-falls\\copied_output\\run_13June\\results"

OLD_DB_PATH = os.path.join(LOG_DIR_PATH, "optuna_optimization.log")
OLD_STUDY_NAME = "matsim_mode_optimization"

NEW_DB_PATH = os.path.join(LOG_DIR_PATH , "optimization_migrated_trials.log")
NEW_STUDY_NAME = "matsim_mode_choice_optimization"


def calculate_fixed_metrics(delta_q_history, reward_history, window_size, threshold):
    
    total_iterations = len(reward_history)
    if total_iterations < window_size:
        window_size = total_iterations

    saturation_iteration, smoothed_rewards = policy_evolution_rate(reward_history, window_size, threshold)

    reward_at_convergence, delta_q_at_stabilization, reward_variance = settled_policy_metrics(saturation_iteration, total_iterations, delta_q_history, reward_history, window_size)

    return reward_at_convergence, delta_q_at_stabilization, saturation_iteration

def main():

    print(f"Loading old hyperparameters from: {OLD_DB_PATH}")

    try:

        file_storage = JournalFileStorage(OLD_DB_PATH)
        old_storage_backend = JournalStorage(file_storage)

        new_file_storage = JournalFileStorage(NEW_DB_PATH)
        new_storage_backend = JournalStorage(new_file_storage)

        old_study = optuna.load_study(study_name=OLD_STUDY_NAME, storage=old_storage_backend)

    except Exception as e:
        print(f"Error loading old database: {e}")
        return

    print(f"Creating fresh study: {NEW_STUDY_NAME}")

    new_study = optuna.create_study(
        study_name=NEW_STUDY_NAME,
        storage=new_storage_backend,
        directions=["maximize", "minimize", "minimize"],
        load_if_exists=True
    )

    # Recreate the distributions parameter map to match your search space bounds
    distributions = {
        "alpha": optuna.distributions.FloatDistribution(0.05, 0.40),
        "gamma": optuna.distributions.FloatDistribution(0.85, 0.99),
        "epsilon_decay": optuna.distributions.FloatDistribution(0.92, 0.98),
        "epsilon_minimum": optuna.distributions.FloatDistribution(0.001, 0.10),
        "discontinuity": optuna.distributions.FloatDistribution(1.0, 3.0),
        "retrieval_cost": optuna.distributions.FloatDistribution(1.0, 4.0),
    }

    migrated_count = 0

    # Loop through the old trials found in the SQLite database
    for old_trial in old_study.trials:
        if old_trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        trial_num = old_trial.number
        folder_name = f"trial_{trial_num}"
        csv_file_path = os.path.join(CSV_DIR_PATH, folder_name, "agent_tracking.csv")

        # Verify the downloaded local CSV file actually exists for this trial
        if not os.path.exists(csv_file_path):
            print(f"[Skipping] No CSV found for Trial #{trial_num} at {csv_file_path}")
            continue

        utilities = []
        deltas = []

        # Read the raw history data arrays directly from the CSV
        with open(csv_file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip CSV header

            for row in reader:
                if row:
                    utilities.append(float(row[3]))
                    deltas.append(float(row[5]))

        # Run the arrays through the corrected metrics equation
        avg_u, var_u, conv_i = calculate_fixed_metrics(deltas, utilities, 50, 0.002)

        recovered_trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            params=old_trial.params,
            distributions=distributions,
            values=[avg_u, var_u, conv_i]
        )

        new_study.add_trial(recovered_trial)

        # Explicit target verification debug block for Trial 346
        if trial_num == 346:
            print("\n" + "="*50)
            print(f"🔥 CRITICAL DEBUG: CALCULATED DATA FOR TRIAL #{trial_num}")
            print("="*50)
            print(f"Total Rows read from CSV File: {len(utilities)}")
            print(f"Value unpacked into avg_u (Obj 1):  {avg_u}")
            print(f"Value unpacked into var_u (Obj 2):  {var_u}")
            print(f"Value unpacked into conv_i (Obj 3): {conv_i}")
            print("="*50 + "\n")
        
        migrated_count += 1

    print(f"\nSuccessfully migrated {migrated_count} trials to the clean journal study.")

# ==========================================
# Main Script
# ==========================================
if __name__ == "__main__":
    main()