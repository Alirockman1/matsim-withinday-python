import csv
import os
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.storages.journal import JournalFileBackend

from withinday.core.pymatsim import run_simulation

class HyperParameterOptimizer:

    MATSIM_JAR = "/app/simulation.jar"

    def __init__(self, study_name, scenario, objective, run_flag):
        """
        Initializes the distributed worker tuning node.
        """
        self._study_name = study_name
        self._scenario = scenario
        self._objective_name = objective
        self._matsim_config_path = f"/app/scenarios/{scenario}/input/config.xml"
        self._shared_folder_directory = f"/app/scenarios/{scenario}/shared_storage/{objective}"
        self._run_flag = run_flag

    def initialize_shared_database(self):
        """
        Validates existence of the sqlite schema file on shared storage, initializing it if absent.
        """
        os.makedirs(self._shared_folder_directory, exist_ok=True)
    
        # Connect to Optuna Journal Storage on shared workspace
        storage_file = os.path.join(self._shared_folder_directory, f"optimization_{self._run_flag}.log")
        print(f"Temp Log: {storage_file}")
        file_backend = JournalFileBackend(storage_file)
        storage_engine = JournalStorage(file_backend)

        array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        target_trial_number = array_task_id - 1
            
        # Automatically intercepts and sets up the schema tables safely if file is empty
        optuna.create_study(
            study_name=self._study_name,
            storage=storage_engine,
            directions=[
                "maximize",
                "minimize",
                "minimize"
            ],
            load_if_exists=True
        )

        print("Study initialized successfully.")
        
        return storage_engine, target_trial_number

    def _calculate_multi_objective_metrics(self, delta_q_history, reward_history, window_size=20, threshold=0.1):
        """
        Calculates advanced multi-objective metrics using a rolling window approach.
        
        Returns:
            tuple: (final_moving_avg_utility, final_moving_variance, convergence_iter)
        """
        total_iterations = len(reward_history)
        if total_iterations < window_size:
            window_size = total_iterations
        
        # 2. Objective 1: Maximize the final moving average value
        final_window = reward_history[-window_size:]
        reward_moving_average = sum(final_window) / window_size if window_size > 0 else 0.0
        
        # 3. Objective 2: Minimize the variance within this final rolling window
        reward_moving_variance = sum((x - reward_moving_average) ** 2 for x in final_window) / window_size if window_size > 0 else 0.0

        # 4. Objective 3: Minimize Q-convergence iteration
        convergence_iteration = None
        value_stable_for_iteration = 20
        
        for i in range(len(delta_q_history)):
            window_slice = delta_q_history[i : i + value_stable_for_iteration]
            if len(window_slice) < value_stable_for_iteration:
                break
            if all(d <= threshold for d in window_slice):
                convergence_iteration = i
                break

        if convergence_iteration is None:
            convergence_iteration = total_iterations * 1.5

        return reward_moving_average, reward_moving_variance, convergence_iteration

    def run_optuna(self, storage_engine, trial_number):
        """
        Connects to the persistent global storage study registry, 
        asks for a single trial, executes the simulation, and tells Optuna the results.
        Designed to be driven safely by a Slurm Array Job.
        """
        # 1. Load the study
        study = optuna.load_study(
            study_name=self._study_name, 
            storage=storage_engine
        )
        
        # 3. Ask Optuna for the next trial parameters dynamically
        trial = study.ask()
        
        try:
            # Extract hyperparameters chosen by Optuna for this trial
            alpha = trial.suggest_float("alpha", 0.01, 0.9)
            gamma = trial.suggest_float("gamma", 0.5, 0.99)
            epsilon_decay = trial.suggest_float("epsilon_decay", 0.8, 0.99)
            min_epsilon = trial.suggest_float("epsilon_minimum", 0.0001, 0.01)
            w_1 = trial.suggest_float("w_1", 0.1, 5.0)
            w_2 = trial.suggest_float("w_2", 0.1, 5.0)

            # Define trial-specific output path
            print(trial.number)
            trial_output_directory = os.path.join(self._shared_folder_directory, f"trial_{trial_number}")
            os.makedirs(trial_output_directory, exist_ok=True)

            # Ensure MATSim maps its output base to this trial folder
            os.environ["MATSIM_OUTPUT_BASE"] = trial_output_directory

            # Package configuration properties
            rl_extra_configs = {
                "agentModeChoice.trainingCutoffIteration": str(os.environ.get("MAX_TRAINING_ITERATION", "300")),
                "agentModeChoice.alpha": str(alpha),
                "agentModeChoice.gamma": str(gamma),
                "agentModeChoice.epsilonDecay": str(epsilon_decay),
                "agentModeChoice.epsilonMinimum": str(min_epsilon),
                "agentModeChoice.discontinuityPenalty": str(w_1),
                "agentModeChoice.retrievalCostPenalty": str(w_2)
            }

            # Execute the MATSim simulation run script
            print(f"Starting Trial {trial.number} with parameters: {rl_extra_configs}")
            run_simulation(
                run_script="modechoice.rl.java.core.RunRLModeChoice", 
                extra_config=rl_extra_configs
            )

            # Read metric columns from agent_tracking.csv after completion
            csv_path = os.path.join(trial_output_directory, "agent_tracking.csv")
            
            utilities = []
            deltas = []

            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip CSV header

                for row in reader:
                    if row:
                        utilities.append(float(row[3]))
                        deltas.append(float(row[5]))

            # Calculate composite objective metrics
            average_utility, utility_variance, convergence_iteration = self._calculate_multi_objective_metrics(deltas, utilities)    
            
            print(f"Successfully completed Trial {trial.number} with scores: {[average_utility, utility_variance, convergence_iteration]}")

            # 4. Report final scores back to Optuna storage using study.tell()
            study.tell(trial, [average_utility, utility_variance, convergence_iteration])

        except Exception as e:
            print(f"Error executing or parsing tracking file for trial {trial.number}: {e}")
            # Safely mark the trial as failed in Optuna storage
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise e
