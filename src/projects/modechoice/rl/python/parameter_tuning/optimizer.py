import csv
import os
import optuna
from  withinday.core.nodes import WorkerNode
import project.rl.parameter_tuning.initialize_db as optimization_tuner

class HyperParameterOptimizer:

    MATSIM_JAR = "/app/simulation.jar"
    DB_FILE_PATH = "/app/shared_storage/optuna_optimization.log"

    def __init__(self, study_name, scenario_name):
        """
        Initializes the distributed worker tuning node."""
        
        self._study_name = study_name
        self._matsim_config_path = f"/app/scenarios/{scenario_name}/input/config.xml"

    def _calculate_multi_objective_metrics(self, delta_q_history, reward_history, window_size=20, threshold=0.1):
        """
        Calculates advanced multi-objective metrics using a rolling window approach.
        
        Returns:
            tuple: (final_moving_avg_utility, final_moving_variance, convergence_iter)
        """

        # Ensure we have enough data points to compute the window
        total_iterations = len(reward_history)
        if total_iterations < window_size:
            window_size = total_iterations
        
        # 2. Objective 1: Maximize the final moving average value
        final_window = reward_history[-window_size:]
        reward_moving_average = sum(final_window) / window_size
        
        # 3. Objective 2: Minimize the variance within this final rolling window
        reward_moving_variance = sum((x - reward_moving_average) ** 2 for x in final_window) / window_size

        # 4. Objective 3: Minimize Q-convergence iteration (unchanged)
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

    def _objective(self, trial: optuna.Trial):
        """
        The objective hook invoked by Optuna to run an individual isolated simulation trial.
        """

        # Initialize the objective values
        utilities = []
        deltas = []

        # Parameters
        alpha_val = trial.suggest_float("alpha", 0.05, 0.40)
        gamma_val = trial.suggest_float("gamma", 0.85, 0.99)
        decay_val = trial.suggest_float("epsilon_decay", 0.92, 0.98)
        min_val = trial.suggest_float("epsilon_minimum", 0.001, 0.10)

        # --- Behavioral Weights (Coarser Stepping) ---
        disc_val = trial.suggest_float("discontinuity", 1.0, 3.0)
        retrieval_val = trial.suggest_float("retrieval_cost", 1.0, 4.0)

        alpha = str(alpha_val)
        gamma = str(gamma_val)
        epsilon_decay = str(decay_val)
        epsilon_minimum = str(min_val)
        discontinuity_weight = str(disc_val)
        retrieval_cost_weight = str(retrieval_val)

        # Local paths
        base_output_directory = os.environ.get("MATSIM_OUTPUT_BASE")
        optuna_directory = os.path.join(base_output_directory, "Optuna")
        trial_directory = os.path.join(optuna_directory, f"trial_{trial.number}")
        os.makedirs(trial_directory, exist_ok=True)

        # Execute matsim
        worker = WorkerNode(
            output_directory=trial_directory,
            config_file_path=self._matsim_config_path,
            matsim_iteration=os.environ.get("MATSIM_ITERATION"),
            num_threads=os.environ.get("NUM_THREADS"),
            training_iteration=os.environ.get("MAX_TRAINING_ITERATION"),
            learning_rate=alpha,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            min_epsilon=epsilon_minimum,
            penalty_weights=[discontinuity_weight, retrieval_cost_weight],
            java_heap=os.environ.get("JAVA_HEAP")
        )

        worker.run()

        # Retrieve the data for iteration based run
        csv_file_path = os.path.join(trial_directory, "agent_tracking.csv")

        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"Expected simulation log missing at resolved location: {csv_file_path}")

        with open(csv_file_path, mode="r", encoding="utf-8") as f:
            # Skip the header row if your MATSim tracking CSV outputs column titles
            reader = csv.reader(f)
            header = next(reader) 
            
            for row in reader:
                if not row:
                    continue
                try:
                    utilities.append(float(row[3]))
                    deltas.append(float(row[5]))
                except (ValueError, IndexError) as e:
                    # Guard rail against formatting glitches or missing values
                    print(f"[Warning] Skipping malformed logging data row: {row}. Error: {e}")

        # 5. Extract results and feed composite metrics calculations back to database storage
        avgerage_utility, utility_variance, convergence_iteration = self._calculate_multi_objective_metrics(deltas, utilities)

        return avgerage_utility, utility_variance, convergence_iteration

    def run_optuna(self, n_trials=1, n_jobs=1):
        """
        Connects to the persistent global storage study registry and claims a trial allocation block.
        """
        
        storage_path = optimization_tuner.initialize_shared_database(self.DB_FILE_PATH, self._study_name)

        self.db_url = storage_path

        study = optuna.load_study(
            study_name=self._study_name, 
            storage=storage_path
        )

        study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs)
