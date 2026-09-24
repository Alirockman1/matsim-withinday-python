import os
from withinday.core.pymatsim import run_simulation

def run_rl_pipeline():
    os.environ["SERVICE_CLASS"] = "modechoice.rl.python.networking.bridge_service.ReinforcementLearningBridgeService"
    params_string = os.environ.get("PARAMS", "")
    params_dictionary = {}
    
    if params_string:
        for item in params_string.split():
            if "=" in item:
                key, value = item.split("=", 1)
                params_dictionary[key] = value

    # Package reinforcement learning hyperparameters into extra_config
    rl_extra_configs = {
        "agentModeChoice.trainingCutoffIteration": os.environ.get("MAX_TRAINING_ITERATION"),
        "agentModeChoice.alpha": params_dictionary.get("alpha"),
        "agentModeChoice.gamma": params_dictionary.get("gamma"),
        "agentModeChoice.epsilonDecay": params_dictionary.get("epsilon_decay"),
        "agentModeChoice.epsilonMinimum": params_dictionary.get("epsilon_minimum"),
        "agentModeChoice.discontinuityPenalty": params_dictionary.get("w_1", 1.0),
        "agentModeChoice.retrievalCostPenalty": params_dictionary.get("w_2", 1.0)
    }

    # Build upon the base runner by injecting the RL config parameters
    run_simulation(run_script="modechoice.rl.java.core.RunRLModeChoice",extra_config=rl_extra_configs)

if __name__ == "__main__":
    run_rl_pipeline()