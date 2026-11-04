import os
from withinday.core.pymatsim import run_simulation


def run_berlin_rl_pipeline():
    os.environ["SERVICE_CLASS"] = "modechoice.rl.python.networking.bridge_service.ReinforcementLearningBridgeService"
    params_string = os.environ.get("PARAMS", "")
    params_dictionary = {}

    if params_string:
        for item in params_string.split():
            if "=" in item:
                key, value = item.split("=", 1)
                params_dictionary[key] = value

    # Package reinforcement learning hyperparameters into extra_config. They override the defaults
    # the runner reads from the scenario's modeChoiceModel.xml.
    rl_extra_configs = {
        "qsim.numberOfThreads": os.environ.get("NUM_THREADS"),
        "agentModeChoice.trainingCutoffIteration": os.environ.get("MAX_TRAINING_ITERATION"),
        "agentModeChoice.alpha": params_dictionary.get("alpha"),
        "agentModeChoice.gamma": params_dictionary.get("gamma"),
        "agentModeChoice.epsilonDecay": params_dictionary.get("epsilon_decay"),
        "agentModeChoice.epsilonMinimum": params_dictionary.get("epsilon_minimum"),
        "agentModeChoice.discontinuityPenalty": params_dictionary.get("w_1", 1.0),
        "agentModeChoice.retrievalCostPenalty": params_dictionary.get("w_2", 1.0),
    }

    # The Open Berlin runner brings its own analysis chain; it needs output this run does not write.
    run_simulation(
        run_script="modechoice.rl.java.core.RunBerlinRLModeChoice",
        extra_config=rl_extra_configs,
        extra_args=["--post-processing", "disabled"],
    )


if __name__ == "__main__":
    run_berlin_rl_pipeline()
