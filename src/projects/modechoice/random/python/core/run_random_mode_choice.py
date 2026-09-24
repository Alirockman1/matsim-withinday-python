import os
from withinday.core.pymatsim import run_simulation

def run_random_mode_choice_pipeline():
    os.environ["SERVICE_CLASS"] = "modechoice.random.python.networking.bridge_service.RandomBridgeService"

    # Build upon the base runner by injecting the RL config parameters
    run_simulation(run_script="modechoice.random.java.core.RunRandomModeChoiceWithinDay")

if __name__ == "__main__":
    run_random_mode_choice_pipeline()