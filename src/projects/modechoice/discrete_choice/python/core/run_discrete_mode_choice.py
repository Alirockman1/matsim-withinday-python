import os
from withinday.core.pymatsim import run_simulation

def run_discrete_mode_choice_pipeline():
    os.environ["SERVICE_CLASS"] = ""

    # Build upon the base runner by injecting the RL config parameters
    run_simulation(run_script="modechoice.discrete_choice.java.core.RunDiscreteModeChoiceWithinDay")

if __name__ == "__main__":
    run_discrete_mode_choice_pipeline()