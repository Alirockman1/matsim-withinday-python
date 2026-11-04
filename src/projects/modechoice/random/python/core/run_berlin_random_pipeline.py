import os
from withinday.core.pymatsim import run_simulation

def run_berlin_random_pipeline():
    os.environ["SERVICE_CLASS"] = "modechoice.random.python.networking.bridge_service.RandomBridgeService"

    # The Open Berlin runner brings its own analysis chain; it needs output this run does not write.
    run_simulation(
        run_script="modechoice.random.java.core.RunBerlinRandomModeChoice",
        extra_config={"qsim.numberOfThreads": os.environ.get("NUM_THREADS")},
        extra_args=["--post-processing", "disabled"],
    )

if __name__ == "__main__":
    run_berlin_random_pipeline()
