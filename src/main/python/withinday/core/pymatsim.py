import os
import time
from withinday.core.nodes import WorkerNode

def run_simulation(run_script="org.matsim.withinday.core.RunWithinDay", extra_config=None):
    if extra_config is None:
        extra_config = {}

    start_time = time.perf_counter()

    scenario = os.environ.get("SCENARIO")
    output_dir = os.environ.get("MATSIM_OUTPUT_BASE")
    objective = os.environ.get("OBJECTIVE")

    config_path = f"/app/scenarios/{scenario}/input/config.xml"
    
    if objective == "multi":
        num_agent_per_iter = int(os.environ.get("AGENTS_PER_ITERATION"))
    else:
        num_agent_per_iter = 1

    extra_config.update({"withinday.agentsPerIteration": str(num_agent_per_iter)})
    
    worker = WorkerNode(
        java_run_script=run_script,
        output_directory=output_dir,
        config_file_path=config_path,
        matsim_iteration=os.environ.get("MATSIM_ITERATION"),
        num_threads=os.environ.get("NUM_THREADS"),
        replanner_class=os.environ.get("REPLANNER_CLASS", "default"),
        observer_class=os.environ.get("OBSERVER_CLASS", "default"),
        java_heap=os.environ.get("JAVA_HEAP", "12g"),
        extra_config=extra_config
    )

    try:
        worker.run()
    finally:
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        print(f"\nTotal Runtime: {elapsed_seconds:.2f} seconds\n")

if __name__ == "__main__":
    run_simulation()