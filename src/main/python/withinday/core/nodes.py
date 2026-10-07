import os
import subprocess
from typing import Dict, Any, Optional

class WorkerNode:
    """
    Represents a distributed worker tuning node responsible for configuring and 
    executing MATSim simulation runs via subprocess execution. It handles JVM heap sizing, 
    thread allocation, configuration parameters, and custom within-day replanner/observer bindings."""

    def __init__(self, java_run_script: str, output_directory: str, config_file_path: str, matsim_iteration: Any, 
                num_threads: Any, replanner_class: Optional[str] = "default", observer_class: Optional[str] = "default",
                java_heap: str = "12g", extra_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the distributed worker tuning node with simulation parameters and resource allocations.

        Args:
            java_run_script (str): The fully qualified main Java class name to execute.
            output_directory (str): The target directory path for MATSim output files.
            config_file_path (str): The path to the MATSim XML configuration file.
            matsim_iteration (Any): The number of iterations to run (last iteration).
            num_threads (Any): The number of parallel computation threads to allocate.
            replanner_class (Optional[str]): Custom within-day replanner class name, if applicable.
            observer_class (Optional[str]): Custom within-day observer class name, if applicable.
            java_heap (str): The maximum Java heap size (e.g., '12g', '16g'). Defaults to '12g'.
            extra_config (Optional[Dict[str, Any]]): Additional key-value configuration parameters to inject."""

        self._java_run_script = java_run_script
        self._output_dir = output_directory
        self._config_path = config_file_path
        
        self._iteration= matsim_iteration
        self._threads = num_threads
        self._java_heap = java_heap

        self._replanner_class = replanner_class
        self._observer_class = observer_class
        self._extra_config = extra_config or {}

    def run(self):
        """
        Constructs the Java subprocess execution command with all configured parameters, 
        environment variables, and dynamic overrides, then executes the simulation run.

        Raises:
            subprocess.CalledProcessError: If the underlying Java simulation process fails with a non-zero exit code."""
            
        cmd = [
            "java",
            f"-Xmx{self._java_heap}",
            "-Djava.awt.headless=true",
            "-cp", "/app/simulation.jar",
            f"{self._java_run_script}",
            f"{self._config_path}",
            f"--config:controller.outputDirectory={self._output_dir}",
            f"--config:controller.lastIteration={self._iteration}",
            f"--config:global.numberOfThreads={self._threads}"
        ]

        # Injects project-specific config parameters dynamically
        for key, value in self._extra_config.items():
            if value is not None:
                cmd.append(f"--config:{key}={value}")

        if self._replanner_class and self._replanner_class != "default":
            cmd.append(f"--config:withinday.replanner={self._replanner_class}")

        if self._observer_class and self._observer_class != "default":
            cmd.append(f"--config:withinday.observer={self._observer_class}")

        subprocess.run(cmd, check=True, env=os.environ)