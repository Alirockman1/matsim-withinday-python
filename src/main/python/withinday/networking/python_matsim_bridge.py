import os
import importlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger("matsim_bridge")

class BaseSimulationBridgeService:
    """
    Generic Abstract Class for MATSim Python Bridge Integrations.
    
    Developers extend this class to implement custom decision algorithms
    (e.g., Rule-Based Heuristics, Mathematical Optimization, Machine Learning, RL)
    without needing to modify the underlying networking layer (`main.py`)."""

    def __init__(self):
        self._lock = Lock()
        self.session_config: Dict[str, Any] = {}
        self.trip_memory: Dict[str, Any] = {}

    def configure_session(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses scenario parameters and prepares global context or models at MATSim start.
        
        :param config_data: Dictionary containing configuration key-value pairs from MATSim.
        :return: Status response dictionary."""

        with self._lock:
            self.session_config = config_data
            logger.info("Simulation session configured successfully.")
            return {
                "status": "configured", 
                "session_id": config_data.get("scenarioName", "default")
            }

    def request_decision(self, agent_tag: int, observation: Any) -> str:
        """
        Main decision loop method. Takes an agent's current state/observation 
        from MATSim and returns the selected action or mode choice string.
        
        :param observation: Pydantic model (`ObserverData`) or dict containing state parameters.
        :return: Selected mode choice string (e.g., "car", "bike", "pt")."""

        raise NotImplementedError("Subclasses must implement request_decision()")

    def process_feedback(self, agent_tag: int, feedback: Any) -> Dict[str, Any]:
        """
        Called after MATSim executes a leg/trip. Receives real execution results 
        (travel times, distances, rewards, scores) to update agent memory or online models.
        
        :param feedback: Pydantic model (`ArrivalData`) or dict containing trip execution metrics.
        :return: Processing status dictionary."""

        raise NotImplementedError("Subclasses must implement process_feedback()")

    def checkpoint_state(self, iteration) -> bool:
        """
        Triggered periodically (e.g., end of iteration or server shutdown).
        Persists current model weights, statistics, or memory state to disk.
        
        :return: True if checkpoint succeeded, False otherwise."""

        raise NotImplementedError("Subclasses must implement checkpoint_state()")

    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Returns diagnostic metrics, decision statistics, or active model properties.
        
        :return: Dictionary of service runtime metrics."""

        with self._lock:
            return {
                "active_agent_memories": len(self.agent_memories),
                "configured": bool(self.session_config)
            }

    def reset_episode():
        """
        Reset the internal memory stored within the bridge."""

        pass

def load_bridge_service() -> Optional[BaseSimulationBridgeService]:
    """
    Dynamically imports and instantiates the bridge service class specified 
    by the `SERVICE_CLASS` environment variable.
    
    Example environment variable format:
      SERVICE_CLASS="custom_services.ReinforcementLearningBridgeService"
    
    :return: An instance of a class extending `BaseSimulationBridgeService`."""
    
    service_path = os.getenv("SERVICE_CLASS", "none")

    if not service_path or service_path.lower() == "none":
        logger.info("Running baseline mode: No external bridge service loaded.")
        return None
    
    try:
        module_name, class_name = service_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        service_class = getattr(module, class_name)
        
        if not issubclass(service_class, BaseSimulationBridgeService):
            raise TypeError(f"Class '{class_name}' must inherit from BaseSimulationBridgeService")
            
        logger.info(f"Successfully loaded bridge service class: {service_path}")
        return service_class()
    
    except Exception as e:
        logger.error(f"Failed to load service '{service_path}'. Error: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize service class '{service_path}'") from e
