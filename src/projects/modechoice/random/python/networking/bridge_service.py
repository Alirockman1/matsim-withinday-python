import logging
from typing import Dict, Any

from python_matsim_bridge import BaseSimulationBridgeService
from modechoice.random.python.utils.StateUtils import *
from modechoice.random.python.core.Agent import RandomModeAgent as Agent

logger = logging.getLogger("matsim_bridge")

class RandomBridgeService(BaseSimulationBridgeService):
    """Custom bridge service implementing random mode choice selection.
    """

    def __init__(self):
        super().__init__()
        self.agent = None
        self.trip_memory: Dict[str, Any] = {}
        self.daily_stats: Dict[str, Any] = {}

    def configure_session(self, config_data: Dict[str, Any]):
        """
        Initializes the Decentralized Q-Learning Agent using parameters 
        passed from MATSim's config.xml.
        """
        with self._lock:
            self.session_config = config_data

            # Instantiate Agent
            self.agent = Agent(random_seed=config_data.get("randomSeed"))

            logger.info("Random model for the WithinDayReplanner initialized.")
            return {"status": "Agent Initialized"}

    def request_decision(self, observation: Any):
        """
        Receives trip observation state, records memory, and selects
        the agent's mode choice via trhe random model.
        """
        with self._lock:
            if not self.agent:
                raise RuntimeError("Agent has not been initialized.")

            agent_id = observation.agentID

            if agent_id not in self.trip_memory or not self.trip_memory[agent_id]:
                initiate_memory(observation, self.trip_memory)

            chosen_mode = get_chosen_action(agent_id, self.agent, self.trip_memory)
            return str(chosen_mode)

    def checkpoint_state(self) -> bool:
        """
        Random model has no state to persist, so we simply return True 
        to satisfy the interface and allow clean shutdown.
        """
        return True