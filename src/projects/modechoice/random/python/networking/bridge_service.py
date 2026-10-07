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
        self.agents: Dict[int, Agent] = {}

    def _get_or_create_agent(self, agent_tag: int) -> Agent:
        """
        Retrieves an existing agent instance for the given agent tag,
        or creates a new one if it doesn't exist yet.
        """
        if agent_tag not in self.agents:
            # Instantiate with the unique ID and session config dictionary
            self.agents[agent_tag] = Agent(agent_tag)     
        return self.agents[agent_tag]  

    def configure_session(self, config_data: Dict[str, Any]):
        """
        Initializes the Decentralized Q-Learning Agent using parameters 
        passed from MATSim's config.xml.
        """
        with self._lock:
            self.session_config = config_data

            # Update class global parameter
            Agent.configure_global_parameters(config_data)

            logger.info("Random model for the WithinDayReplanner initialized.")
            return {"status": "Agent Initialized"}

    def request_decision(self, agent_tag: int, observation: Any):
        """
        Receives trip observation state, records memory, and selects
        the agent's mode choice via trhe random model.
        """
        with self._lock:
            agent = self._get_or_create_agent(agent_tag)

            if agent_tag not in self.trip_memory or not self.trip_memory[agent_tag]:
                initiate_memory(agent_tag, observation, self.trip_memory)

            chosen_mode = get_chosen_action(agent_tag, agent, self.trip_memory)
            return str(chosen_mode)

    def checkpoint_state(self) -> bool:
        """
        Random model has no state to persist, so we simply return True 
        to satisfy the interface and allow clean shutdown.
        """
        return True