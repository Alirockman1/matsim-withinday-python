import os
import logging
from typing import Dict, Any

from python_matsim_bridge import BaseSimulationBridgeService
from modechoice.rl.python.utils.SessionManager import *
from modechoice.rl.python.utils.StateUtils import prepare_state, get_chosen_action
from modechoice.rl.python.core.RLAgent import DecentralizedQLearningAgent as QLearningAgent

logger = logging.getLogger("matsim_bridge")

class ReinforcementLearningBridgeService(BaseSimulationBridgeService):
    """
    Custom Reinforcement Learning Bridge Service implementing Decentralized Q-Learning.
    This class handles RL state initialization, action selection, and policy updates.
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

            # Parse available transport modes
            mode_string = config_data.get("modes", "")
            mode_list = [m.strip() for m in mode_string.split(",") if m.strip()]

            # Instantiate Q-Learning Agent
            self.agent = QLearningAgent(
                mode_list,
                alpha=config_data.get("alpha"),
                gamma=config_data.get("gamma"),
                initial_epsilon=config_data.get("epsilon"),
                epsilon_decay=config_data.get("epsilonDecay"),
                epsilon_minimum=config_data.get("epsilonMinimum"),
                max_iteration=config_data.get("trainingCutoffIteration")
            )

            model_type = config_data.get("modelType", "Q-Learning")
            model_file = config_data.get("modelFileName")

            # Load existing Q-Table weights if present
            if model_file and os.path.exists(model_file):
                logger.info(f"Loading existing Q-table from: {model_file}")
                self.agent.load_q_table(model_file)

            logger.info("RL Agent successfully initialized.")
            return {"status": "Agent Initialized", "model": model_type}

    def request_decision(self, observation: Any):
        """
        Receives trip observation state, records memory, and selects
        the agent's mode choice via the RL policy.
        """
        with self._lock:
            if not self.agent:
                raise RuntimeError("RL Agent has not been initialized.")

            agent_id = observation.agentID
            state = prepare_state(observation, self.trip_memory)
            self.agent.init_state(agent_id, state)

            chosen_mode = get_chosen_action(agent_id, self.agent, self.trip_memory)
            return str(chosen_mode)

    def process_feedback(self, feedback: Any):
        """
        Calculates trip rewards upon trip arrival, updates the Q-table policy.
        """
        with self._lock:
            if not self.agent:
                raise RuntimeError("RL Agent has not been initialized.")

            agent_id = feedback.agentID
            agent_memory = self.trip_memory.get(agent_id, None)

            if not agent_memory:
                raise KeyError(f"No active memory record found for agent ID: '{agent_id}'")

            # Compute step reward and transition state
            reward, next_state, termination = update_experience(feedback, self.agent, self.daily_stats)

            # Update Q-table policy
            self.agent.update_policy(
                agent_id, 
                agent_memory['state'], 
                agent_memory['mode'], 
                reward, 
                next_state, 
                print_tabel=False
            )

            delta_q = float(getattr(self.agent, "delta_q", 0.0))
            
            return {"deltaQ": delta_q}

    def checkpoint_state(self):
        """
        Persists the current Q-table to disk.
        """
        with self._lock:
            model_file = self.session_config.get("modelFileName")
            if self.agent and model_file:
                # Ensure target directory exists
                filePath = os.path.abspath(model_file)
                os.makedirs(os.path.dirname(filePath), exist_ok=True)

                self.agent.file_path = filePath
                self.agent.save_q_table()
                
                logger.info(f"Q-Table successfully saved to {model_file}")
                return True
            return False

    def get_service_metrics(self):
        """
        Returns the active Q-Table (with stringified state keys), latest delta_q,
        agent IDs, and population metadata for inspection/debugging.
        """
        with self._lock:
            if not self.agent or not hasattr(self.agent, "_q_table"):
                return {
                    "status": "uninitialized",
                    "q_table": {},
                    "agents": []
                }

            # 1. Convert Q-Table state keys to strings to ensure JSON serialization succeeds
            safe_q_table = {}
            for agent_id, state_map in self.agent._q_table.items():
                safe_q_table[str(agent_id)] = {
                    str(state_key): q_values 
                    for state_key, q_values in state_map.items()
                }

            # 2. Extract population metadata and active agent IDs from trip memory
            agents_info = []
            for agent_id, memory in self.trip_memory.items():
                agents_info.append({
                    "agent_id": str(agent_id),
                    "population": memory.get("population", "default")
                })

            # 3. Retrieve latest delta_q safely
            latest_delta_q = float(getattr(self.agent, "delta_q", 0.0))

            return {
                "latest_delta_q": latest_delta_q,
                "agents": agents_info,
                "q_table": safe_q_table
            }