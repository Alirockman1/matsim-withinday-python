import os
import logging
from typing import Dict, Any

from python_matsim_bridge import BaseSimulationBridgeService
from modechoice.rl.python.utils.SessionManager import *
from modechoice.rl.python.utils.StateUtils import prepare_state, get_chosen_action
from modechoice.rl.python.core.RLAgent import DecentralizedQLearningAgent as Agent

logger = logging.getLogger("matsim_bridge")

class ReinforcementLearningBridgeService(BaseSimulationBridgeService):
    """
    Custom Reinforcement Learning Bridge Service implementing Decentralized Q-Learning.
    This class handles RL state initialization, action selection, and policy updates.
    """

    def __init__(self):
        super().__init__()
        self.agents: Dict[int, Agent] = {}

    def _get_or_create_agent(self, agent_tag: int) -> Agent:
        """
        Retrieves an existing Q-Learning agent instance for the given agent tag,
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
            model_type = config_data.get("modelType", "Q-Learning")

            logger.info("RL Agent successfully initialized.")
            return {"status": f"{model_type} Initialized"}

    def request_decision(self, agent_tag: int, dataset: str, observation: Any):
        """
        Receives trip observation state, initializes state representation, and selects the mode.
        """
        with self._lock:
            agent = self._get_or_create_agent(agent_tag)
            state = prepare_state(agent_tag, observation, self.trip_memory, True)

            if not agent.terminal and dataset != "test":
                agent.init_state(state)

            chosen_mode = get_chosen_action(agent_tag, agent, self.trip_memory, dataset)
            return str(chosen_mode)

    def process_feedback(self, agent_tag: int, feedback: Any):
        """
        Records the reward and terminal status into the agent's trip memory.
        """
        with self._lock:
            if not self.agents:
                raise RuntimeError("RL Agents have not been initialized.")

            agent = self._get_or_create_agent(agent_tag)
            
            agent.terminal = feedback.isTerminal
            reward   = feedback.reward
            matsim_score = feedback.matsimScore
            agent_trip_memory = self.trip_memory.get(agent_tag, None)

            if not agent_trip_memory:
                raise KeyError(f"No active memory record found for agent ID: '{agent_tag}'")

            # Compute step reward and transition state
            update_reward(reward, matsim_score, agent_trip_memory)

    def call_update_policy(self, agent_tag: int):
        """
        Triggers the Q-table policy update using the stored transition 
        (previous_state -> previous_action -> reward -> next_state).
        """
        with self._lock:
            agent = self._get_or_create_agent(agent_tag)
            agent_trip_memory = self.trip_memory.get(agent_tag, None)

            if not agent_trip_memory:
                raise KeyError(f"No active memory record found for agent ID: '{agent_tag}'")

            # Determine next state based on terminal flag
            if agent.terminal:
                previous_state = agent_trip_memory.get("state")
                previous_action = agent_trip_memory.get("mode")
                next_state = None
            else:
                previous_state = agent_trip_memory.get("previous_state", None)
                previous_action = agent_trip_memory.get("previous_mode", None)
                next_state = agent_trip_memory.get("state")

            # Safely grab the latest reward from history
            reward_history = agent_trip_memory.get("reward_history", [])
            latest_reward = reward_history[-1] if reward_history else 0.0

            # Update Q-table policy
            if previous_state is not None and previous_action is not None:
                agent.update_policy(
                    previous_state, 
                    previous_action, 
                    latest_reward, 
                    next_state, 
                    print_tabel=True
                )

            delta_q = float(getattr(agent, "delta_q", -999))
            return {"response": delta_q}
    
    def reset_episode(self):
        """
        Clears agent trip memory and active states at the end of an iteration.
        """
        with self._lock:
            if hasattr(self, "trip_memory"):
                self.trip_memory.clear()
            
            Agent.increment_simulation_iteration

            for agent_ID , agent in self.agents.items():
                agent.terminal = False
            
            return {"status": "memory_cleared"}

    def checkpoint_state(self, iteration):
        """
        Persists the current Q-table to disk.
        """
        with self._lock:
            for _, agent in self.agents.items():
                agent.dump(iteration)
                
            logger.info(f"Q-Table successfully saved.")
            return True
        return False

    def update_shared_policy(self, include_states: bool = True, include_q_table: bool = True):
        """
        Aggregates the per-agent summaries (each agent builds its own via agent.summary()).
 
        Pass include_states=False / include_q_table=False for a light per-iteration summary;
        both are True by default to keep the previous output shape.
        """
        with self._lock:
            if not self.agents:
                return {
                    "status": "uninitialized",
                    "agents": [],
                    "q_table": {},
                }
 
            agents_info = [agent.summary(include_states, include_q_table) for agent in self.agents.values()]
 
            metrics = {
                "status": "active",
                "agents": agents_info,
            }
 
            if include_q_table:
                # Same shape as before: {agent_id: {state: {mode: q}}} at the top level
                metrics["q_table"] = {info["agent_id"]: info.pop("q_table") for info in agents_info}
 
            return metrics