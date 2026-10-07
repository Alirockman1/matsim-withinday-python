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
            model_file = config_data.get("modelFileName")

            # Load existing Q-Table weights if present
            if model_file and os.path.exists(model_file):
                logger.info(f"Loading existing Q-table from: {model_file}")
                #self.agent.load_q_table(model_file)

            logger.info("RL Agent successfully initialized.")
            return {"status": f"{model_type} Initialized"}

    def request_decision(self, agent_tag: int, observation: Any):
        """
        Receives trip observation state, initializes state representation, and selects the mode.
        """
        with self._lock:
            agent = self._get_or_create_agent(agent_tag)
            state = prepare_state(agent_tag, observation, self.trip_memory)

            print(f"Terminal Status: The agent {observation.agentID} is terminal: {agent.terminal}")

            if not agent.terminal:
                agent.init_state(state)

            chosen_mode = get_chosen_action(agent_tag, agent, self.trip_memory)
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
            #print(latest_reward)
            #print(previous_state)
            #print(previous_action)

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

    def get_service_metrics(self):
        """
        Returns the active Q-Table (with stringified state keys), latest delta_q,
        agent IDs, and population metadata for inspection/debugging.
        """
        with self._lock:
            if not hasattr(self, "agents") or not self.agents:
                return {
                    "status": "uninitialized",
                    "q_table": {},
                    "agents": []
                }

            safe_q_table = {}
            agents_info = []
            max_delta_q = 0.0

            for agent_id, agent_instance in self.agents.items():
                agent_id_str = str(agent_id)
                
                # 1. Convert each agent's Q-Table state keys to strings
                agent_q_map = {}
                if hasattr(agent_instance, "_q_table"):
                    for state_key, q_values in agent_instance._q_table.items():
                        agent_q_map[str(state_key)] = q_values
                safe_q_table[agent_id_str] = agent_q_map

                # 2. Extract population metadata from trip memory
                memory = self.trip_memory.get(agent_id, {})
                population_type = memory.get("population", "default") if isinstance(memory, dict) else "default"
                
                agents_info.append({
                    "agent_id": agent_id_str,
                    "population": population_type
                })

                # 3. Track maximum delta_q across the population
                if hasattr(agent_instance, "delta_q"):
                    max_delta_q = max(max_delta_q, float(agent_instance.delta_q))

            return {
                "latest_delta_q": max_delta_q,
                "agents": agents_info,
                "q_table": safe_q_table
            }