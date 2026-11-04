import random
import pickle
import os

from modechoice.rl.python.core.Memory import Memory

class DecentralizedQLearningAgent:
    RANDOM_SEED = 42

    _config = {}
    _alpha = 0.1
    _gamma = 0.9
    _initial_epsilon = 1.0
    _epsilon_min = 0.01
    _epsilon_decay = 0.99
    _output_directory = ""
    _file_path = ""
    _simulation_iteration = 0
    max_training_iteration = 100
    _modes = []
    _checkpoint_model = ""
    
    @classmethod
    def configure_global_parameters(cls, config_args: dict):
        """
        Dynamically configures class-level shared parameters from any arbitrary dictionary 
        passed by MATSim/Java, without needing to know every key ahead of time.
        """
        cls._config = config_args or cls._config

        # Extract known parameters safely with default fallbacks
        cls._alpha = float(config_args.get("alpha", cls._alpha))
        cls._gamma = float(config_args.get("gamma", cls._gamma))
        cls._initial_epsilon = float(config_args.get("epsilon", cls._initial_epsilon))
        cls._epsilon_min = float(config_args.get("epsilonMinimum", cls._epsilon_min))
        cls._epsilon_decay = float(config_args.get("epsilonDecay", cls._epsilon_decay))
        cls.max_training_iteration = int(config_args.get("trainingCutoffIteration", cls.max_training_iteration))
        cls._output_directory = str(config_args.get("outputDirectory"))
        cls._checkpoint_model = str(config_args.get("checkpointModel"))
        cls._simulation_iteration = 0
        
        # Parse available modes globally
        mode_string = config_args.get("modes", "")
        if isinstance(mode_string, str):
            cls._modes = [m.strip() for m in mode_string.split(",") if m.strip()]
        else:
            cls._modes = list(mode_string)

        random.seed(cls.RANDOM_SEED)

    @classmethod
    def increment_simulation_iteration(cls):
        cls._simulation_iteration =+ 1

    @classmethod
    def get_epsilon(cls):
        return cls._epsilon

    @classmethod
    def set_epsilon(cls, value):
        cls._epsilon = float(value)

    @classmethod
    def get_file_path(cls):
        return cls._file_path

    @classmethod
    def set_file_path(cls, path):
        cls._file_path = path

    @classmethod
    def get_modes(cls):
        return cls._modes


    def __init__(self, unique_id, args={}):
        """Initializes a unique instance for a specific ID."""
        self._id = unique_id
        self._q_table = {}
        self._current_iteration = 0
        self._delta_q = 0.0
        self._epsilon = self._initial_epsilon
        self._terminal = False
        self._previous_iteration = 0
        self._memory = Memory()

        for key, value in args.items():
            setattr(self, key, value)

        self._load_data_on_startup()

    @property
    def q_table(self):
        return self._q_table

    @property
    def id(self):
        return self._id
    
    @property
    def delta_q(self):
        return self._delta_q
    
    @property
    def terminal(self):
        return self._terminal

    @terminal.setter
    def terminal(self, isterminal):
        self._terminal = isterminal

    @property
    def state_coverage(self):
        return self._memory.get_data()

    def summary(self, include_states=True, include_q_table=False):
        """
        Summarize this agent for monitoring (plain, JSON-safe values only).
 
        :param include_states:  also include the per-state visit / action counts (can be large).
        :param include_q_table: also include this agent's full Q-table (can be very large).
        """
        data = {
            "iteration": self._current_iteration,
            "q_table_states": len(self._q_table),
            **self._memory.summary(),
        }

        if include_states:
            data["states"] = self._memory.details()
        if include_q_table:
            data["q_table"] = {str(state): dict(q_values) for state, q_values in self._q_table.items()}
        return data
    
    def init_state(self, state_tuple):
        ''' Initializes the state and action pair in this agent's Q-table. '''

        if state_tuple not in self._q_table:
            # Initialize all possible modes
            self._q_table[state_tuple] = {mode: 600.0 for mode in self._modes}

        self._memory.record_state(state_tuple, self._modes)

    def decay_epsilon(self, iteration):
        ''' Updates the epsilon value for the current iteration. '''
        self._current_iteration = iteration + self._previous_iteration
        self._epsilon = max(self._epsilon_min, self._initial_epsilon * (self._epsilon_decay ** self._current_iteration))
        
    def choose_action_train(self, state, available_modes):
        ''' Epsilon-Greedy selection for mode choices for this specific agent. ''' 
        # Exploration (Random)
        if random.random() < self._epsilon and self._current_iteration < self.max_training_iteration:
            chosen_action = random.choice(available_modes)
        # Exploitation
        else:
            agent_policy = self._q_table[state]
            filtered_policy = {mode: agent_policy[mode] for mode in available_modes}
            chosen_action = max(filtered_policy, key=filtered_policy.get)

        self._memory.record_action(state, chosen_action)
        return chosen_action

    def choose_action_evaluation(self, state, available_modes):
        ''' Evaluate the saved policy in the q_table.
	    Fall back to the first available mode if the state doesnt exist.
	    The fallback has to come from available_modes: a hard coded mode name only exists
	    in the scenario it was written for, and MATSim aborts the mobsim with
	    UnknownModeException on any mode its TripRouter does not know. '''

        try:
            agent_policy = self._q_table[state]
            filtered_policy = {mode: agent_policy[mode] for mode in available_modes}
            return max(filtered_policy, key=filtered_policy.get)
        except (KeyError, TypeError):
            return available_modes[0]

    def update_policy(self, state, action, reward, next_state=None, print_tabel=False):
        ''' 
        The Bellman Equation Update for this agent.
        "Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]" '''

        if self._simulation_iteration >= self.max_training_iteration:
            return
        
        current_q_value = self._q_table[state][action]

        if next_state is None:
            target = reward # No furture update required
        # Get Max Q(s', a') for the next state
        else:
            future_reward_value = max(self._q_table[next_state].values())
            # Discount the future reward to the current reward
            target = reward + self._gamma * future_reward_value

        # Update Rule: Q(s,a) = Q(s,a) + alpha * [Reward + gamma * MaxQ(s') - Q(s,a)]
        self._q_table[state][action] += self._alpha * (target - current_q_value)
        new_q_value = self._q_table[state][action]

        # Incremental change in q value
        old_q_value = current_q_value
        self._delta_q = abs(new_q_value - old_q_value)

        if print_tabel:
            self.print_q_table()

    def print_q_table(self):
        ''' Print the bit-encoded Q-table within the Command Prompt window using a clean tabular layout. '''

        width = 125
        print(f"\n{'=' * width}")
        print(f"{'PERSONAL Q-TABLE (20-BIT VECTOR VIEW): ' + str(self._id):^{width}}")
        print(f"{'=' * width}")

        # Optimized header matching the exact 20-bit feature set
        header = (f"{'CUR ZONE':<10} | {'TGT ZONE':<10} | "
                  f"{'DEP BIN':<8} | {'DISTANCE':<12} | {'ASSET':<14} | {'CONGESTION':<12} || "
                  f"{'CAR':<7} | {'PT':<7} | {'BIKE':<7} | {'WALK':<7}")
        print(header)
        print("-" * width)

        for state_bits, actions in self._q_table.items():
            state_bits_string = [str(bit) for bit in state_bits]

            # Handle terminal states or unexpected short keys safely
            if "TERMINAL" in state_bits or len(state_bits) < 20:
                terminal_row = (f"{'TERMINAL STATE':<10} | {' ':<10} | "
                                f"{'-':<8} | {'-':<12} | {'TERMINAL':<14} | {'-':<12} || "
                                f"{actions.get('car', 0.0):>7.2f} | {actions.get('pt', 0.0):>7.2f} | "
                                f"{actions.get('bike', 0.0):>7.2f} | {actions.get('pedestrian', 0.0):>7.2f}")
                print(terminal_row)
                print("." * width)
                continue

            # 1. Unpack bits according to the exact 20-bit breakdown:
            print(state_bits_string)
            curr_zone_bits = "".join(state_bits_string[0:4])
            tgt_zone_bits  = "".join(state_bits_string[4:8])
            time_bits      = "".join(state_bits_string[8:13])
            dist_bits      = "".join(state_bits_string[13:16])
            asset_bits     = "".join(state_bits_string[16:18])
            cong_bits      = "".join(state_bits_string[18:20])

            # 2. Decode Values & Labels
            curr_zone = int(curr_zone_bits, 2)
            tgt_zone  = int(tgt_zone_bits, 2)
            tau_val   = int(time_bits, 2)
            
            dist_idx   = int(dist_bits, 2)
            dist_label = {
                0: "0-1 km", 
                1: "1-2 km", 
                2: "2-5 km", 
                3: "5-10 km",
                4: "10+ km"
            }.get(dist_idx, "UNKNOWN")

            asset_idx   = int(asset_bits, 2)
            asset_label = {
                0: "NONE", 
                1: "CAR ONLY", 
                2: "BIKE ONLY", 
                3: "CAR + BIKE"
            }.get(asset_idx, "UNKNOWN")
            
            cong_idx   = int(cong_bits, 2)
            cong_label = {
                0: "FREE", 
                1: "SLOW", 
                2: "MILD", 
                3: "HEAVY"
            }.get(cong_idx, "UNKNOWN")

            # 3. Print formatted row with Q-values
            row = (f"{curr_zone:<10} | {tgt_zone:<10} | "
                   f"{tau_val:<8} | {dist_label:<12} | {asset_label:<14} | {cong_label:<12} || "
                   f"{actions.get('car', 0.0):>7.2f} | "
                   f"{actions.get('pt', 0.0):>7.2f} | "
                   f"{actions.get('bike', 0.0):>7.2f} | "
                   f"{actions.get('pedestrian', 0.0):>7.2f}")
            print(row)
            
            # Section break between states
            print("." * width)

        print(f"{'=' * width}\n")

    def dump(self, iteration):
        """Save or update the shared Q-table checkpoint."""

        if iteration > self.max_training_iteration:
            save_directory = self._output_directory
        else:
            save_directory = os.path.join(self._output_directory, "ITERS", f"it.{iteration}")

        try:
            os.makedirs(save_directory, exist_ok=True)

            save_file_path = os.path.join(save_directory, "q_table.pkl")

            if os.path.isfile(save_file_path) and os.path.getsize(save_file_path) > 0:
                with open(save_file_path, "rb") as f:
                    checkpoint = pickle.load(f)

                checkpoint.setdefault("q_tables", {})
                checkpoint.setdefault("memories", {})

                checkpoint["q_tables"][self._id] = self._q_table
                checkpoint["memories"][self._id] = self._memory.get_data()
            else:
                checkpoint = {
                    "iteration": iteration + self._previous_iteration + 1,
                    "config_arg": self._config,
                    "q_tables": {self._id: self._q_table},
                    "memories": {self._id: self._memory.get_data()},
                }

            with open(save_file_path, "wb") as f:
                pickle.dump(checkpoint, f)

            print(f"PYTHON SERVICE: Checkpoint successfully saved to {save_file_path} for agent {self._id}.")
        except Exception as e:
            print(f"PYTHON SERVICE: Failed to save checkpoint: {e}")

    def _load_data_on_startup(self):
        """Load decentralized Q-tables and populate self._q_table."""

        load_file_path = self._checkpoint_model

        if not load_file_path or not str(load_file_path).strip():
            print("PYTHON SERVICE: No checkpoint path provided. Starting fresh.")
            return False

        if not os.path.isfile(load_file_path):
            print(f"PYTHON SERVICE: Checkpoint not found: {load_file_path}")
            return False

        if os.path.getsize(load_file_path) == 0:
            print(f"PYTHON SERVICE: Checkpoint file is empty: {load_file_path}")
            return False

        try:
            with open(load_file_path, "rb") as f:
                saved_data = pickle.load(f)

            self._previous_iteration = saved_data["iteration"]
            self.configure_global_parameters(saved_data["config_arg"])
            q_tables = saved_data.get("q_tables", {})
            memories = saved_data.get("memories", {})

            self._parse_loaded_data_for_agent(q_tables, memories)

            print(f"PYTHON SERVICE: Loaded tables for {self._id}")
            self.print_q_table()

            return True
        except Exception as e:
            print(f"PYTHON SERVICE: Failed to load checkpoint: {e}")
            return False

    def _parse_loaded_data_for_agent(self, q_tables, memories):
        self._q_table = {}

        q_table = q_tables.get(self._id) or q_tables.get(str(self._id))
        memory = memories.get(self._id) or memories.get(str(self._id))

        for state, action_values in q_table.items():
            if not isinstance(action_values, dict):
                print(f"PYTHON SERVICE: Skipping invalid state {state!r}.")
                continue

            self._q_table[state] = {}

            for action, q_value in action_values.items():
                self._q_table[state][action] = q_value
        
        if memory:
            self._memory.load_data(memory)