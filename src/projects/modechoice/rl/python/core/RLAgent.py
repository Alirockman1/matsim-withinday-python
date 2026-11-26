import random
import pickle
import os

from modechoice.rl.python.core.Memory import Memory


class DecentralizedQLearningAgent:
    RANDOM_SEED = 42
    _CHECKPOINT_PARAMS = {
        "alpha": ("_alpha", float),
        "gamma": ("_gamma", float),
        "epsilon": ("_initial_epsilon", float),
        "epsilonMinimum": ("_epsilon_min", float),
        "epsilonDecay": ("_epsilon_decay", float),
        "tieTolerance": ("_tie_tolerance", float),
    }

    _config = {}
    _alpha = 0.1
    _alpha_min = 0.01
    _gamma = 0.9
    _initial_epsilon = 1.0
    _epsilon_min = 0.01
    _epsilon_decay = 0.99
    _tie_tolerance = 1.0          # Q-values within this many reward units of the best count as tied
    _output_directory = ""
    _file_path = ""
    _simulation_iteration = 0
    _random_seed = RANDOM_SEED
    max_training_iteration = 100
    _modes = []
    _checkpoint_model = ""

    @classmethod
    def configure_global_parameters(cls, config_args: dict):
        """
        Dynamically configures class-level shared parameters from any arbitrary dictionary
        passed by MATSim/Java, without needing to know every key ahead of time.
        """
        config_args = config_args or {}
        cls._config = config_args or cls._config

        # Extract known parameters safely with default fallbacks
        cls._alpha = float(config_args.get("alpha", cls._alpha))
        cls._gamma = float(config_args.get("gamma", cls._gamma))
        cls._initial_epsilon = float(config_args.get("epsilon", cls._initial_epsilon))
        cls._epsilon_min = float(config_args.get("epsilonMinimum", cls._epsilon_min))
        cls._epsilon_decay = float(config_args.get("epsilonDecay", cls._epsilon_decay))
        cls._tie_tolerance = float(config_args.get("tieTolerance", cls._tie_tolerance))
        cls.max_training_iteration = int(config_args.get("trainingCutoffIteration", cls.max_training_iteration))
        cls._output_directory = str(config_args.get("outputDirectory", cls._output_directory))
        cls._checkpoint_model = str(config_args.get("checkpointModel", cls._checkpoint_model))
        cls._simulation_iteration = 0
        cls._random_seed = int(config_args.get("randomSeed", cls.RANDOM_SEED))

        print(f"PYTHON SERVICE: random seed = {cls._random_seed}")

        # Parse available modes globally
        mode_string = config_args.get("modes", "")
        if isinstance(mode_string, str):
            cls._modes = [m.strip() for m in mode_string.split(",") if m.strip()]
        else:
            cls._modes = list(mode_string)

        random.seed(cls._random_seed)

    @classmethod
    def apply_checkpoint_parameters(cls, saved_config: dict):
        """
        Adopt the learning parameters (and the modes) a loaded Q-table was trained with, so the policy is
        used / continued under the parameters that produced it. Only the keys in _CHECKPOINT_PARAMS and
        the modes are taken from the checkpoint; run-specific settings stay as configured for this run.
        Idempotent and side-effect free otherwise: it neither re-seeds the RNG nor resets iteration counters.
        """
        saved_config = saved_config or {}
        applied = {}

        for key, (attr, cast) in cls._CHECKPOINT_PARAMS.items():
            if key not in saved_config:
                continue  # e.g. older checkpoints without tieTolerance: keep the current value
            value = cast(saved_config[key])
            if getattr(cls, attr) != value:
                print(f"PYTHON SERVICE: {key}: {getattr(cls, attr)} -> {value} (from checkpoint)")
            setattr(cls, attr, value)
            applied[key] = saved_config[key]

        if "modes" in saved_config:
            mode_value = saved_config["modes"]
            modes = ([m.strip() for m in mode_value.split(",") if m.strip()]
                     if isinstance(mode_value, str) else list(mode_value))
            if modes != cls._modes:
                print(f"PYTHON SERVICE: modes: {cls._modes} -> {modes} (from checkpoint)")
            cls._modes = modes
            applied["modes"] = saved_config["modes"]

        # keep the config that gets written into the next checkpoint consistent with what is in effect
        cls._config = {**cls._config, **applied}

    @classmethod
    def increment_simulation_iteration(cls):
        cls._simulation_iteration += 1

    @classmethod
    def get_file_path(cls):
        return cls._file_path

    @classmethod
    def set_file_path(cls, path):
        cls._file_path = path

    @classmethod
    def get_modes(cls):
        return cls._modes

    def __init__(self, unique_id, args=None):
        """Initializes a unique instance for a specific ID."""
        self._id = unique_id
        self._q_table = {}
        self._current_iteration = 0
        self._delta_q = 0.0
        self._epsilon = self._initial_epsilon
        self._terminal = False
        self._previous_iteration = 0
        self._memory = Memory()

        for key, value in (args or {}).items():
            setattr(self, key, value)

        self._load_data_on_startup()

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, value):
        self._epsilon = float(value)

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

    def _greedy(self, state, available_modes, break_ties_randomly=False):
        """
        Greedy action with tolerance-based tie handling.

        All modes whose Q-value is within `_tie_tolerance` of the best are treated as tied.
          - training:   pick uniformly among the tied modes with the seeded RNG, so new states are not
                        always explored in the same fixed order and the seed actually matters.
          - evaluation: deterministic: most-tried action first, then highest Q, then the order of the
                        `modes` list from the config (nothing is hard-coded).
        """
        q = self._q_table[state]
        modes = sorted(available_modes)  # sorted: independent of set/hash ordering -> reproducible
        best = max(q[m] for m in modes)
        tied = [m for m in modes if best - q[m] <= self._tie_tolerance]

        if len(tied) == 1:
            return tied[0]
        if break_ties_randomly:
            return random.choice(tied)
        order = {m: i for i, m in enumerate(self._modes)}
        return max(
            tied,
            key=lambda m: (self._memory.get_action_count(state, m),   # 1. most-tried action
                           q[m],                                      # 2. highest Q
                           -order.get(m, len(order))),                # 3. position in the configured modes
        )

    def get_alpha(self, state, action):
        ''' Learning rate from n(s,a) and n(s,a)/n(s), with the Q init counted as a prior. '''
        n_sa = max(1, self._memory.get_action_count(state, action))
        n_s = max(n_sa, sum(self._memory.get_action_count(state, m) for m in self._modes))

        n_eff = n_sa * n_sa / n_s                 # samples discounted by staleness
        prior = 1.0 / self._alpha - 1.0           # pseudo-samples carried by the init value
        alpha = 1.0 / (n_eff + prior)

        return min(self._alpha, max(self._alpha_min, alpha))

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
            # Initialize all possible modes (optimistic initial value)
            self._q_table[state_tuple] = {mode: 600.0 for mode in self._modes}

        self._memory.record_state(state_tuple, self._modes)

    def decay_epsilon(self, iteration):
        ''' Updates the epsilon value for the current iteration. '''
        self._current_iteration = iteration + self._previous_iteration
        self._epsilon = max(self._epsilon_min,
                            self._initial_epsilon * (self._epsilon_decay ** self._current_iteration))

    def choose_action_train(self, state, available_modes):
        ''' Epsilon-Greedy selection for mode choices for this specific agent. '''
        training = self._current_iteration < self.max_training_iteration

        # Exploration (Random)
        if training and random.random() < self._epsilon:
            chosen_action = random.choice(available_modes)
        # Exploitation (random tie-breaking only while training)
        else:
            chosen_action = self._greedy(state, available_modes, break_ties_randomly=training)

        self._memory.record_action(state, chosen_action)
        return chosen_action

    def choose_action_evaluation(self, state, available_modes):
        ''' Evaluate the saved policy in the q_table (deterministic).
            Return default pedestrian mode if the state doesn't exist. '''

        try:
            return self._greedy(state, available_modes, break_ties_randomly=False)
        except (KeyError, TypeError):
            return "pedestrian"

    def update_policy(self, state, action, reward, next_state=None, print_tabel=False):
        '''
        The Bellman Equation Update for this agent.
        "Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]" '''

        if self._current_iteration >= self.max_training_iteration:
            return

        current_q_value = self._q_table[state][action]

        if next_state is None:
            target = reward  # No future update required
        # Get Max Q(s', a') for the next state
        else:
            future_reward_value = max(self._q_table[next_state].values())
            # Discount the future reward to the current reward
            target = reward + self._gamma * future_reward_value

        # Update Rule: Q(s,a) = Q(s,a) + alpha * [Reward + gamma * MaxQ(s') - Q(s,a)]
        #alpha = self.get_alpha(state, action)
        self._q_table[state][action] += self._alpha * (target - current_q_value)
        new_q_value = self._q_table[state][action]

        # Incremental change in q value
        self._delta_q = abs(new_q_value - current_q_value)

        if print_tabel:
            self.print_q_table()

    def print_q_table(self):
        ''' Print the bit-encoded Q-table within the Command Prompt window using a clean tabular layout. '''

        width = 125
        print(f"\n{'=' * width}")
        print(f"{'PERSONAL Q-TABLE (20-BIT VECTOR VIEW): ' + str(self._id):^{width}}")
        print(f"{'=' * width}")

        # Header matching the exact 20-bit feature set
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

            # 1. Unpack bits according to the exact 20-bit breakdown
            curr_zone_bits = "".join(state_bits_string[0:4])
            tgt_zone_bits = "".join(state_bits_string[4:8])
            time_bits = "".join(state_bits_string[8:13])
            dist_bits = "".join(state_bits_string[13:16])
            asset_bits = "".join(state_bits_string[16:18])
            cong_bits = "".join(state_bits_string[18:20])

            # 2. Decode Values & Labels
            curr_zone = int(curr_zone_bits, 2)
            tgt_zone = int(tgt_zone_bits, 2)
            tau_val = int(time_bits, 2)

            dist_idx = int(dist_bits, 2)
            dist_label = {
                0: "0-1 km",
                1: "1-2 km",
                2: "2-5 km",
                3: "5-10 km",
                4: "10+ km"
            }.get(dist_idx, "UNKNOWN")

            asset_idx = int(asset_bits, 2)
            asset_label = {
                0: "NONE",
                1: "CAR ONLY",
                2: "BIKE ONLY",
                3: "CAR + BIKE"
            }.get(asset_idx, "UNKNOWN")

            cong_idx = int(cong_bits, 2)
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

        if (iteration + self._previous_iteration) > self.max_training_iteration:  # Test phase
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

            q_tables = saved_data.get("q_tables", {})
            memories = saved_data.get("memories", {})

            if not self._parse_loaded_data_for_agent(q_tables, memories):
                return False  # fresh start: the iteration offset stays 0

            # Iterations already explored by this Q-table; added to the current iteration in decay_epsilon()
            self._previous_iteration = saved_data["iteration"]

            # The Q-table is only meaningful under the parameters it was learned with: adopt them.
            self.apply_checkpoint_parameters(saved_data.get("config_arg", {}))
            self._epsilon = max(self._epsilon_min,
                                self._initial_epsilon * (self._epsilon_decay ** self._previous_iteration))

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

        if not q_table:
            print(f"PYTHON SERVICE: No saved Q-table for agent {self._id}. Starting fresh.")
            return False

        for state, action_values in q_table.items():
            if not isinstance(action_values, dict):
                print(f"PYTHON SERVICE: Skipping invalid state {state!r}.")
                continue

            self._q_table[state] = {}

            for action, q_value in action_values.items():
                self._q_table[state][action] = q_value

        if memory:
            self._memory.load_data(memory)

        return True