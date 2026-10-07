import random
import pickle
import os

class DecentralizedQLearningAgent:

    RANDOM_SEED = 42
    _alpha = 0.1
    _gamma = 0.9
    _initial_epsilon = 1.0
    _epsilon_min = 0.01
    _epsilon_decay = 0.99
    _output_directory = ""
    _file_path = ""
    max_training_iteration = 100
    _modes = []
    
    @classmethod
    def configure_global_parameters(cls, config_args: dict):
        """
        Dynamically configures class-level shared parameters from any arbitrary dictionary 
        passed by MATSim/Java, without needing to know every key ahead of time.
        """
        config = config_args or {}

        # Extract known parameters safely with default fallbacks
        cls._alpha = float(config.get("alpha", cls._alpha))
        cls._gamma = float(config.get("gamma", cls._gamma))
        cls._initial_epsilon = float(config.get("epsilon", cls._initial_epsilon))
        cls._epsilon_min = float(config.get("epsilonMinimum", cls._epsilon_min))
        cls._epsilon_decay = float(config.get("epsilonDecay", cls._epsilon_decay))
        cls.max_training_iteration = int(config.get("trainingCutoffIteration", cls.max_training_iteration))
        cls._output_directory = str(config_args.get("outputDirectory"))
        
        # Parse available modes globally
        mode_string = config.get("modes", "")
        if isinstance(mode_string, str):
            cls._modes = [m.strip() for m in mode_string.split(",") if m.strip()]
        else:
            cls._modes = list(mode_string)

        random.seed(cls.RANDOM_SEED)

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

        for key, value in args.items():
            setattr(self, key, value)

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

    def init_state(self, state_tuple):
        ''' Initializes the state and action pair in this agent's Q-table. '''

        if state_tuple not in self._q_table:
            # Initialize all possible modes
            self._q_table[state_tuple] = {mode: 600.0 for mode in self._modes}

    def decay_epsilon(self, iteration):
        ''' Updates the epsilon value for the current iteration. '''
        self._current_iteration = iteration 
        self._epsilon = max(self._epsilon_min, self._initial_epsilon * (self._epsilon_decay ** iteration))
        
    def choose_action(self, state, available_modes):
        ''' Epsilon-Greedy selection for mode choices for this specific agent. ''' 
        # Exploration (Random)
        if random.random() < self._epsilon and self._current_iteration < self.max_training_iteration:
            return random.choice(available_modes)
        # Exploitation
        else:
            agent_policy = self._q_table[state]
            filtered_policy = {mode: agent_policy[mode] for mode in available_modes}
            return max(filtered_policy, key=filtered_policy.get)

    def update_policy(self, state, action, reward, next_state=None, print_tabel=False):
        ''' 
        The Bellman Equation Update for this agent.
        "Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]" '''

        if self._current_iteration >= self.max_training_iteration:
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
        ''' Print the bit-encoded Q-table within the Command Prompt window using a grid layout. '''

        # Width tailored for tighter, packed columns
        width = 118

        print(f"\n{'=' * width}")
        print(f"{'PERSONAL Q-TABLE (GRID VIEW): ' + str(self._id):^{width}}")
        print(f"{'=' * width}")

        # The new optimized header
        header = (f"{'DEPARTURE GRID':<18} | {'ARRIVAL GRID':<18} | "
                  f"{'DEP BIN':<8} | {'ASSETS':<12} || "
                  f"{'CAR':<7} | {'PT':<7} | {'BIKE':<7} | {'WALK':<7}")
        print(header)
        print("-" * width)

        for state_bits, actions in self._q_table.items():
            state_bits_string = [str(bit) for bit in state_bits]

            if "TERMINAL" in state_bits or len(state_bits) < 136:
                terminal_row = (f"{'TERMINAL STATE':<18} | {' ':<18} | "
                                f"{'-':<8} | {'TERMINAL':<12} || "
                                f"{actions.get('car', 0.0):>7.2f} | {actions.get('pt', 0.0):>7.2f} | "
                                f"{actions.get('bike', 0.0):>7.2f} | {actions.get('pedestrian', 0.0):>7.2f}")
                print(terminal_row)
                print("." * width)
                continue

            # 1. Unpack the components
            dep = state_bits_string[0:64]
            arr = state_bits_string[64:128]
            #flx = state_bits_string[133]
            tme = "".join(state_bits_string[128:133])
            ast = "".join(state_bits_string[134:136])

            # 2. Decode Labels
            tau_val = int(tme, 2)
            asset_idx = int(ast, 2)
            asset_label = {0: "NONE", 1: "CAR ONLY", 2: "BIKE ONLY", 3: "CAR + BIKE"}.get(asset_idx, "NONE")
            #flex_label = "FLEXIBLE" if flx == '1' else "CONSTRAINED"

            # 3. Slice bits into a 4x16 Matrix (4 chunks of 16 bits each)
            d_chunks = ["".join(state_bits_string[0:16]), "".join(state_bits_string[16:32]), "".join(state_bits_string[32:48]), "".join(state_bits_string[48:64])]
            a_chunks = ["".join(state_bits_string[64:80]), "".join(state_bits_string[80:96]), "".join(state_bits_string[96:112]), "".join(state_bits_string[112:128])]

            # 4. Print 4 unified rows combining the Grid segments and Metrics
            # Row 1 contains the core labels and values
            row1 = (f"{d_chunks[0]:<18} | {a_chunks[0]:<18} | "
                    f"{str(tau_val):<8} | {asset_label:<12} || "
                    f"{actions.get('car', 0.0):>7.2f} | "
                    f"{actions.get('pt', 0.0):>7.2f} | "
                    f"{actions.get('bike', 0.0):>7.2f} | "
                    f"{actions.get('pedestrian', 0.0):>7.2f}")
            print(row1)

            # Row 2, 3, and 4 continue the bit grids while leaving metrics columns clean
            row2 = f"{d_chunks[1]:<18} | {a_chunks[1]:<18} | {' ':<8} | {' ':<12} || {' ' * 37}"
            print(row2)

            row3 = f"{d_chunks[2]:<18} | {a_chunks[2]:<18} | {' ':<8} | {' ':<12} || {' ' * 37}"
            print(row3)

            row4 = f"{d_chunks[3]:<18} | {a_chunks[3]:<18} | {' ':<8} | {' ':<12} || {' ' * 37}"
            print(row4)
            
            # Section break
            print("." * width)

        print(f"{'=' * width}\n")

    def dump(self, iteration):
        ''' Saves the Q-table to a file using pickle '''

        if iteration > self.max_training_iteration:
            save_directoy = self._output_directory
        else:    
            save_directory = f"{self._output_directory}/ITERS/it.{iteration}"

        print(f"The directory to save the table is: {save_directory}")
        
        try:
            # Ensure the directory exists before dumping
            os.makedirs(save_directory, exist_ok=True)
            save_file_path = os.path.join(save_directory, "q_table.pkl")
            
            with open(save_file_path, 'wb') as f:
                pickle.dump(self._q_table, f)
                
            print(f"PYTHON SERVICE: Q-Table successfully saved to {save_directory}.")
        except Exception as e:
            print(f"PYTHON SERVICE: Failed to save Q-Table: {e}")




    def load_models(self, load_file_path):
        ''' Loads decentralized tables from file. '''
        if os.path.exists(load_file_path):
            try:
                with open(load_file_path, 'rb') as f:
                    self._agent_tables = pickle.load(f)
                print(f"PYTHON SERVICE: Loaded tables for {len(self._agent_tables)} unique agents.")
            except Exception as e:
                print(f"PYTHON SERVICE: Failed to load: {e}.")
        else:
            print("PYTHON SERVICE: No model found. Starting decentralized training fresh.")