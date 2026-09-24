import random
import pickle
import os

class DecentralizedQLearningAgent:

    RANDOM_SEED = 42

    def __init__(self, mode_list, alpha, gamma, initial_epsilon, epsilon_decay, epsilon_minimum, max_iteration=100):
        self._alpha = alpha                            # Learning rate
        self._gamma = gamma                            # Discount factor
        self._initial_epsilon = initial_epsilon        # Epsilon-greedy factor
        self._epsilon_min = epsilon_minimum
        self._epsilon_decay = epsilon_decay
        self._model_file = "model.pkl"
        self.max_training_iteration = max_iteration
        self.modes = mode_list
        
        # Initialize an empty Q-Table
        self._q_table = {}

        random.seed(self.RANDOM_SEED)

    @property
    def q_table(self):
        return self._q_table
    
    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, modes_list):
        self._modes = modes_list

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, path):
        self._file_path = path

    @property
    def q_table(self):
        return self._q_table
    
    @property
    def delta_q(self):
        return self._delta_q
    
    @property
    def epsilon(self):
        return self._epsilon


    def init_state(self, agent_id, state_tuple):
        ''' Initializes the state and action pair in the Q-table. '''

        if agent_id not in self._q_table:
            self._q_table[agent_id] = {}

        if state_tuple not in self._q_table[agent_id]:
            # Initialize all possible modes
            self._q_table[agent_id][state_tuple] = {mode: 600.0 for mode in self.modes}

    def save_q_table(self):
        ''' Saves the Q-table to a file using pickle '''

        try:
            with open(self._file_path, 'wb') as f:
                pickle.dump(self._q_table, f)
            print(f"PYTHON SERVICE: Q-Table successfully saved to {self._file_path}.")
        except Exception as e:
            print(f"PYTHON SERVICE: Failed to save Q-Table: {e}.")

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

    def decay_epsilon(self, iteration):
        ''' Updates the epsilon value for the current iteration. ''' 
        self._epsilon = max(self._epsilon_min, self._initial_epsilon * (self._epsilon_decay ** iteration))
        self._current_iteration = iteration
        
    def choose_action(self, agent_id, state, available_modes):
        ''' Epsilon-Greedy selection for mode choices. ''' 
        
        agent_q_table = self._q_table[agent_id]

        # Get the greedy epsilon value
        current_epsilon = self._epsilon
        current_iteration = self._current_iteration

        # Exploration (Random)
        if random.random() < current_epsilon and current_iteration < self.max_training_iteration:
            action = random.choice(available_modes)
        # Exploitation
        else:
            agent_policy = agent_q_table[state]
            # Filter policy for only modes available for the agent
            filtered_policy = {mode: agent_policy[mode] for mode in available_modes}
            action = max(filtered_policy, key=filtered_policy.get)

        return action

    def update_policy(self, agent_id, state, action, reward, next_state=None, print_tabel=False):
        ''' 
        The Bellman Equation Update 
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)] '''

        if self._current_iteration >= self.max_training_iteration:
            return
        
        agent_q_table = self._q_table[agent_id]

        old_q = agent_q_table[state][action]
        
        # Get current Q(s,a)
        current_q = agent_q_table[state][action]

        if next_state is None:
            target = reward # No furture update required
        # Get Max Q(s', a') for the next state
        else:
            future_reward_value = max(self._q_table[agent_id][next_state].values())
            # Discount the future reward to the current reward
            target = reward + self._gamma * future_reward_value

        # Update Rule: Q(s,a) = Q(s,a) + alpha * [Reward + gamma * MaxQ(s') - Q(s,a)]
        agent_q_table[state][action] += self._alpha * (target - current_q)

        new_q = agent_q_table[state][action]

        # Incremental change in q value
        self._delta_q = abs(new_q - old_q)

        if print_tabel:
            self.print_q_table(agent_id)

    def print_q_table(self, agent_id):
        ''' Print the bit-encoded Q-table within the Command Prompt window using a grid layout. '''
        
        # Check if agent exists in Q-table
        if agent_id not in self._q_table or not self._q_table[agent_id]:
            return

        # Width tailored for tighter, packed columns
        width = 118

        print(f"\n{'=' * width}")
        print(f"{'PERSONAL Q-TABLE (GRID VIEW): ' + str(agent_id):^{width}}")
        print(f"{'=' * width}")

        # The new optimized header
        header = (f"{'DEPARTURE GRID':<18} | {'ARRIVAL GRID':<18} | "
                  f"{'DEP BIN':<8} | {'FLEXIBILITY':<12} | {'ASSETS':<12} || "
                  f"{'CAR':<7} | {'PT':<7} | {'BIKE':<7} | {'WALK':<7}")
        print(header)
        print("-" * width)

        sorted_table = sorted(self._q_table[agent_id].items(), key=lambda x: x[0])

        for state_bits, actions in sorted_table:
            # Emergency fallback filter for short/terminal states
            if "TERMINAL" in state_bits or len(state_bits) < 151:
                terminal_row = (f"{'TERMINAL STATE':<18} | {' ':<18} | "
                                f"{'-':<8} | {'-':<12} | {'TERMINAL':<12} || "
                                f"{actions.get('car', 0.0):>7.2f} | {actions.get('pt', 0.0):>7.2f} | "
                                f"{actions.get('bike', 0.0):>7.2f} | {actions.get('pedestrian', 0.0):>7.2f}")
                print(terminal_row)
                print("." * width)
                continue

            # 1. Unpack the components
            dep = state_bits[0:64]
            arr = state_bits[64:128]
            tme = state_bits[128:146]
            flx = state_bits[146]
            ast = state_bits[147:151]

            # 2. Decode Labels
            tau_val = tme.find('1')
            flex_label = "FLEXIBLE" if flx == '1' else "CONSTRAINED"
            
            asset_idx = ast.find('1')
            asset_label = {0: "NONE", 1: "CAR ONLY", 2: "BIKE ONLY", 3: "CAR + BIKE"}.get(asset_idx, "NONE")

            # 3. Slice bits into a 4x16 Matrix (4 chunks of 16 bits each)
            d_chunks = [dep[0:16], dep[16:32], dep[32:48], dep[48:64]]
            a_chunks = [arr[0:16], arr[16:32], arr[32:48], arr[48:64]]

            # 4. Print 4 unified rows combining the Grid segments and Metrics
            # Row 1 contains the core labels and values
            row1 = (f"{d_chunks[0]:<18} | {a_chunks[0]:<18} | "
                    f"{str(tau_val):<8} | {flex_label:<12} | {asset_label:<12} || "
                    f"{actions.get('car', 0.0):>7.2f} | "
                    f"{actions.get('pt', 0.0):>7.2f} | "
                    f"{actions.get('bike', 0.0):>7.2f} | "
                    f"{actions.get('pedestrian', 0.0):>7.2f}")
            print(row1)

            # Row 2, 3, and 4 continue the bit grids while leaving metrics columns clean
            row2 = f"{d_chunks[1]:<18} | {a_chunks[1]:<18} | {' ':<8} | {' ':<12} | {' ':<12} || {' ' * 37}"
            print(row2)

            row3 = f"{d_chunks[2]:<18} | {a_chunks[2]:<18} | {' ':<8} | {' ':<12} | {' ':<12} || {' ' * 37}"
            print(row3)

            row4 = f"{d_chunks[3]:<18} | {a_chunks[3]:<18} | {' ':<8} | {' ':<12} | {' ':<12} || {' ' * 37}"
            print(row4)
            
            # Section break
            print("." * width)

        print(f"{'=' * width}\n")