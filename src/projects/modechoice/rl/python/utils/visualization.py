import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

from pathlib import Path

sns.set()

class IterationProgression:

    MAX_ITERATION = 600
    MOVING_WINDOW = 30

    def __init__(self, output_directory_path):
        self._output_directory_path = output_directory_path

    def plot_cumalative_reward_progression(self):

        csv_file = os.path.join(self._output_directory_path, 'mode_choice_results.csv')

        cumalative_reward_df = pd.read_csv(csv_file)
        reward_per_iteration = cumalative_reward_df[cumalative_reward_df['Iteration'] <= self.MAX_ITERATION].copy()

        # Converting to numpy arrays for calculation
        iterations = reward_per_iteration['Iteration'].values
        rewards = reward_per_iteration['Total Reward'].values

        # Generate weighted_moving_average values and rolling Standard Deviation
        weighted_moving_average_trend = self.calculate_weighted_moving_average(rewards, window=self.MOVING_WINDOW)
        rolling_std = reward_per_iteration['Total Reward'].rolling(window=self.MOVING_WINDOW, min_periods=1).std().values

        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

        # A. Raw Needle Spikes (Raw Reward data)
        ax.plot(iterations, rewards, color='#488AC7', alpha=0.2, linewidth=0.6)

        # B. Shaded Performance Variance (using weighted_moving_average and Rolling STD)
        ax.fill_between(iterations, weighted_moving_average_trend - rolling_std, weighted_moving_average_trend + rolling_std, 
                        color='#488AC7', alpha=0.1, zorder=2)

        # C. Bold Weighted Moving Average Line
        ax.plot(iterations, weighted_moving_average_trend, color='#488AC7', linewidth=2.5, 
                label='Weighted Moving Average (weighted_moving_average)', zorder=4)

        # D. Convergence Marker at Iteration 157
        ax.axvline(x=283, color='#27AE60', linestyle='--', linewidth=2, zorder=5)

        # --- 4. FORMATTING & ANNOTATIONS ---
        ax.text(78, 95, "Optimization Phase", color="#848D7F", fontsize=16, fontstyle='italic', ha='center')
        ax.text(228, 95, "Steady State (Exploitation)", color='#848D7F', fontsize=16, fontstyle='italic', ha='center')

        # Axis Labels and Title
        ax.set_title('Agent Mode Choice Convergence (weighted_moving_average Trend)', fontsize=22, pad=15, fontweight='bold')
        ax.set_ylabel('Cumulative Score (Utility)', fontsize=18)
        ax.set_xlabel('Iteration', fontsize=18)

        # Tick font sizes and Grid
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_yticks(np.arange(100, 350, 50))
        ax.yaxis.grid(True, linestyle='--', color='#DDDDDD', linewidth=0.8)

        # Aesthetic Cleanup
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')

        ax.set_xlim(0, self.MAX_ITERATION)
        ax.set_ylim(80, 320)

        plt.tight_layout()
        plt.show()

    def plot_example_curve(self, csv_file=None):
        
        np.random.seed(42)
        iterations = np.arange(0, self.MAX_ITERATION+1)
        
        # --- 2. LOGIC FOR CONVERGENCE AT 157 ---
        rewards = []
        for i in iterations:
            if i < 40:
                base = np.random.choice([152, 187, 214])
                noise = np.random.normal(0, 20)
            elif i < 157:
                progress = (i - 40) / (157 - 40)
                base = 152 + (257 - 152) * progress
                noise = np.random.normal(0, 8)
                if np.random.random() > 0.92: base -= 30 
            else:
                base = 257 
                noise = np.random.normal(0, 3) 
                if np.random.random() > 0.98: noise -= 15 
            rewards.append(base + noise)

        df = pd.DataFrame({'Iteration': iterations, 'Reward': rewards})
        df['SMA'] = df['Reward'].rolling(window=10, min_periods=1).mean()
        df['STD'] = df['Reward'].rolling(window=10, min_periods=1).std()

        # --- 3. THE PLOT (12x4 Wide) ---
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

        # Raw Reward "Needle" Spikes
        ax.plot(df['Iteration'], df['Reward'], color='#488AC7', alpha=0.2, linewidth=0.6)

        # Shaded Variance
        ax.fill_between(df['Iteration'], df['SMA'] - df['STD'], df['SMA'] + df['STD'], 
                        color='#488AC7', alpha=0.1, zorder=2)

        # Main Bold Trend Line
        ax.plot(df['Iteration'], df['SMA'], color='#488AC7', linewidth=2.5, zorder=4)

        # --- 4. CONVERGENCE MARKER AT 157 ---
        ax.axvline(x=157, color='#27AE60', linestyle='--', linewidth=2, zorder=5)

        # Phase Labels (increased size to 16 as requested)
        ax.text(78, 95, "Optimization Phase", color="#848D7F", fontsize=16, fontstyle='italic', ha='center')
        ax.text(228, 95, "Steady State (Exploitation)", color='#848D7F', fontsize=16, fontstyle='italic', ha='center')

        # --- 5. CLEANING & FORMATTING ---
        ax.set_title('Agent Mode Choice Convergence', fontsize=22, pad=15)
        ax.set_ylabel('Cumulative Score (Utility)', fontsize=18)
        ax.set_xlabel('Iteration', fontsize=18)

        # INCREASE FONT OF NUMBERS (Ticks)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # GRID: Horizontal line every 50 reward units
        # We manually set the ticks and then enable the grid
        ax.set_yticks(np.arange(100, 350, 50))
        ax.yaxis.grid(True, linestyle='--', color='#DDDDDD', linewidth=0.8)
        ax.xaxis.grid(False) # Clean look

        # Spine Cleanup
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        ax.set_xlim(0, self.MAX_ITERATION)
        ax.set_ylim(80, 320)

        plt.tight_layout()
        plt.show()

    def reward_dataframe(self):
        # Setup the path to the ITERS folder
        iteration_directory = Path(self._output_directory_path) / "ITERS"
        
        if not iteration_directory.exists():
            print(f"Error: {iteration_directory} does not exist.")
            return None

        # Find all subfolders starting with 'it.'
        iteration_folders = [f for f in iteration_directory.iterdir() if f.is_dir() and f.name.startswith('it.')]
        print(f"Found {len(iteration_folders)} iteration folders.")

        data_list = []

        reward_pattern = re.compile(r"TOTAL DAY REWARD.*[:\s]+([-+]?\d*\.\d+|\d+)")

        for folder in sorted(iteration_folders, key=lambda x: int(x.name.split('.')[-1])):
            iteration_num = int(folder.name.split('.')[-1])
            log_file = folder / "simulation_history.log"

            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Find all rewards in the file
                    rewards = reward_pattern.findall(content)
                    
                    # Convert strings to floats
                    rewards = [float(r) for r in rewards]
                    
                    if rewards:
                        # We take the average reward for this iteration or sum them
                        avg_reward = sum(rewards) / len(rewards)
                        data_list.append({
                            "iteration": iteration_num,
                            "total_reward": avg_reward,
                            "trip_count": len(rewards)
                        })
            else:
                print(f"Warning: No log file found in {folder.name}")

        # Create DataFrame
        df = pd.DataFrame(data_list)
        return df
    
    def calculate_weighted_moving_average(self, data, window=10):
        weights = np.arange(1, window + 1) # [1, 2, ..., 10]
        weighted_moving_average = []
        
        for i in range(len(data)):
            if i < window - 1:
                # For initial iterations where history is short, use SMA
                weighted_moving_average.append(np.mean(data[:i+1]))
            else:
                # Apply weighted calculation to the window
                sub_data = data[i - window + 1 : i + 1]
                weighted_moving_average.append(np.sum(sub_data * weights) / weights.sum())
        return np.array(weighted_moving_average)

# --- EXECUTION ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.facecolor'] = 'white'

project_root = "scenarios\\sioux-falls\\modified\\output"
iteration_progression = IterationProgression(project_root)
iteration_progression.plot_cumalative_reward_progression()
#iteration_progression.plot_example_curve()
