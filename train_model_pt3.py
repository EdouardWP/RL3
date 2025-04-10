"""
Training script for the Rocket environment using Stable-Baselines3.
This script trains a model for the 'starship' rocket to land.

Features:
- Fresh training from scratch
- Continue training from a checkpoint
- Improved rewards for precision landing
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

import gymnasium as gym
from gymnasium import spaces
from rocket_env import RocketEnv
from rocket import Rocket

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# Directory setup
log_dir = "sb3_rocket_logs/"
model_dir = "sb3_rocket_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Training configuration - can be modified via command line
# Continue from checkpoint
CONTINUE_TRAINING = False
CHECKPOINT_TO_LOAD = None  # Use the latest if None
# Use improved rewards for precision landing
USE_IMPROVED_REWARDS = False
# Total timesteps for training (less if continuing)
TOTAL_TIMESTEPS = 5_000_000
ADDITIONAL_TIMESTEPS = 1_000_000

#-------------------
# Improved Rewards Implementation 
#-------------------

class ImprovedRocket(Rocket):
    """
    Extension of the Rocket class with improved reward function for landing precision.
    Overrides the calculate_reward method to incorporate better incentives.
    """
    
    def __init__(self, *args, **kwargs):
        # Store previous state for trajectory smoothness calculation
        self.prev_state = None
        super(ImprovedRocket, self).__init__(*args, **kwargs)
    
    def reset(self, state_dict=None):
        """Reset the environment and clear previous state."""
        self.prev_state = None
        return super().reset(state_dict)
    
    def calculate_reward(self, state):
        """
        Calculate improved reward that encourages:
        1. Precise landing (exponential distance scaling)
        2. Early orientation correction (instead of last-second)
        3. Smooth trajectory
        4. Appropriate descent velocity
        5. Moving toward target (not away from it)
        6. Initial heading guidance toward target
        """
        # Get the base reward from the original function
        base_reward = super().calculate_reward(state)
        
        # Additional reward components
        additional_reward = 0.0
        
        # Get distance to target
        dist_to_target = ((state['x'] - self.target_x)**2 + (state['y'] - self.target_y)**2)**0.5
        
        # 1. Exponential distance scaling - more reward when very close to target
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        
        dist_x = abs(state['x'] - self.target_x)
        dist_y = abs(state['y'] - self.target_y)
        dist_norm = dist_x / x_range + dist_y / y_range
        
        # Exponential scaling that gives much higher reward when very close
        dist_factor = np.exp(-dist_norm * 5)
        precision_reward = 0.1 * dist_factor
        additional_reward += precision_reward
        
        # 2. Orientation-distance coupling - encourage being upright earlier when close
        orientation_factor = 1.0 - abs(state['theta']) / (0.5*np.pi)
        
        # More reward for being upright when close to target
        # Stronger effect when closer to target
        proximity_factor = np.exp(-dist_to_target / 100.0)
        approach_reward = 0.1 * orientation_factor * proximity_factor
        additional_reward += approach_reward
        
        # 3. Trajectory smoothness - penalize rapid changes
        if self.prev_state is not None:
            # Calculate change in control inputs
            f_change = abs(state['f'] - self.prev_state['f'])
            phi_change = abs(state['phi'] - self.prev_state['phi'])
            
            # Reward smoothness (less change is better)
            smoothness_reward = 0.02 * (1.0 - min(1.0, (f_change + phi_change) / 2.0))
            additional_reward += smoothness_reward
        
        # 4. Velocity matching - reward appropriate descent velocity based on height
        if self.task == 'landing':
            # Ideal descent speed increases with height (slower near ground)
            height_above_ground = state['y'] - self.H/2.0
            ideal_vy = -min(5.0, max(1.0, height_above_ground / 20))
            velocity_diff = abs(state['vy'] - ideal_vy)
            velocity_reward = 0.05 * np.exp(-velocity_diff)
            additional_reward += velocity_reward
            
        # 5. Direction toward target - reward moving toward target, penalize moving away
        if self.prev_state is not None:
            # Calculate if we're getting closer to or further from the target
            prev_dist = ((self.prev_state['x'] - self.target_x)**2 + (self.prev_state['y'] - self.target_y)**2)**0.5
            curr_dist = ((state['x'] - self.target_x)**2 + (state['y'] - self.target_y)**2)**0.5
            
            # Reward getting closer, penalize getting further
            moving_toward_target = prev_dist - curr_dist
            
            # Scale reward based on distance (larger effect when further away)
            far_scale = min(1.0, dist_to_target / 300.0)  # Scale up to 300m away
            direction_reward = 0.2 * np.tanh(moving_toward_target * 0.1) * (0.5 + 0.5 * far_scale)
            additional_reward += direction_reward
            
            # STRONGER penalty for moving away from target with high velocity
            vx_toward_target = np.sign(self.target_x - state['x']) * state['vx']
            if vx_toward_target < 0:  # Moving away horizontally
                # Increase penalty and make it more severe when far from target
                heading_penalty = -0.2 * abs(vx_toward_target) / 10.0 * (0.5 + 0.5 * far_scale)
                additional_reward += heading_penalty
                
        # 6. NEW: Initial heading guidance - encourage immediately moving toward target
        # Calculate ideal heading vector (normalized)
        dx = self.target_x - state['x']
        dy = self.target_y - state['y']
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            ideal_heading_x = dx / dist
            ideal_heading_y = dy / dist
            
            # Current velocity direction (normalized)
            vel_mag = np.sqrt(state['vx']**2 + state['vy']**2)
            if vel_mag > 0:
                current_heading_x = state['vx'] / vel_mag
                current_heading_y = state['vy'] / vel_mag
                
                # Dot product gives alignment (-1 to 1)
                heading_alignment = ideal_heading_x * current_heading_x + ideal_heading_y * current_heading_y
                
                # Transform to 0-1 range and reward
                heading_alignment = (heading_alignment + 1) / 2.0
                
                # Scale by distance - more important when far away
                far_effect = min(1.0, dist_to_target / 400.0)  # Scale up to 400m
                heading_reward = 0.15 * heading_alignment * (0.2 + 0.8 * far_effect)
                additional_reward += heading_reward
                
                # Extra penalty for heading completely wrong way (more than 90 degrees off)
                if heading_alignment < 0:
                    wrong_way_penalty = -0.1 * abs(heading_alignment) * far_effect
                    additional_reward += wrong_way_penalty
        
        # Store current state for next comparison
        self.prev_state = state.copy()
        
        # Combine base reward with additional components
        return base_reward + additional_reward
        
    def step(self, action):
        """Take a step and calculate the improved reward."""
        next_state, _, done, info = super().step(action)
        reward = self.calculate_reward(self.state)
        return next_state, reward, done, info


class ImprovedRocketEnv(RocketEnv):
    """
    Extends RocketEnv to use the ImprovedRocket with better reward function.
    All other functionality remains the same.
    """
    
    def __init__(self, task='landing', rocket_type='starship', max_steps=800, render_mode=None):
        """Initialize with ImprovedRocket instead of standard Rocket."""
        # Skip the parent __init__ and call grandparent __init__ directly
        gym.Env.__init__(self)
        
        # Create the improved rocket simulation
        self.rocket = ImprovedRocket(max_steps=max_steps, task=task, rocket_type=rocket_type)
        self.task = task
        self.rocket_type = rocket_type
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Define action and observation spaces (same as parent)
        self.action_space = spaces.Discrete(self.rocket.action_dims)
        
        # Determine observation space bounds
        low_bounds = np.array([
            self.rocket.world_x_min,             # x position
            self.rocket.world_y_min,             # y position
            -100.0,                              # vx velocity
            -100.0,                              # vy velocity
            -np.pi,                              # theta orientation
            -np.pi,                              # vtheta angular velocity
            -np.pi/2,                            # phi nozzle angle
            -0.01                                # f thrust (allow small negative values)
        ], dtype=np.float32)
        
        high_bounds = np.array([
            self.rocket.world_x_max,             # x position
            self.rocket.world_y_max,             # y position
            100.0,                               # vx velocity
            100.0,                               # vy velocity
            np.pi,                               # theta orientation
            np.pi,                               # vtheta angular velocity
            np.pi/2,                             # phi nozzle angle
            3.0 * self.rocket.g                  # f thrust (max in action table)
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        # For visualization
        self.last_frame = None


# Custom callback for visualization every 100 episodes
class VisualizeCallback(BaseCallback):
    def __init__(self, verbose=0, render_freq=100, use_improved_rewards=False):
        super(VisualizeCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.episode_count = 0
        self.vis_env = None
        self.use_improved_rewards = use_improved_rewards
    
    def _on_training_start(self):
        # Create a separate environment for visualization
        if self.use_improved_rewards:
            self.vis_env = ImprovedRocketEnv(task='landing', rocket_type='starship', render_mode='human')
        else:
            self.vis_env = RocketEnv(task='landing', rocket_type='starship', render_mode='human')
        
    def _on_step(self):
        # Check if episode is done
        dones = self.locals.get('dones')
        if dones is not None and any(dones):
            self.episode_count += 1
            
            # Visualize every render_freq episodes
            if self.episode_count % self.render_freq == 0:
                print(f"\nVisualizing episode {self.episode_count}...")
                obs, _ = self.vis_env.reset()
                done = False
                total_reward = 0
                
                # Use the current model to play one episode
                while not done:
                    # If using VecNormalize, normalize the observation
                    if hasattr(self.model.env, 'normalize_obs'):
                        norm_obs = self.model.env.normalize_obs(np.array([obs]))
                        action, _ = self.model.predict(norm_obs, deterministic=True)
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Take the action
                    obs, reward, terminated, truncated, _ = self.vis_env.step(action.item())
                    total_reward += reward
                    done = terminated or truncated
                    
                    # Small delay to make visualization viewable
                    time.sleep(0.01)
                
                print(f"Visualization episode reward: {total_reward:.2f}")
        
        return True
    
    def _on_training_end(self):
        if self.vis_env is not None:
            self.vis_env.close()

# Callback for saving evaluations during training
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1, file_prefix=""):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_rewards = []
        self.evaluations_timesteps = []
        self.file_prefix = file_prefix
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            self.evaluations_rewards.append(mean_reward)
            self.evaluations_timesteps.append(self.n_calls)
            
            # Log the evaluation
            if self.verbose > 0:
                print(f"Evaluation at step {self.n_calls}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
                
            # Save the evaluation results
            np.savez(
                f"{log_dir}{self.file_prefix}evaluations.npz",
                timesteps=self.evaluations_timesteps,
                rewards=self.evaluations_rewards
            )
            
            # Plot the evaluation results
            plt.figure(figsize=(10, 6))
            plt.plot(self.evaluations_timesteps, self.evaluations_rewards)
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Reward")
            plt.title(f"Evaluation Rewards ({self.file_prefix})")
            plt.savefig(f"{log_dir}{self.file_prefix}eval_rewards.png")
            plt.close()
        
        return True

def make_env(use_improved_rewards=False):
    """Create and wrap the environment."""
    if use_improved_rewards:
        env = ImprovedRocketEnv(task='landing', rocket_type='starship', render_mode=None)
    else:
        env = RocketEnv(task='landing', rocket_type='starship', render_mode=None)
    env = Monitor(env, log_dir)
    return env

def find_latest_checkpoint():
    """Find the latest checkpoint in the model directory."""
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("ppo_") and f.endswith(".zip")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    
    # Sort by step number
    # Handle different filename formats - extract the step number more safely
    def extract_step_number(filename):
        try:
            # Try to extract a number from the filename
            # Match the last number before .zip
            parts = filename.split('_')
            for part in reversed(parts):
                if part.endswith('.zip'):
                    num_part = part.split('.')[0]
                    if num_part.isdigit():
                        return int(num_part)
                elif part.isdigit():
                    return int(part)
            # If no number is found, use the filename for lexical sort
            return filename
        except:
            # In case of any error, fall back to string comparison
            return filename
    
    checkpoints.sort(key=extract_step_number)
    return checkpoints[-1]

def train_from_scratch(use_improved_rewards=False):
    """Train a model from scratch using PPO algorithm."""
    # Create the vectorized environment
    env = DummyVecEnv([lambda: make_env(use_improved_rewards)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = make_env(use_improved_rewards)
    
    reward_type = "improved" if use_improved_rewards else "standard"
    print(f"Starting fresh training with {reward_type} rewards...")
    
    # Set up the callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=f"ppo_{reward_type}"
    )
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        file_prefix=f"{reward_type}_"
    )
    
    visualize_callback = VisualizeCallback(
        render_freq=100,  # Visualize every 100 episodes
        use_improved_rewards=use_improved_rewards
    )
    
    # Create and train the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, visualize_callback]
    )
    
    # Save the final model
    model.save(f"{model_dir}{reward_type}_final_model")
    env.save(f"{model_dir}{reward_type}_vec_normalize.pkl")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, env

def continue_training(checkpoint_path=None, use_improved_rewards=False):
    """Continue training from a checkpoint."""
    # Find the checkpoint to load
    if checkpoint_path is None:
        checkpoint_file = CHECKPOINT_TO_LOAD if CHECKPOINT_TO_LOAD else find_latest_checkpoint()
        checkpoint_path = os.path.join(model_dir, checkpoint_file)
    
    reward_type = "improved" if use_improved_rewards else "continued"
    print(f"Continuing training from: {checkpoint_path} with {reward_type} rewards")
    
    # Create the vectorized environment
    env = DummyVecEnv([lambda: make_env(use_improved_rewards)])
    
    # Try to load the saved VecNormalize statistics
    try:
        env = VecNormalize.load(f"{model_dir}vec_normalize.pkl", env)
        # Keep training flag to True to continue collecting running statistics
        env.training = True
        print("Loaded normalization statistics")
    except:
        print("Could not load normalization statistics, creating new ones")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Load the model
    model = PPO.load(checkpoint_path, env=env)
    
    # Use a lower learning rate when continuing training with improved rewards
    # This helps with stability when changing the reward function
    if use_improved_rewards:
        print("Adjusting learning rate for stability with improved rewards")
        model.learning_rate = 1e-4  # Lower learning rate for stability
    
    # Create evaluation environment
    eval_env = make_env(use_improved_rewards)
    
    # Set up the callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=f"ppo_{reward_type}"
    )
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        file_prefix=f"{reward_type}_"
    )
    
    visualize_callback = VisualizeCallback(
        render_freq=100,  # Visualize every 100 episodes
        use_improved_rewards=use_improved_rewards
    )
    
    # Continue training
    print(f"Continuing training for {ADDITIONAL_TIMESTEPS} additional timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, visualize_callback],
        reset_num_timesteps=False  # Don't reset the timestep counter
    )
    
    # Save the final model
    model.save(f"{model_dir}{reward_type}_final_model")
    env.save(f"{model_dir}{reward_type}_vec_normalize.pkl")
    
    training_time = time.time() - start_time
    print(f"Continued training completed in {training_time:.2f} seconds")
    
    return model, env

def test(model, env, num_episodes=5, use_improved_rewards=False):
    """Test the trained model."""
    # Create a test environment with rendering
    if use_improved_rewards:
        test_env = ImprovedRocketEnv(task='landing', rocket_type='starship', render_mode='human')
    else:
        test_env = RocketEnv(task='landing', rocket_type='starship', render_mode='human')
    
    # Test the model
    obs, _ = test_env.reset()
    
    rewards = []
    for episode in range(num_episodes):
        episode_reward = 0
        done = False
        obs, _ = test_env.reset()
        
        print(f"Testing episode {episode+1}/{num_episodes}")
        
        while not done:
            # VecNormalize normalization
            obs_normalized = env.normalize_obs(np.array([obs]))
            
            # Use the model to predict the action
            action, _ = model.predict(obs_normalized, deterministic=True)
            
            # Take the action
            obs, reward, terminated, truncated, _ = test_env.step(action.item())
            episode_reward += reward
            done = terminated or truncated
            
            # Add a small delay to make the rendering visible
            time.sleep(0.01)
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward:.2f}")
    
    test_env.close()
    
    # Print and save the results
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return rewards

def analyze_results(file_prefix=""):
    """Analyze and visualize the training results."""
    # Load evaluation results
    try:
        evaluations = np.load(f"{log_dir}{file_prefix}evaluations.npz")
        timesteps = evaluations['timesteps']
        rewards = evaluations['rewards']
        
        # Plot the evaluation rewards
        plt.figure(figsize=(12, 8))
        plt.plot(timesteps, rewards)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title(f"Evaluation Rewards ({file_prefix})")
        plt.savefig(f"{log_dir}{file_prefix}final_evaluation.png")
        plt.show()
        
        print(f"Analysis complete. Results saved to {log_dir}{file_prefix}final_evaluation.png")
    except:
        print(f"No evaluation data found for prefix '{file_prefix}'")
    
    # Load and plot the training rewards if available
    try:
        monitor_data = np.load(f"{log_dir}monitor.npz")
        steps = monitor_data['l']
        returns = monitor_data['r']
        
        plt.figure(figsize=(12, 8))
        plt.plot(np.cumsum(steps), returns, alpha=0.5)
        plt.plot(np.cumsum(steps), np.convolve(returns, np.ones(100)/100, mode='same'))
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title(f"Training Rewards ({file_prefix})")
        plt.legend(["Rewards", "Moving Average (100)"])
        plt.savefig(f"{log_dir}{file_prefix}training_rewards.png")
        plt.show()
    except:
        print("Monitor data not available for plotting")

def visualize_training(model_path, num_episodes=10, use_improved_rewards=False):
    """Visualize the model's performance on multiple episodes."""
    print(f"Visualizing model from {model_path}...")
    
    # Create environment with rendering
    if use_improved_rewards:
        env = ImprovedRocketEnv(task='landing', rocket_type='starship', render_mode='human')
    else:
        env = RocketEnv(task='landing', rocket_type='starship', render_mode='human')
    
    # Load the model
    model = PPO.load(model_path)
    
    # Run episodes
    rewards = []
    success_count = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Track if rocket gets closer to target
        initial_distance = None
        min_distance = float('inf')
        
        print(f"\nEpisode {episode+1}/{num_episodes}:")
        
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Calculate distance to target
            rocket = env.rocket
            distance = ((rocket.state['x'] - rocket.target_x)**2 + 
                        (rocket.state['y'] - rocket.target_y)**2)**0.5
            
            # Track initial and minimum distance
            if initial_distance is None:
                initial_distance = distance
            min_distance = min(min_distance, distance)
            
            # Add a small delay to make the rendering visible
            time.sleep(0.01)
        
        # Calculate distance improvement
        distance_improvement = initial_distance - min_distance
        improvement_percent = (distance_improvement / initial_distance) * 100 if initial_distance > 0 else 0
        
        # Check if landing was successful
        landing_success = False
        if hasattr(env.rocket, 'landed') and env.rocket.landed:
            landing_success = True
            success_count += 1
        
        rewards.append(episode_reward)
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Distance improvement: {improvement_percent:.1f}% (from {initial_distance:.1f}m to {min_distance:.1f}m)")
        print(f"  Landing successful: {landing_success}")
    
    env.close()
    
    # Print summary
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = (success_count / num_episodes) * 100
    
    print("\nVisualization Summary:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    
    return rewards

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a rocket landing model with various options')
    
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue training from a checkpoint')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint file to load (if not specified, will use the latest)')
    
    parser.add_argument('--improved-rewards', action='store_true',
                        help='Use improved rewards for precision landing')
    
    parser.add_argument('--timesteps', type=int, default=5_000_000,
                        help='Total timesteps for training from scratch')
    
    parser.add_argument('--additional-timesteps', type=int, default=1_000_000,
                        help='Additional timesteps when continuing training')
    
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze results without training')
    
    parser.add_argument('--test', action='store_true',
                        help='Test the model without training')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the model performance')
    
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to visualize or test')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the model to test or visualize')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Update global variables based on arguments
    CONTINUE_TRAINING = args.continue_training
    if args.checkpoint:
        CHECKPOINT_TO_LOAD = args.checkpoint
    USE_IMPROVED_REWARDS = args.improved_rewards
    TOTAL_TIMESTEPS = args.timesteps
    ADDITIONAL_TIMESTEPS = args.additional_timesteps
    
    # Check environment
    print("Checking environment...")
    if USE_IMPROVED_REWARDS:
        env = ImprovedRocketEnv(task='landing', rocket_type='starship')
    else:
        env = RocketEnv(task='landing', rocket_type='starship')
    check_env(env)
    env.close()
    print("Environment check passed!")
    
    # Determine what to do
    if args.analyze:
        # Just analyze results
        print("Analyzing results...")
        prefix = "improved_" if USE_IMPROVED_REWARDS else ""
        if CONTINUE_TRAINING:
            prefix = "continued_"
        analyze_results(prefix)
    elif args.test:
        # Just test the model
        if args.model_path:
            print(f"Testing model from {args.model_path}...")
            model = PPO.load(args.model_path)
            env = DummyVecEnv([lambda: make_env(USE_IMPROVED_REWARDS)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            test(model, env, use_improved_rewards=USE_IMPROVED_REWARDS)
        else:
            print("Please specify a model path with --model-path")
    elif args.visualize:
        # Visualize the model
        visualize_training(args.model_path, args.episodes, USE_IMPROVED_REWARDS)
    else:
        # Train or continue training
        if CONTINUE_TRAINING:
            # Continue from checkpoint
            model, env = continue_training(
                checkpoint_path=args.checkpoint, 
                use_improved_rewards=USE_IMPROVED_REWARDS
            )
        else:
            # Train from scratch
            model, env = train_from_scratch(
                use_improved_rewards=USE_IMPROVED_REWARDS
            )
        
        # Test the trained model
        print("\nTesting the trained model...")
        test_rewards = test(model, env, use_improved_rewards=USE_IMPROVED_REWARDS)
        
        # Analyze the results
        print("\nAnalyzing results...")
        prefix = "improved_" if USE_IMPROVED_REWARDS else ""
        if CONTINUE_TRAINING:
            prefix = "continued_"
        analyze_results(prefix)
    
    print("Done!") 