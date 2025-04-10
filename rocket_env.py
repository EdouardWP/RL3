"""
RocketEnv - OpenAI Gymnasium-compatible environment for Rocket simulation.
This implements a Gym-compatible wrapper around the Rocket class.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rocket import Rocket
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class RocketEnv(gym.Env):
    """
    RocketEnv - A Gymnasium-compatible environment for the Rocket simulation.
    
    This environment wraps the Rocket class, providing a standardized interface
    that follows OpenAI Gym/Gymnasium conventions. The environment supports both
    hover and landing tasks.
    
    Observation space: 
        8-dimensional vector representing rocket state
        [x, y, vx, vy, theta, vtheta, phi, f]
    
    Action space:
        Discrete(9) - representing combinations of thrust and nozzle angle
    
    Reward:
        Task dependent (hover or landing) with components for:
        - Distance to target
        - Rocket orientation
        - Landing success (if applicable)
    
    Episode termination:
        - Crash
        - Successful landing (landing task)
        - Max steps reached
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}
    
    def __init__(self, task='hover', rocket_type='starship', max_steps=800, render_mode=None):
        """
        Initialize the Rocket environment.
        
        Args:
            task (str): 'hover' or 'landing'
            rocket_type (str): 'falcon' or 'starship'
            max_steps (int): Maximum number of steps per episode
            render_mode (str): The render mode to use ('human' or 'rgb_array')
        """
        super(RocketEnv, self).__init__()
        
        # Create the rocket simulation
        self.rocket = Rocket(max_steps=max_steps, task=task, rocket_type=rocket_type)
        self.task = task
        self.rocket_type = rocket_type
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.rocket.action_dims)
        
        # Determine observation space bounds based on Rocket environment
        low_bounds = np.array([
            self.rocket.world_x_min,             # x position
            self.rocket.world_y_min,             # y position
            -100.0,                              # vx velocity
            -100.0,                              # vy velocity
            -np.pi,                              # theta orientation
            -np.pi,                              # vtheta angular velocity
            -np.pi/2,                            # phi nozzle angle
            -0.01                                # f thrust (allow small negative values for numerical stability)
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

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a new initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options for reset
            
        Returns:
            observation (numpy.ndarray): Initial state
            info (dict): Additional information
        """
        super().reset(seed=seed)
        
        # Reset the rocket simulation
        observation = self.rocket.reset()
        
        # Additional info dictionary
        info = {}
        
        return observation, info

    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action (int): Action to take (index into action table)
            
        Returns:
            observation (numpy.ndarray): New state
            reward (float): Reward for this step
            terminated (bool): Whether the episode is terminated
            truncated (bool): Whether the episode is truncated (e.g., due to max steps)
            info (dict): Additional information
        """
        # Take action in the rocket simulation
        observation, reward, done, info = self.rocket.step(action)
        
        # Check if episode is terminated or truncated
        terminated = done
        truncated = self.rocket.step_id >= self.max_steps
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.
        
        Returns:
            frame (numpy.ndarray): RGB frame if mode is 'rgb_array', else None
        """
        if self.render_mode is None:
            return None
            
        # Render the rocket simulation
        frame = self.rocket.render(window_name='RocketEnv', wait_time=1)
        self.last_frame = frame
        
        if self.render_mode == 'rgb_array':
            return frame
        
        return None

    def close(self):
        """
        Close the environment.
        """
        # Clean up resources
        import cv2
        cv2.destroyAllWindows()


# Basic test code if this file is run directly
if __name__ == "__main__":
    # Create and test the environment
    env = RocketEnv(task='landing', rocket_type='starship', render_mode='human')
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Test reset
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    
    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}, Action: {action}, Reward: {reward:.4f}, Done: {terminated or truncated}")
        
        if terminated or truncated:
            print("Episode finished early")
            break
    
    env.close()