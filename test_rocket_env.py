"""
Test script for the RocketEnv Gymnasium environment.
This script demonstrates how to use the RocketEnv with standard Gym patterns.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from rocket_env import RocketEnv
import time

def test_random_agent():
    """Run a random agent in the environment and record rewards."""
    
    # Create the environment
    env = RocketEnv(task='landing', render_mode='human')
    print(f"Created environment with observation space {env.observation_space} and action space {env.action_space}")
    
    # Run 3 episodes
    episodes = 3
    all_rewards = []
    
    for episode in range(episodes):
        # Reset environment
        observation, info = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{episodes}")
        
        # Episode loop
        while not done:
            # Sample random action
            action = env.action_space.sample()
            
            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}, Current reward: {total_reward:.2f}")
            
            # Short delay to make visualization viewable
            time.sleep(0.01)
        
        print(f"Episode {episode+1} finished after {step} steps with total reward {total_reward:.2f}")
        all_rewards.append(total_reward)
    
    # Close the environment
    env.close()
    
    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, episodes+1), all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Random Agent Performance')
    plt.savefig('random_agent_rewards.png')
    plt.show()
    
    return all_rewards

def verify_gym_compatibility():
    """Verify that our environment works with standard Gym utilities."""
    
    try:
        # Register our environment
        gym.register(
            id='Rocket-v0',
            entry_point='rocket_env:RocketEnv',
            max_episode_steps=800,
        )
        
        # Create the environment using the Gym API
        env = gym.make('Rocket-v0', render_mode=None)
        
        # Basic verification
        print("\nGym Compatibility Test:")
        print(f"Environment registered and created successfully: {env}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test reset and step
        observation, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reset and step work correctly.")
        print(f"Observation shape: {observation.shape}")
        print(f"Reward from random action: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"Error in gym compatibility test: {e}")
        return False

if __name__ == "__main__":
    print("Testing RocketEnv with random agent...")
    rewards = test_random_agent()
    print(f"Average reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")
    
    # Verify Gym compatibility
    is_compatible = verify_gym_compatibility()
    print(f"\nGym compatibility test {'passed' if is_compatible else 'failed'}") 