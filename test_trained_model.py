"""
Test script for evaluating a trained Stable-Baselines3 model on the Rocket environment.
This can be run after training to visualize and evaluate the model's performance.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from rocket_env import RocketEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Directory setup
log_dir = "sb3_rocket_logs/"
model_dir = "sb3_rocket_models/"

def make_env():
    """Create the environment."""
    return RocketEnv(task='landing', rocket_type='starship', render_mode=None)

def load_model():
    """Load the trained model and normalized environment."""
    # Try to load the final model first, then fall back to the latest checkpoint
    try:
        model_path = f"{model_dir}final_model.zip"
        if not os.path.exists(model_path):
            # Find the latest checkpoint
            checkpoints = [f for f in os.listdir(model_dir) if f.startswith("ppo_rocket_") and f.endswith(".zip")]
            if not checkpoints:
                raise FileNotFoundError(f"No model files found in {model_dir}")
            
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            model_path = os.path.join(model_dir, checkpoints[-1])
            print(f"Final model not found, using latest checkpoint: {model_path}")
        
        # Load the model
        model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load the normalized environment if available
        env = DummyVecEnv([make_env])
        try:
            env = VecNormalize.load(f"{model_dir}vec_normalize.pkl", env)
            env.training = False  # Don't update the normalization statistics
            env.norm_reward = False  # Don't normalize rewards when testing
            print("Loaded normalized environment")
        except FileNotFoundError:
            print("Normalized environment not found, using unnormalized environment")
        
        return model, env
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def visualize_test(model, env, num_episodes=5):
    """Visualize the model's performance in the environment."""
    # Create a test environment with rendering
    test_env = RocketEnv(task='landing', rocket_type='starship', render_mode='human')
    
    rewards = []
    landing_success = 0
    
    for episode in range(num_episodes):
        episode_reward = 0
        done = False
        obs, _ = test_env.reset()
        steps = 0
        
        print(f"\nTesting episode {episode+1}/{num_episodes}")
        
        while not done:
            # Normalize observations if using VecNormalize
            if isinstance(env, VecNormalize):
                obs_normalized = env.normalize_obs(np.array([obs]))
                action, _ = model.predict(obs_normalized, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Take the action
            obs, reward, terminated, truncated, info = test_env.step(action.item())
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Check if this was a successful landing
            if terminated and reward > 0:
                landing_success += 1
                print(f"Successful landing in episode {episode+1}!")
            
            # Add a small delay for visualization
            time.sleep(0.01)
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1} finished after {steps} steps with reward: {episode_reward:.2f}")
    
    test_env.close()
    
    # Print summary statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = landing_success / num_episodes * 100
    
    print("\nTest Results Summary:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Landing success rate: {success_rate:.1f}% ({landing_success}/{num_episodes})")
    
    return rewards, success_rate

def formal_evaluation(model, env, num_eval_episodes=20):
    """Formally evaluate the model without rendering."""
    # Create evaluation environment
    eval_env = make_env()
    
    # If we have a normalized environment, wrap the evaluation environment
    if isinstance(env, VecNormalize):
        # We need to wrap the evaluation environment in a DummyVecEnv first
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Then create a VecNormalize with the same normalization stats
        eval_env = VecNormalize(
            eval_env,
            norm_obs=env.norm_obs,
            norm_reward=False,  # Don't normalize rewards when evaluating
            clip_obs=env.clip_obs,
            clip_reward=env.clip_reward,
            gamma=env.gamma,
            epsilon=env.epsilon
        )
        
        # Copy normalization stats
        eval_env.obs_rms = env.obs_rms
        
        print("Evaluation environment normalized with training stats")
    
    # Evaluate the model
    print(f"\nFormal evaluation using {num_eval_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=num_eval_episodes,
        deterministic=True
    )
    
    print(f"Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return mean_reward, std_reward

if __name__ == "__main__":
    # Load the trained model
    print("Loading trained model...")
    model, env = load_model()
    
    if model is None:
        print("Failed to load model. Please make sure you have trained a model first.")
        exit(1)
    
    # Visualize test episodes
    print("\nVisualizing model performance...")
    rewards, success_rate = visualize_test(model, env, num_episodes=5)
    
    # Formal evaluation
    mean_reward, std_reward = formal_evaluation(model, env, num_eval_episodes=20)
    
    # Save results to file
    with open(f"{log_dir}test_results.txt", "w") as f:
        f.write(f"Test Visualization Results:\n")
        f.write(f"Landing success rate: {success_rate:.1f}%\n")
        f.write(f"Mean reward during visualization: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}\n\n")
        f.write(f"Formal Evaluation Results:\n")
        f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    
    print(f"\nResults saved to {log_dir}test_results.txt")
    print("Done!") 