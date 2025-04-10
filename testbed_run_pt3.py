import numpy as np
from rocket import Rocket
from rocket_env_pt2 import RocketEnv
from stable_baselines3 import PPO
import os
import glob
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained Stable-Baselines3 PPO model')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='Path to the trained model file (.zip)')
    parser.add_argument('--improved-rewards', action='store_true', 
                        help='Use improved rewards environment')
    args = parser.parse_args()

    # Setup
    task = 'landing'
    max_steps = 800
    
    # Default to the latest model if none specified
    model_dir = "sb3_rocket_models/"
    if args.model_path is None:
        # Find latest checkpoint
        model_files = glob.glob(os.path.join(model_dir, "*.zip"))
        if model_files:
            args.model_path = sorted(model_files)[-1]  # Get the last one
            print(f"Using latest model: {args.model_path}")
        else:
            print("No model files found in directory. Please specify a model path.")
            exit(1)
    
    # Check if we're using improved rewards
    use_improved = args.improved_rewards
    if "improved" in args.model_path:
        use_improved = True
        print("Detected 'improved' in model path, using improved environment")
    
    # Create the environment
    if use_improved:
        # Import the ImprovedRocketEnv class from train_model_pt3.py
        from train_model_pt3 import ImprovedRocketEnv
        env = ImprovedRocketEnv(task=task, rocket_type='starship', render_mode='human')
        print("Using ImprovedRocketEnv")
    else:
        env = RocketEnv(task=task, rocket_type='starship', render_mode='human')
        print("Using standard RocketEnv")

    # Load the model
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)
    
    # Run a test episode
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_id = 0
    
    print("Starting test episode...")
    
    start_pos = None
    min_distance = float('inf')
    
    while not done and step_id < max_steps:
        # Use the model to predict actions
        action, _ = model.predict(obs, deterministic=True)
        
        # Take a step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_id += 1
        
        # Track distance to target
        if start_pos is None:
            start_pos = (env.rocket.state['x'], env.rocket.state['y'])
            target_pos = (env.rocket.target_x, env.rocket.target_y)
            start_dist = np.sqrt((start_pos[0] - target_pos[0])**2 + 
                                (start_pos[1] - target_pos[1])**2)
            
        # Calculate current distance to target
        current_dist = np.sqrt((env.rocket.state['x'] - env.rocket.target_x)**2 + 
                              (env.rocket.state['y'] - env.rocket.target_y)**2)
        min_distance = min(min_distance, current_dist)
        
        # Print status every 100 steps
        if step_id % 100 == 0:
            print(f"Step {step_id}, Current Reward: {total_reward}")
        
        # Small delay to make visualization viewable
        time.sleep(0.01)
    
    # Print final results
    print(f"\nTest completed in {step_id} steps")
    print(f"Total reward: {total_reward}")
    print(f"Starting distance: {start_dist:.2f}m")
    print(f"Minimum distance to target: {min_distance:.2f}m")
    print(f"Distance improvement: {(start_dist - min_distance):.2f}m ({((start_dist - min_distance)/start_dist*100):.1f}%)")
    
    success = hasattr(env.rocket, 'landed') and env.rocket.landed
    print(f"Landing successful: {success}")
    
    env.close() 