import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import numpy as np
import os

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

if __name__ == '__main__':
    # Choose between 'hover' and 'landing'
    task = 'landing'
    
    # Set training parameters
    max_m_episode = 1000  # Total episodes to train (reduced for testing)
    max_steps = 800      # Maximum steps per episode
    
    # Create the environment
    env = Rocket(task=task, max_steps=max_steps)
    print(f"Environment created: {env.state_dims} states, {env.action_dims} actions")
    
    # Create output folder for checkpoints
    ckpt_folder = os.path.join('./', task + '_entropy_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    
    # Initialize the actor-critic network with entropy regularization
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    print("Actor-Critic network created with entropy regularization")
    
    # Track rewards for plotting
    all_rewards = []
    
    # Training loop
    for episode_id in range(max_m_episode):
        # Reset environment at the start of each episode
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []
        
        # Run the episode
        for step_id in range(max_steps):
            # Get action from policy
            action, log_prob, value = net.get_action(state)
            
            # Take action in environment
            state, reward, done, _ = env.step(action)
            
            # Store results
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            
            # Render every 100 episodes
            if episode_id % 100 == 1:
                env.render()
            
            # If episode is done or max steps reached, update the network
            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break
        
        # Track total reward for this episode
        episode_reward = np.sum(rewards)
        all_rewards.append(episode_reward)
        
        print(f'Episode {episode_id}: Reward = {episode_reward:.3f}')
        
        # Save checkpoint and plot every 100 episodes
        if episode_id % 100 == 1:
            # Plot rewards
            plt.figure(figsize=(10, 5))
            plt.plot(all_rewards)
            plt.plot(np.convolve(all_rewards, np.ones(50)/50, mode='valid'))
            plt.title('Rewards with Entropy Regularization')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.legend(['Episode Reward', 'Moving Average (50)'])
            plt.savefig(os.path.join(ckpt_folder, f'entropy_rewards_{episode_id:08d}.jpg'))
            plt.close()
            
            # Save checkpoint
            torch.save({
                'episode_id': episode_id,
                'REWARDS': all_rewards,
                'model_G_state_dict': net.state_dict()
            }, os.path.join(ckpt_folder, f'entropy_ckpt_{episode_id:08d}.pt'))
    
    print("Training complete!") 