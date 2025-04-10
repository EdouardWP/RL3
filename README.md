# Rocket Landing with Reinforcement Learning

## Project Overview

This project implements a reinforcement learning solution for the challenging rocket landing problem, inspired by SpaceX's vertical landing technology. Using three complementary approaches, I've developed a system that can successfully land a rocket on a target with high precision.

The project is divided into three progressive parts:
1. A custom Actor-Critic algorithm with entropy regularization
2. An OpenAI Gymnasium-compatible environment for rocket simulation
3. A Stable-Baselines3 implementation with improved reward functions

The simulation includes physics-based dynamics such as gravity, thrust control, and aerodynamic effects. The goal is to train an agent that can control the rocket's thrust and nozzle angle to perform either hovering or precision landing tasks.

## Files by Assignment Part

### Part 1: Actor-Critic with Entropy Regularization
- `actor_critic_entropy_pt1.md`: Explanation of Actor-Critic algorithm and entropy regularization implementation
- `policy.py`: Implementation of the Actor-Critic neural network architecture
- `test_entropy_ac.py`: Script to test and visualize the Actor-Critic algorithm

### Part 2: OpenAI Gymnasium Environment
- `rocket_env_pt2.py`: OpenAI Gymnasium-compatible environment for the rocket simulation
- `test_rocket_env.py`: Script to test the Gymnasium environment with a random agent
- `rocket.py`: Core rocket physics simulation (base implementation)

### Part 3: Stable-Baselines3 Implementation with Improved Rewards
- `train_model_pt3.py`: Main training script using PPO algorithm with improved reward function
- `testbed_run_pt3.py`: Testing script to evaluate trained models
- `sb3_rocket_models/`: Directory containing trained models

## How to Run

### Setup
```bash
pip install gymnasium numpy matplotlib stable-baselines3
```

### Testing Each Part

#### Part 1: Actor-Critic with Entropy Regularization
```bash
python test_entropy_ac.py
```
This will run the Actor-Critic implementation with entropy regularization on the landing task. It displays a visualization of the rocket attempting to land and saves checkpoints to the landing_entropy_ckpt folder.

#### Part 2: OpenAI Gymnasium Environment 
```bash
python test_rocket_env.py
```
This tests the Gym-compatible environment with a random agent. It verifies that the environment follows the Gymnasium API standards by running random actions and displaying the rocket's behavior.

#### Part 3: Stable-Baselines3 Implementation with Improved Rewards

1. **Quick Test** (fastest way to see results):
```bash
python testbed_run_pt3.py --improved-rewards
```
This loads the best trained model and shows the rocket landing with improved rewards.

2. **Train from scratch**:
```bash
python train_model_pt3.py --improved-rewards
```

3. **Continue training**:
```bash
python train_model_pt3.py --continue --improved-rewards
```

4. **Visualize with stats** (multiple episodes with performance statistics):
```bash
python train_model_pt3.py --visualize --improved-rewards --episodes 5
```
This visualizes 5 test episodes using the trained model and displays performance statistics.

## Learning Outcomes

Each part of this implementation demonstrates different aspects of reinforcement learning:

- **Part 1**: Understanding of the Actor-Critic architecture and the importance of entropy regularization
- **Part 2**: Ability to convert simulations into standardized environments following Gymnasium conventions
- **Part 3**: Practical application of state-of-the-art RL algorithms and reward function engineering

## Key Improvements in Part 3

The improved reward function enhances landing precision through:
- Exponential distance scaling for better targeting
- Orientation-distance coupling for early attitude correction
- Velocity matching based on height
- Trajectory smoothness rewards
- Directional incentives toward target