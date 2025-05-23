import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    task = 'hover'   # 'hover' or 'landing'
    task = 'landing' # 'hover' or 'landing'

    max_steps = 800
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))
    if ckpt_dir: ckpt_dir = ckpt_dir[-1]  # last ckpt
    print(ckpt_dir)

    env = Rocket(task=task, max_steps=max_steps)

    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if ckpt_dir and os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False, map_location=device)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')
        if env.already_crash:
            break
 