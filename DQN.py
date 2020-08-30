import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from envs.env import make_env

# import Replay_Memory as replay_mamory
# import QNet

# make environment
env = make_env('Pong-v4', seed=1)
env.reset()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # define replay memory, Q function, target Q function
# mem = replay_mamory(10000)
# policy_net = QNet().to(device)
# target_net = QNet().to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

# # training setting
# batch_size = 32
# gamma = 0.99
# eps_start = 1
# eps_end = 0.1
# eps_decay = 1000000
# target_update = 10000
# lr = 0.00025

# optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.95)

# # training