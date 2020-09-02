import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import time

import torch
import torch.nn as nn
import torch.optim as optim

from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from replay_memory import ReplayBuffer
from QNet import QNet

# set up
path = 'res/model/yelin.pth'
env_id = 'PongNoFrameskip-v4'

# make env
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

# load model
model = QNet(env.observation_space.shape, env.action_space.n)
model.load_state_dict(torch.load(path))
model.cuda()

# evaluation: play pong game with trained QNet
num_frames = 10000
epsilon = 0
total_rewards = []

state = env.reset()
for frame_idx in tqdm(range(1, num_frames+1)):
    action = model.act(state, epsilon).cpu().numpy()
    next_state, reward, done, info = env.step(action)
    state = next_state
    env.render()
    time.sleep(.5)
env.close