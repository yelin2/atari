import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from replay_memory import ReplayBuffer
from QNet import QNet

# plot reward and loss
def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig('res/graph/yelin/fig %s.png'%(frame_idx))
    plt.clf()

# set atari env
env_id = 'PongNoFrameskip-v4'
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

# define model, optimizer
resume = False
path = 'res/model/qnet_adam.pth'
model = QNet(env.observation_space.shape, env.action_space.n)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.00001)

# set training options
replay_initial = 10000
replay_buffer = ReplayBuffer(100000)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final)*math.exp(-1.*frame_idx/epsilon_decay)

epsilon = 1.
epsilons = []

num_frames = 1400000
batch_size = 32
gamma = 0.99

losses = []
total_reward = []
episode_reward = 0

# compute loss
def compute_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).cuda()
    next_state = torch.FloatTensor(np.float32(next_state)).cuda()
    action = torch.LongTensor(action).cuda()
    reward = torch.FloatTensor(reward).cuda()
    done = torch.FloatTensor(done).cuda()
    
    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.cuda().unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0].detach()
    target_q_value = reward + gamma * next_q_value * (1-done)

    loss = (q_value - target_q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# load model
if resume:
    model.load_state_dict(torch.load(path))

# start training
state = env.reset()
for frame_idx in tqdm(range(1, num_frames + 1), mininterval=1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        total_reward.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        loss = compute_loss(batch_size)
        losses.append(loss.item())

    if frame_idx % 10000 == 0:
        plot(frame_idx, total_reward, losses)

# save model
torch.save(model.state_dict(), 'res/model/yelin.pth')