import numpy as np
import random

import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNet, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, self.num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x.cuda())
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu(x)
        out = self.fc2(x)
        return out

    # select action with epsilon greedy
    def act(self, state, epsilon):        
        if random.random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
            with torch.no_grad():
                q_value = self.forward(state)
                action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action