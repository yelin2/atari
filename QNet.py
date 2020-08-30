'''
Author: yelin lee
network architecture for approximate Q-value function
implement
    - fixed Q-target
    - experience memory
'''

import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self):
        super(QNet,self).__init__()

    def forward(self):
        return 1
    
    