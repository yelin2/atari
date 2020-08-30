'''
Author: yelin lee
environment wrapper for atari
implement
    - clipreward: cliping reward 1, 0, -1
    - obsTransform: transform 216x116 RGB image to 84x84 gray scale image
    - No-opearation reset: sample initial states by taking random number of no-ops on reset
    - Fire reset: take action on reset for env that are fixed until firing
'''

import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
# cv2.ocl.setUseOpenCL(False)
# from .wrappers import TimeLimit

