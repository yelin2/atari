import gym

import numpy as np
import time

'''
    Pong environment
        100 step 동안 1번 정도만 왔다갔다 함...
'''
env = gym.make('Pong-v4')
env.reset()

# check action, observation space
# print(env.action_space)
# print(env.observation_space)

# check observation space more...
# print(env.observation_space.high)
# print(env.observation_space.low)

for _ in range(100):
    o, r, d, i = env.step(env.action_space.sample())
    # print(o.shape)
    # (210, 160, 3)
    env.render()
    time.sleep(.5)
env.close()