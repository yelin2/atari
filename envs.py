import cv2
import gym
import numpy as np
from gym.spaces.box import Box


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


def _process_frame42(frame):
    # 1. crop [210, 160, 3]->[160, 160, 3]
    frame = frame[34:34 + 160, :160]
    # 2. resize [42, 42, 3] 한번에 resize 해버리면 pixel이 뭉개짐?
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    # 3. channel에 대해 mean 값 취함
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    # 4. 255로 나눠서 0에서 1사이의 값으로 만들어줌
    frame *= (1.0 / 255.0)
    # 5. move axis [w, h, c]->[c, w, h]
    frame = np.moveaxis(frame, -1, 0)
    return frame


# reshape [3, 160, 210]->[1, 42, 42], normalize [0, 1]
class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        # super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    # def _observation(self, observation):
    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        # super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    # def _observation(self, observation):
    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
