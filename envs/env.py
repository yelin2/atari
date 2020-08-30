import gym

import os
# try:
#     from mpi4py import MPI
# except ImportError:
#     MPI = None

from utils import logger
from bench import Monitor
from atari_wrappers import make_atari, wrap_deepmind

# skip Monitor
def make_env(env_id, seed=None, reward_scale=1.0, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
	wrapper_kwargs = wrapper_kwargs or {}
	env_kwargs = env_kwargs or {}
	env = make_atari(env_id)

	env.seed(seed)

	env = Monitor(env, logger_dir and os.path.join(logger_dir), allow_early_resets=True)
	env = wrap_deepmind(env)

	return env
