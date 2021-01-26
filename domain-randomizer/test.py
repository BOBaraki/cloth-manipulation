import randomizer
import gym
from gym import envs
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time

#env = RandomizedEnvWrapper(gym.make('ResidualNoisyHookRandomizedEnv-v0'), seed=123)
env = RandomizedEnvWrapper(envs.make('RandomizedGen3SidewaysFold-v0'), seed=12)
#env = gym.make('FetchHookRandomizedEnv-v0')
env.randomize(['default'])
obs = env.reset()
for i in range(2000):
    obs, _, done, _ = env.step(np.zeros((4)))
    env.render()
    if i % 100 == 0:
        env.randomize(['default'])
        env.reset()

