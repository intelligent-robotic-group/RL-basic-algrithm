import numpy as np
import gym_bandits
import gym
from gym import envs



if __name__ == '__main__':


    print(envs.registry.all())
    env = gym.make('Copy-v0')
    for i_episode in range(100):
        observation = env.reset()
        for t in range(10000):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("{} timesteps taken for the episode".format(t + 1))
                break