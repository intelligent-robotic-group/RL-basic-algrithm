#######################################################################
# Copyright (C)
# 14th Mar. 2019
# Author: Zhong Jie
# chapter 2-8:Gradient bandit algrithms
#######################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm      #num of arms
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice([action for action, q in enumerate(self.q_estimation) if q == q_best])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation = self.q_estimation + self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

def simulate(runs, time, bandit):
    best_action_counts = np.zeros((runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for r in tqdm(range(runs)):  #显示进度条一共2000次
        bandit.reset()           #初始化参数
        for t in range(time): #1000步
            action = bandit.act()
            reward = bandit.step(action)
            rewards[ r, t] = reward
            if action == bandit.best_action:
                best_action_counts[ r, t] = 1
    best_action_counts = best_action_counts.mean(axis=0) #对所有run的最好action做统计百分比
    rewards = rewards.mean(axis=0)
    return best_action_counts, rewards

def fig2_5(runs=2000, time=1000):
    bandits1 = Bandit(step_size=0.1, gradient=True, gradient_baseline=False, true_reward=4.0)
    best_action_counts1, rewards1 = simulate(runs, time, bandits1)
    bandits2 = Bandit(step_size=0.1, gradient=True, gradient_baseline=True, true_reward=4.0)
    best_action_counts2, rewards2 = simulate(runs, time, bandits2)
    bandits3 = Bandit(step_size=0.4, gradient=True, gradient_baseline=False, true_reward=4.0)
    best_action_counts3, rewards3 = simulate(runs, time, bandits3)
    bandits4 = Bandit(step_size=0.4, gradient=True, gradient_baseline=True, true_reward=4.0)
    best_action_counts4, rewards4 = simulate(runs, time, bandits4)

    plt.figure(figsize=(10, 10))
    plt.plot(best_action_counts1, label='stepsize=0.1 without baseline')
    plt.plot(best_action_counts2, label='stepsize=0.1 with baseline')
    plt.plot(best_action_counts3, label='stepsize=0.4 without baseline')
    plt.plot(best_action_counts4, label='stepsize=0.4 with baseline')
    plt.xlabel('steps')
    plt.ylabel('%optimal actions')
    plt.savefig('Gradient.png')
    plt.close()



if __name__ == '__main__':
    fig2_5()