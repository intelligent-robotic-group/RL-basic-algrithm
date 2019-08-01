#######################################################################
# Copyright (C)
# 7th Mar. 2019
# Author: Zhong Jie
#######################################################################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self,k_arm=10,epsilon=0.1,initial=0., step_size=0.1, sample_averages=False, true_reward=0., UCB_param=None):
        self.k = k_arm  # num of arms
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.UCB_param = UCB_param

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)


    def act(self):
        if np.random.rand()<self.epsilon:      #  exploration epsilon greedy
            return np.random.choice(self.indices)
        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                     self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])
        q_best = np.max(self.q_estimation)
        return np.random.choice([action for action, q in enumerate(self.q_estimation) if q == q_best])

    def step(self,action):   # 更新奖励估计Q

        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
        else:
            self.q_estimation[action] +=self.alfa*(reward-self.q_estimation[action])
        return reward

def simulate(runs,time,bandit):
    best_action_counts = np.zeros((runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for r in range(runs):  #显示进度条一共2000次
        print(r)
        bandit.reset()           #初始化参数
        for t in range(time): #1000步
            action = bandit.act()
            reward = bandit.step(action)
            rewards[r, t] = reward
            if action == bandit.best_action:
                 best_action_counts[r, t] = 1
    best_action_counts = best_action_counts.mean(axis=0) #对所有run的最好action做统计百分比
    rewards = rewards.mean(axis=0)
    return best_action_counts, rewards

def fig2_4(runs=2000, time=1000):

    bandits = Bandit(sample_averages=True, UCB_param=2, epsilon=0)
    best_action_counts, rewards = simulate(runs, time, bandits)
    bandits1 = Bandit(sample_averages=True)
    best_action_counts1, rewards1 = simulate(runs, time, bandits1)

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='UCB')
    plt.plot(rewards1, label='eplilon-greedy')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(best_action_counts, label='UCB')
    plt.plot(best_action_counts1, label='eplilon-greedy')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()


    plt.savefig('figUCB0.png')
    plt.close()
if __name__ == '__main__':
    fig2_4()
