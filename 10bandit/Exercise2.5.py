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
    def __init__(self,k_arm=10,epsilon=0.1,initial=0., step_size=0.1, sample_averages=True,true_reward=0.,alfa=0.1):
        self.k = k_arm  # num of arms
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.alfa=alfa

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
        q_best = np.max(self.q_estimation)
        return np.random.choice([action for action, q in enumerate(self.q_estimation) if q == q_best])

    def step(self,action):   # 更新奖励估计Q
        # generate the reward under N(real reward, 0.01)
        self.q_true=np.random.normal(0,0.01,self.k) + self.q_true  #变化的q*
        self.best_action = np.argmax(self.q_true)
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
    for r in tqdm(range(runs)):  #显示进度条一共2000次
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

def exercise2_5(runs=2000, time=100):

    bandits = Bandit(sample_averages=True)
    best_action_counts, rewards = simulate(runs, time, bandits)
    bandits1 = Bandit(sample_averages=False)
    best_action_counts1, rewards1 = simulate(runs, time, bandits1)

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='sample_averages method')
    plt.plot(rewards1, label='constant_stepsize')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(best_action_counts, label='sample_averages method')
    plt.plot(best_action_counts1, label='constant_stepsize')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()


    plt.savefig('exercise2_5.png')
    plt.close()
if __name__ == '__main__':
    exercise2_5()
    print("done!")