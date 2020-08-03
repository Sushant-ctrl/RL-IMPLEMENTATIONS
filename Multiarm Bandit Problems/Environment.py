import numpy as np
import random


class GaussianSampledBandit(object):

    def __init__(self, k, variance, mean):

        self.variance = np.array(variance)
        self.q_star = np.array(mean)
        self.num_arms = k
        self.times_played = np.zeros(self.num_arms)
        self.tot_reward = 0
        self.acc_reward = []
        self.avg_reward = []
        self.arm_history_dict = {}
        for i in range(self.num_arms):
            self.arm_history_dict[i] = []

    def arm_history(self):

        for arm in range (self.num_arms):
            self.arm_history_dict[arm].append(self.times_played[arm])

    def calc_avg_reward(self):

        n = np.sum(self.times_played)
        self.avg_reward.append(self.tot_reward/n)

    def play_arm(self, arm):

        self.times_played[arm] += 1
        self.arm_history()
        reward = random.gauss(self.q_star[arm],self.variance[arm])
        self.tot_reward = reward + self.tot_reward
        self.calc_avg_reward()
        self.acc_reward.append(self.tot_reward)
        return reward

    def times_each_arm_played(self):

        return self.times_played

    def return_total_reward(self):

        return self.tot_reward

    def reset_game(self):

        for i in range (self.num_arms):
            self.arm_history_dict[i]=[]
        self.times_played = np.zeros(self.num_arms)
        self.acc_reward = []
        self.avg_reward = []
        self.tot_reward = 0

    def return_cumilative_rewards(self):

        return self.acc_reward

    def return_avg_rewards(self):

        return self.avg_reward

    def return_arm_history(self):

        return self.arm_history_dict