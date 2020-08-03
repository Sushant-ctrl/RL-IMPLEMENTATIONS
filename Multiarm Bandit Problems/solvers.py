import numpy as np
import random
from Environment import GaussianSampledBandit
import math


class epsilon_greedy (GaussianSampledBandit):

    def __init__(self, Bandit, k, num_iters, epsilon):

        self.Bandit = Bandit
        self.epsilon = epsilon
        self.no_of_arms = k
        self.Q_values = np.ones(k)
        self.num_iters = num_iters


    def epsilon_greedy_policy(self):

        if random.random() >= self.epsilon:
            arm_choice = random.randint(0, self.no_of_arms-1)

        else:
            arm_choice = np.argmax(self.Q_values)

        return arm_choice

    def pull_bandit(self, arm):

        reward = self.Bandit.play_arm(arm)
        return reward

    def eps_decay(self, t):
        
        if (t%10 == 0) & (self.epsilon >= 0.02):
            self.epsilon = self.epsilon - 0.000002


    def play_game(self):

        for t in range(self.num_iters):

            arm = self.epsilon_greedy_policy()
            reward = self.pull_bandit(arm)
            update_Q = self.Q_values[arm] + (reward - self.Q_values[arm])/self.Bandit.times_each_arm_played()[arm]
            self.Q_values[arm] =update_Q
            self.eps_decay(t)

        return self.Bandit.times_each_arm_played(), self.Q_values, self.Bandit.q_star


class softmax (GaussianSampledBandit):

    def __init__(self, bandit, k, num_iters, beta):

        self.Bandit = bandit
        self.no_of_arms = k
        self.num_iters = num_iters
        self.b = beta
        self.Q_values = np.ones(k)

    def softmax_implementaion(self):

        e_powered_Q = np.exp(self.Q_values/self.b)
        probablity = np.zeros(self.no_of_arms)
        for arm in range (self.no_of_arms):
            probablity[arm] = e_powered_Q[arm]/np.sum(e_powered_Q)

        arm = random.choices(np.arange(10),weights=probablity)

        return arm[0]

    def pull_bandit(self,arm):

        reward = self.Bandit.play_arm(arm)
        return reward

    def beta_decay(self,t):

        if (t%10 == 0) & (self.b/2 > 0.1):
            self.b = self.b / 3

    def play_game(self):

        for t in range(self.num_iters):

            arm = self.softmax_implementaion()
            reward = self.pull_bandit(arm)
            update_Q = self.Q_values[arm] + (reward - self.Q_values[arm])/self.Bandit.times_each_arm_played()[arm]
            self.Q_values[arm] =update_Q
            self.beta_decay(t)


        return self.Bandit.times_each_arm_played(), self.Q_values, self.Bandit.q_star

class UCB1(GaussianSampledBandit):

    def __init__(self, bandit, k, num_iters):

        self.Bandit = bandit
        self.no_of_arms = k
        self.num_iters = num_iters
        self.Q_values = np.zeros(k)
        self.max_regret = []
        self.step_regret = []
        self.acc_regret = []

    def pull_bandit(self, arm):

        reward = self.Bandit.play_arm(arm)
        return reward

    def UCB1_policy(self, n):

        temp = (2*np.log(n))/self.Bandit.times_each_arm_played()
        arm = np.argmax(self.Q_values + np.sqrt(temp))
        return arm

    def Regret(self, reward, n, arm):


        deltai = np.max(self.Bandit.q_star) - self.Bandit.q_star[arm]
        if (deltai != 0):
            self.max_regret.append((8 * np.log(n))/deltai + (1+np.pi*np.pi/3)*deltai)
        else:
            self.max_regret.append(self.max_regret[n-2])

        step_regret = self.Bandit.q_star[arm]-reward
        self.step_regret.append(step_regret)
        acc_regret = sum(self.step_regret)
        self.acc_regret.append(acc_regret)

    def play_game(self):

        for n in range(self.no_of_arms):

            reward = self.pull_bandit(n)
            self.Q_values[n] = self.Q_values[n] + reward
            self.Regret(reward, n+1, n)

        for n in range(self.no_of_arms,self.num_iters):

            arm = self.UCB1_policy(n)
            reward = self.pull_bandit(arm)
            self.Q_values[arm] = self.Q_values[arm] + (reward + self.Q_values[arm])/n
            self.Regret(reward, n+1, arm)

        return self.step_regret, self.acc_regret, self.max_regret, self.Bandit.times_each_arm_played(),\
               self.Q_values, self.Bandit.q_star







