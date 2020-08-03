import matplotlib.pyplot as plt
import numpy as np

class plots():

    def __init__(self, iters,k):
        self.t = np.arange(iters)
        self.arms = k

    def cuml_reward_plotter(self, rewards, rewards1):

        t = self.t
        plt.plot(t, rewards,'r')
        plt.plot(t, rewards1,'b')
        plt.show()

    def avg_reward_plotter(self, rewards, rewards1):

        t = self.t
        plt.plot( rewards, label = 'Epsilon Greedy')
        plt.plot( rewards1, label = 'Softmax')
        plt.xlabel('Nth game')
        plt.ylabel('Average reward')
        plt.title('AVERAGE REWARD PLOT')
        plt.show()

    def arm_history_plotter(self, dict, dict1):

        t = self.t
        for i in range (self.arms):
            plt.plot(t, dict[i])

        for i in range (self.arms):
            plt.plot(t, dict1[i])

        plt.show()

    def regret_plotter(self, step_regret, accumulated_regret, max_regret):

        print(len((step_regret)),\
        len((accumulated_regret)))
        print(step_regret)
        t = self.t
        plt.plot(step_regret, 'r')
        plt.plot(accumulated_regret, 'b')
        plt.plot(max_regret, 'g')
        plt.show()

