import matplotlib.pyplot as plt
import numpy as np


def reward_plot(reward_list):
    plt.plot(reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()

def comparitive_reward_plot(reward_list_Q,reward_list_SARSA):

    t = np.arange(len(reward_list_Q))

    plt.plot(t, reward_list_Q,'r',t, reward_list_SARSA,'b')
    #plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()