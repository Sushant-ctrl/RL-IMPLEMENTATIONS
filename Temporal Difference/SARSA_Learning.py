import gym
import gym_gridworlds
import numpy as np
import random
from utils import reward_plot
from collections import defaultdict

env = gym.make('Cliff-v0')
#env = gym.make('WindyGridworld-v0')

Q = defaultdict(lambda :(np.zeros(env.action_space.n)))
epsilon = 0.1
gamma = 1
alpha = 0.6


def epsilon_greedy_step(state, epsilon=0.2):
    # epsilon = epsilon_decay

    prob = random.random()

    if prob >= epsilon:
        action = np.argmax(Q[state])

    else:
        action = env.action_space.sample()

    return action


def reward_history(reward, reward_list):
    reward_list[-1] += reward
    return reward_list

def SARSA_control(num_episodes, gamma, alpha):

    e = epsilon
    reward_list = []

    for episode in range (num_episodes):

        done = False
        state = env.reset()
        action = epsilon_greedy_step(state, e)
        reward_list.append(0)
        while not done:

            next_state, reward, done,_ = env.step(action)
            next_action = epsilon_greedy_step(next_state, e)

            Q[state][action] += alpha * (reward + gamma*Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action

            reward_list = reward_history(reward, reward_list)

        if episode % 100 == 0:
            print('SARSA',reward_list[-1])
            e /= 2

    return Q, reward_list

Q, reward_list = SARSA_control(num_episodes= 1000, gamma= 1, alpha=0.5)
#reward_plot(reward_list)