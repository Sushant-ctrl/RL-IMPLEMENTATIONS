import gym
import gym_gridworlds
import numpy as np
import random
from  collections import defaultdict
from utils import reward_plot

epsilon = 0.1
gamma = 1
alpha = 0.6

env = gym.make('Cliff-v0')
# env = gym.make('WindyGridworld-v0')

Q = defaultdict(lambda : (np.zeros(env.action_space.n)))

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

def Q_learning(num_episodes, gamma, alpha):

    e = epsilon
    reward_list = []

    for episode in range(num_episodes):
        done = False
        state = env.reset()
        reward_list.append(0)

        while not done:

            action = epsilon_greedy_step(state, e)
            next_state, reward, done, _ = env.step(action)

            greedy_action = epsilon_greedy_step(next_state, 0)

            Q[state][action] += alpha * (reward + gamma*Q[next_state][greedy_action] - Q[state][action])

            state = next_state

            reward_list = reward_history(reward, reward_list)

        if episode % 50 == 0:
            print(reward_list[-1])
            e /= 2

    return Q, reward_list


Q, reward_list = Q_learning(num_episodes=401, gamma=1, alpha=0.5)
#reward_plot(reward_list)
