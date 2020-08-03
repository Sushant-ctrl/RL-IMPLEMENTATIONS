import gym
import numpy as np
from utils import plot_value_function
from collections import defaultdict

env = gym.make('Blackjack-v0')

gamma = 1

def create_random_policy(nA):

    A = np.ones(nA, dtype=float) / nA

    def policy(observarion):
        return A

    return policy

def create_greedy_policy(Q):

    def policy(observarion):

        Q_values = Q[observarion]
        greedy_action = np.argmax(Q_values)
        A = np.ones(len(Q_values))
        A[greedy_action] = 1

        return A

    return policy

def MonteCarlo_Off_policy(env, num_episodes, behavior_policy, gamma):

    # Cumilative sum
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    # Q values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    Target_Policy = create_greedy_policy(Q)

    for episodes in range(num_episodes):
        if episodes % 100 == 0:
            print(episodes)
        state = env.reset()
        done = False
        episode_list = []

        while not done:

            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p = probs)
            next_state, reward, done, _ = env.step(action)
            episode_list.append((state, action, reward))
            state = next_state

        G = 0.0
        W = 1.0

        for t in range(len(episode_list))[::-1]:

            state, action, reward = episode_list[t]
            G = gamma * G + reward

            C[state][action] = C[state][action] + W
            Q[state][action] += (W/C[state][action])*(G - Q[state][action])

            if action != np.argmax(Target_Policy(state)):
                break;

            W = W / behavior_policy(state)[action]
    return Q, Target_Policy

random_policy = create_random_policy(env.action_space.n)
Q, policy = MonteCarlo_Off_policy(env, num_episodes=500000, behavior_policy=random_policy, gamma=1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
print(Q)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plot_value_function(V, title="Optimal Value Function")

