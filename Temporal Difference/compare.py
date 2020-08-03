from SARSA_Learning import SARSA_control
from Q_Learning import Q_learning
from utils import comparitive_reward_plot

_, reward_list_Q = Q_learning(num_episodes=1000, gamma=1, alpha=0.5)

_, reward_list_SARSA = SARSA_control(num_episodes= 1000, gamma= 1, alpha=0.5)

comparitive_reward_plot(reward_list_Q, reward_list_SARSA)