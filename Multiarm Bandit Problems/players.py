import numpy as np
from Environment import GaussianSampledBandit
from solvers import epsilon_greedy
from solvers import UCB1
from solvers import softmax
from plotters import plots


k = 10
variance = np.array([0.1, 0.5, 0.7, 0.73, 0.756, 0.789, 0.81, 0.83, 0.855, 0.865])
mean = np.array([1, 2, 9.03, 19.2, 9.01, 9, 1.345, 0.99, 0, 4.09])
num_iteration = 20000


Bandit = GaussianSampledBandit(k, variance, mean)
plot = plots(num_iteration, k)


def play_epsilon_greedy_solution():

    epsilon = 0.9


    Bandit.reset_game()
    print(Bandit.tot_reward)

    epsilon_greedy_strategy = epsilon_greedy(Bandit, k, num_iteration, epsilon)
    Pulls, Q_values, q_stars = epsilon_greedy_strategy.play_game()
    print(Pulls)
    print(Q_values)
    print(q_stars)
    print(Bandit.tot_reward)

    return Bandit.return_cumilative_rewards(), Bandit.return_arm_history(), Bandit.return_avg_rewards()

def play_softmax_solution():


    beta = 10000

    Bandit.reset_game()
    print(Bandit.tot_reward)

    softmax_strategy = softmax(Bandit, k, num_iteration, beta)
    Pulls, Q_values, q_stars = softmax_strategy.play_game()
    print(Pulls)
    print(Q_values)
    print(q_stars)
    print(Bandit.tot_reward)

    return Bandit.return_cumilative_rewards(), Bandit.return_arm_history(), Bandit.return_avg_rewards()

def play_UCB1_solution():

    Bandit.reset_game()
    print(Bandit.tot_reward)

    UCB1_strategy = UCB1(Bandit, k, num_iteration)
    step_regret, accumulated_regret, max_regret, Pulls, Q_values, q_stars = UCB1_strategy.play_game()

    print(Pulls)
    print(Q_values)
    print(q_stars)
    print(Bandit.tot_reward)

    return step_regret, accumulated_regret, max_regret, Bandit.return_cumilative_rewards(),\
           Bandit.return_arm_history(), Bandit.return_avg_rewards()

def plotting():

    # rewards_eps_greedy, arm_dict_eps_greedy, avg_rewards_eps_greedy = play_epsilon_greedy_solution()
    # rewards_softmax, arm_dict_softmax, avg_rewards_softmax = play_softmax_solution()
    step_regret_UCB1, accumulated_regret_UCB1, max_regret_UCB1, rewards_UCB1, arm_dict_UCB1, avg_rewards_UCB1 = play_UCB1_solution()

    plot.regret_plotter(step_regret_UCB1, accumulated_regret_UCB1,max_regret_UCB1)
    # plot.avg_reward_plotter(avg_rewards_eps_greedy,avg_rewards_softmax)
    # plot.cuml_reward_plotter (rewards_eps_greedy, rewards_softmax)
    # plot.arm_history_plotter (arm_dict_eps_greedy, arm_dict_softmax)

plotting()
