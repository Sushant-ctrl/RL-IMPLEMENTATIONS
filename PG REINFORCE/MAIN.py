import gym
import numpy as np
import torch
from REINFORCE import Policy_net
from REINFORCE import update_policy
from UTILS import plot

env = gym.make("CartPole-v0")
PATH = "CartPole_REINFORCE1"
Net = Policy_net(env.observation_space.shape[0], env.action_space.n,120,20,3e-4)
#Net.load_state_dict(torch.load(PATH))

max_episodes = 3000
num_steps = []
avg_num_steps = []

def main():

    for episodes in range (max_episodes):

        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        steps = 0

        while not done:

            env.render()
            steps = steps + 1
            action, log_prob = Net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        update_policy(Net, rewards, log_probs)
        num_steps.append(steps)
        avg_num_steps.append(np.mean(num_steps[-50:]))
        #plot(num_steps, avg_num_steps)

        if episodes % 10 == 0:
            torch.save(Net.state_dict(),PATH)
            print('Episodes:',episodes,'Total steps =',num_steps[-1],'Average steps =',avg_num_steps[-1])


main()

