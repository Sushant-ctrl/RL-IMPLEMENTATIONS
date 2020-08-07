import torch
import gym
from REINFORCE import Policy_net

env = gym.make("CartPole-v0")
PATH = "CartPole_REINFORCE"
Net = Policy_net(env.observation_space.shape[0], env.action_space.n,120,20,3e-4)
Net.load_state_dict(torch.load(PATH))

def Demo():

    while True:
        state = env.reset()
        env.render()
        done = False
        steps = 0

        while not done:

            steps = steps + 1
            action, log_prob = Net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

        print("Survived for ",steps,"steps")


Demo()