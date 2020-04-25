import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features=state_size, out_features=420)
        self.fc2 = nn.Linear(in_features=420, out_features=130)
        self.out = nn.Linear(in_features=130, out_features=action_size)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

state_size = 4
action_size = 2

model = DQN(state_size, action_size)
model.load_state_dict(torch.load('CartPole_solved_model.pt'))
model.eval()


def select_action(state, model):
    s = torch.tensor(state, dtype=torch.float)
    q = model(s)
    action = torch.argmax(q)
    return int(action)

import gym

env = gym.make('CartPole-v0')
env.reset()


while True:
    tot_sur = 0
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = select_action(state,model)
        state, reward, done, info = env.step(action)
        tot_sur = tot_sur + 1
        time.sleep(0.002)
    print('survived for' + str(tot_sur) + 'time steps')

