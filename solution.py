import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple
import gym
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from PIL import display

env = gym.make('CartPole-v0').unwrapped

Experience = namedtuple('Experience',('state', 'action', 'next_state', 'reward', 'done'))

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

class E_greedy():

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        #return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
        eps = self.start - self.decay*current_step
        if eps >= self.end:
            return eps
        else:
            return self.end


class Agent():

    def __init__(self, strategy, action_size, device):
        self.current_step = 0
        self.strategy = strategy
        self.action_size = action_size

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate >= random.random():
            #exploration
            return random.randrange(self.action_size),rate
        else:
            #we do not use the gradient because this is exploitation and here we are not training the network
            with torch.no_grad():
                q = policy_net(torch.tensor(state, dtype=torch.float).to(device))
                action = torch.argmax(q)
            return int(action),rate


class Replay_Memory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.counter = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.counter % self.capacity] = experience
        self.counter += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


def plot(values, ep, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Traning")
    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.plot(values)
    plt.plot(get_moving_avg(moving_avg_period, values))
    plt.pause(0.001)
    plt.figure(1)
    plt.title("e greedy")
    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    print("Episodes", len (values), "\n" , moving_avg_period, "episode moving avg:", get_moving_avg(moving_avg_period, values)[-1])

    if is_ipython:
        display.clear_output(wait=True)


def get_moving_avg(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values)>period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))

        return moving_avg.numpy()

    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))
    t1 = np.asarray(batch.state)
    t2 = np.asarray(batch.action)
    t3 = np.asarray(batch.reward)
    t4 = np.asarray(batch.next_state)
    t5 = np.asarray(batch.done)

    return t1, t2, t3, t4, t5



batch_size = 32
eps_start = 1
eps_stop = 0.01
eps_decay = 0.00001
gamma = 0.99
target_update = 5
memory_size = 1000000
alpha = 0.0001
num_episodes = 1000
action_size = 2
state_size = 4
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
device='cpu'

strategy = E_greedy(eps_start, eps_stop, eps_decay)
agent = Agent(strategy, action_size, device)
memory = Replay_Memory(memory_size)

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr= alpha)

class q_values():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)
    device = 'cpu'

    def get_current(policy_net, states, actions):
        states = torch.tensor(states, dtype = torch.float).to(device)
        actions = (torch.tensor(actions).unsqueeze(-1)).argmax(dim=-1)
        return policy_net.forward(states).gather(dim=1, index=actions.unsqueeze(-1))

    def get_next(target_net, next_states):
        next_states = torch.tensor(next_states, dtype = torch.float).to(device)
        q = policy_net.forward(next_states).detach()
        q_star = q.argmax(dim=-1)
        return q.gather(dim=1, index=q_star.unsqueeze(-1))

episode_duration = []
eps = []
for episode in range(num_episodes):

    state = env.reset()
    done = False
    counter = 0

    while not done:

        counter = counter+1
        action,ep = agent.select_action(state, policy_net)
        next_state, reward, done, info = env.step(action)
        memory.push(Experience(state, action, next_state, reward, done-1))
        state = next_state
        if memory.can_provide_sample(batch_size):

            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states, terminal_states = extract_tensors(experiences)
            terminal_states = torch.tensor(terminal_states, dtype=torch.float).unsqueeze(dim=-1).to(device)
            current_q_values = q_values.get_current(policy_net, states, actions)
            rewards = torch.tensor(rewards,dtype=torch.float).unsqueeze(dim=-1).to(device)
            next_q_values = q_values.get_next(target_net, next_states)
            target_qvalues = next_q_values * terminal_states * gamma + rewards

            loss = F.mse_loss(current_q_values, target_qvalues)
            optimizer.zero_grad()
            #loss.backward()
            #optimiser.step()
            a = list(policy_net.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            b = list(policy_net.parameters())[0].clone()
            print(torch.equal(a.data, b.data))

    eps.append(ep)
    episode_duration.append(counter)
    plot(episode_duration,ep, 100)
    #plot(eps,100)
    print(counter)
    # state = np.zeros(np.shape(state))

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
