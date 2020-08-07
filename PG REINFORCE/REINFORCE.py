import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

gamma = 0.9

class Policy_net(nn.Module):

    def __init__(self,input,output,l1,l2,alpha):
        super(Policy_net,self).__init__()

        self.num_actions = output
        self.state_size = input
        self.linear1 = nn.Linear(self.state_size,l1)
        self.linear2 = nn.Linear(l1,l2)
        self.linear3 = nn.Linear(l2,self.num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self,state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)

        return x

    def get_action(self,state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        best_action = np.random.choice(self.num_actions, p= np.squeeze(probs.detach().numpy()))
        log_probs = torch.log(probs.squeeze(0)[best_action])

        return best_action, log_probs

def update_policy(Policy_net, rewards, log_probs):

    discounted_rewards = []

    for t in range(len(rewards)):

        Gt = 0
        pw = 0

        for r in rewards[t:]:

            Gt = Gt + gamma ** pw * r
            pw = pw + 1

        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean())/discounted_rewards.std()

    policy_grads = []

    for log_prob, Gt in zip(log_probs, discounted_rewards):

        policy_grads.append(-log_prob * Gt)

    Policy_net.optimizer.zero_grad()
    policy_grads = torch.stack(policy_grads).sum()
    policy_grads.backward()
    Policy_net.optimizer.step()