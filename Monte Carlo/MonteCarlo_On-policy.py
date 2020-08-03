import gym
import numpy as np
import math
import random

#### Monte Carlo using epsilon soft control

gamma = 1
epsilon = 0.6

Q = {}
P = {}
R = {}
N = {}


card_count = list(np.arange(4,22))
dealer_showing = list(np.arange(1,11))
useabe_ace = [True, False]
action_space = [0,1]
score = {'Win':1,'Draws':0,'Losses':0}

#### Initialization

for i in (card_count):
    for j in (dealer_showing):
        for k in (useabe_ace):
            for l in (action_space):
                N[(i,j,k),l] = 0
                Q[(i,j,k),l] = 0
            P[(i,j,k)] = random.choice(action_space)

env = gym.make('Blackjack-v0')
for episode in range(500000):

    state = env.reset()
    done = False
    G = []
    history  = []
    while not done:
        e = random.random()
        if  1 - epsilon/2 >= e:
            action = P[state]
        else:
            if P[state] == 1:action = 0
            else: action = 1
        next_state, reward, done, info = env.step(action)
        G.append(reward)
        history.append([state,action])
        N[(state,action)] += 1
        state = next_state
    for i in range (len(G)):
        for j in range (i,len(G)):
            G[i] = G[i] + math.pow(gamma,j-i)*G[j]
        h = tuple(history[i])
        Q[h] = (Q[h]*(N[h]-1) + G[i])/N[h]

    for i in (card_count):
        for j in (dealer_showing):
            for k in (useabe_ace):
                P[(i, j, k)] = np.argmax([Q[(i,j,k),0],Q[(i,j,k),1]])

    # if episode>=49000:
    if (reward >= 1) :
        score['Win'] += 1
    if reward == 0:
        score['Draws'] += 1
    if reward == -1:
        score['Losses'] +=1
    if episode%400 == 0:
        print(episode)
        print (score)
        print("win ratio",score['Win']/4)
        score = {'Win': 1, 'Draws': 0, 'Losses': 0}
        epsilon = epsilon - 0.01
env.close()
