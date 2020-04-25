# DQN LOW STATE

## THE ENVIRONMENT

### Description

For studying and implementing I have used the CartPole environment of the OpenAI gym. In this environmnent there is a cart which can travel on a frictionless rail and it has a pole attached to the cart with the means of a hinge.The pole is free to rotate about this hinge.

### The Task

The main task is to select actions such that the pole remains vertical on the cart for maximum time steps i.e. 200 for v-0.

### Observation

Type: Box(4)

![](https://i.imgur.com/UxZL8FP.png)

### Action

Type: Discrete(2)

![](https://i.imgur.com/ApgXudC.png)

### Reward

Reward is 1 for every step taken, including the termination step.

### Episode Termination


   1. Pole Angle is more than ±12°
   2. Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
   3. Episode length is greater than 200.
---

## Solution

The environment is solved using the **Deep Q-Learning** in the low state.Low state means the simplest possible form of state space.

Deep Q learning is a form of q learning. The `q(s,a)` values are estimated using a deep neural network as a function approximator to find the q values. This network takes in the state values as input and gives the optimal q value corresponding to all the actions as output.

The task then turns out to train the network to give the optimal q values  $q_*(s,a)$ 

For training two networks are used, policy net and target net. The target net produces the target, towards which the policy net is trained. The parameters of the policy net are copied into the target net after certain fixed number of episodes.

Mean squared losses and ADAM optimiser is used to train the network.

Experience Replay is also used.



## Training graphs and Videos

### Trained CartPole

![](https://i.imgur.com/FMBR42W.gif)

### Training curves

#### Reward vs Episode
![](https://i.imgur.com/QZflkt0.png)

#### Loss curve
![](https://i.imgur.com/uPLsonL.png)

#### Epsilon Decay
![](https://i.imgur.com/CJKwKhG.png)

## Packages used

* OPEN AI gym
* Pytorch
* Numpy
* Matplotlib
* Namedtuple 
* Time
