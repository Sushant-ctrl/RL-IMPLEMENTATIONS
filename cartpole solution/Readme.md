# DQN LOW STATE
###### tags: `Reinforcement Learning` , `Cartpole`, `DQN`

## THE ENVIRONMENT

### Description

For studying and implementing I have used the CartPole environment of the OpenAI gym. In this environmnent there is a cart which can travel on a frictionless rail and it has a pole attached to the cart with the means of a hinge.The pole is free to rotate about this hinge.

For more details on the environment -> [CartPole Environment](https://github.com/openai/gym/wiki/CartPole-v0)

### The Task

The main task is to select actions such that the pole remains vertical on the cart for maximum time steps i.e. 200 for v-0.

### Solution

The environment is solved using **Deep Q-Learning with Experience Replay** in the low state.Low state means the observations are taken directly from the environment and not from images.

In Deep Q learning the `q(s,a)` values are estimated using a deep neural network as a function approximator. This network takes in the state values as input and gives the optimal q value corresponding to all the actions as output.

The task then turns out to train the network to give the optimal q values  q<sub>*</sub>(s,a) 

For training two networks are used, policy net and target net. The target net produces the target, towards which the policy net is trained. The parameters of the policy net are copied into the target net after certain fixed number of episodes.By doing this fixed targets are obtained to update the function approximator.

Mean squared loss and ADAM optimiser is used to train the network.



## Results

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

* OpenAI Gym
* PyTorch
* NumPy
* Matplotlib
* namedtuple 
* time
