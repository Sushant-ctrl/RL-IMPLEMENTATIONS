# Monte Carlo Methods

---
###### Tags: `Monte Carlo` `Model Free Control` `Model Free Prediction`

---

## Overview

Monte Carlo methods require only experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Learning from actual experience is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior.

Monte Carlo methods work for episodic tasks i.e. all the episodes must terminate at some finite time.

All implementations are done on the `Blackjack-v0` env of the OpenAI gym.

The following concepts are implemented
- On-Policy 
  - [x] Estimation of Q values
  - [ ] Control assuming exploring starts
  - [x] Control using $\epsilon$-soft policies
- Off-Policy
  - [x] Estimation using Importance Sampling
  - [X] Control using weighted Importance Sampling

### On-Policy
On policy is quite intutive to understand and use we use the current policy to calculate the returns and we calculate the action and state value functions of the current policy and then use it to improve the policy.

### Off-Policy
Speaking in layman terms in off policy the value functions of any policy is calculated using another are calculated following some other policy and then scaling them accordingly.

### Results for Off Policy
- **With no useabe Ace**
![](https://i.imgur.com/G7nNVj1.png)
- **With useable Ace**
![](https://i.imgur.com/Agt80RF.png)


#### Resources For Learning:
- For detailed proofs and theory refer [Sutton and Barto](http://incompleteideas.net/book/RLbook2018.pdf)
- For summaries and quick read refer [Slides of Elena and Xi](https://www.kth.se/social/files/58b941d5f276542843812288/RL04-Monte-Carlo.pdf)
- For blackjack environment refer the [Github page of OpenAi gym](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py)
