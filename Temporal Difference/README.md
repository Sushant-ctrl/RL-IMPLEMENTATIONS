# Temporal Difference Methods

---

###### Tags: `TD Learning` `Model Free Control` `Model Free Prediction`

### Overview 
TD learning mehtods are methods that are model free and don't require the task to be episodic. This method can be applied for evaluation and control of continuing tasks.
It is computationally cheaper than the Monte Carlo mehtod due to the fact that the updates are made online and we need not wait till the end of every episode.
TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).
Here the target is value function of the next state and not the true return.

>Which is better TD or MC, is a Question open for debate and research

### Implementations 

- [x] SARSA
- [x] Q-Learning
- [x] Comparitive study of SARSA and Q-Learning

### Results

- SARSA on WindyGridworld

![](https://i.imgur.com/CZCvWwi.png)
 
- Q-Learning on Cliff-walk

![](https://i.imgur.com/VexuFPE.png)

- Comparision of Q-learning and SARSA on Cliff walk 
  The red one is from Q-Learning and the blue from SARSA
![](https://i.imgur.com/TT9W4bi.png)

### Resources 

- For detailed proofs and theory refer [Sutton and Barto.](http://incompleteideas.net/book/RLbook2018.pdf)
- For understanding the environment visit the [github page.](https://github.com/podondra/gym-gridworlds)
