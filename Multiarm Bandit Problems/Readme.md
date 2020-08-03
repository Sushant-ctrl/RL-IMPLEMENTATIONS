# Multi-Armed Bandit Problems

-Sushant Kumar

## INTRODUCTION:

 The problem of Multiarm Bandit commonly known as K-arm bandit problem, is a problem from the field of probablity theory and statistics. It is a topic that researchers in the field of Reinforcement learning work on.

### What is the K-Armed Bandit Problem?
 Imagine a casino in Las Vegas.
 There are k slot machines each having one lever to pull.
 You decide to play these slot machines and earn money.
 Each machine has a different distribution from which our reward is sampled.
 
 > Note: In all the slot machines referred here are stationary i.e. The distribution doesn't change with time.
 
 
 For further details on what the problem is refer the wiki page of **[Multiarmed Bandit Problems](https://en.wikipedia.org/wiki/Multi-armed_bandit)**
 
 ## Aspects on which solution of a K-Armed Bandit problem is judged.
 
 There are 3 views to each solution to the k-armed bandit problem.
 
 * Correctness Question
   Is the algorithm giving best possible reward?
 * Rate of Convergence Question
   Is the algorithm converging on a particular policy i.e. the best possible policy?
 * Sample Complexity Question
   How fast is the algorithm converges to the best solution?
   
 ## Algorithms Implemeted 
 
 The following algorithms are implemented in the above code.
 
- [x] Epsilon Greedy 
- [x] Softmax
- [x] UCB1
- [ ] Median Elemination
- [ ] Thompson Sampling
 
 ## Resources
 * [Sutton and Barto Chapter-2](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
 * [NPTEL RL by Prof. Balaraman Ravindran Week 1&2](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)


## Results:
![](https://i.imgur.com/9TrUupv.png)
