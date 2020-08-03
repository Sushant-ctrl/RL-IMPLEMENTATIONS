# TABULAR RL

Tabular RL was implemented using the frozen lake environment of the OPEN AI gym.

## IMPLEMENTATIONS

- [x] Policy evaluation and iteration
- [x] Value evaluation

## The Environment

The environment is a puzzle of a frozen lake. The agent is given a starting point denoted by 'S' and it has to reach the goal represented by 'G'. The steps that are frozen are represented by 'F' and the holes are represented by 'H'.
The agent recieves a reward 1 on reaching the goal, -1 for falling in the hole and 0 on all other steps.
The episode terminates upon reaching the hole or getting to the goal.

**There are two environments provided Deterministic and stochastic in the code you may switch between the two for implementations. There are 2 maps 4x4 and 8x8.**

## RESULTS
![](https://i.imgur.com/l9l64OA.gif)

