[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"



# Project 3: Collaboration and Competition

## Project Details

### Introduction

The objective of the project is to train the agents to play Tennis using the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.


![tennis](assets/trained_agent.gif)
### Rewards
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

### State Space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Solution Criteria
In order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting Started

### Setup

1. Clone the repository from https://github.com/vgudapati/DRLND_Collaboration_Competetion.git
2. Setup the dependencies as described [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md).
3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
4. Place the file in the DRLND GitHub repository, in the `DRLND_Collaboration_Competetion` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

## Future work - improving agent's performance and extensions

In addition to the current work, we can do the following to improve performance:

1. Implement the other Multi-Agent algorithms such as:
    - Multi Agent PPO as presented in this paper (https://arxiv.org/pdf/1710.03748.pdf)
    - Multi Agent DQN as presented in this report (http://cs231n.stanford.edu/reports/2016/pdfs/122_Report.pdf). While using MADQN, we can try various combinations of DQN algorithms and as we know the most effective one is the rainbow method. This significantly improves performance on DQN networks. 
    - We can implement the suggestion of Gaussians mixture for action-value distribution as described in the D4PG paper along with the MAD4PG algorithm. (https://arxiv.org/pdf/1804.08617.pdf)

2. In addition to the above papers, we can use the traditional optimizations for a deep neural network by finding out the optimal learning rates, batch sizes and other hyper parameters.
