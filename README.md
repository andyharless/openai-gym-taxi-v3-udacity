# openai-gym-taxi-v3-udacity

## Attempts to solve OpenAI Gym's Taxi-v3 RL environment

The skeleton of this code is from [Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/lab-taxi).  Their version uses Taxi-v2, but this version uses v3, since v2 is deprecated.  (But I also did version 2 [here](https://github.com/andyharless/openai-gym-taxi-v2-udacity).)

The environment is from [here](https://gym.openai.com/envs/Taxi-v3/)

To do the simple demo, on Linux or Mac with Docker installed, make `taxi.sh` executable and run it:
```
git clone https://github.com/andyharless/openai-gym-taxi-v3-udacity.git
cd openai-gym-taxi-v3-udacity
chmod u+x taxi.sh
./taxi.sh
```

It should produce a score (best average reward of 100) of 9.24

(I cheated by using a good seed.)

This version uses a variation on standard Q-learning.  The policy is epsilon-greedy, but when the non-greedy action is chosen, instead of being sampled from a uniform distribution, it is sampled from a distribution that reflects two things:
   - a preference for actions with higher Q values (i.e. "greedy but flexible")
   - a preference for novel actions (those that have recently been less often chosen in the current state)
The latter are tracked via a "path memory" table (same shape as the Q table), which counts how often each action is taken in each state.  At the end of each episode, path memories from the previous episode decay geometrically.  The sampling distribution for stochastic actions is the softmax of a linear combination of the Q values (with a positive coefficient) and the path memory values (with a negative coefficient).
