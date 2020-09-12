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

It should produce a score (best average reward of 100) of 9.12

Of course, I cheated by using a good seed.  If you don't cheat, the typical score is more like 9.0

