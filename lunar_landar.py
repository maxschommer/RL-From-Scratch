import gym
import numpy as np
from scipy.special import softmax

env = gym.make('LunarLander-v2')

alpha = 0.01

num_states = len(env.observation_space.high)
num_actions = env.action_space.n

theta = np.random.normal(0, 1,  size=(num_actions, num_states))
# print(theta)


for i_episode in range(2000):
    hist = []
    vt = 0
    s = env.reset()  # reset for each new trial
    for t in range(10000):
        env.render()
        pi = softmax(np.matmul(theta, s))
        # print(pi)
        a = np.random.choice(np.arange(len(pi)), p=pi)
        s_prime, r, done, info = env.step(a)
        vt += r
        hist.append((s, a, r))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        else:
            s = s_prime

    for (s, a, r) in hist:
        theta[a, :] += alpha * np.mean(theta)
        # print(s, a, r)


# Goal is to map from states to actions with some feature vector and weights.
# Features can just be all of the states, actions are the set of actions
# If there are m actions, and n state values, then then each action will
# have n weights
