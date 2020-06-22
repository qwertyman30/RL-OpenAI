from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch

from agent.bc_agent import BCAgent
from utils import *

def preprocessing_state(state, history_length=1):
    state = rgb2gray(state).reshape(-1, 96, 96)
    if history_length != 0:
        state = np.concatenate([ [state[0] for i in range(history_length)], state ])
        state = np.array([state[i:i+history_length+1].T for i in range(len(state) - history_length)])
    
    return state

def id_to_action(action_id, max_speed=0.8):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0,0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0,0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=1):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    env.viewer.window.dispatch_events()

    while True:

        state = preprocessing_state(state, history_length=history_length)

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...

        state = torch.Tensor(state).cuda()
        state = state.view((-1, 1+history_length, 96, 96))
        a = agent.predict(state)
        #a = torch.nn.functional.softmax(a, dim=1)
        a = torch.max(a.data, 1)[1]
        a = id_to_action(a)

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward

if __name__ == "__main__":

    rendering = True                      
        
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = BCAgent(lr=0.001, history_length=3)
    agent.load("models\\agent1_15k_epoch94_50.pt")
    agent.net.eval()

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        print(i)
        episode_reward = run_episode(env, agent, rendering=rendering, history_length=3)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std() 

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')