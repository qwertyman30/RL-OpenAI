from __future__ import print_function

import gym
from agent.dqn_agent import DQNAgentCar
from train_carracing import run_episode
from agent.networks import *
import numpy as np
import os
from datetime import datetime
import json

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    history_length = 0

    #TODO: Define networks and load agent
    # ....
    Q = CNN2(n_classes=5, history_length=history_length)
    Q_target = CNN2(n_classes=5, history_length=history_length)
    agent = DQNAgentCar(Q, Q_target, num_actions=5)
    agent.load("C:\\Users\\Monish\\Desktop\\workspace\\exercise3_R\\reinforcement_learning\\models_carracing\\dqn_agent_2.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True, history_length=history_length)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

