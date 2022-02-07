import numpy as np
import imageio
from pdb import set_trace
import os
import shelve
from collections import deque
import argparse

# Parse cmdline args
parser = argparse.ArgumentParser(description='Domain Adaptive Imitation Learning (DAIL)')
parser.add_argument('--expert_dataset_dir', default='./saved_dataset/temp', type=str)
parser.add_argument('--learner_dataset_dir', default='./saved_dataset/temp', type=str)
parser.add_argument('--save_dataset_dir', default='./saved_dataset/temp', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    num_transitions = int(1e7)
    ## expert domain
    npzfile = np.load(args.expert_dataset_dir, allow_pickle=True)
    obs = npzfile['obs']
    acs = npzfile['acs']
    nacs = npzfile['nacs']
    tot_reward = npzfile['tot_reward']
    expert_deque = deque(maxlen=num_transitions)
    steps = 0
    for i in range(obs.shape[0]):
        if steps>=num_transitions:
            continue
        for j in range(obs[i].shape[0] - 1):
            done = False
            expert_transition = (
                obs[i][j], nacs[i][j], 0., obs[i][j + 1],
                0.0 if done else 1.0, acs[i][j], 0., 0., 0.)
            steps += 1
            expert_deque.append(expert_transition)
    print("-----------------------")
    print("Expert")
    print("Num Transitions: {}".format(steps))
    print("Avg Reward: {}".format(np.mean(tot_reward)))

    ## learner domain
    npzfile = np.load(args.learner_dataset_dir, allow_pickle=True)
    obs = npzfile['obs']
    acs = npzfile['acs']
    nacs = npzfile['nacs']
    tot_reward = npzfile['tot_reward']
    learner_deque = deque(maxlen=num_transitions)
    steps = 0
    for i in range(obs.shape[0]):
        if steps >= num_transitions:
            continue
        ep_reward = 0.
        for j in range(obs[i].shape[0] - 1):
            done = False
            learner_transition = (
                obs[i][j], nacs[i][j], 0., obs[i][j + 1],
                0.0 if done else 1.0, acs[i][j], 0., 0., 0.)
            steps += 1
            learner_deque.append(learner_transition)
    print("-----------------------")
    print("Learner")
    print("Num Transitions: {}".format(steps))
    print("Avg Reward: {}".format(np.mean(tot_reward)))

    hybrid_dataset = shelve.open(args.save_dataset_dir, writeback=True)
    hybrid_dataset['expert'] = expert_deque
    hybrid_dataset['learner'] = learner_deque
    hybrid_dataset.close()
