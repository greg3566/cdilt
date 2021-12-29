
import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
import json, sys, os
from os import path
import time
import shutil
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import importlib
import argparse
import time
import sys
import random

# os settings
sys.path.append(os.getcwd() + '/..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Experiment modules
from dail.environment import create_env
from dail.replaymemory import create_replay_memory
from dail.agents import ddpg
from dail.utils import *
#from dail.environments import register
from dail import reacher_env

# Parse cmdline args
parser = argparse.ArgumentParser(description='Domain Adaptive Imitation Learning (DAIL)')
parser.add_argument('--logdir', default='./logs/temp', type=str)

parser.add_argument('--save_expert_dir', default='./saved_expert/temp', type=str)
parser.add_argument('--save_learner_dir', default='./saved_learner/temp', type=str)
parser.add_argument('--save_dataset_dir', default='./temp/temp.pickle', type=str)

parser.add_argument('--load_expert_dir', default='./saved_expert/temp', type=str)
parser.add_argument('--load_learner_dir', default='./saved_learner/temp', type=str)
parser.add_argument('--load_dataset_dir', default='./temp/empty.pickle', type=str)

parser.add_argument('--expert_dataset_dir', default='./saved_dataset/temp', type=str)
parser.add_argument('--learner_dataset_dir', default='./saved_dataset/temp', type=str)

parser.add_argument('--edomain', required=True, type=str)
parser.add_argument('--ldomain', required=True, type=str)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--exp_id', default='reacher', type=str)
parser.add_argument('--doc', default='', type=str)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--render', default=1, type=int)
parser.add_argument('--agent_type', type=str, required=True)
parser.add_argument('--algo', type=str, required=True)
parser.add_argument('--n_demo', type=int, default=0)



args = parser.parse_args()

# Clear out the save logs directory (for tensorboard)
if os.path.isdir(args.logdir) and args.logdir != './logs/temp':
    shutil.rmtree(args.logdir)

# GPU settings
#if args.gpu > -1:
#    print("GPU COMPATIBLE RUN...")
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Create the environment
expert_domain = args.edomain
learner_domain = args.ldomain
env_params = {'expert': expert_domain, 'learner': learner_domain}
seed_dict = {'expert': args.seed, 'learner': args.seed}
env = create_env(env_params, seed_dict)

# Seeding
print('Seeding: {}'.format(args.seed))
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Print experiment details
print('Booting with exp params: {}...'.format(args.exp_id))
time.sleep(2)

# Experiment parameters
mod = importlib.import_module('saved_params.'+args.exp_id+'_params')
params = mod.generate_params(env=env)

# Replay memory for each domain
replay_memory = create_replay_memory(env=env, params=params)

# Train the agent
agent = ddpg.DDPGAgent(params=params,
                       cmd_args=args,
                       env=env,
                       replay_memory=replay_memory,
                       save_expert_dir=args.save_expert_dir,
                       save_learner_dir=args.save_learner_dir,
                       save_dataset_dir=args.save_dataset_dir,
                       load_expert_dir=args.load_expert_dir,
                       load_learner_dir=args.load_learner_dir,
                       load_dataset_dir=args.load_dataset_dir,
                       logdir=args.logdir,
                       render=args.render,
                       gpu=args.gpu,
                       is_transfer=(args.agent_type=='transfer'))



if __name__ == '__main__':
    print('----------------------------')
    if args.agent_type == 'expert':
        print("Training Expert")
        print("Save={}".format(args.save_expert_dir))
        agent.train_expert(from_ckpt=False)

    elif args.agent_type == 'expert_from_ckpt':
        print("Training Expert from Checkpoint")
        print("Load={}".format(args.load_expert_dir))
        print("Save={}".format(args.save_expert_dir))
        agent.train_expert(from_ckpt=True)

    elif args.agent_type == 'create_alignment_taskset':
        print("Creating Alignment Taskset with")
        print("Expert={}".format(args.load_expert_dir))
        print("Self={}".format(args.load_learner_dir))
        agent.create_alignment_taskset()

    elif args.agent_type == 'gama':
        print("GAMA with")
        print("Expert={}".format(args.load_expert_dir))
        print("Self={}".format(args.save_learner_dir))
        agent.gama(from_ckpt=False)

    elif args.agent_type == 'zeroshot':
        print("Zeroshot Evaluation")
        print("Expert={}".format(args.load_expert_dir))
        print("Self={}".format(args.load_learner_dir))
        agent.zeroshot()

    elif args.agent_type == 'rollout_expert':
        print("Rollout expert({})".format(args.load_expert_dir))
        agent.rollout_expert()

    elif args.agent_type == 'create_demo':
        print("Create demonstrations dataset and save ({})".format(args.save_dataset_dir))
        agent.create_demonstrations(num_demo=args.n_demo)

    elif args.agent_type == 'bc':
        print("Behavioral Cloning on Target Expert")
        print("Dataset={}".format(args.load_dataset_dir))
        print("Save={}".format(args.save_expert_dir))
        agent.bc(num_demo=args.n_demo)

    elif args.agent_type == 'bc_from_ckpt':
        print("Behavioral Cloning on Target Expert from Checkpoint")
        print("Dataset={}".format(args.load_dataset_dir))
        print("Load={}".format(args.load_expert_dir))
        print("Save={}".format(args.save_expert_dir))
        agent.bc(from_ckpt=True)

    else:
        print("Unrecognized experiment type")
        exit(1)
    print('----------------------------')



