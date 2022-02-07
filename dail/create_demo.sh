#!/bin/bash

python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_corner.npz --edomain reacher2_corner --ldomain reacher2_corner --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_wall.npz --edomain reacher2_wall --ldomain reacher2_wall --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0

python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_very_slow_corner.npz --edomain reacher2_very_slow_corner --ldomain reacher2_very_slow_corner --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_very_slow_wall.npz --edomain reacher2_very_slow_wall --ldomain reacher2_very_slow_wall --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0

python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_slow_corner.npz --edomain reacher2_slow_corner --ldomain reacher2_slow_corner --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_slow_wall.npz --edomain reacher2_slow_wall --ldomain reacher2_slow_wall --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0

python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_fast_corner.npz --edomain reacher2_fast_corner --ldomain reacher2_fast_corner --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_fast_wall.npz --edomain reacher2_fast_wall --ldomain reacher2_fast_wall --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0

python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_very_fast_corner.npz --edomain reacher2_very_fast_corner --ldomain reacher2_very_fast_corner --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_very_fast_wall.npz --edomain reacher2_very_fast_wall --ldomain reacher2_very_fast_wall --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0

python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher3_corner.npz --edomain reacher3_corner --ldomain reacher3_corner --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher3_wall.npz --edomain reacher3_wall --ldomain reacher3_wall --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0

python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_corner.npz --edomain reacher2_corner --ldomain reacher2_corner --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
python train.py --load_expert_dir None --save_dataset_dir expert_data/demo/0207/reacher2_wall.npz --edomain reacher2_wall --ldomain reacher2_wall --gpu 5 --agent_type create_demo --algo ddpg --n_demo 2000 --seed 0
