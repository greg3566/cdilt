#!/bin/bash

# gen weight = 0.001, lr 1e-5
# 10/10 worked!

# start new tmux sesson
TAG="SR2R"
SESS_NAME="align_timestep_${TAG}_p_20_5"
DOC="empty"
VENV_DIR="~/anaconda3/envs/dail"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME -n 1

BEGIN=1
END=3
TOTAL_GPU=1

for ((i=BEGIN; i<=END; i++)); do
gpu_num=2

PYTHON_CMD="source activate ${VENV_DIR} && python train.py --algo ddpg --agent_type gama --load_dataset_dir ./expert_data/taskset/${TAG}.pickle --load_expert_dir None --save_learner_dir ./saved_alignments/${TAG}_p_20_5/seed_${i} --logdir ./logs/${TAG}_p_20_5/seed_${i} --edomain reacher2_slow_wall --ldomain reacher2_wall --seed ${i} --gpu ${gpu_num}"

if [ $i -ne $BEGIN ]
then
    tmux selectp -t $SESS_NAME:1
    tmux split-window -h
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
else
    tmux selectp -t $SESS_NAME:1
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
fi

sleep 30

tmux select-layout tiled
done
tmux a -t $SESS_NAME

