#!/bin/bash

# gen weight = 0.001, lr 1e-5
# 10/10 worked!

# start new tmux sesson
SESS_NAME="align_anttest"
DOC="empty"
VENV_DIR="venv/bin/activate"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME -n 1

BEGIN=2
END=3
TOTAL_GPU=1
GPUS=(0)

for ((i=BEGIN; i<=END; i++)); do
gpu_num=${GPUS[$((i % TOTAL_GPU))]}

PYTHON_CMD="source ${VENV_DIR} && CUDA_VISIBLE_DEVICES=${gpu_num} python train.py --algo ddpg --agent_type gama --load_dataset_dir ./expert_data/taskset/Antv1test.pickle --load_expert_dir None --save_learner_dir ./saved_alignments/Antv1test/seed_${i} --logdir ./logs/Antv1test/seed_${i} --edomain Antv1_1 --ldomain Antv1_1 --seed ${i} --gpu ${gpu_num}"

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

