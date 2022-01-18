#!/bin/bash

# gen weight = 0.001, lr 1e-5
# 10/10 worked! the alignment is good 10/10

# start new tmux sesson
SESS_NAME="eval_t_r2r"
DOC="empty"
#VENV_DIR="../.virtualenv/cdil/bin/activate"
VENV_DIR="venv/bin/activate"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME -n 1

BEGIN=1
END=3
TOTAL_GPU=1

for ((i=BEGIN; i<=END; i++)); do
gpu_num=$((i % TOTAL_GPU))

PYTHON_CMD="source ${VENV_DIR} && python train.py --algo ddpg --agent_type zeroshot --load_expert_dir ./target_expert/reacher2_corner/alldemo --load_learner_dir ./saved_alignments/timestep/12goals/tss/seed_${i} --edomain reacher2_corner --ldomain reacher2_slow_corner --seed 100${i} --doc t_r2r"

if [ $i -ne $BEGIN ]
then
    tmux selectp -t $SESS_NAME:1
    tmux split-window -h
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
else
    tmux selectp -t $SESS_NAME:1
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
fi

sleep 0.5

tmux select-layout tiled
done
tmux a -t $SESS_NAME

