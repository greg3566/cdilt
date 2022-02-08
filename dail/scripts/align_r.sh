#!/bin/bash

# gen weight = 0.001, lr 1e-5
# 10/10 worked!

DATE=0207
# get parameter
while getopts t:s:g: flag
do
    case "${flag}" in
        t) TAG=${OPTARG};;
        s) SUF=${OPTARG};;
        g) gpu_num=${OPTARG};;
    esac
done
case "${TAG}" in
    R2VS_R2) EDO="reacher2_very_slow_wall";;
    R2S_R2) EDO="reacher2_slow_wall";;
    R2_R2) EDO="reacher2_wall";;
    R2F_R2) EDO="reacher2_fast_wall";;
    R2VF_R2) EDO="reacher2_very_fast_wall";;
    R3_R2) EDO="reacher3_wall";;
esac

echo "${TAG} ${SUF} ${gpu_num} ${EDO}"

# start new tmux sesson
SESS_NAME="align_timestep_${TAG}${SUF}"
DOC="empty"
VENV_DIR="~/anaconda3/envs/dail"
tmux kill-session -t $SESS_NAME
tmux new-session -d -s $SESS_NAME -n 1

BEGIN=1
END=5
#TOTAL_GPU=1

for ((i=BEGIN; i<=END; i++)); do
# gpu_num=1

PYTHON_CMD="source activate ${VENV_DIR} && python train.py --algo ddpg --agent_type gama --load_dataset_dir ./expert_data/taskset/${DATE}/${TAG}.pickle --load_expert_dir None --save_learner_dir ./saved_alignments/${TAG}${SUF}/seed_${i} --logdir ./logs/${TAG}${SUF}/seed_${i} --edomain ${EDO} --ldomain reacher2_wall --seed ${i} --gpu ${gpu_num}"

if [ $i -ne $BEGIN ]
then
    tmux selectp -t $SESS_NAME:1
    tmux split-window -h
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
else
    tmux selectp -t $SESS_NAME:1
    tmux send-keys -t $SESS_NAME:1 "$PYTHON_CMD" "C-m"
fi

sleep 50

tmux select-layout tiled
done
tmux a -t $SESS_NAME

