#!/bin/bash

DATE=0207

# reacher
ESUFs=("_very_slow" "_slow" "" "_fast" "_very_fast")
EDOMs=("R2VS" "R2S" "R2" "R2F" "R2VF")

for i in ${!ESUFs[@]}; do
    echo "reacher2${ESUFs[$i]}"
    python dataset_merger.py --expert_dataset_dir expert_data/demo/${DATE}/reacher2${ESUFs[$i]}_wall.npz --learner_dataset_dir expert_data/demo/${DATE}/reacher2_wall.npz --save_dataset_dir expert_data/taskset/${DATE}/${EDOMs[$i]}_R2.pickle
done

python dataset_merger.py --expert_dataset_dir expert_data/demo/${DATE}/reacher3_wall.npz --learner_dataset_dir expert_data/demo/${DATE}/reacher2_wall.npz --save_dataset_dir expert_data/taskset/${DATE}/R3_R2.pickle


# ant
python dataset_merger.py --expert_dataset_dir expert_data/demo/${DATE}/Antv5_alignment.npz --learner_dataset_dir expert_data/demo/${DATE}/Antv4_alignment.npz --save_dataset_dir expert_data/taskset/${DATE}/A5_A4.pickle

