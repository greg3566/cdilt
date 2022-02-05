# cdilt
Cross-Domain Imitation Learning with Time-step multiplier
https://www.notion.so/CDIL-3f91f7e5aa10466a9d958b660cae3486

## TODO
1. Implement time-step multiplier (done / should be parmeterized)
2. Implement state-action entanglement (maybe done / should be parmeterized)
3. Implement entangle loss (maybe done / should be parameterized)
4. New task
5. Implement latant domain
     1. Implement cycle loss
     2. Implement interpolation loss
6. training multi joint agent
     1. 4-legged Ant (original) (Eevee)
     2. 3-legged Ant (Eevee)
     3. 6-legged Ant (Togepi)
     4. Humanoid (jeongwoo local)

# dail
## directories
alignment_expert
* Alignment task, Expert domain, Expert policy
* -> gama

saved_alignments
* alignment from Expert domain to Learner domain
* <- gama
* -> zeroshot

target_demo
* Target task Expert domain Expert demonstratoin
* <- create_demo
* -> bc

target_expert
* Target task Expert domain BC policy
* <- bc
* -> zeroshot

## agent_type
expert
* Training Expert
* -> alignment_expert / (AtLdEp) / (TtEdEp) / (TtLdEp)
* -> input: task, output: policy

create_alignment_taskset
* Creating Alignment Taskset
* <- alignment_expert
* <- (AtLdEp)
* -> alignment_taskset
* -> input: policy, output: demonstration set(Dx, Dy)

gama
* GAMA
* <- alignment_taskset
* <- alignment_expert
* -> saved_alignments
* -> input: demonstration set, expert policy output: f,g

zeroshot
* Zeroshot evaluation
* <- target_expert
* <- saved_alignments
* -> input: bc_policy(pi_y,tau), f, g, output: pi_x,tau(but not save) and metric

rollout_expert
* Rollout expert
* <- alignment_expert / (AtLdEp) / (TtEdEp) / (TtLdEp)
* -> input: policy, domain, task, output: reward

create_demo
* Create demonstratins dataset and save
* <- (TtEdEp)
* -> target_demo
* -> input: expert policy, domain, output: demo(D_y,tau)

bc
* Behavioral Cloning on Target Expert
* <- target_demo
* -> target_expert
* -> input: expert domain demo, output: policy(pi_y,tau)

## How to make baseline (Target task, Learner domain)
1. Random  
    0.0 for scaled performance
    1. ??
2. Self Demonstrations  
    upper bound : bc from self domain demonstration
    1. train --agent_type=expert with Target task Learner domain
    2. change names
    3. train --agent_type=create_demo
    4. train --agent_type=bc
3. Expert  
    1. train --agent_type=expert with Target task Learner domain


## Env settings for multi joint agent
'''
cp -R mujoco_envs {ENV}/lib/python3.7/site-packages/gym/envs/mujoco
'''
and run with spinning up codes
(will be updated more detail)

'''
ssh jaeseok@147.46.111.251
jaeseok
conda activate cdil
'''

environment setting
'''
python -m spinup.run sac_tf1 --env Antv1_1-v0 --exp_name Antv1_1 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_2-v0 --exp_name Antv1_2 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_3-v0 --exp_name Antv1_3 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_4-v0 --exp_name Antv1_4 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_5-v0 --exp_name Antv1_5 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_6-v0 --exp_name Antv1_6 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_7-v0 --exp_name Antv1_7 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_8-v0 --exp_name Antv1_8 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_9-v0 --exp_name Antv1_9 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_10-v0 --exp_name Antv1_10 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_11-v0 --exp_name Antv1_11 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv1_12-v0 --exp_name Antv1_12 --epochs 1000 --save_freq 10

python -m spinup.run sac_tf1 --env Antv2_1-v0 --exp_name Antv2_1 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_2-v0 --exp_name Antv2_2 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_3-v0 --exp_name Antv2_3 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_4-v0 --exp_name Antv2_4 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_5-v0 --exp_name Antv2_5 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_6-v0 --exp_name Antv2_6 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_7-v0 --exp_name Antv2_7 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_8-v0 --exp_name Antv2_8 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_9-v0 --exp_name Antv2_9 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_10-v0 --exp_name Antv2_10 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_11-v0 --exp_name Antv2_11 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv2_12-v0 --exp_name Antv2_12 --epochs 1000 --save_freq 10

python -m spinup.run sac_tf1 --env Antv3_1-v0 --exp_name Antv3_1 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_2-v0 --exp_name Antv3_2 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_3-v0 --exp_name Antv3_3 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_4-v0 --exp_name Antv3_4 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_5-v0 --exp_name Antv3_5 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_6-v0 --exp_name Antv3_6 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_7-v0 --exp_name Antv3_7 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_8-v0 --exp_name Antv3_8 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_9-v0 --exp_name Antv3_9 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_10-v0 --exp_name Antv3_10 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_11-v0 --exp_name Antv3_11 --epochs 1000 --save_freq 10
python -m spinup.run sac_tf1 --env Antv3_12-v0 --exp_name Antv3_12 --epochs 1000 --save_freq 10
'''
