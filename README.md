# cdilt
Cross-Domain Imitation Learning with Time-step multiplier

## TODO
1. Implement time-step multiplier (done / should be parmeterized)
2. Implement state-action entanglement (maybe done / should be parmeterized)
3. Implement entangle loss (maybe done / should be parameterized)
4. New task
5. Implement latant domain
     1. Implement cycle loss
     2. Implement interpolation loss

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

create_alignment_taskset
* Creating Alignment Taskset
* <- alignment_expert
* <- (AtLdEp)
* -> alignment_taskset

gama
* GAMA
* <- alignment_taskset
* <- alignment_expert
* -> saved_alignments

zeroshot
* Zeroshot evaluation
* <- target_expert
* <- saved_alignments

rollout_expert
* Rollout expert
* <- alignment_expert / (AtLdEp) / (TtEdEp) / (TtLdEp)

create_demo
* Create demonstratins dataset and save
* <- (TtEdEp)
* -> target_demo

bc
* Behavioral Cloning on Target Expert
* <- target_demo
* -> target_expert

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
