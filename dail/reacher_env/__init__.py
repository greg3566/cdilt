from gym.envs.registration import register

# MuJoCO environments
#register(id='Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof:Reacher2DOFEnv', max_episode_steps=60)
#register(id='Reacher2DOFCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_corner:Reacher2DOFCornerEnv', max_episode_steps=60)
#register(id='Reacher2DOFWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_wall:Reacher2DOFWallEnv', max_episode_steps=60)

# dynamics altered
register(id='Reacher2DOFSto-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_sto:Reacher2DOFStoEnv', max_episode_steps=60)
register(id='Reacher2DOFAct-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act:Reacher2DOFActEnv', max_episode_steps=60)
register(id='Reacher2DOFActCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act_corner:Reacher2DOFActCornerEnv', max_episode_steps=60)
register(id='Reacher2DOFActWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act_wall:Reacher2DOFActWallEnv', max_episode_steps=60)

# embodiment altered
#register(id='Reacher3DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof:Reacher3DOFEnv', max_episode_steps=60)
#register(id='Reacher3DOFCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof_corner:Reacher3DOFCornerEnv', max_episode_steps=60)
#register(id='Reacher3DOFWall-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof_wall:Reacher3DOFWallEnv', max_episode_steps=60)

# push environments
register(id='Reacher2DOFPush-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_push:Reacher2DOFPushEnv', max_episode_steps=500)
register(id='Reacher2DOFActPush-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act_push:Reacher2DOFActPushEnv', max_episode_steps=500)
register(id='Reacher3DOFPush-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof_push:Reacher3DOFPushEnv', max_episode_steps=500)

# viewpoint altered
register(id='TP_Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.tp_reacher_2dof:TP_Reacher2DOFEnv', max_episode_steps=60)
register(id='TP_WRITE_Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.tp_write_reacher_2dof:TP_WRITE_Reacher2DOFEnv', max_episode_steps=500)
register(id='WRITE_Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.write_reacher_2dof:WRITE_Reacher2DOFEnv', max_episode_steps=500)


# Longer reachers
register(id='Reacher4DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_4dof:Reacher4DOFEnv', max_episode_steps=60)
register(id='Reacher5DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_5dof:Reacher5DOFEnv', max_episode_steps=70)
register(id='Reacher6DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_6dof:Reacher6DOFEnv', max_episode_steps=80)

# Timestep altered
register(id='Reacher2DOFSlow-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_slow:Reacher2DOFSlowEnv', max_episode_steps=150)
register(id='Reacher2DOFSlowCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_slow_corner:Reacher2DOFSlowCornerEnv', max_episode_steps=150)
register(id='Reacher2DOFSlowWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_slow_wall:Reacher2DOFSlowWallEnv', max_episode_steps=150)

# Ant v1 (Original)
register(id='Antv1_1-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_1', max_episode_steps=1000)
register(id='Antv1_2-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_2', max_episode_steps=1000)
register(id='Antv1_3-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_3', max_episode_steps=1000)
register(id='Antv1_4-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_4', max_episode_steps=1000)
register(id='Antv1_5-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_5', max_episode_steps=1000)
register(id='Antv1_6-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_6', max_episode_steps=1000)
register(id='Antv1_7-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_7', max_episode_steps=1000)
register(id='Antv1_8-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_8', max_episode_steps=1000)
register(id='Antv1_9-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_9', max_episode_steps=1000)
register(id='Antv1_10-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_10', max_episode_steps=1000)
register(id='Antv1_11-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_11', max_episode_steps=1000)
register(id='Antv1_12-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_12', max_episode_steps=1000)
register(id='Antv1_multi_goal-v0', entry_point='dail.reacher_env.mujoco.ant:Antv1_multi_goal', max_episode_steps=1000)


# Ant v2 (6legged)
register(id='Antv2_1-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_1', max_episode_steps=1000)
register(id='Antv2_2-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_2', max_episode_steps=1000)
register(id='Antv2_3-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_3', max_episode_steps=1000)
register(id='Antv2_4-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_4', max_episode_steps=1000)
register(id='Antv2_5-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_5', max_episode_steps=1000)
register(id='Antv2_6-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_6', max_episode_steps=1000)
register(id='Antv2_7-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_7', max_episode_steps=1000)
register(id='Antv2_8-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_8', max_episode_steps=1000)
register(id='Antv2_9-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_9', max_episode_steps=1000)
register(id='Antv2_10-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_10', max_episode_steps=1000)
register(id='Antv2_11-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_11', max_episode_steps=1000)
register(id='Antv2_12-v0', entry_point='dail.reacher_env.mujoco.ant_6legged:Antv2_12', max_episode_steps=1000)


# Ant v3 (long leg)
register(id='Antv3_1-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_1', max_episode_steps=1000)
register(id='Antv3_2-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_2', max_episode_steps=1000)
register(id='Antv3_3-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_3', max_episode_steps=1000)
register(id='Antv3_4-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_4', max_episode_steps=1000)
register(id='Antv3_5-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_5', max_episode_steps=1000)
register(id='Antv3_6-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_6', max_episode_steps=1000)
register(id='Antv3_7-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_7', max_episode_steps=1000)
register(id='Antv3_8-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_8', max_episode_steps=1000)
register(id='Antv3_9-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_9', max_episode_steps=1000)
register(id='Antv3_10-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_10', max_episode_steps=1000)
register(id='Antv3_11-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_11', max_episode_steps=1000)
register(id='Antv3_12-v0', entry_point='dail.reacher_env.mujoco.ant_long:Antv3_12', max_episode_steps=1000)


# Ant v4
register(id='Antv4_alignment-v0', entry_point='dail.reacher_env.mujoco.ant_reacher:Antv4_alignment', max_episode_steps=200)
register(id='Antv4_target-v0', entry_point='dail.reacher_env.mujoco.ant_reacher:Antv4_target', max_episode_steps=200)

# Ant v5
register(id='Antv5_alignment-v0', entry_point='dail.reacher_env.mujoco.ant_6legged_reacher:Antv5_alignment', max_episode_steps=200)
register(id='Antv5_target-v0', entry_point='dail.reacher_env.mujoco.ant_6legged_reacher:Antv5_target', max_episode_steps=200)

# reacher
register(id='Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof:Reacher2DOFEnv', max_episode_steps=60)
register(id='Reacher2DOFCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof:Reacher2DOFCornerEnv', max_episode_steps=60)
register(id='Reacher2DOFWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof:Reacher2DOFWallEnv', max_episode_steps=60)


register(id='Reacher3DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof:Reacher3DOFEnv', max_episode_steps=60)
register(id='Reacher3DOFCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof:Reacher3DOFCornerEnv', max_episode_steps=60)
register(id='Reacher3DOFWall-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof:Reacher3DOFWallEnv', max_episode_steps=60)

register(id='Reacher2DOFVerySlowCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_very_slow:Reacher2DOFVerySlowCornerEnv', max_episode_steps=240)
register(id='Reacher2DOFVerySlowWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_very_slow:Reacher2DOFVerySlowWallEnv', max_episode_steps=240)

register(id='Reacher2DOFSlowCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_slow:Reacher2DOFSlowCornerEnv', max_episode_steps=120)
register(id='Reacher2DOFSlowWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_slow:Reacher2DOFSlowWallEnv', max_episode_steps=120)

register(id='Reacher2DOFFastCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_fast:Reacher2DOFFastCornerEnv', max_episode_steps=30)
register(id='Reacher2DOFFastWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_fast:Reacher2DOFFastWallEnv', max_episode_steps=30)


register(id='Reacher2DOFVeryFastCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_very_fast:Reacher2DOFVeryFastCornerEnv', max_episode_steps=15)
register(id='Reacher2DOFVeryFastWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_very_fast:Reacher2DOFVeryFastWallEnv', max_episode_steps=15)

