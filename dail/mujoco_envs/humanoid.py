import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    # ypos = sim.data.yipos
    # print((np.sum(mass * xpos, 0) / np.sum(mass)))
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0], (np.sum(mass * xpos, 0) / np.sum(mass))[1]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.steps = 0
        self.i_episode = 0
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        self.steps += 1
        xpos_before, ypos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        xpos_after, ypos_after = mass_center(self.model, self.sim)
        vertical_cost = (abs(ypos_after) - abs(ypos_before)) / self.dt
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (xpos_after - xpos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - vertical_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        if self.steps % 100 == 0:
            print("[%d] %.2f %.2f" %(self.i_episode, xpos_after, ypos_after))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_vertical=-vertical_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        self.steps = 0
        self.i_episode += 1
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
