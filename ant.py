import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        # original
        # mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        # utils.EzPickle.__init__(self)

        # single goal
        self.goal = [1.0, 1.0]
        print("START Ant Env (updated 2022.01.12. 19:50 KST)")
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

        # multi goal
        # self.goal = [1.0, 1.0]
        # mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        # utils.EzPickle.__init__(self)
        

    def step(self, a):
        self.steps += 1
        # original
        # xposbefore = self.get_body_com("torso")[0]
        # self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        # forward_reward = (xposafter - xposbefore) / self.dt
        # ctrl_cost = 0.5 * np.square(a).sum()
        # contact_cost = (
        #     0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # )
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        # done = not notdone
        # ob = self._get_obs()
        # if self.steps % 100 == 0:
        #     print("[%d] %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1]))
        # return (
        #     ob,
        #     reward,
        #     done,
        #     dict(
        #         reward_forward=forward_reward,
        #         reward_ctrl=-ctrl_cost,
        #         reward_contact=-contact_cost,
        #         reward_survive=survive_reward,
        #     ),
        # )

        # no goal but constrain direction
        # xposbefore = self.get_body_com("torso")[0]
        # yposbefore = self.get_body_com("torso")[1]
        # self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        # yposafter = self.get_body_com("torso")[1]
        # forward_reward = (xposafter - xposbefore) / self.dt
        # vertical_cost = (abs(yposafter) - abs(yposbefore)) / self.dt
        # ctrl_cost = 0.5 * np.square(a).sum()
        # contact_cost = (
        #     0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # )
        # survive_reward = 1.0
        # reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        # done = not notdone
        # ob = self._get_obs()
        # if self.steps % 100 == 0:
        #     print("[%d] %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1]))
        # return (
        #     ob,
        #     reward,
        #     done,
        #     dict(
        #         reward_forward=forward_reward,
        #         reward_vertical=-vertical_cost,
        #         reward_ctrl=-ctrl_cost,
        #         reward_contact=-contact_cost,
        #         reward_survive=survive_reward,
        #     ),
        # )

        # single goal
        xposbefore = self.get_body_com("torso")[0:2]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0:2]
        vecbefore = xposbefore - self.goal
        vecafter = xposafter - self.goal
        dist_reward = (np.linalg.norm(vecbefore) - np.linalg.norm(vecafter)) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = dist_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        if self.steps % 100 == 0:
            print("[%d] dist: %.3f" %(self.i_episode, np.linalg.norm(vecafter)))
        if np.linalg.norm(vecafter) < 0.1:
            done = True
            reward += (self.max_steps - self.steps) * 1.0
            print("Success")
            print(xposafter)
            print(np.linalg.norm(vecafter))
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_dist=dist_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

        # multi goal
        torso_pos = self.get_body_com("torso")[0:2]
        vec = torso_pos - self.goal
        dist_cost = np.linalg.norm(vec)
        self.do_simulation(a, self.frame_skip)
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = - dist_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        if dist_cost < 0.1:
            done = True
            print("Success")
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_dist=-dist_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        # original
        # return np.concatenate(
        #     [
        #         self.sim.data.qpos.flat[2:],
        #         self.sim.data.qvel.flat,
        #         np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        #     ]
        # )

        # single goal
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0
        # original
        # qpos = self.init_qpos + self.np_random.uniform(
        #     size=self.model.nq, low=-0.1, high=0.1
        # )
        # qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        # self.set_state(qpos, qvel)
        # return self._get_obs()

        # single goal
        self.goal = [2.0, 0.0]
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

