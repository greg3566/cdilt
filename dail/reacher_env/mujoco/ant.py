import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
import random


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = math.pi / 4
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
        

    def _step(self, a):
        self.steps += 1
        # original
        # xposbefore = self.get_body_com("torso")[0]
        # self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        # forward_reward = (xposafter - xposbefore) / self.dt
        # ctrl_cost = 0.5 * np.square(a).sum()
        # contact_cost = (
        #     0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        # )
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        # done = not notdone
        # ob = self._get_obs()
        # if self.steps % 100 == 0 and False:
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
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        delta_x = xposafter - xposbefore
        delta_y = yposafter - yposbefore
        forward_reward = (delta_x * math.cos(self.theta) + delta_y * math.sin(self.theta)) / self.dt
        vertical_cost = (abs(-xposafter * math.sin(self.theta) + yposafter * math.cos(self.theta)) - abs(-xposbefore * math.sin(self.theta) + yposafter * math.cos(self.theta))) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        print(ob)
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

        # single goal
        # xposbefore = self.get_body_com("torso")[0:2]
        # self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0:2]
        # vecbefore = xposbefore - self.goal
        # vecafter = xposafter - self.goal
        # dist_reward = (np.linalg.norm(vecbefore) - np.linalg.norm(vecafter)) / self.dt
        # ctrl_cost = 0.5 * np.square(a).sum()
        # contact_cost = (
        #     0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        # )
        # survive_reward = 1.0
        # reward = dist_reward - ctrl_cost - contact_cost + survive_reward
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        # done = not notdone
        # if self.steps % 100 == 0 and False:
        #     print("[%d] dist: %.3f" %(self.i_episode, np.linalg.norm(vecafter)))
        # if np.linalg.norm(vecafter) < 0.1:
        #     done = True
        #     reward += (self.max_steps - self.steps) * 1.0
        #     print("Success")
        #     print(xposafter)
        #     print(np.linalg.norm(vecafter))
        # ob = self._get_obs()
        # return (
        #     ob,
        #     reward,
        #     done,
        #     dict(
        #         reward_dist=dist_reward,
        #         reward_ctrl=-ctrl_cost,
        #         reward_contact=-contact_cost,
        #         reward_survive=survive_reward,
        #     ),
        # )

        # multi goal
        torso_pos = self.get_body_com("torso")[0:2]
        vec = torso_pos - self.goal
        dist_cost = np.linalg.norm(vec)
        self.do_simulation(a, self.frame_skip)
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
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
        #         self.model.data.qpos.flat[2:],
        #         self.model.data.qvel.flat,
        #         np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        #     ]
        # )
        # single goal
        return np.concatenate(
            [
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0
        self.theta = math.pi / 4

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

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 10.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -10.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 80.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 100.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 170.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_6(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -170.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_7(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -80.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_8(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -100.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_9(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 45.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_10(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = 135.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_11(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -45.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_12(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta = -135.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.steps += 1

        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        if self.steps % 100 == 0 and False:
            print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)


class Antv1_multi_goal(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.i_episode = 0
        self.max_steps = 1000
        self.steps = 0
        self.theta_list = [10, -10, 80, 100, -80, -100, 170, -170, 45, 135, -45, -135]
        self.theta = 0.0 / 180.0 * math.pi
        self.st = math.sin(self.theta)
        self.ct = math.cos(self.theta)
        self.diverge_penalty_ratio = 2.0

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)        

    def _step(self, a):
        self.steps += 1
        
        # no goal but constrain direction
        xb = self.get_body_com("torso")[0]
        yb = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xa = self.get_body_com("torso")[0]
        ya = self.get_body_com("torso")[1]
        dx = xa - xb
        dy = ya - yb
        forward_reward = (dx * self.ct + dy * self.st) / self.dt
        vertical_cost = (abs(-xa * self.st + ya * self.ct) - abs(-xb * self.st + yb * self.ct)) / self.dt
        if vertical_cost > 0:
            vertical_cost = vertical_cost * self.diverge_penalty_ratio
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - vertical_cost - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        # if self.steps % 100 == 0 and False:
        #     print("[%d] %.2f %.2f %.2f" %(self.i_episode, self.get_body_com("torso")[0], self.get_body_com("torso")[1], math.atan2(self.get_body_com("torso")[1], self.get_body_com("torso")[0])))
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_vertical=-vertical_cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )


    def _get_obs(self):
        # single goal
        x = self.model.data.qpos.flat[0]
        y = self.model.data.qpos.flat[1]
        r = math.sqrt(x ** 2 + y ** 2)
        ct = x / r
        st = y / r
        return np.concatenate(
            [
                [r, ct, st],
                self.model.data.qpos.flat[2:],
                self.model.data.qvel.flat,
                np.clip(self.model.data.cfrc_ext, -1, 1).flat,
                [self.ct, self.st],
            ]
        )

    def reset_model(self):
        self.i_episode += 1
        self.steps = 0
        idx = random.randint(0, 11)
        self.theta = self.theta_list[idx] / 180.0 * math.pi
        self.ct = math.cos(self.theta)
        self.st = math.sin(self.theta)
        print("[%d] direction %.2f" %(self.i_episode, self.theta))

        # single goal
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()



    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_state_from_obs(self, obs):
        r = obs[0]
        ct = obs[1]
        st = obs[2]
        pos = [r * ct, r * st]
        qpos = np.concatenate([pos, obs[3:16]], axis=0)
        qvel = obs[16:30]

        self.set_state(qpos, qvel)
