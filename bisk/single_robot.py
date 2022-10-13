# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Iterable, List

import gym
import numpy as np

from bisk.base import BiskEnv
from bisk.features import make_featurizer
from bisk.helpers import (add_ball, add_box, add_capsule, add_fwd_corridor,
                          add_robot, root_with_floor)

log = logging.getLogger(__name__)


class BiskSingleRobotEnv(BiskEnv):
    def __init__(
        self, robot: str, features: str = 'joints', allow_fallover: bool = False
    ):
        super().__init__()
        self.allow_fallover = allow_fallover

        self.robot = robot.lower()
        self.world_scale = 1.0
        if self.robot.startswith('humanoidcmu') or self.robot.startswith(
            'humanoidamass'
        ):
            # The CMU/AMASS robots are about 1.3 the size of the default
            # Humanoid
            self.world_scale = 1.3

        root = root_with_floor()
        _, robot_pos = add_robot(
            root, robot, 'robot', init=getattr(self, 'init_robot', None)
        )

        frameskip = 5
        fs = root.find('numeric', 'robot/frameskip')
        if fs is not None:
            frameskip = int(fs.data[0])
        self.init_sim(root, frameskip)
        if self.robot.startswith('halfcheetah'):
            # qpos is x_pos, z_pos, y_rot
            self.init_qpos[0] = robot_pos[0]
            self.init_qpos[1] = robot_pos[2]
        elif self.robot.startswith('walker'):
            # qpos is z_pos, x_pos, y_rot
            self.init_qpos[0] = robot_pos[2]
            self.init_qpos[1] = robot_pos[0]
        elif self.robot.startswith('humanoidcmu') or self.robot.startswith(
            'humanoidamass'
        ):
            # initialize to upright position
            self.init_qpos[0:3] = [0.0, 0.0, 1.2]
            self.init_qpos[3:7] = [0.859, 1.0, 1.0, 0.859]
        else:
            # TODO Verify that this actually corresponds to the torso position?
            self.init_qpos[0:3] = robot_pos

        self.featurizer = self.make_featurizer(features)
        self.observation_space = self.featurizer.observation_space

    @property
    def is_2d(self):
        # TODO sth more proper? But it needs to be callable from init_sim, i.e.
        # before the simulator instance is constructed.
        return (
            self.robot.startswith('halfcheetah')
            or self.robot.startswith('walker')
            or self.robot == 'testcube2d'
        )

    @property
    def robot_pos(self) -> np.ndarray:
        if self.robot.startswith('humanoidcmu') or self.robot.startswith(
            'humanoidamass'
        ):
            return self.p.named.data.xpos['robot/lowerneck']
        else:
            return self.p.named.data.xpos['robot/torso']

    @property
    def robot_speed(self) -> np.ndarray:
        return self.p.named.data.subtree_linvel['robot/torso']

    def make_featurizer(self, features: str):
        return make_featurizer(features, self.p, self.robot, 'robot')

    def reset_state(self):
        noise = 0.1
        qpos = self.init_qpos + self.np_random.uniform(
            low=-noise, high=noise, size=self.p.model.nq
        )
        qvel = self.init_qvel + noise * self.np_random.standard_normal(self.p.model.nv)
        self.p.data.qpos[:] = qpos
        self.p.data.qvel[:] = qvel
        self.featurizer.reset()

    def get_observation(self):
        return self.featurizer()

    def fell_over(self) -> bool:
        if self.robot.startswith('humanoid'):
            zpos = self.robot_pos[2]
            return bool(zpos < 0.9)
        elif self.robot.startswith('halfcheetah'):
            # Orientation pointing upwards and body almost on the ground
            up = self.p.named.data.xmat['robot/torso', 'zz']
            zpos = self.p.named.data.qpos['robot/rootz']
            if up < -0.8 and zpos < 0.12:
                return True
        elif self.robot.startswith('walker'):
            zpos = self.p.named.data.qpos['robot/rootz']
            r = self.p.named.data.qpos['robot/rooty']
            if zpos < 0.9 or r < -1.4 or r > 1.4:
                return True
        return False

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not self.allow_fallover and self.fell_over():
            terminated = True
            info['fell_over'] = True
        return obs, reward, terminated, truncated, info

    def add_box(
        self,
        root: 'dm_control.mjcf.RootElement',
        name: str,
        size: Iterable[float],
        rgba: Iterable[float] = None,
        with_body: bool = False,
        **kwargs,
    ) -> 'dm_control.mjcf.Element':
        size = [s * self.world_scale for s in size]
        if 'pos' in kwargs:
            kwargs['pos'] = [p * self.world_scale for p in kwargs['pos']]
        return add_box(root, name, size, rgba, with_body, **kwargs)

    def add_capsule(
        self,
        root: 'dm_control.mjcf.RootElement',
        name: str,
        rgba: Iterable[float] = None,
        with_body: bool = False,
        **kwargs,
    ) -> 'dm_control.mjcf.Element':
        if 'fromto' in kwargs:
            kwargs['fromto'] = [p * self.world_scale for p in kwargs['fromto']]
        if 'size' in kwargs:
            kwargs['size'] = [p * self.world_scale for p in kwargs['size']]
        if 'pos' in kwargs:
            kwargs['pos'] = [p * self.world_scale for p in kwargs['pos']]
        return add_capsule(root, name, rgba, with_body, **kwargs)

    def add_fwd_corridor(self, root: 'dm_control.mjcf.RootElement', W=4):
        return add_fwd_corridor(root, W * self.world_scale)


class BiskSingleRobotWithBallEnv(BiskSingleRobotEnv):
    def __init__(
        self, robot: str, features: str = 'joints', allow_fallover: bool = False
    ):
        super().__init__(robot, features, allow_fallover)

        self.ball_qpos_idx: List[int] = []
        self.ball_qvel_idx: List[int] = []
        if self.is_2d:
            for j in ['ball-x', 'ball-z', 'ball-ry']:
                qppos = self.p.named.model.jnt_qposadr[j]
                self.ball_qpos_idx.append(qppos)
                qvpos = self.p.named.model.jnt_dofadr[j]
                self.ball_qvel_idx.append(qvpos)
        else:
            qppos = self.p.named.model.jnt_qposadr['ball']
            for i in range(3):
                self.ball_qpos_idx.append(qppos + i)
            qvpos = self.p.named.model.jnt_dofadr['ball']
            for i in range(6):
                self.ball_qvel_idx.append(qvpos + i)

        obs_base = self.featurizer.observation_space
        obs_env = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.ball_qpos_idx) + len(self.ball_qvel_idx),),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            [('ball', obs_env), ('observation', obs_base)]
        )

    def init_sim(self, root: 'mjcf.RootElement', frameskip: int = 5):
        ball_size = 0.15 * self.world_scale
        add_ball(
            root,
            'ball',
            size=ball_size,
            mass=0.1 * self.world_scale,
            twod=self.is_2d,
        )

        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()

        # Small noise for ball
        noise = 0.01
        qpos = self.init_qpos + self.np_random.uniform(
            low=-noise, high=noise, size=self.p.model.nq
        )
        qvel = self.init_qvel + noise * self.np_random.standard_normal(self.p.model.nv)
        self.p.data.qpos[self.ball_qpos_idx] = qpos[self.ball_qpos_idx]
        self.p.data.qvel[self.ball_qvel_idx] = qvel[self.ball_qvel_idx]

    def get_observation(self):
        ball_qpos = self.p.data.qpos[self.ball_qpos_idx].ravel().copy()
        ball_qvel = self.p.data.qvel[self.ball_qvel_idx]
        # Ball X/Y position is relative to robot's torso
        ball_qpos[0] -= self.robot_pos[0]
        if not self.is_2d:
            ball_qpos[1] -= self.robot_pos[1]
        else:
            # Normalize Y rotation to [-pi,pi], as MuJoCo produces large values
            # occasionally.
            ball_qpos[2] = np.arctan2(
                np.sin(ball_qpos[2]), np.cos(ball_qpos[2])
            )

        return {
            'observation': super().get_observation(),
            'ball': np.concatenate([ball_qpos, ball_qvel]).astype(np.float32),
        }
