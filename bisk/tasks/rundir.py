# Copyright (c) 2022-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from bisk.single_robot import BiskSingleRobotEnv


class BiskRunDirEnv(BiskSingleRobotEnv):
    '''
    Dense-reward task: move at a specific angle.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        allow_fallover: bool,
        heading_deg: float,
        **kwargs):
        super().__init__(robot, features, allow_fallover, **kwargs)

        heading = np.deg2rad(heading_deg)
        # torso orientation: X/Y are switched
        self.dir = np.asarray([np.sin(heading), np.cos(heading), 0])

    def step(self, action):
        pos_before = self.robot_pos.copy()
        return_vals = list(super().step(action)); obs, reward, terminated, truncated, info = return_vals if gym.__version__ == '0.26.1' else return_vals[:3] + [None] + [return_vals[-1]]
        pos_after = self.robot_pos
        displacement = pos_after - pos_before

        rdir = np.dot(self.dir, self.p.named.data.xmat['robot/torso'].reshape(3,3))
        reward = np.dot(rdir, displacement)

        if info.get('fell_over', False):
            terminated = True
            reward = -1
        return (obs, reward, terminated, truncated, info) if gym.__version__ == '0.26.1' else (obs, reward, terminated, info)
