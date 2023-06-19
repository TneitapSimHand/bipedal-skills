# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from bisect import bisect_left

import gym
import numpy as np
from dm_control import mjcf

from bisk.helpers import asset_path
from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskGapsEnv(BiskSingleRobotEnv):
    '''
    Jump over gaps to progress in X-direction.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        allow_fallover: bool,
        shaped: bool,
        max_size: float,
        min_gap: float,
        max_gap: float,
        fixed_size: bool,
        **kwargs,
    ):
        super().__init__(robot, features, **kwargs)
        self.shaped = shaped
        self.max_size = max(0.5, max_size)
        self.fixed_size = fixed_size
        self.min_gap = min_gap
        self.max_gap = max_gap

        obs_base = self.featurizer.observation_space
        obs_env = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            [('next_gap_platform', obs_env), ('observation', obs_base)]
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        W = 8
        self.add_fwd_corridor(root, W)
        root.find('geom', 'floor').remove()

        # Base platform
        H = 0.1
        root.asset.add(
            'material',
            name='mat_base',
            reflectance=0.5,
            shininess=1,
            specular=1,
            texrepeat=[1, 1],
            texuniform=True,
            texture='tex_plane',
        )
        root.asset.add(
            'texture',
            name='tex_lava',
            type='2d',
            file=f'{asset_path()}/lava.png',
        )
        root.asset.add(
            'material',
            name='mat_gaps',
            reflectance=0.5,
            shininess=1,
            specular=1,
            texrepeat=[1, 1],
            texuniform=True,
            texture='tex_lava',
        )
        self.add_box(
            root,
            f'base',
            size=[(W + 4) / 2, W, H],
            pos=[(-W + 4) / 2, 0, -H],
            conaffinity=1,
            material='mat_base',
        )

        # 200 platforms should be enough for everybody
        self.n_platforms = 200
        root.asset.add(
            'material',
            name='mat_platform',
            reflectance=0.5,
            shininess=1,
            specular=1,
        )
        for i in range(self.n_platforms):
            o = (i % 2) * 0.1
            self.add_box(
                root,
                f'platform-{i}',
                size=[1, W, H],
                pos=[2, 0, -H],
                material='mat_platform',
                rgba=[0.2 + o, 0.3 + o, 0.4 + o, 1.0],
            )
            # Gaps are placed 5cm below
            g = self.add_box(
                root,
                f'gap-{i}',
                size=[1, W, H],
                pos=[2, 0, -H - 0.05],
                material='mat_gaps',
            )

        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()
        self.max_platforms_reached = 0
        xpos = 4 * self.world_scale
        if self.fixed_size:
            gaps = np.zeros(self.n_platforms) + self.min_gap * self.world_scale
            sizes = (
                np.zeros(self.n_platforms) + self.max_size * self.world_scale
            )
        else:
            if self.robot.startswith('quadruped'):
                gaps = (
                    self.np_random[0].uniform(0.8, 1.55, size=(self.n_platforms,))
                    * self.world_scale
                )
                ms = max(self.max_size * 2, 2.0)
                sizes = (
                    self.np_random[0].uniform(2.0, ms, size=(self.n_platforms,))
                    * self.world_scale
                )
            elif self.robot.startswith('humanoid'):
                gaps = (
                    self.np_random[0].uniform(
                        self.min_gap, self.max_gap, size=(self.n_platforms,)
                    )
                    * self.world_scale
                )
                sizes = (
                    self.np_random[0].uniform(
                        1.0, self.max_size, size=(self.n_platforms,)
                    )
                    * self.world_scale
                )
            else:
                gaps = (
                    self.np_random[0].uniform(
                        self.min_gap, self.max_gap, size=(self.n_platforms,)
                    )
                    * self.world_scale
                )
                sizes = (
                    self.np_random[0].uniform(
                        0.5, self.max_size, size=(self.n_platforms,)
                    )
                    * self.world_scale
                )

        self.gap_starts = []
        self.platform_starts = []
        for i in range(self.n_platforms):
            self.gap_starts.append(xpos)
            self.p.named.model.geom_size[f'gap-{i}'][0] = gaps[i] / 2
            self.p.named.model.geom_pos[f'gap-{i}'][0] = xpos + gaps[i] / 2
            xpos += gaps[i]
            self.platform_starts.append(xpos)
            self.p.named.model.geom_size[f'platform-{i}'][0] = sizes[i] / 2
            self.p.named.model.geom_pos[f'platform-{i}'][0] = (
                xpos + sizes[i] / 2
            )
            xpos += sizes[i]

    def next_gap_platform_index(self):
        xpos = self.robot_pos[0]
        nxp = bisect_left(self.platform_starts, xpos)
        nxg = bisect_left(self.gap_starts, xpos)
        return nxg, nxp

    def get_observation(self):
        nxg, nxp = self.next_gap_platform_index()
        xpos = self.robot_pos[0]
        if nxg < len(self.gap_starts):
            next_gap_d = self.gap_starts[nxg] - xpos
        else:
            next_gap_d = 1.0
        if nxp < len(self.platform_starts):
            next_platform_d = self.platform_starts[nxp] - xpos
        else:
            next_platform_d = 1.0
        next_platform_reached = nxp < self.max_platforms_reached
        return {
            'observation': super().get_observation(),
            'next_gap_platform': np.array(
                [next_gap_d, next_platform_d, not next_platform_reached],
                dtype=np.float32
            ),
        }

    def on_step_single_frame(self):
        for c in self.p.data.contact:
            names = self.p.named.model.name_geomadr.axes.row.names
            nams = sorted([names[c.geom1], names[c.geom2]])
            if nams[0].startswith('gap') and nams[1].startswith('robot/'):
                self.touched_gap = True
                break

    def step_simulation(self):
        super().step_simulation()
        self.max_platforms_reached = max(
            self.max_platforms_reached, self.next_gap_platform_index()[1]
        )

    def step(self, action):
        mpbefore = self.max_platforms_reached
        self.touched_gap = False
        xpos1 = self.robot_pos[0]
        return_vals = list(super().step(action)); obs, reward, terminated, truncated, info = return_vals if gym.__version__ == '0.26.1' else return_vals[:3] + [None] + [return_vals[-1]]
        xpos2 = self.robot_pos[0]

        score = 1 if self.max_platforms_reached > mpbefore else 0
        info['score'] = score
        info['shaped_reward'] = xpos2 - xpos1
        reward = info['shaped_reward'] if self.shaped else score

        if info.get('fell_over', False):
            terminated = True
            reward = -1
        if self.touched_gap:
            terminated = True
            reward = -1
            info['score'] -= 1
        return (obs, reward, terminated, truncated, info) if gym.__version__ == '0.26.1' else (obs, reward, terminated, info)
