# Copyright (c) 2022-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from bisect import bisect_left
from typing import List

import gym
import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import mjlib

from bisk.features import make_featurizer
from bisk.helpers import add_capsule, asset_path
from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskButterfliesEnv(BiskSingleRobotEnv):
    '''
    Chasing butterflies. This is similar to more classic food gathering tasks,
    but in three dimensions. The (humanoid) robot is equipped with a dip net and
    has to collect as many "butterflies" (spheres floating in the air) as
    possible within an episode. The butterflies are projected to a sphere around
    the robot's head, and a fixed-size long/lat grid contains distances to the
    closest butterflies in that direction.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        allow_fallover: bool,
        goal_area: int,
        n_butterflies: int,
        zoff: int,
        shaped: bool = False,
        **kwargs,
    ):
        self.n_butterflies = n_butterflies
        super().__init__(robot, features, allow_fallover, **kwargs)
        self.goal_area = goal_area
        self.shaped = shaped
        self.zoff = zoff

        obs_base = self.featurizer.observation_space
        obs_env = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10 * 10,),  # 10x10 long/lat grid
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            [('butterflies', obs_env), ('observation', obs_base)]
        )

    def make_featurizer(self, features: str):
        return make_featurizer(
            features, self.p, self.robot, 'robot', exclude=r'robot/net'
        )

    def init_robot(self, robot: 'mjcf.RootElement', name: str):
        size = 0.01 * self.world_scale
        if self.robot in {'humanoid', 'humanoidpc'}:
            raise NotImplementedError()
        elif self.robot in {'humanoidcmupc', 'humanoidamasspc'}:
            hand = robot.find('body', 'lhand')
            handg = hand.find('geom', 'lhand')
            zpos = handg.size[0]
            net = hand.add(
                'body',
                name='net',
                pos=handg.pos + [0, 0, -zpos / 2],
                xyaxes=[0, -1, 0, 0, 0, 1],
            )
            dclass = handg.dclass.dclass
        elif self.robot == 'testcube':
            torso = robot.find('body', 'torso')
            torsog = torso.find('geom', 'torso')
            zpos = torsog.size[2]
            net = torso.add('body', name='net', pos=[0, 0, zpos])
            if torsog.dclass:
                dclass = torsog.dclass.dclass
            else:
                dclass = None
        else:
            raise NotImplementedError(
                'Humanoid robot required for BiskButterfliesEnv'
            )

        robot.asset.add(
            'texture',
            name='tex_net',
            type='2d',
            file=f'{asset_path()}/net.png',
        )
        robot.asset.add(
            'material',
            name='mat_net',
            reflectance=0.5,
            shininess=0.2,
            specular=1,
            texrepeat=[10, 10],
            texuniform=False,
            texture='tex_net',
        )

        net_length = 0.5 * self.world_scale
        net_mass = 0.01
        if self.robot == 'testcube':
            net_radius = 0.5 * self.world_scale
        else:
            net_radius = 0.15 * self.world_scale
        net.add(
            'geom',
            name='net_handle_geom',
            type='capsule',
            fromto=[0, 0, 0, 0, 0, net_length],
            size=[size],
            mass=net_mass,
            dclass=dclass,
        )
        net.add(
            'geom',
            name='net_geom',
            type='ellipsoid',
            pos=[0, 0, net_length + net_radius],
            size=(net_radius, net_radius, 1e-3),
            xyaxes=[1, 0, 0, 0, 0, 1],
            mass=net_mass,
            dclass=dclass,
            contype=3,  # collide with body and butterflies
            material='mat_net',
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        try:
            from matplotlib import pyplot as plt

            cmap = plt.get_cmap('rainbow')
        except:
            cmap = lambda x: [1, 0, 0, 1]

        root.asset.add(
            'material',
            name='butterfly',
            reflectance=0,
            shininess=0,
            specular=0,
            emission=0.5,
        )
        for i in range(self.n_butterflies):
            root.worldbody.add(
                'geom',
                name=f'butterfly_{i}',
                type='sphere',
                pos=(i, 0.0, 1.0),
                size=(0.1,),
                rgba=cmap(i / self.n_butterflies),
                material='butterfly',
                gap=1.0,  # high gap so that there won't be actual contact forces
                conaffinity=2,  # only collide with the net
            )

        super().init_sim(root, frameskip)

    def get_observation(self):
        nd = self.p.named.data
        bf_geoms = [
            idx
            for idx, name in enumerate(nd.geom_xpos.axes.row.names)
            if name.startswith('butterfly')
            and not self.butterflies_caught[int(name.split('_')[1])]
        ]

        grid = np.zeros((10, 10), dtype=np.float32)
        if len(bf_geoms) == 0:
            return {
                'observation': super().get_observation(),
                'butterflies': grid.flatten(),
            }

        try:
            bf_rpos = np.dot(
                nd.geom_xpos[bf_geoms] - nd.xpos['robot/head'],
                nd.xmat['robot/head'].reshape(3, 3),
            )
        except KeyError:
            bf_rpos = np.dot(
                nd.geom_xpos[bf_geoms] - nd.xpos['robot/torso'],
                nd.xmat['robot/torso'].reshape(3, 3),
            )
        bf_dist = np.linalg.norm(bf_rpos, axis=1)
        bf_npos = np.divide(bf_rpos, bf_dist.reshape(bf_rpos.shape[0], 1)).T
        lat = np.rad2deg(np.arccos(bf_npos[1]))
        lon = np.rad2deg(np.arctan2(bf_npos[0], bf_npos[2]))
        lat10 = np.floor(lat / 18).astype(np.int32)
        lon10 = np.floor((lon + 180) / 36).astype(np.int32)

        expdist = np.exp(-bf_dist)
        for i, (x, y) in enumerate(zip(lat10, lon10)):
            grid[x][y] = max(grid[x][y], expdist[i])

        return {
            'observation': super().get_observation(),
            'butterflies': grid.flatten(),
        }

    def reset_state(self):
        super().reset_state()

        poss = self.np_random.uniform(-1.0, 1.0, size=(self.n_butterflies, 3))
        scale = (
            np.asarray([self.goal_area, self.goal_area, 0.5]) * self.world_scale
        )
        off = np.asarray([0, 0, self.zoff * self.world_scale])
        if self.robot == 'testcube':
            off[2] = 1
        for i in range(self.n_butterflies):
            self.p.named.model.geom_pos[f'butterfly_{i}'] = (
                poss[i] * scale + off
            )
            self.p.named.model.geom_rgba[f'butterfly_{i}'][3] = 1

        self.butterflies_caught = np.zeros(self.n_butterflies, dtype=np.int32)

    def on_step_single_frame(self):
        contact = self.p.data.contact
        gnames = self.p.named.model.geom_type.axes.row.names
        for c1, c2 in zip(contact.geom1, contact.geom2):
            if not (
                gnames[c1].startswith('butterfly_')
                and gnames[c2] == 'robot/net_geom'
            ):
                continue
            id = int(gnames[c1].split('_')[1])
            if self.butterflies_caught[id] == 0:
                log.debug(f'contact: {gnames[c1]} - {gnames[c2]}')
            self.butterflies_caught[id] = 1
            self.p.named.model.geom_rgba[gnames[c1]][3] = 0.1

    def step(self, action):
        bfs_caught_before = self.butterflies_caught.sum()
        obs, reward, terminated, truncated, info = super().step(action)
        bfs_caught_after = self.butterflies_caught.sum()
        score = bfs_caught_after - bfs_caught_before
        info['score'] = score

        # TODO: what's a good shaped reward here?
        # Based on distance to the closest butterfly?
        info['shaped_reward'] = score

        reward = info['shaped_reward'] if self.shaped else score

        if info.get('fell_over', False):
            terminated = True
            reward = -1
        return obs, reward, terminated, False, info
