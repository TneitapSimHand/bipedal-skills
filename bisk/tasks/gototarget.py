# Copyright (c) 2022-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import gym
import numpy as np
from dm_control import mjcf
from dm_control.utils import rewards

from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskGoToTargetEnv(BiskSingleRobotEnv):
    '''
    Simple 1D/2D navigation, a port of dm_control's GoToTarget task.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        allow_fallover: bool,
        shaped: bool,
        tolerance: float,
        goal_area: float,
        num_targets: int = 1,
        goal_switch_steps: int = 10,
        single_target: bool = False,
        on_circle: bool = False,
        **kwargs,
    ):
        self.shaped = shaped
        self.goal_area = goal_area
        self.tolerance = tolerance
        self.goals = np.zeros((num_targets, 2))
        self.goal_switch_steps = goal_switch_steps
        self.on_circle = on_circle
        self.steps_to_switch = 0
        self.single_target = single_target
        super().__init__(robot, features, allow_fallover, **kwargs)

        obs_base = self.featurizer.observation_space
        if self.is_2d:
            obs_env = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1 * num_targets,),
                dtype=np.float32,
            )
        else:
            obs_env = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3 * num_targets,),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(
            [('targets', obs_env), ('observation', obs_base)]
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        for i in range(self.goals.shape[0]):
            root.worldbody.add(
                'site',
                name=f'target_{i}',
                type='sphere',
                pos=(0.0, 0.0, 1.0),
                size=(0.1,),
                rgba=(0.9, 0.6, 0.6, 1.0 if i == 0 else 0.2),
            )
            root.worldbody.add(
                'site',
                name=f'target_tolerance_{i}',
                type='ellipsoid',
                pos=(0.0, 0.0, 1.0),
                size=(self.tolerance, self.tolerance, 1e-3),
                rgba=(0.9, 0.6, 0.6, 0.2 if i == 0 else 0.05),
            )

        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()
        self.sample_goal(all=True)

    def get_observation(self):
        if self.is_2d:
            targets = self.goals[:, 0:1] - self.robot_pos[0:1]
        else:
            targets = np.zeros((self.goals.shape[0], 3))
            targets[:, :2] = self.goals - self.robot_pos[:2]
            targets = np.dot(
                targets, self.p.named.data.xmat['robot/torso'].reshape(3, 3)
            )
        return {
            'observation': super().get_observation(),
            'targets': targets.flatten().astype(np.float32),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        dist = np.linalg.norm(self.goals[0] - self.robot_pos[:2])
        if dist < self.tolerance:
            score = 1
            self.steps_to_switch -= 1
        else:
            score = 0
        info['score'] = score
        info['distance'] = dist

        info['shaped_reward'] = -0.1 * (
            1
            - rewards.tolerance(
                dist,
                (0, 0),
                margin=self.goal_area / 2,
            )
        )
        reward = info['shaped_reward'] if self.shaped else score

        if self.steps_to_switch <= 0:
            self.sample_goal()
            obs = self.get_observation()

        if info.get('fell_over', False):
            terminated = True
            reward = -1
        if score == 1 and self.single_target:
            terminated = True
        return obs, reward, terminated, truncated, info

    def sample_goal(self, all: bool = False):
        if all:
            if self.on_circle:
                self.goals = self.np_random.standard_normal(self.goals.shape)
                self.goals /= np.maximum(
                    np.linalg.norm(self.goals, axis=1).reshape(-1, 1), 1e-5
                )
                self.goals *= self.goal_area
            else:
                self.goals = self.np_random.uniform(
                    -self.goal_area, self.goal_area, size=self.goals.shape
                )
        else:
            self.goals = np.roll(self.goals, -1, axis=0)
            if self.on_circle:
                self.goals[-1] = self.np_random.standard_normal(2)
                self.goals[-1] /= np.maximum(
                    np.linalg.norm(self.goals[-1]), 1e-5
                )
                self.goals[-1] *= self.goal_area
            else:
                self.goals[-1] = self.np_random.uniform(
                    -self.goal_area, self.goal_area, size=(2,)
                )

        if self.is_2d:
            self.goals[:, 1] = 0
        for i in range(self.goals.shape[0]):
            self.p.named.model.site_pos[f'target_{i}'][0:2] = self.goals[i]
            self.p.named.model.site_pos[f'target_{i}'][2] = 0
            self.p.named.model.site_pos[f'target_tolerance_{i}'][
                0:2
            ] = self.goals[i]
            self.p.named.model.site_pos[f'target_tolerance_{i}'][2] = 0
        self.steps_to_switch = self.goal_switch_steps
