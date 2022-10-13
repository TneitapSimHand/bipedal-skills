# Copyright (c) 2022-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import gym
import numpy as np
from dm_control.utils import rewards

from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskVelocityControlEnv(BiskSingleRobotEnv):
    '''
    Track a randomly changing velocity. From MoCapAct:
    https://github.com/microsoft/MoCapAct/blob/e11713c/mocapact/tasks/velocity_control.py
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        allow_fallover: bool,
        shaped: bool,
        max_speed: float = 4.5,
        reward_margin: float = 0.75,
        direction_exponent: float = 1.0,
        steps_before_changing_velocity: int = 166,
        **kwargs
    ):
        super().__init__(robot, features, allow_fallover, **kwargs)
        self.shaped = shaped
        self.max_speed = max_speed
        self.reward_margin = reward_margin
        self.direction_exponent = direction_exponent
        self.steps_before_changing_velocity = steps_before_changing_velocity

        obs_base = self.featurizer.observation_space
        obs_task = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            [('target', obs_task), ('observation', obs_base)]
        )

    def sample_move_speed(self):
        self.move_speed = self.np_random.uniform(high=self.max_speed)
        if self.is_2d:
            # Go forward or backward
            self.move_angle = self.np_random.choice([0, np.pi])
        else:
            self.move_angle = self.np_random.uniform(high=2 * np.pi)
        self.move_speed_counter = 0

    def reset_state(self):
        super().reset_state()
        self.sample_move_speed()

    def get_observation(self):
        sin, cos = np.sin(self.move_angle), np.cos(self.move_angle)
        phase = self.move_speed_counter / self.steps_before_changing_velocity
        return {
            'observation': super().get_observation(),
            'target': np.array([self.move_speed, sin, cos, phase]),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        xvel, yvel = self.robot_speed[:2]

        speed = np.linalg.norm([xvel, yvel])
        speed_error = self.move_speed - speed
        speed_reward = np.exp(-((speed_error / self.reward_margin) ** 2))
        if np.isclose(xvel, 0.0) and np.isclose(yvel, 0.0):
            dot = 0.0
            angle_reward = 1.0
        else:
            direction = np.array([xvel, yvel])
            direction /= np.linalg.norm(direction)
            direction_tgt = np.array(
                [np.cos(self.move_angle), np.sin(self.move_angle)]
            )
            dot = direction_tgt.dot(direction)
            angle_reward = ((dot + 1) / 2) ** self.direction_exponent

        speed_match = np.abs(speed_error) < 0.1
        angle_match = dot > np.cos(np.deg2rad(15))
        score = 1.0 if speed_match and angle_match else 0.0
        if self.shaped:
            reward = speed_reward * angle_reward
        else:
            reward = score
        info['score'] = score

        self.move_speed_counter += 1
        if self.move_speed_counter >= self.steps_before_changing_velocity:
            self.sample_move_speed()
            obs = self.get_observation()

        if info.get('fell_over', False):
            terminated = True
            reward = -1
        return obs, reward, terminated, truncated, info
