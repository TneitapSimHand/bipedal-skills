# Copyright (c) 2022-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import pytest
import numpy as np

import bisk


@pytest.fixture
def env():
    env = gym.make('BiskGoToSphere-v1', robot='testcube')
    obs, _ = env.reset(seed=0)
    yield env
    env.close()


def test_scripted_policy(env):
    for i in range(4):
        obs, reward, terminated, truncated, info = env.step([0, 0, 0])

    retrn = 0
    while not (terminated or truncated):
        target = obs['targets'][:2]
        dir = target / np.linalg.norm(target)
        dx, dy = 0, 0
        if np.abs(target[0]) > np.abs(target[1]):
            dx = np.sign(target[0])
        else:
            dy = np.sign(target[1])
        obs, reward, terminated, truncated, info = env.step([dx, dy, 0])
        retrn += reward

    assert terminated
    assert not truncated
    assert retrn == 1
