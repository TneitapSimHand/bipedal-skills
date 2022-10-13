# Copyright (c) 2021-present, Facebook, Inc.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import pytest

import bisk


@pytest.fixture
def env():
    env = gym.make('BiskPoleBalance-v1', robot='testcube')
    obs, _ = env.reset(seed=0)
    yield env
    env.close()


def test_rewards(env):
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step([1, 0, 0])
        if terminated:
            assert reward == 0
        else:
            assert reward == 1
