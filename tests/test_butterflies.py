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
    env = gym.make('BiskButterflies-v1', robot='testcube')
    obs, _ = env.reset(seed=0)
    yield env
    env.close()


def test_fixed_policy(env):
    # fmt off
    policy = []
    # Drop to floor
    for i in range(5):
        policy.append([0, 0, 0])
    # Go in a circle
    for d in (-1,1):
        for i in range(15):
            policy.append([d,0,0])
        for i in range(15):
            policy.append([0,d,0])
    # And the other way
    for d in (1,-1):
        for i in range(15):
            policy.append([d,0,0])
        for i in range(15):
            policy.append([0,d,0])

    retrn = 0
    for action in policy:
        obs, reward, terminated, truncated, info = env.step(action)
        assert (not terminated and not truncated)
        retrn += reward
    assert retrn == 4
