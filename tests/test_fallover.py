# Copyright (c) 2022-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import numpy as np
import pytest

import bisk


def test_walker_fallover():
    env = gym.make('BiskGoalWall-v1', robot='walker')
    env.reset(seed=0)
    while True:
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape)
        )
        if terminated or truncated:
            break
    assert terminated
    assert not truncated
    env.close()


def test_walker_continuous():
    env = gym.make('BiskGoalWallC-v1', robot='walker')
    env.reset(seed=0)
    while True:
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape)
        )
        if terminated or truncated:
            break
    assert truncated
    assert not terminated
    env.close()


def test_humanoidcmupc_fallover():
    env = gym.make('BiskGoalWall-v1', robot='humanoidcmupc')
    env.reset(seed=0)
    while True:
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape)
        )
        if terminated or truncated:
            break
    assert terminated
    assert not truncated
    env.close()


def test_humanoidcmupc_continuous():
    env = gym.make('BiskGoalWallC-v1', robot='humanoidcmupc')
    env.reset(seed=0)
    while True:
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape)
        )
        if terminated or truncated:
            break
    assert truncated
    assert not terminated
    env.close()
