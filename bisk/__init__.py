# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

__version__ = "2.0"

from gym.envs.registration import register

from bisk.base import BiskEnv
from bisk.single_robot import BiskSingleRobotEnv


def register_all(robot, shaped, fallover):
    fallover_text = 'C' if fallover else ''
    shaped_text = 'Shaped' if shaped else ''
    register(
        id=f'BiskGoToTargets{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.gototarget:BiskGoToTargetEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'tolerance': 0.5,
            'goal_area': 8.0,
            'num_targets': 2,
            'goal_switch_steps': 1,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskGoToSphere{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.gototarget:BiskGoToTargetEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'tolerance': 0.5,
            'goal_area': 4.0,
            'num_targets': 1,
            'single_target': True,
            'on_circle': True,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskHurdles{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.hurdles:BiskHurdlesEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'max_height': 0.3,
            'fixed_height': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskLimbo{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.limbo:BiskLimboEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'notouch': False,
            'min_height': 'auto',
            'fixed_height': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskHurdlesLimbo{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.hurdleslimbo:BiskHurdlesLimboEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'notouch': False,
            'min_bar_height': 'auto',
            'max_hurdle_height': 0.3,
            'fixed_height': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskGaps{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.gaps:BiskGapsEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'max_size': 2.5,
            'min_gap': 0.2,
            'max_gap': 0.7,
            'fixed_size': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskStairs{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.stairs:BiskStairsEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'step_height': 0.2,
            'step_length_min': 0.5,
            'step_length_max': 1.0,
            'num_flights': 2,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskStairsCont{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.stairs:BiskStairsEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'step_height': 0.2,
            'step_length_min': 0.5,
            'step_length_max': 1.0,
            'num_flights': 10,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskGoalWall{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.goalwall:BiskGoalWallEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
            'init_distance': 2.5,
            'touch_ball_reward': 0,
        },
        max_episode_steps=250,
    )
    register(
        id=f'BiskVelocityControl{shaped_text}{fallover_text}{robot}-v1',
        entry_point=f'bisk.tasks.velocitycontrol:BiskVelocityControlEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'allow_fallover': fallover_text,
            'shaped': shaped,
        },
        max_episode_steps=1000,
    )
    if shaped:
        register(
            id=f'BiskRunDir{shaped_text}{fallover_text}{robot}-v1',
            entry_point=f'bisk.tasks.rundir:BiskRunDirEnv',
            kwargs={
                'robot': robot,
                'features': 'joints',
                'allow_fallover': fallover_text,
                'heading_deg': 0,
            },
            max_episode_steps=1000,
        )
    if not shaped:
        register(
            id=f'BiskPoleBalance{shaped_text}{fallover_text}{robot}-v1',
            entry_point=f'bisk.tasks.polebalance:BiskPoleBalanceEnv',
            kwargs={
                'robot': robot,
                'features': 'joints',
                'allow_fallover': fallover_text,
                'pole_mass': 0.5,
                'pole_length': 0.5,
                'n_poles': 1,
            },
            max_episode_steps=1000,
        )
        register(
            id=f'BiskButterflies{shaped_text}{fallover_text}{robot}-v1',
            entry_point=f'bisk.tasks.butterflies:BiskButterfliesEnv',
            kwargs={
                'robot': robot,
                'features': 'joints',
                'allow_fallover': fallover_text,
                'shaped': shaped,
                'goal_area': 4,
                'n_butterflies': 10,
                'zoff': 1.6,
            },
            max_episode_steps=1000,
        )


for robot in (
    '',
    'HalfCheetah',
    'Walker',
    'Humanoid',
    'HumanoidPC',
    'HumanoidCMUPC',
    'HumanoidAMASSPC',
):
    for shaped in (False, True):
        for fallover in (False, True):
            register_all(robot, shaped, fallover)
