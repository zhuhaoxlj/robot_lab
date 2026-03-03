# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Termination functions for locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_fallen(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Check if the robot has fallen down by checking if the base z-axis is pointing downward.

    Args:
        env: The learning environment.
        asset_cfg: The configuration for the rigid body.
        threshold: The threshold for the projected gravity z-component. If the z-component
            is less than this threshold (i.e., the robot is tilted too much), the robot
            is considered fallen.

    Returns:
        A boolean tensor indicating which environments have a fallen robot.
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 2] is the z-component of gravity in the base frame
    # When robot is upright, this should be close to -1.0 (gravity points down in world, projected up in base)
    # When robot is fallen, this will be closer to 0 or positive
    return asset.data.projected_gravity_b[:, 2] > threshold


def is_too_low(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_height: float) -> torch.Tensor:
    """Check if the robot's base height is too low, indicating it has fallen.

    Args:
        env: The learning environment.
        asset_cfg: The configuration for the rigid body.
        min_height: The minimum allowed height (in meters). If the robot's base height
            is below this threshold, the robot is considered fallen.

    Returns:
        A boolean tensor indicating which environments have a robot that is too low.
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    # Get the z-position of the robot base in world frame
    base_height = asset.data.root_pos_w[:, 2]
    # Terminate if height is below threshold
    return base_height < min_height
