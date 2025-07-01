# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)



def kernel_func(x:torch.tensor) -> torch.Tensor:
    """Kernel function for rewards."""
    sensitivity = 1.0
    return 2 / (torch.exp(-x * sensitivity) + torch.exp(x * sensitivity))


def pose_regularization(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Regularization reward for the robot's pose."""
    """This function computes a regularization reward for the robot's pose. It penalizes the robot for deviating from its initial pose."""
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos
    nominal_joint_names = ["Left_Hip_Pitch","Left_Hip_Roll","Left_Hip_Yaw",
                           "Left_Knee_Pitch","Left_Ankle_Pitch","Left_Ankle_Roll",
                           "Right_Hip_Pitch","Right_Hip_Roll","Right_Hip_Yaw",
                           "Right_Knee_Pitch","Right_Ankle_Pitch","Right_Ankle_Roll"]
    nominal_joint_pos = torch.tensor([-0.2, 0.0, 0.0,
                         0.4, -0.25, 0.0,
                        -0.2, 0.0, 0.0,
                        0.4, -0.25, 0.0], device=env.device)
    target_joint_indices = [robot.data.joint_names.index(name) for name in nominal_joint_names]

    arm_angles = joint_pos[:, target_joint_indices]
    reward = kernel_func(torch.norm(arm_angles - nominal_joint_pos,p=2 ,dim=1)) * env.step_dt
    return reward

# メモ、bVbの方はbase link frameの速度、IVbはinertial reference frameの速度
def command_tracking(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    xy_vel_w_err = command[:, :2] - robot.data.root_lin_vel_w[:, :2] # World frame linear velocity
    xy_vel_b_err = command[:, :2] - robot.data.root_lin_vel_b[:, :2] # robot frame linear velocity
    ang_vel_error = command[:, 2] - robot.data.root_ang_vel_w[:, 2]
    target = torch.cat([xy_vel_w_err,xy_vel_b_err,ang_vel_error.unsqueeze(1)],dim=1)
    norm = torch.norm(target, p=2, dim=1)
    Cv = env.common_step_counter / (4800.0) 
    if Cv > 1.0:
        Cv = 1.0
    if env.common_step_counter < 500: # 最初の500ステップは報酬を0にする
        Cv = 0.0
    return Cv * kernel_func(norm) * env.step_dt


def foot_clearance(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for the feet being above the ground."""
    robot = env.scene[asset_cfg.name]
    # 各インデックスの取得
    foot_names = ["left_foot_link", "right_foot_link"]
    foot_ids = [robot.data.body_names.index(name) for name in foot_names]
    foot_joint_names = ["Left_Ankle_Roll","Right_Ankle_Roll"]
    foot_joint_ids = [robot.data.joint_names.index(name) for name in foot_joint_names]
    # print(robot.data.body_link_pos_w.shape)   # torch.Size([4096, 24, 3])
    # print(robot.data.joint_pos.shape)         # torch.Size([4096, 23])
    # 脚の高さ、0 = 左足、 1 = 右足
    foot_heights = robot.data.body_link_pos_w[:, foot_ids, 2]
    # 脚の角度、0 = 左足、 1 = 右足
    foot_angles = robot.data.joint_pos[:, foot_joint_ids]
    # print(foot_heights.shape)                   # torch.Size([4096, 2])
    # print(foot_angles.shape)                    # torch.Size([4096, 2])
    swing_leg_index = torch.argmax(foot_heights, dim=1)
    stance_leg_index = torch.argmin(foot_heights, dim=1)
    # print(swing_leg_index.shape)                #torch.Size([4096])
    # print(stance_leg_index.shape)               #torch.Size([4096])

    # 遊脚がどちらかを判断する
    # if foot_heights[0] > foot_heights[1]:
    #     swing_leg_index = 0
    #     stance_leg_index = 1
    # else:
    #     swing_leg_index = 1
    #     stance_leg_index = 0

    # 何かの重み？(論文読んでも説明が無くて意味不明)
    w_phi = 4.0
    # 目標の足上げ高さ?(論文読んでも説明が無くて意味不明)
    pz_des = 0.15
    swing_heights = torch.gather(foot_heights,dim=1,index=swing_leg_index.unsqueeze(1))     # torch.Size([4096, 1])
    stance_heights = torch.gather(foot_heights,dim=1,index=stance_leg_index.unsqueeze(1))   # torch.Size([4096, 1])
    foot_angles0 = foot_angles[:,0].unsqueeze(1) # torch.Size([4096, 1])
    foot_angles1 = foot_angles[:,1].unsqueeze(1) # torch.Size([4096, 1])
    target = torch.cat([(pz_des - swing_heights) * w_phi,
                                    foot_angles0,
                                    foot_angles1,
                                    stance_heights * w_phi
                                    ],dim = 1) # torch.Size([4096, 4])

    norm = torch.norm(target,p=2,dim=1)        # torch.Size([4096]) 
    Cf = env.common_step_counter / (4800.0) 
    if Cf > 1.0:
        Cf = 1.0
    return Cf * kernel_func(norm) * env.step_dt