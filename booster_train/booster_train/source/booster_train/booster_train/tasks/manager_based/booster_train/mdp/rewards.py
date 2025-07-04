# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import joint_vel
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# このカーネル関数はxの絶対値が大きいほど値が小さくなる。逆に、小さいほど値が大きくなる。
# つまり、以下の報酬系は全てノルムが小さい方が良いことになる。
def kernel_func(x:torch.tensor,sensitivity:float = 1.0) -> torch.Tensor:
    """Kernel function for rewards."""
    return 2 / ((torch.exp(-x * sensitivity) + torch.exp(x * sensitivity)))
# sensitivityの値は今の所論文の数値を入れている


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
    foot_angles = joint_pos[:, target_joint_indices]
    reward = kernel_func(torch.norm(foot_angles - nominal_joint_pos,p=2 ,dim=1),sensitivity = 3.0) * env.step_dt
    return reward

global_Cf : torch.tensor = None  # foot_clearance用のCf。初期値は0.05
global_Cv : torch.tensor = None  # commnd_tracking用のCf。初期値は0.01

# (num_envs,1)を返す
def getGlobalCf(env,pow : float = 0.995 ,init_value : float = 0.05):
    global global_Cf
    if global_Cf is None:
        global_Cf = torch.full((env.num_envs,1),init_value,device=env.device)
    else:
        terminated = env.termination_manager.terminated.unsqueeze(1) # (num_envs,1)に形を合わせる
        global_Cf = torch.where(terminated,global_Cf ** (pow),global_Cf)
    return global_Cf

# (num_envs,1)を返す
def getGlobalCv(env,pow : float = 0.95 ,init_value : float = 0.01):
    global global_Cv
    if global_Cv is None:
        global_Cv = torch.full((env.num_envs,1),init_value,device=env.device)
    else:
        terminated = env.termination_manager.terminated.unsqueeze(1)   # (num_envs,1)に形を合わせる
        global_Cv = torch.where(terminated,global_Cv ** (pow),global_Cv)
    return global_Cv

# メモ、bVbの方はbase link frameの速度、IVbはinertial reference frameの速度
def command_tracking(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    xy_vel_w_err = command[:, :2] - robot.data.root_lin_vel_w[:, :2] # World frame linear velocity
    xy_vel_b_err = command[:, :2] - robot.data.root_lin_vel_b[:, :2] # robot frame linear velocity
    ang_vel_error = command[:, 2] - robot.data.root_ang_vel_w[:, 2]
    target = torch.cat([xy_vel_w_err,xy_vel_b_err,ang_vel_error.unsqueeze(1)],dim=1)
    norm = torch.norm(target, p=2, dim=1)
    Cv = getGlobalCv(env)
    sum = Cv * (kernel_func(norm,sensitivity = 9.0).unsqueeze(1)) * env.step_dt
    return sum.squeeze()

def foot_z_distance(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    # 各インデックスの取得
    foot_names = ["left_foot_link", "right_foot_link"]
    foot_ids = [robot.data.body_names.index(name) for name in foot_names]
    # 脚の高さ、0 = 左足、 1 = 右足
    foot_heights_left = robot.data.body_link_pos_w[:, foot_ids[0], 2]
    foot_heights_right = robot.data.body_link_pos_w[:, foot_ids[1], 2]
    # このカーネル関数は小さい方が大きいので、逆数にする
    # 多分 0.4 ~ 0.01、つまり逆数にすると2.5 ~ 100くらいの間で推移するはずなので1.0 ~ 0.8くらい?
    # return kernel_func( 1.0 / (foot_heights_left - foot_heights_right),sensitivity=0.2) * env.step_dt
    return torch.abs(foot_heights_left - foot_heights_right) * env.step_dt


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


    # 何かの重み？(論文読んでも説明が無くて意味不明)
    w_phi = 5.0
    # 目標の足上げ高さ?(論文読んでも説明が無くて意味不明)
    pz_des = 0.40
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
    sum = getGlobalCf(env) * (kernel_func(norm,sensitivity = 3.0).unsqueeze(1)) * env.step_dt
    return sum.squeeze()

# 平均に対して早すぎる関節があるときペナルティ
def velocity_reguralize(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    target_names = [
                    "Left_Hip_Pitch","Left_Hip_Roll",
                    "Left_Knee_Pitch","Left_Ankle_Pitch","Left_Ankle_Roll",
                    "Right_Hip_Pitch","Right_Hip_Roll",
                    "Right_Knee_Pitch","Right_Ankle_Pitch","Right_Ankle_Roll"]
    leg_joint_ids = [robot.data.joint_names.index(name) for name in target_names]
    leg_velocities = joint_vel(env,asset_cfg)[:,leg_joint_ids]
    # print("--------")
    # print(leg_joint_ids)
    # print(leg_velocities[:5])
    mean = torch.mean(torch.abs(leg_velocities),dim=1).unsqueeze(1)
    # print(mean[:5])
    diffs = torch.abs(leg_velocities) - mean
    diffs = torch.clamp(diffs,min=0.0)
    sum = torch.sum(diffs,dim=1) / 10.0
    # print(sum[:5])
    return kernel_func(sum,sensitivity=5.0)*env.step_dt