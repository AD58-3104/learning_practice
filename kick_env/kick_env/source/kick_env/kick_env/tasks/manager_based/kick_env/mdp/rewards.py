# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from .observations import ball_pos_rel as get_ball_pos_rel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_command_tracking(env: ManagerBasedRLEnv, command_name: str = "target_pos" ,asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)[:,:2]
    ball_pos_rel = get_ball_pos_rel(env)
    ball_distance = torch.norm(ball_pos_rel,dim=1,p=2).unsqueeze(1) #l2ノルム計算してボールの距離を出す
    command_norm = torch.norm(command,dim=1,p=2).unsqueeze(1)
    # 単位難しい。ラジアンなので基本0~3.14の間
    radian = (command * ball_pos_rel).sum(dim=1) / (ball_distance * command_norm).squeeze()
    radian = torch.abs(torch.acos(radian))
    radian = torch.nan_to_num(radian,nan=0.0) # nanになったものを置き換える
    ball_command_distance = torch.abs(ball_distance - command_norm)
    # print(ball_pos_rel.shape)
    # print(command.shape)
    # print(command_norm.shape)
    # print((ball_distance * command_norm).shape)
    # radian = torch.nn.functional.cosine_similarity(command,ball_pos_rel,dim=1)
    # print(ball_command_distance.shape)
    # print(ball_distance.shape)
    # print(radian.shape)
    # print("----------------")
    # print(f"ball_distance = {ball_distance[:1]}")
    # print(f"radian = {radian[:1]}")
    # print(f"ball_command_distance = {ball_command_distance[:1]}")
    # print(f"command = {command[:1]}")
    # print(f"ball_pos_rel = {ball_pos_rel[:1]}")
    # print(f"ball_distance = {ball_distance[:1]}")
    # print(f"command_norm = {command_norm[:1]}")
    # print(f"radian = {radian[:1]}")
    # print(f"ball_command_distance = {ball_command_distance[:1]}")
    
    #bd = 0~10くらい、rad = 0~3.14、bcd = 0~10mくらい？
    # radを*3くらいすれば3つのスケールは合いそう
    return (ball_distance - (radian.unsqueeze(1) * 3.3) - ball_command_distance).squeeze()