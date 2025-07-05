
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ボールとロボットの相対位置を計算する
def ball_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    ball: Articulation = env.scene["soccer_ball"]
    robot_pos = robot.data.root_pos_w[:,:2]
    ball_pos = ball.data.root_pos_w[:,:2]
    rel_pos = ball_pos - robot_pos
    # print(rel_pos[:1])
    return rel_pos
