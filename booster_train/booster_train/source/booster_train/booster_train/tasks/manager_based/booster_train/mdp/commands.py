from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTermCfg
from dataclasses import MISSING



# カリキュラム学習用のコマンド
# 最初~threashouldまではこの中で設定したパラメータでコマンド生成
# threshould以降はCfgで与えられたパラメータでコマンド生成
class CurriculumCommand(UniformVelocityCommand):
    # cfg : CurriculumCommandCfg
    cfg_curriculum : CurriculumCommandCfg
    def __init__(self, cfg: CurriculumCommandCfg, env: ManagerBasedEnv):
        self.current_time = 0.0
        self.env_dt = env.step_dt
        self.curriculum_start_step = cfg.curriculum_start_step
        self.curriculum_end_step = cfg.curriculum_end_step
        self.cfg_curriculum = cfg
        self.started = False
        super().__init__(cfg.init_config,env)
    
    def compute(self, dt: float):
        super().compute(dt)
        self.current_time += dt


    def _resample(self, env_ids: Sequence[int]):
        current_step  = self.current_time / self.env_dt
        if  (current_step > float(self.curriculum_start_step)):
            percentile = float(current_step - self.curriculum_start_step) / float(self.curriculum_end_step - self.curriculum_start_step)
            val = (percentile * self.cfg_curriculum.final_target_value) + self.cfg_curriculum.start_value
            current_ranges = UniformVelocityCommandCfg.Ranges(
                        lin_vel_x=(-val, val),
                        lin_vel_y=(-val, val),
                        ang_vel_z=(-val, val)
                    )
            self.cfg = self.cfg_curriculum.target_config.replace(ranges=current_ranges)
            if not self.started:
                self.started = True
                print("Start Changing velocity command config!!!")
                print(self.cfg.ranges)
                print(percentile)
                print(val)
                print(current_ranges)
        super()._resample(env_ids)


@configclass
class CurriculumCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = CurriculumCommand
    curriculum_start_step : int = 1000
    curriculum_end_step : int = 7000
    resampling_time_range = (10.0, 10.0)  # これは必要なので入れる。しかし下のやつと同じにしないとおかしくなるので注意
    # 一番最初からstart_stepまで利用するconfig
    init_config : UniformVelocityCommandCfg = UniformVelocityCommandCfg(
                    asset_name="robot",
                    resampling_time_range=(10.0, 10.0),
                    rel_standing_envs=0.02,
                    rel_heading_envs=1.0,
                    heading_command=False,
                    heading_control_stiffness=0.5,
                    debug_vis=True,
                    ranges=UniformVelocityCommandCfg.Ranges(
                        lin_vel_x=(0.0, 0.0), 
                        lin_vel_y=(0.0, 0.0), 
                        ang_vel_z=(0.0, 0.0)
                    ))
    # start_step直後の値
    start_value : float = 0.0
    # 目標値
    final_target_value : float = 0.6

    # あるだけ。後で消す予定
    target_config : UniformVelocityCommandCfg = UniformVelocityCommandCfg(
                    asset_name="robot",
                    resampling_time_range=(10.0, 10.0),
                    rel_standing_envs=0.02,
                    rel_heading_envs=1.0,
                    heading_command=False,
                    heading_control_stiffness=0.5,
                    debug_vis=True,
                    ranges=UniformVelocityCommandCfg.Ranges(
                        lin_vel_x=(-0.4, 0.4), 
                        lin_vel_y=(-0.4, 0.4), 
                        ang_vel_z=(-0.4, 0.4)
                    ))