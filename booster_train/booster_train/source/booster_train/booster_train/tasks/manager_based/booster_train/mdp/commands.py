from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTermCfg
from dataclasses import MISSING
import math



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
        self.previous_command = torch.zeros(env.num_envs, 3, device=env.device)
        self.previous_command[:,0] = cfg.vcore # 最初のコマンド用に設定しておく
        self.percent_calc = PercentileCalculator(env)
        super().__init__(cfg.init_config,env)
        # 拡張する
        self.vel_command_b = torch.zeros(self.num_envs, 6, device=env.device)
    
    def compute(self, dt: float):
        super().compute(dt)
        self.current_time += dt

    # self.command_vel_bは前半の3つがサンプリングされた目標速度、後半の3つが補完された速度
    def _update_command(self):
        super()._update_command()
        percent = self.percent_calc.update()
        # print(f"prev_com {self.previous_command}")
        # print(self.previous_command.shape)
        command_inter = (self.vel_command_b[:,3:6] - self.previous_command) * percent + self.previous_command
        self.vel_command_b[:,[3,4,5]] = command_inter

    def _resample(self, env_ids: Sequence[int]):
        current_step  = self.current_time / self.env_dt
        if  (current_step >= float(self.curriculum_start_step)):
            percentile = float(current_step - self.curriculum_start_step) / float(self.curriculum_end_step - self.curriculum_start_step)
            val = (percentile * self.cfg_curriculum.final_target_value)
            x_high = (self.cfg_curriculum.final_target_value - self.cfg_curriculum.vcore) * percentile
            current_ranges = UniformVelocityCommandCfg.Ranges(
                        lin_vel_x=(-val, x_high),
                        lin_vel_y=(-val, val),
                        ang_vel_z=(-val, val)
                    )
            self.cfg = self.cfg_curriculum.target_config.replace(ranges=current_ranges)
            if not self.started:
                self.started = True
                # print("Start Changing velocity command config!!!")
                # print(self.cfg.ranges)
                # print(percentile)
                # print(val)
                # print(current_ranges)
        # print("-------")
        # print(f"current_step = {current_step}")
        # print(f"range = {current_ranges}")
        # print(f"resampling time = {self.cfg.resampling_time_range}")
        # print(f"env_ids = {env_ids[:10]}")
        self.previous_command = self.vel_command_b[:,:3] # リサンプリング前に保存する
        super()._resample(env_ids)
        self.vel_command_b[env_ids,0] += self.cfg_curriculum.vcore # リセットされたやつだけvcoreを足す
        self.percent_calc.reset(env_ids) #リサンプリングされた環境はリセットする
        # print(self.vel_command_b[:1,:])

# リサンプリングしてからの長さを計算する
class PercentileCalculator(object):
    def __init__(self,env):
        self.start_step = torch.zeros((env.num_envs,1),device=env.device)
        self.current_step = 0
        self.max_episode_length = env.max_episode_length
        # リサンプリングしてからその速度に達するまでの目標ステップ(それに向けて速度は漸増する)
        self.target_step = (self.max_episode_length / 3.0) 

    # 目標速度に到達するまでの各環境のパーセンテージを返す
    def update(self):
        self.current_step += 1
        # target_stepまでの割合
        return torch.clip((self.current_step - self.start_step)  / self.target_step,0,1.0)

    def reset(self,env_ids):
        self.start_step[env_ids] = self.current_step


@configclass
class CurriculumCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = CurriculumCommand
    curriculum_start_step : int = 0
    curriculum_end_step : int = 5000
    resampling_time_range =(20.0, 20.0)  # これは必要なので入れる。しかし下のやつと同じにしないとおかしくなるので注意
    # 一番最初からstart_stepまで利用するconfig
    init_config : UniformVelocityCommandCfg = UniformVelocityCommandCfg(
                    asset_name="robot",
                    resampling_time_range=(20.0, 20.0),
                    rel_standing_envs=0.02,
                    rel_heading_envs=1.0,
                    heading_command=False,
                    heading_control_stiffness=0.5,
                    debug_vis=True,
                    ranges=UniformVelocityCommandCfg.Ranges(
                        lin_vel_x=(0.4, 0.4), 
                        lin_vel_y=(0.0, 0.0), 
                        ang_vel_z=(0.0, 0.0)
                    ))

    vcore : float = 0.4
    # 目標値
    final_target_value : float = 0.6

    # あるだけ。後で消す予定
    target_config : UniformVelocityCommandCfg = UniformVelocityCommandCfg(
                    asset_name="robot",
                    resampling_time_range=(20.0, 20.0),
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