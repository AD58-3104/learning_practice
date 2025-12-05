from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

class EpicodeLengthObserver(object):
    def __init__(self):
        self.start_step : torch.tensor = None
        self.current_step : int = 0
        self.initialized : bool = False

    def initialize(self,env):
        if self.start_step is None:
            self.start_step = torch.zeros((env.num_envs,1),device=env.device)
            self.initialized = True

    # それぞれのエピソード計測をアップデート
    def update(self,env):
        if not self.initialized:
            self.initialize(env) 
        self.current_step += 1
        terminated = getattr(env,'termination_manager').terminated.unsqueeze(1)
        self.start_step = torch.where(terminated,self.current_step,self.start_step)

    # 各環境のエピソード長を返す
    def get_episode_len(self):
        return self.current_step - self.start_step 

episode_length_buf_for_phase_info : EpicodeLengthObserver = EpicodeLengthObserver()

def get_episode_length_buf_for_phaseinfo(env):
    global episode_length_buf_for_phase_info
    episode_length_buf_for_phase_info.update(env) # 先にアップデートしないと最初に壊れる
    return episode_length_buf_for_phase_info.get_episode_len()

def phase_infomation(env : ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.tensor:
    cycle_step = (1.2) / env.step_dt # 0.7秒に一歩.step_dtは(s)なので、周期が何ステップか
    if not hasattr(env,'episode_length_buf'):
        # なんか最初に一回呼ばれる事があるっぽい。しかもその時ManagerBasedRLEnvはまだ__init__が呼ばれてないっぽいのでそれを避けるためにやる
        # 多分形を知るために最初に一回呼ばれるっぽい。クソバカ。
        print("ManagerBasedRLEnv has not initialized yet!!!")
        return torch.zeros(env.num_envs,2) # (num_envs,2)
    episode_length_buf = env.episode_length_buf
    wt = episode_length_buf / cycle_step * (2 * torch.pi)  # 今周期のどこか
    phase = torch.abs(torch.stack([torch.sin(wt),torch.cos(wt + (torch.pi / 2))],dim=1))   #sinが右脚、cosが左脚
    return phase
