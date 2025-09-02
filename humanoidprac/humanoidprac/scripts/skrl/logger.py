from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import json

class SettingLogger():
    def __init__(self,env_cfg:ManagerBasedRLEnvCfg):
        print("Set up joint config logger...")
        self.env_cfg = env_cfg

    def log_setting(self,filepath):
        joint_cfg = self.env_cfg.events.change_joint_torque
        data = {
            "joint_names": joint_cfg.params["asset_cfg"].joint_names,
            "joint_torques": joint_cfg.params["joint_torque"]
        }
        self.write_to_json(data,filepath)

    def write_to_json(self,data:dict,filepath):
        with open(filepath,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
        return 
    

LOG_DATA_NAMES = [
    "torques",
    "velocities",
]

def get_logfile_name(file_name,data_name):
    name, ext = file_name.rsplit('.', 1)
    return f"{name}_{data_name}.{ext}"

class ExperimentValueLogger:
    def __init__(self,finish_step: int, log_file_name: str = 'exp_logdata/log.csv', target_envs: list = None ):
        self.finish_step = finish_step
        self.step_count = 0
        self.log_file_name = log_file_name
        self.log_files = {}
        self.finish_full_episode = 0
        self.episode_count = 1  # 最初は1から始める
        for data_name in LOG_DATA_NAMES:
            filename = get_logfile_name(log_file_name, data_name)
            self.log_files[data_name] = open(filename, 'w')
            print(f"[Exp Data Logger] Open logfile {filename}")

        if target_envs is None or len(target_envs) == 0:
            self.target_envs = [0]  # 特に指定が無いなら、1つの環境だけを対象にする
        else:
            self.target_envs = target_envs

    def log(self, env: ManagerBasedRLEnv) -> bool:
        self.step_count += 1
        self.write_torque_data(env)
        self.write_velocity_data(env)
        self.log_success_rate(env)
        if self.step_count >= self.finish_step:
            self._finish_logging()
            return True
        if self.step_count % 500 == 0:
            print(f"[Exp Data Logger] Step {self.step_count} logged.")
        return False

    def get_joint_name_str(self,env: ManagerBasedRLEnv):
        robot = env.scene["robot"]
        joint_names = robot.data.joint_names
        return ','.join(joint_names)

    def write_torque_data(self,env: ManagerBasedRLEnv):
        if self.step_count == 1:
            joint_name_str = self.get_joint_name_str(env)
            self.log_files["torques"].write(joint_name_str + '\n')
        robot = env.scene["robot"]
        torques = robot.data.applied_torque[self.target_envs]
        flaten = torques.flatten()
        val_str = ','.join(map(str,flaten.tolist()))
        self.log_files["torques"].write(val_str + '\n')

    def write_velocity_data(self,env: ManagerBasedRLEnv):
        if self.step_count == 1:
            joint_name_str = self.get_joint_name_str(env)
            self.log_files["velocities"].write(joint_name_str + '\n')
        robot = env.scene["robot"]
        velocities = robot.data.joint_vel[self.target_envs]
        flaten = velocities.flatten()
        val_str = ','.join(map(str,flaten.tolist()))
        self.log_files["velocities"].write(val_str + '\n')

    def log_success_rate(self,env: ManagerBasedRLEnv):
        terminated = env.termination_manager.terminated[0]
        timeout = env.termination_manager.time_outs[0]
        if terminated or timeout:
            if timeout:
                self.finish_full_episode += 1
            self.episode_count += 1
        return

    def print_success_rate(self):
        if self.episode_count > 0:
            success_rate = float(self.finish_full_episode) / float(self.episode_count)
            with open("exp_logdata/success_rate.txt", "a") as f:
                f.write(f"Success rate: {success_rate:.2f}\n")
            print(f"[Exp Data Logger] Success rate: {success_rate:.2f}")
        else:
            with open("exp_logdata/success_rate.txt", "a") as f:
                f.write("No episodes completed.\n")
            print("[Exp Data Logger] No episodes completed.")

    def _finish_logging(self):
        for f in self.log_files.values():
            f.close()
        self.print_success_rate()
        print(f"[Exp Data Logger] Step {self.step_count} reached.")
        print("[Exp Data Logger] Logging finished.")