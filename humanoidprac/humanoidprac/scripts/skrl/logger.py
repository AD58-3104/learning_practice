from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
import torch.nn.functional as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import json
import time

class SettingLogger():
    def __init__(self,env_cfg:ManagerBasedRLEnvCfg):
        print("Set up joint config logger...")
        self.env_cfg = env_cfg

    def log_setting(self,filepath):
        joint_cfg = self.env_cfg.events.change_joint_torque
        # joint_cfg はNoneになる可能性がある(イベントをoffにする場合など)
        if joint_cfg is not None:
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
    "max_torques",
    "velocity_tracking_rate",
    "success_rate",
]
TMP_LOG_FILE = "exp_logdata/tmp_log.csv"

def get_logfile_name(file_name,data_name):
    name, ext = file_name.rsplit('.', 1)
    return f"{name}_{data_name}.{ext}"

class ExperimentValueLogger:
    # log_file_nameには保存する基本ディレクトリの絶対パスを含めた基本名が入る
    def __init__(self,finish_step: int, log_file_name: str = TMP_LOG_FILE, target_envs: list = None ):
        self.finish_step = finish_step
        self.step_count = 0
        self.log_file_name = log_file_name
        self.log_files = {}
        self.file_names = []
        self.finish_full_episode = 0
        self.episode_count = 1  # 最初は1から始める
        import os
        self.log_dir_name = os.path.dirname(log_file_name)
        for data_name in LOG_DATA_NAMES:
            filename = get_logfile_name(log_file_name, data_name)
            self.file_names.append(filename)
            self.log_files[data_name] = open(filename, 'w')
            print(f"[Exp Data Logger] Open logfile {filename}")

        if target_envs is None or len(target_envs) == 0:
            self.target_envs = [0]  # 特に指定が無いなら、1つの環境だけを対象にする
        else:
            self.target_envs = target_envs
        self.start_time = time.perf_counter()

    def log(self, env: ManagerBasedRLEnv) -> bool:
        self.step_count += 1
        self.write_max_torque_data(env)
        self.write_torque_data(env)
        self.write_velocity_data(env)
        self.write_velocity_tracking_rate(env)
        self.log_success_rate(env)
        if self.step_count >= self.finish_step:
            self._finish_logging()
            return True
        if self.step_count % 500 == 0:
            elapsed = time.perf_counter() - self.start_time
            step_per_sec = 500.0 / elapsed
            print(f"Dir [{self.log_dir_name}]")
            print(f"[Exp Data Logger] Step {self.step_count} logged. {step_per_sec:.4f} [step/sec]")
            self.start_time = time.perf_counter()
        return False

    def get_joint_name_str(self,env: ManagerBasedRLEnv):
        robot = env.scene["robot"]
        joint_names = robot.data.joint_names
        return ','.join(joint_names)

    """
    各関節に対して全ての環境の最大トルクを記録する。
    """
    def write_max_torque_data(self,env: ManagerBasedRLEnv):
        if self.step_count == 1:
            joint_name_str = self.get_joint_name_str(env)
            self.log_files["max_torques"].write(joint_name_str + '\n')
        robot = env.scene["robot"]
        torques = robot.data.applied_torque[self.target_envs]
        abs_max_indices = torch.argmax(torch.abs(torques), dim=0)
        torques = torch.gather(torques, 0, abs_max_indices.unsqueeze(0)).squeeze(0)
        flaten = torques.flatten()
        val_str = ','.join(map(str,flaten.tolist()))
        self.log_files["max_torques"].write(val_str + '\n')

    def write_torque_data(self,env: ManagerBasedRLEnv):
        if self.step_count == 1:
            joint_name_str = self.get_joint_name_str(env)
            self.log_files["torques"].write(joint_name_str + '\n')
        robot = env.scene["robot"]
        torques = torch.sum(robot.data.applied_torque[self.target_envs], dim=0, keepdim=True)
        torques = torques / len(self.target_envs)
        flaten = torques.flatten()
        val_str = ','.join(map(str,flaten.tolist()))
        self.log_files["torques"].write(val_str + '\n')

    def write_velocity_data(self,env: ManagerBasedRLEnv):
        if self.step_count == 1:
            joint_name_str = self.get_joint_name_str(env)
            self.log_files["velocities"].write(joint_name_str + '\n')
        robot = env.scene["robot"]
        velocities = torch.sum(robot.data.joint_vel[self.target_envs], dim=0, keepdim=True)
        velocities = velocities / len(self.target_envs)
        flaten = velocities.flatten()
        val_str = ','.join(map(str,flaten.tolist()))
        self.log_files["velocities"].write(val_str + '\n')

    # expカーネルでの速度追従率を計算する。誤差が最小の場合1.0
    def track_rate_lin_vel_xy_yaw_frame(self,env: ManagerBasedRLEnv) -> torch.Tensor:
        command_name = "base_velocity"
        asset = env.scene["robot"]
        vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
        # lin_vel_direction_error = F.cosine_similarity(env.command_manager.get_command(command_name)[:, :2],vel_yaw[:, :2],dim=1)
        lin_vel_magnitude_error_avg = torch.sum(torch.norm(vel_yaw[:, :2] - env.command_manager.get_command(command_name)[:, :2],dim=1)) / env.num_envs
        return torch.exp(-lin_vel_magnitude_error_avg / 1.0 ) 

    # 角速度の追従率を計算する。誤差が最小の場合1.0
    def track_rate_ang_vel_z_world(self,env: ManagerBasedRLEnv) -> torch.Tensor:
        command_name = "base_velocity"
        asset = env.scene["robot"]
        ang_vel_magnitude_error_avg = torch.sum(torch.square((asset.data.root_ang_vel_w[:, 2]) - (env.command_manager.get_command(command_name)[:, 2]))) / env.num_envs
        # 方向誤差はまあ計算しなくても良いことにする。多分真逆に進むという事は無いので。
        return torch.exp(-ang_vel_magnitude_error_avg / 1.0 )

    def write_velocity_tracking_rate(self,env: ManagerBasedRLEnv):
        if self.step_count == 1:
            joint_name_str = "lin_vel_xy_yaw,ang_vel_z_world"
            self.log_files["velocity_tracking_rate"].write(joint_name_str + '\n')
        lin_vel_error = self.track_rate_lin_vel_xy_yaw_frame(env)
        ang_vel_error = self.track_rate_ang_vel_z_world(env)
        val_str = ','.join(map(str,[lin_vel_error.item(), ang_vel_error.item()]))
        self.log_files["velocity_tracking_rate"].write(val_str + '\n')

    def log_success_rate(self,env: ManagerBasedRLEnv):
        terminated = torch.sum(env.termination_manager.terminated.int())
        timeout = torch.sum(env.termination_manager.time_outs.int())
        self.episode_count += terminated + timeout
        self.finish_full_episode += timeout
        return

    def print_success_rate(self):
        self.log_files["success_rate"].write(f"Total episodes,Total success episodes, Success rate[%]\n")
        if self.episode_count > 0:
            success_rate = float(self.finish_full_episode) / float(self.episode_count) * 100.0
            self.log_files["success_rate"].write(f"{self.episode_count},{self.finish_full_episode},{success_rate:.2f}\n")
            print(f"[Exp Data Logger] Success rate: {success_rate:.2f}")
        else:
            self.log_files["success_rate"].write("No episodes completed.\n")
            print("[Exp Data Logger] No episodes completed.")

    def _finish_logging(self):
        self.print_success_rate()
        for f in self.log_files.values():
            f.close()
        import shutil
        import os
        for name in self.file_names:
            shutil.copyfile(name, os.path.join("exp_logdata", os.path.basename(name)))
        print(f"[Exp Data Logger] Step {self.step_count} reached.")
        print("[Exp Data Logger] Logging finished.")


# NNの判別器を訓練するために利用する観測を保存するクラス
class DiscriminatorObsDataLogger:
    def __init__(self, env: ManagerBasedRLEnv, log_file_name: str = "discriminator_obs.dat"):
        self.log_file_name = log_file_name
        self.log_file = open(log_file_name, 'w')
        print(f"[Discriminator Data Logger] Open logfile {log_file_name}")
        obs_num = env.observation_space.shape[0]
        obs_rep = ','.join([f"observation_{i}" for i in range(obs_num)])
        self.log_file.write(f"step,terminated,{obs_rep}\n")

    def log(self, step: int, observations: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        term_or_trunc = terminated | truncated
        self.log_file.write(f"{step},{term_or_trunc[0].item()},{','.join(map(str, observations[0].tolist()))}\n")

    def close(self):
        self.log_file.close()
        print(f"[Discriminator Data Logger] Logfile {self.log_file_name} closed.")