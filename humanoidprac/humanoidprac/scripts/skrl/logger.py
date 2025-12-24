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
import numpy as np

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

class ClassSuccessRateLogger:
    def __init__(self, classifier,log_dir: str = "."):
        self.classifier = classifier
        self.log_dir = log_dir
        self.total_failure = [0 for _ in range(classifier.num_of_classes)]
        self.total_epicode_count = [0 for _ in range(classifier.num_of_classes)]

    def log(self, terminated: torch.Tensor, truncated: torch.Tensor):
        term_or_trunc = terminated | truncated
        for class_idx in range(self.classifier.num_of_classes):
            term_or_trunc_in_class = self.classifier.mask_other_classes(class_idx, term_or_trunc)
            self.total_epicode_count[class_idx] += torch.sum(term_or_trunc_in_class).item()
            self.total_failure[class_idx] += torch.sum(self.classifier.mask_other_classes(class_idx, terminated)).item()

    def write_result(self):
        import os
        with open(os.path.join(self.log_dir, "class_success_rate.log"), 'w') as f:
            f.write("Class Index, Total Episodes, Total Failures, Success Rate [%]\n")
            for class_idx in range(self.classifier.num_of_classes):
                if self.total_epicode_count[class_idx] > 0:
                    success_rate = (1.0 - float(self.total_failure[class_idx]) / float(self.total_epicode_count[class_idx])) * 100.0
                else:
                    success_rate = 0.0
                f.write(f"{class_idx}, {self.total_epicode_count[class_idx]}, {self.total_failure[class_idx]}, {success_rate:.2f}\n")
        print("[Class Success Rate Logger] Result written to class_success_rate.log")

# NNの判別器を訓練するために利用する観測を保存するクラス
class DiscriminatorObsDataLogger:
    def __init__(self, env: ManagerBasedRLEnv):
        self.log_file_names = []
        self.num_envs = env.num_envs
        self.data = {}
        # ファイルの初期化
        for i in range(self.num_envs):
            filename = f"./nn_data/discriminator_obs_env_{i}.npz"
            self.log_file_names.append(filename)
            self.data[i] = []
        # データ形式
        # step, terminated | truncated, data...
        # データは[2]に来る

    def log(self, step: int, observations: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        term_or_trunc = terminated | truncated
        for env_idx in range(self.num_envs):
            obs_list = observations[env_idx].tolist()
            self.data[env_idx].append(
                    np.array([step, term_or_trunc[env_idx].item()] + obs_list))

    def close(self):
        for env_idx in range(self.num_envs):
            filename = self.log_file_names[env_idx]
            data_array = np.array(self.data[env_idx])
            np.savez_compressed(filename, data=data_array)
        print(f"[Discriminator Data Logger] Logfiles closed.")

class DiscriminatorObsDataLoggerOptimized:
    def __init__(self, env, max_steps: int):
        self.save_dir = "./nn_data"
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.num_envs = env.num_envs
        self.max_steps = max_steps
        
        # バッファは最初のlog呼び出し時に確保します（obsの次元数を知るため）
        self.buffer = None 
        self.current_write_idx = 0
        
        self.log_file_names = [os.path.join(self.save_dir, f"discriminator_obs_env_{i}.npz") for i in range(self.num_envs)]

    def log(self, step: int, observations: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        """
        step: 現在のステップ数（整数）
        observations: (num_envs, obs_dim) のTensor
        terminated, truncated: (num_envs,) のTensor
        """
        
        # --- 初回のみ実行: バッファの確保 ---
        if self.buffer is None:
            obs_dim = observations.shape[1]
            # 保存するデータの次元: [step(1) + terminated(1) + observations(obs_dim)]
            data_dim = 1 + 1 + obs_dim
            # CPU上に巨大なメモリ領域を確保
            self.buffer = torch.zeros((self.num_envs, self.max_steps, data_dim), dtype=torch.float32, device='cpu')

        # バッファあふれ防止
        if self.current_write_idx >= self.max_steps:
            return

        # --- GPU上での一括処理 (forループなし) ---
        device = observations.device
        
        # 1. Step数 (スカラー -> 全環境分の列ベクトルへ拡張)
        step_tensor = torch.full((self.num_envs, 1), step, device=device, dtype=torch.float32)
        
        # 2. 終了フラグ (論理和 -> float変換 -> 次元追加)
        term_tensor = (terminated | truncated).float().unsqueeze(1)
        
        # 3. データの結合 [step, term, obs...]
        # ここですべてGPUテンソルとして結合します
        batch_data = torch.cat([step_tensor, term_tensor, observations], dim=1)

        # --- CPUへの一括転送 ---
        # 計算の裏で転送を行う non_blocking=True でさらに高速化
        self.buffer[:, self.current_write_idx, :] = batch_data.to('cpu', non_blocking=True)
        
        self.current_write_idx += 1

    def close(self):
        print(f"[Discriminator Data Logger] Saving data...")
        
        # 実際に記録されたステップ数まで有効なデータをスライス
        if self.buffer is not None:
            valid_data = self.buffer[:, :self.current_write_idx, :].numpy()
            
            for env_idx in range(self.num_envs):
                filename = self.log_file_names[env_idx]
                np.savez_compressed(filename, data=valid_data[env_idx])
        else:
            print("Warning: No data was logged.")
            
        print(f"[Discriminator Data Logger] Logfiles closed.")

class JointTorqueLogger:
    def __init__(self, env: ManagerBasedRLEnv, joint_num: int, target_torque: float):
        self.log_file_names = []
        self.num_envs = env.num_envs
        self.joint_num = joint_num
        self.target_torque = target_torque
        self.data = {}
        # dataは環境id毎にステップ順のデータを持つ
        # ファイルの初期化
        for i in range(self.num_envs):
            filename = f"./nn_data/joint_torque_env_{i}.npz"
            self.log_file_names.append(filename)
            self.data[i] = []
        # データ形式 
        # step, terminated | truncated, data...

    def log(self, env, terminated: torch.tensor, truncated: torch.Tensor):
        term_or_trunc = terminated | truncated
        robot = env.scene["robot"]
        joint_effort_limits = robot.data.joint_effort_limits  # (num_envs, joint_num)
        joint_effort_limits_bool = (joint_effort_limits < (self.target_torque + 1.0)).long()
        for env_idx in range(self.num_envs):
            self.data[env_idx].append(
                    np.array([env.common_step_counter, term_or_trunc[env_idx].item()] + joint_effort_limits_bool[env_idx].tolist(), dtype=np.int32))

    def close(self):
        for env_idx in range(self.num_envs):
            filename = self.log_file_names[env_idx]
            data_array = np.array(self.data[env_idx])
            np.savez_compressed(filename, data=data_array)
        print(f"[Joint Torque Logger] Logfiles closed.")


class JointTorqueLoggerOptimized:
    def __init__(self, env, joint_num: int, target_torque: float, max_steps: int):
        self.save_dir = "./nn_data"
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.num_envs = env.num_envs
        self.joint_num = joint_num
        self.target_torque = target_torque
        
        # 【高速化ポイント1】事前に大きなバッファ（メモリ領域）をCPU側に確保します
        # 形状: (環境数, ステップ数, データ次元数) 
        # データ次元 = step(1) + terminated(1) + joint_bools(joint_num)
        self.data_dim = 1 + 1 + joint_num
        
        # 注意: メモリ圧迫を避けるためCPU上のTorch Tensorとして確保（NumPyよりTorch間のコピーが速いため）
        self.buffer = torch.zeros((self.num_envs, max_steps, self.data_dim), dtype=torch.int32, device='cpu')
        
        self.current_step_idx = 0 
        self.log_file_names = [os.path.join(self.save_dir, f"joint_torque_env_{i}.npz") for i in range(self.num_envs)]

    def log(self, env, terminated: torch.Tensor, truncated: torch.Tensor):
        # バッファあふれ防止
        if self.current_step_idx >= self.buffer.shape[1]:
            return

        # --- GPU上での計算 (ここがポイント) ---
        # 全て Tensor演算で行い、forループは使いません
        
        # 1. Terminated / Truncated 情報 (num_envs, 1)
        term_or_trunc = (terminated | truncated).float().unsqueeze(1) 
        
        # 2. Joint Limits Boolean (num_envs, joint_num)
        robot = env.scene["robot"]
        joint_effort_limits = robot.data.joint_effort_limits
        
        # 比較演算もGPU上で一括処理
        # (num_envs, joint_num) のBoolean Tensorを作成
        target_val = self.target_torque + 1.0
        joint_bools = (joint_effort_limits < target_val).float()

        # 3. Step Counter (num_envs, 1)
        # スカラー値を環境数分拡張します
        step_tensor = torch.full((self.num_envs, 1), env.common_step_counter, device=env.device, dtype=torch.int32)

        # 4. データの結合 (Concatenate)
        # GPU上で横方向に結合: [step, term, bool_0, bool_1, ...]
        current_step_data = torch.cat([step_tensor, term_or_trunc, joint_bools], dim=1)

        # --- CPUへの転送 (1ステップに1回だけ) ---
        # .to('cpu', non_blocking=True) を使うと、計算の裏で転送できる場合がありさらに速い
        self.buffer[:, self.current_step_idx, :] = current_step_data.to('cpu', non_blocking=True)
        
        self.current_step_idx += 1

    def close(self):
        print(f"[Joint Torque Logger] Saving data...")
        
        # 実際に記録されたステップ数まででスライス
        valid_data = self.buffer[:, :self.current_step_idx, :].numpy()
        
        for env_idx in range(self.num_envs):
            filename = self.log_file_names[env_idx]
            # 各環境のデータを保存
            # valid_data[env_idx] は (recorded_steps, data_dim) の形状
            np.savez_compressed(filename, data=valid_data[env_idx])
            
        print(f"[Joint Torque Logger] Logfiles closed.")

class DiscriminatorTester:
    """NNの判別器をテストするためのクラス"""
    def __init__(self, target_torque: float, joint_num: int, num_envs: int):
        self.target_torque = target_torque
        self.joint_num = joint_num
        self.total_count = torch.zeros((num_envs, joint_num), device="cuda")
        self.success_count = torch.zeros((num_envs, joint_num), device="cuda")
        self.detect_count = torch.zeros((num_envs, joint_num), device="cuda")

    def log(self, env: ManagerBasedRLEnv, discriminator_outputs: torch.Tensor):
        robot = env.scene["robot"]
        joint_effort_limits = robot.data.joint_effort_limits  # (num_envs, joint_num)
        joint_effort_limits_bool = (joint_effort_limits < (self.target_torque + 1.0))
        # リミット掛かってる環境の総数を記録する
        self.total_count += joint_effort_limits_bool.long()
        self.success_count += (joint_effort_limits_bool & discriminator_outputs.bool()).long()
        self.detect_count += discriminator_outputs.long()

    def write_result(self):
        log_dir = "./play_logs"
        import os
        with open(os.path.join(log_dir, "discriminator_test_result.log"), 'w') as f:
            f.write("Joint Index, Total limited num, Total Success detected, Success Rate [%]\n")
            # >>> Right_legs joint_ids: [1, 4, 8, 12]
            # >>> Left_legs joint_ids: [0, 3, 7, 11]
            for joint_idx in [1, 4, 8, 12, 0, 3, 7, 11]:
                total_tested = int(torch.sum(self.total_count[:, joint_idx]).item())
                total_success = int(torch.sum(self.success_count[:, joint_idx]).item())
                if total_tested > 0:
                    success_rate = (float(total_success) / float(total_tested)) * 100.0
                else:
                    success_rate = 0.0
                f.write(f"{joint_idx}, {total_tested}, {total_success}, {success_rate:.2f}\n")
                print(f"[Discriminator Tester] Joint {joint_idx}: Success Rate {success_rate:.2f}% ({total_success}/{total_tested})")
            for joint_idx in range(self.joint_num):
                total_detected = int(torch.sum(self.detect_count[:, joint_idx]).item())
                f.write(f"Joint {joint_idx} Total detected: {total_detected}\n")
                print(f"[Discriminator Tester] Joint {joint_idx}: Total detected {total_detected}")
        print("[Discriminator Tester] Result written to discriminator_test_result.log")