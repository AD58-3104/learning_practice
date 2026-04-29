# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
8 関節分の単一ポリシーモデル (`agent_<joint_name>.pt`) を 1 フォルダから読み込み、
故障通知された関節 ID に応じて対応エージェントを切り替えながら推論する play スクリプト。

play_hydra.py をベースに、test_jointnet / log_nn_data / online_testing /
use_delayed_failure_info 等の旧モードは削除し、関節名キー切替のみをサポートする。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import ast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play 8-joint single-policy agents from a folder.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--models_dir", type=str, required=True, help="Folder containing 8 agent_<joint>.pt files.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--finish_step", type=int, default=3000, help="終了ステップ数")
parser.add_argument(
    "--joint_cfg", type=str, default="",
    help="上書きする joint の設定を辞書形式で指定する. 例: '{\"left_knee_joint\": 150}'"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import humanoidprac.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play 8-joint single-policy agents."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # joint の effort_limit を上書きする (任意)
    if args_cli.joint_cfg:
        param = ast.literal_eval(args_cli.joint_cfg)
        print(f"[INFO] Overriding joint configuration : {env_cfg.scene.robot.actuators['legs'].effort_limit}")
        for joint_name, torque in param.items():
            env_cfg.scene.robot.actuators["legs"].effort_limit[joint_name] = torque
            print(f"[INFO] Overriding joint configuration with: {joint_name} -> {torque}Nm")

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    log_dir = "play_logs"
    os.makedirs(log_dir, exist_ok=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # 推論用に学習由来のロギングを抑止する
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append("../")
    sys.path.append("../../")

    target_envs = torch.arange(1, env.num_envs)  # 全環境の平均値を取る
    import logger
    exp_val_logger = logger.ExperimentValueLogger(
        finish_step=args_cli.finish_step,
        log_file_name=os.path.join(log_dir, "play_log.csv"),
        target_envs=target_envs,
    )

    # 健康状態モデル
    env.reset()
    health_agent_model = "normal_agent.pt"
    tmp_rn = Runner(env, agent_cfg)
    tmp_rn.agent.load(health_agent_model)
    health_agent = tmp_rn.agent

    # 8 関節分のエージェントをロード (関節名キー dict)
    from custom_parallel_trainer import JointAgentLoader
    loader = JointAgentLoader(env=env, agent_cfg=agent_cfg, models_dir=args_cli.models_dir)
    agents_by_joint_name = loader.get_agents_by_joint_name()

    # 関節別生存率ロガー
    joint_survival_logger = logger.JointSurvivalRateLogger(
        env=env,
        joint_names=JointAgentLoader.EXPECTED_JOINT_NAMES,
        log_dir=log_dir,
    )

    # CustomParallelAgentTrainer.__init__ は agents=[*] を 1 つは要求する仕様。
    # 新 eval メソッドは self.agents を参照しないので、dict の最初のエージェントをダミーで渡す。
    dummy_first_agent = next(iter(agents_by_joint_name.values()))

    from custom_parallel_trainer import CustomParallelAgentTrainer
    trainer = CustomParallelAgentTrainer(
        env=env,
        agents=[dummy_first_agent],
        health_agent=health_agent,
        agents_scope=[],
        cfg=agent_cfg["trainer"],
    )

    try:
        trainer.eval_with_delayed_failure_info_by_joint_name(
            agents_by_joint_name=agents_by_joint_name,
            expdata_logger=exp_val_logger,
            joint_survival_logger=joint_survival_logger,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
