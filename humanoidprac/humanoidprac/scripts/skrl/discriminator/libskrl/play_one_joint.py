# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
1 つの単一ポリシー (.pt ファイル 1 個) を読み込み、その policy が学習対象とした 1 関節
だけを故障発生対象に絞って評価する play スクリプト。

target_joint_cfg.joint_names を [<joint_name>] のみへ上書きするため、env 内で発生する
故障は常にこの 1 関節のみとなる。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import ast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a single policy against a single joint failure.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the single .pt agent file.")
parser.add_argument(
    "--joint_name", type=str, required=True,
    help="Joint name this policy targets (e.g., 'right_knee'). target_joint_cfg はこれだけに絞られる。",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--finish_step", type=int, default=3000, help="終了ステップ数")
parser.add_argument(
    "--joint_cfg", type=str, default="",
    help="上書きする joint の effort_limit を辞書形式で指定する。例: '{\"left_knee_joint\": 150}'",
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


def _build_single_ppo_agent(env, agent_cfg: dict, suffix: str):
    """train_one.py / JointAgentLoader と同じ構造で PPO エージェントを 1 体構築する。"""
    import copy
    from skrl.utils.model_instantiators.torch import shared_model
    from gymnasium.spaces import Box
    from skrl.agents.torch.ppo import PPO
    from skrl.resources.schedulers.torch import KLAdaptiveRL
    from skrl.memories.torch import RandomMemory
    from numpy import float32

    cfg = copy.deepcopy(agent_cfg)
    cfg["agent"]["experiment"]["experiment_name"] += suffix
    cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveRL

    memory_size = cfg["memory"]["memory_size"]
    if memory_size < 0:
        memory_size = cfg["agent"]["rollouts"]
    memory = RandomMemory(
        memory_size=memory_size,
        num_envs=env.num_envs,
        device=env.device,
    )

    action_space = Box(low=-500.0, high=500.0, shape=env.action_space.shape, dtype=float32)

    roles = ["policy", "value"]
    structure = [cfg["models"]["policy"]["class"], cfg["models"]["value"]["class"]]
    parameters = [cfg["models"]["policy"], cfg["models"]["value"]]
    instance_shared_models = shared_model(
        observation_space=env.observation_space,
        action_space=action_space,
        device=env.device,
        structure=structure,
        roles=roles,
        parameters=parameters,
    )
    models = {
        "policy": instance_shared_models,
        "value": instance_shared_models,
    }
    models["policy"].init_state_dict("policy")
    models["value"].init_state_dict("value")

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg["agent"],
        observation_space=env.observation_space,
        action_space=action_space,
        device=env.device,
    )
    return agent


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Single-policy × single-joint failure play."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # joint の effort_limit を上書きする (任意)
    if args_cli.joint_cfg:
        param = ast.literal_eval(args_cli.joint_cfg)
        for joint_name, torque in param.items():
            env_cfg.scene.robot.actuators["legs"].effort_limit[joint_name] = torque
            print(f"[INFO] Overriding joint configuration: {joint_name} -> {torque}Nm")

    # ★ target_joint_cfg.joint_names をこのスクリプト指定の 1 関節のみに絞る
    try:
        from isaaclab.managers import SceneEntityCfg
        env_cfg.events.change_random_joint_torque_with_delayed_notification.params["target_joint_cfg"] = SceneEntityCfg(
            name="robot",
            joint_names=[args_cli.joint_name],
        )
        print(f"[INFO] target_joint_cfg.joint_names を ['{args_cli.joint_name}'] に上書きしました")
    except (AttributeError, KeyError) as e:
        print(f"[WARN] target_joint_cfg の上書きに失敗: {e}")

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    log_dir = os.path.join("play_logs", f"single_{args_cli.joint_name}")
    os.makedirs(log_dir, exist_ok=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

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

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # 推論用に学習由来のロギングを抑止する
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append("../")
    sys.path.append("../../")

    target_envs = torch.arange(1, env.num_envs)
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

    # 単一ポリシーを構築・読み込み
    if not os.path.isfile(args_cli.model_path):
        raise FileNotFoundError(f"--model_path に指定されたファイルが存在しません: {args_cli.model_path}")
    single_agent = _build_single_ppo_agent(env, agent_cfg, suffix=f"_play_single_{args_cli.joint_name}")
    single_agent.load(args_cli.model_path)
    single_agent.set_running_mode("eval")
    print(f"[INFO] loaded policy from {args_cli.model_path} for joint '{args_cli.joint_name}'")

    # 関節別生存率ロガー (単一関節版でも同じクラスを使う)
    joint_survival_logger = logger.JointSurvivalRateLogger(
        env=env,
        joint_names=[args_cli.joint_name],
        log_dir=log_dir,
    )

    # CustomParallelAgentTrainer.__init__ は agents=[*] を 1 つは要求するので、本ポリシーを
    # そのまま渡す (eval_single_policy_for_joint 内では self.agents は参照されない)。
    from custom_parallel_trainer import CustomParallelAgentTrainer
    trainer = CustomParallelAgentTrainer(
        env=env,
        agents=[single_agent],
        health_agent=health_agent,
        agents_scope=[],
        cfg=agent_cfg["trainer"],
    )

    try:
        trainer.eval_single_policy_for_joint(
            agent=single_agent,
            joint_name=args_cli.joint_name,
            expdata_logger=exp_val_logger,
            joint_survival_logger=joint_survival_logger,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
