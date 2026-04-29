# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train a SINGLE PPO policy for all fault classes with skrl.

failure_model_train.py のクラスごとにエージェントを切り替える学習ではなく、
故障環境を全て一つのポリシーで担当する学習を行う。
CustomParallelAgentTrainer.train_with_delayed_failure_info_single_policy を呼び出す。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a single-policy RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
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
import random
from datetime import datetime

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
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import humanoidprac.tasks  # noqa: F401
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
sys.path.append("../../")

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train a single-policy skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}_single"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # checkpoint
    if args_cli.checkpoint is not None:
        from custom_parallel_trainer import AgentModelLoader
        checkpoint_pathes = AgentModelLoader.get_model_pathes(args_cli.checkpoint)
    else:
        checkpoint_pathes = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    print(f"[DEBUG] env.observation_space: {env.observation_space}")
    print(f"[DEBUG] env.single_observation_space: {env.single_observation_space}")

    # 健康状態モデル
    health_agent_model_path = "normal_agent.pt"
    tmp_rn = Runner(env, agent_cfg)
    tmp_rn.agent.load(health_agent_model_path)
    health_agent = tmp_rn.agent

    # 単一ポリシー用エージェントを1つだけ構築する
    import copy
    from skrl.utils.model_instantiators.torch import shared_model
    from gymnasium.spaces import Box
    from skrl.agents.torch.ppo import PPO
    from skrl.resources.schedulers.torch import KLAdaptiveRL
    from skrl.memories.torch import RandomMemory
    from numpy import float32

    single_agent_cfg = copy.deepcopy(agent_cfg)
    # log_dir 側で既に "_single" 等を付けているので、ここでは experiment_name を上書きしない
    single_agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveRL

    memory_size = single_agent_cfg["memory"]["memory_size"]
    if memory_size < 0:
        memory_size = single_agent_cfg["agent"]["rollouts"]
    memory = RandomMemory(
        memory_size=memory_size,
        num_envs=env.num_envs,
        device=env.device,
    )

    action_space = Box(low=-500.0, high=500.0, shape=env.action_space.shape, dtype=float32)

    roles = ["policy", "value"]
    structure = [single_agent_cfg["models"]["policy"]["class"], single_agent_cfg["models"]["value"]["class"]]
    parameters = [single_agent_cfg["models"]["policy"], single_agent_cfg["models"]["value"]]
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

    single_agent = PPO(
        models=models,
        memory=memory,
        cfg=single_agent_cfg["agent"],
        observation_space=env.observation_space,
        action_space=action_space,
        device=env.device,
    )
    if checkpoint_pathes is not None:
        # 単一ポリシー版なのでチェックポイント先頭のモデルを読み込む
        single_agent.load(checkpoint_pathes[0])
    else:
        single_agent.load(health_agent_model_path)

    # change_random_joint_torque_with_delayed_notification の target_joint_cfg.joint_names を取得
    # train_one_joint.sh から CLI 経由で 1 関節が指定される想定。複数指定でも安全に連結する。
    target_joint_names = []
    try:
        target_joint_cfg = env_cfg.events.change_random_joint_torque_with_delayed_notification.params["target_joint_cfg"]
        if target_joint_cfg.joint_names is not None:
            target_joint_names = list(target_joint_cfg.joint_names)
    except (AttributeError, KeyError) as e:
        print(f"[WARN] target_joint_cfg.joint_names を取得できませんでした: {e}")

    if target_joint_names:
        joint_name_tag = "_".join(target_joint_names)
        single_policy_save_name = f"agent_{joint_name_tag}.pt"
    else:
        single_policy_save_name = None
    print(f"[INFO] target joint(s) for single policy: {target_joint_names} -> save name: {single_policy_save_name}")

    from custom_parallel_trainer import CustomParallelAgentTrainer
    trainer = CustomParallelAgentTrainer(
        env=env,
        agents=[single_agent],
        health_agent=health_agent,
        agents_scope=[],
        cfg=agent_cfg["trainer"],
        single_policy_save_name=single_policy_save_name,
    )

    try:
        trainer.train_with_delayed_failure_info_single_policy()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
