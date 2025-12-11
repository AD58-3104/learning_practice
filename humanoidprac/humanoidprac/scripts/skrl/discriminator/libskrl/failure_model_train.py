# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import ast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
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
parser.add_argument("--joint_cfg",type=str,default="",help="上書きするjointの設定を辞書形式で指定する. 例: '{\"left_knee_joint\": 150}'")

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from humanoidprac.tasks.manager_based.humanoidprac.mdp.events import EnvIdClassifier
from isaaclab_tasks.utils.hydra import hydra_task_config

import humanoidprac.tasks  # noqa: F401
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")
sys.path.append("../../")
# import logger  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # jointの設定を上書きする これもランダムで掛けるやつなので必要無し
    # if args_cli.joint_cfg:
    #     param = ast.literal_eval(args_cli.joint_cfg)
    #     print(f"[INFO] Overriding joint configuration : {env_cfg.scene.robot.actuators['legs'].effort_limit}")
    #     for joint_name, torque in param.items():
    #         env_cfg.scene.robot.actuators["legs"].effort_limit[joint_name] = torque
    #         print(f"[INFO] Overriding joint configuration with: {joint_name} -> {torque}Nm")

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
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
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
    # 設定のログは必要なし
    # joint_cfg_logger = logger.SettingLogger(env_cfg)
    # joint_cfg_logger.log_setting(os.path.join(log_dir, "joint_cfg.json"))

    # get checkpoint path (to resume training)
    # resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

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
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # 学習済みエージェントの読み込み
    target_agents = []
    target_scopes = []
    classifier = EnvIdClassifier(env.num_envs)
    agent_num = classifier.num_of_classes
    # 各クラスの環境数をscopeに設定
    for class_id in range(agent_num):
        target_scopes.append(classifier.get_class_envs(class_id))

    health_agent_model = "normal_agent.pt"
    tmp_rn = Runner(env, agent_cfg)
    tmp_rn.agent.load(health_agent_model)
    # 健康状態モデルのエージェント
    health_agent = tmp_rn.agent

    import copy
    from skrl.utils.model_instantiators.torch import shared_model
    from gymnasium.spaces import Box
    from skrl.agents.torch.ppo import PPO
    for class_id in range(agent_num):
        # エージェントをコンフィグから簡単に構築できるのがRunnerしか無いから使う
        # ログを書き込む場所を別にしたいので、書き換える
        tmp_agent_cfg = copy.deepcopy(agent_cfg)
        tmp_agent_cfg["agent"]["experiment"]["experiment_name"] += f"_model_{class_id}"
        from skrl.resources.schedulers.torch import KLAdaptiveRL
        tmp_agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveRL  # こっちではちゃんとクラスを入れないとダメ
        from skrl.memories.torch import RandomMemory
        # memory_sizeが-1の場合、rolloutsの値を使う
        memory_size = tmp_agent_cfg["memory"]["memory_size"]
        if memory_size < 0:
            memory_size = tmp_agent_cfg["agent"]["rollouts"]
        memory = RandomMemory(
            memory_size=memory_size,
            num_envs= classifier.get_class_envs(class_id),  # memoryの環境数はクラス毎の環境数に合わせる必要がある
            device=env.device,
        )
        # if classifier.get_class_envs(class_id) != target_scopes[class_id]:
        #     raise ValueError(f"環境数の不整合があります。{classifier.get_class_envs(class_id)} != {target_scopes[class_id]}")
        from numpy import float32
        # 環境と同じ行動空間の次元を使う（health_agentとの互換性のため）
        action_space = Box(low=-500.0, high=500.0, shape=env.action_space.shape, dtype=float32)
        
        # shared_modelsにはそれ用のinstantiatorがあるらしいぞ！
        roles = ["policy", "value"]
        structure = [tmp_agent_cfg["models"]["policy"]["class"], tmp_agent_cfg["models"]["value"]["class"]]
        parameters = [tmp_agent_cfg["models"]["policy"], tmp_agent_cfg["models"]["value"]]
        instance_shared_models = shared_model(
            observation_space=env.observation_space,
            action_space=action_space,
            device=env.device,
            structure=structure,
            roles=roles,
            parameters=parameters,
        )
        models = {}
        models["policy"] = instance_shared_models
        models["value"] = instance_shared_models
        models["policy"].init_state_dict("policy")
        models["value"].init_state_dict("value")
        agent = PPO(
            models=models,  # models dict
            memory=memory,  # memory instance, or None if not required
            cfg=tmp_agent_cfg["agent"],  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=action_space,
            device=env.device,
        )
        agent.load(health_agent_model)  # 初期化は健康状態モデルで行う
        target_agents.append(agent)

    from custom_parallel_trainer import CustomParallelAgentTrainer
    trainer = CustomParallelAgentTrainer(
        env=env,
        agents=target_agents,
        health_agent=health_agent,
        agents_scope=target_scopes,
        cfg=agent_cfg["trainer"],
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        # sigintでもモデルを保存したいのでキャッチ
        print("\n[INFO] Training interrupted by user.")
    
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
