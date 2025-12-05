# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl with Hydra configuration support.

This version supports overriding env.events parameters through command line arguments.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import ast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
parser.add_argument("--finish_step",type=int,default=3000,help="終了ステップ数")
parser.add_argument("--joint_cfg",type=str,default="",help="上書きするjointの設定を辞書形式で指定する. 例: '{\"left_knee_joint\": 150}'")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments - split between argparse and hydra args
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
import time
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

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry
from isaaclab_tasks.utils.hydra import hydra_task_config

import humanoidprac.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # jointの設定を上書きする
    if args_cli.joint_cfg:
        param = ast.literal_eval(args_cli.joint_cfg)
        print(f"[INFO] Overriding joint configuration : {env_cfg.scene.robot.actuators['legs'].effort_limit}")
        for joint_name, torque in param.items():
            env_cfg.scene.robot.actuators["legs"].effort_limit[joint_name] = torque
            print(f"[INFO] Overriding joint configuration with: {joint_name} -> {torque}Nm")

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import logger
    joint_cfg_logger = logger.SettingLogger(env_cfg)
    joint_cfg_logger.log_setting(os.path.join(log_dir, "play_joint_cfg.json"))

    learned_models = ["normal_agent.pt","knee_agent.pt"]
    import yaml
    learned_model_conf = yaml.safe_load(open("learned_agent_cfg.yaml"))
    learned_agents = []
    for model in learned_models:
        rn = Runner(env, learned_model_conf)
        rn.agent.load(model)
        learned_agents.append(rn.agent)

    from skrl.memories.torch import RandomMemory
    memory = RandomMemory(
        memory_size=agent_cfg["memory"]["memory_size"],
        num_envs=env.num_envs,
        device=env.device,
    )

    from skrl.utils.model_instantiators.torch import deterministic_model
    import discrete_model
    from gymnasium.spaces import Discrete

    models = {}
    # models["policy"] = discrete_model.MLP(
    #                 observation_space=env.observation_space,
    #                 action_space=Discrete(2),  # 2つのモデルから選ぶ行動
    #                 device=env.device,
    #             )
    models["policy"] = discrete_model.GRU(
                observation_space=env.observation_space,
                action_space=Discrete(2),  # 2つのモデルから選ぶ行動
                device=env.device,
                num_envs=env.num_envs,
                sequence_length=6,  # 6ステップ分の履歴を考慮。
                # これは、ロールアウトの時のステップ数と関連がある。rollout=24,mini_batches=4であれば、
                # 6ステップ分の履歴まで考慮できる。6ステップの、それの約数だけが使える。
                )
    models["value"] = deterministic_model(
                    observation_space=env.observation_space,
                    action_space=Discrete(2),  # discriminator用の離散アクション空間
                    device=env.device,
                    **agent_cfg["models"]["value"],
                )

    from skrl.resources.schedulers.torch import KLAdaptiveRL
    agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveRL

    # from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo import PPO_RNN as PPO
    agent = PPO(
        models=models,  # models dict
        memory=memory,  # memory instance, or None if not required
        cfg=agent_cfg["agent"],  # configuration dict (preprocessors, learning rate schedulers, etc.)
        observation_space=env.observation_space,
        action_space=Discrete(2),
        device=env.device,
    )

    agent.init()
    # load checkpoint (if specified)
    agent.load(resume_path)
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    target_envs = torch.arange(1,env.num_envs)  # 全ての環境のデータの平均値を取る
    if args_cli.finish_step:
        finish_step = args_cli.finish_step
    exp_val_logger = logger.ExperimentValueLogger(finish_step=finish_step, log_file_name=os.path.join(log_dir, "play_log.csv"), target_envs=target_envs)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            select_action = agent.act(obs, timestep=0, timesteps=0)[0]
            outputs = learned_agents[select_action[0]].act(obs, timestep=0, timesteps=0)

            actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
            if exp_val_logger.log(env):
                print("終了ステップに到達しました。")
                break

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()