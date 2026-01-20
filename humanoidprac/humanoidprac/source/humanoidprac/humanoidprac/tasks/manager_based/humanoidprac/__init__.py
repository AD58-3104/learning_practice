# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Humanoidprac-v0-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Humanoidprac-v0-train-random-joint-debuff",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfgRandomJointDebuff",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# PLAY用の環境
gym.register(
    id="Humanoidprac-v0-train-random-joint-debuff-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfgRandomJointDebuff_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:learned_agent_cfg.yaml",
    },
)

# nn_discriminator用のデータ取得環境（これは健常な状態で学習したものを動かしてデータを取る必要がある）
gym.register(
    id="Humanoidprac-nn-disc-data-correction",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfgCorrectLearningData",
        "skrl_cfg_entry_point": f"{agents.__name__}:learned_agent_cfg.yaml",
    },
)

# 並列で故障モデルをトレーニングするタスク
gym.register(
    id="Parallel-failure-train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfgRandomJointDebuff",
        "skrl_cfg_entry_point": f"{agents.__name__}:learned_agent_cfg.yaml",
    },
)
# 👆のタスクのplay用
gym.register(
    id="Parallel-failure-train-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfgRandomJointDebuff_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:learned_agent_cfg.yaml",
    },
)

gym.register(
    id="Humanoidprac-v0-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Discriminatorの訓練環境
gym.register(
    id="Humanoidprac-discriminator",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfgDiscriminator",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_discriminator.yaml",
    },
)

# Discriminatorのplay環境 (ランダムに関節デバフが入る)
gym.register(
    id="Humanoidprac-discriminator-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfgRandomJointDebuff_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_discriminator.yaml",
    },
)

gym.register(
    id="Humanoidprac-discriminator-normal-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actual_env_config:H1FlatEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_discriminator.yaml",
    },
)
