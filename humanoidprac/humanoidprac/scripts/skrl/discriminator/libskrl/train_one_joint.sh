#!/bin/bash
source ~/.bash_functions
_labpython failure_model_train.py --task Parallel-failure-train-v0-one-joint --num_envs 8192 --headless --device 'cuda:1' agent.trainer.timesteps=24000 env.events.change_random_joint_torque_with_delayed_notification.params.target_joint_cfg.joint_names="['right_hip_yaw']"