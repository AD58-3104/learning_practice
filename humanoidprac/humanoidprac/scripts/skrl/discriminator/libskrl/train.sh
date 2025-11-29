#!/bin/bash
source ~/.bash_functions
if [ -z "$1" ]; then
    torque=50.0
else
    torque=$1
fi
echo "Using joint torque limit: $torque"
_labpython failure_model_train.py --task Humanoidprac-v0-train-random-joint-debuff --num_envs 8192 --headless env.events.change_random_joint_torque.params.joint_torque=[$torque] "${@:2}"  agent.trainer.timesteps=500000
