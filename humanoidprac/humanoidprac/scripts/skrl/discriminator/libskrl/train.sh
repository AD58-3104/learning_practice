#!/bin/bash
source ~/.bash_functions
if [ -z "$1" ]; then
    torque=50.0
else
    torque=$1
fi
echo "Using joint torque limit: $torque"
_labpython failure_model_train.py --task Parallel-failure-train-v0 --num_envs 8192 env.events.change_random_joint_torque.params.joint_torque=[$torque] "${@:2}" --headless agent.trainer.timesteps=500000
