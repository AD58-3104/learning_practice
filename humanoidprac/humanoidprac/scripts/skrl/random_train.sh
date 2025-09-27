#!/bin/bash
source ~/.bash_functions
if [ -z "$1" ]; then
    torque=50.0
else
    torque=$1
fi
echo "Using joint torque limit: $torque"
_labpython train.py --task Humanoidprac-v0-train-random-joint-debuff --num_envs 8192 --headless agent.agent.experiment.directory="h1_flat/joint_experiment_ver3" env.events.change_random_joint_torque.params.joint_torque=[$torque] agent.trainer.timesteps=64000