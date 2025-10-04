#!/bin/bash
source ~/.bash_functions
if [ -z "$1" ]; then
    torque=50.0
else
    torque=$1
fi
echo "Using joint torque limit: $torque"
_labpython train.py --task Humanoidprac-discriminator --num_envs 8192 --headless env.events.change_random_joint_torque.params.joint_torque=[$torque]
