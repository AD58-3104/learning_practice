#!/bin/bash
dir=$1

cat $dir/joint_cfg.json
sleep 3
source ~/.bash_functions
_labpython train.py --task Humanoidprac-v0-play --checkpoint $dir/checkpoints/best_agent.pt