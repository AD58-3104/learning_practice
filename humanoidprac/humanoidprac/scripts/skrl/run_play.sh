#!/bin/bash
dir=$1

cat $dir/joint_cfg.json
sleep 3
source ~/.bash_functions
_labpython play_hydra.py --checkpoint $dir/checkpoints/best_agent.pt  ${@:2} 