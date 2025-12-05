#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "モデルのパスを指定してください。"
    echo "使い方: $0 <models_dir>"
    exit 1
fi
sleep 3
source ~/.bash_functions
_labpython play_hydra.py --task Humanoidprac-v0-train-random-joint-debuff-play --num_envs 64 --models_dir "$@"