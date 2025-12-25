#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "モデルのパスを指定してください。"
    echo "使い方: $0 <models_dir>"
    exit 1
fi
sleep 3
source ~/.bash_functions
_labpython play_hydra.py --task Humanoidprac-v0-train-random-joint-debuff-play --num_envs 4096 --headless --models_dir "$@"  agent.trainer.timesteps=100000 --finish_step 1000 --online_testing --test_jointnet --log_nn_data
# _labpython play_hydra.py --task Humanoidprac-nn-disc-data-correction --num_envs 2 --models_dir "$@" --headless agent.trainer.timesteps=100000 --finish_step 1000 --online_testing --test_jointnet
