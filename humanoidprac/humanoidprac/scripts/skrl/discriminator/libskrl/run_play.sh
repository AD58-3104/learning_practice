#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "モデルのパスを指定してください。"
    echo "使い方: $0 <models_dir>"
    exit 1
fi
sleep 3
source ~/.bash_functions
# 本番 H1FlatEnvCfgRandomJointDebuff_PLAY
 _labpython play_hydra.py --task Humanoidprac-v0-train-random-joint-debuff-play --num_envs 8192 --headless --models_dir "$@"  agent.trainer.timesteps=1000 --finish_step 1000
# ネットワークのテスト H1FlatEnvCfgRandomJointDebuff_PLAY
# _labpython play_hydra.py --task Humanoidprac-v0-train-random-joint-debuff-play --num_envs 8192 --headless --models_dir "$@"  agent.trainer.timesteps=10000 --finish_step 1000 --test_jointnet --online_testing
# データ収集  H1FlatEnvCfgCorrectLearningData
# _labpython play_hydra.py --task Humanoidprac-nn-disc-data-correction --num_envs 4096 --headless --models_dir "$@"  agent.trainer.timesteps=1000  --log_nn_data
