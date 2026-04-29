#!/bin/bash
# 8 関節分の単一ポリシーモデルを 1 フォルダから読み込んで推論する起動スクリプト。
# usage: bash run_play_one.sh <models_dir>
#   <models_dir> には agent_<joint_name>.pt の 8 ファイルがまとめて入っている必要がある。

if [ "$#" -lt 1 ]; then
    echo "モデルのパスを指定してください。"
    echo "使い方: $0 <models_dir>"
    exit 1
fi

sleep 3
source ~/.bash_functions

_labpython play_one.py \
    --task Humanoidprac-v0-train-random-joint-debuff-play \
    --num_envs 8192 \
    --headless \
    --models_dir "$1" \
    agent.trainer.timesteps=1000 \
    --finish_step 1000
