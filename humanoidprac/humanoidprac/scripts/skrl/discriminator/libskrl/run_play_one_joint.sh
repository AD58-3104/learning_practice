#!/bin/bash
# 単一ポリシー × 単一関節故障の play 起動スクリプト。
# usage: bash run_play_one_joint.sh <model_path.pt> <joint_name>
#   <model_path>: 単一の .pt エージェントファイル (例: trained_models/<dir>/agent_right_knee.pt)
#   <joint_name>: そのポリシーが対象とする関節名 (例: right_knee)

if [ "$#" -lt 2 ]; then
    echo "使い方: $0 <model_path.pt> <joint_name>"
    echo "  例: $0 trained_models/2026-04-28_20-16-02_ppo/agent_left_knee.pt left_knee"
    exit 1
fi

MODEL_PATH="$1"
JOINT_NAME="$2"

sleep 3
source ~/.bash_functions

_labpython play_one_joint.py \
    --task Humanoidprac-v0-train-random-joint-debuff-play \
    --num_envs 8192 \
    --headless \
    --model_path "$MODEL_PATH" \
    --joint_name "$JOINT_NAME" \
    agent.trainer.timesteps=1000 \
    --finish_step 1000
