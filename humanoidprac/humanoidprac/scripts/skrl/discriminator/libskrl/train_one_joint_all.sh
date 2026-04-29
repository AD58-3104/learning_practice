#!/bin/bash
#
# 8関節について failure_model_train.py を順次実行するループスクリプト。
#
# 旧版は各ラウンドの Isaac Sim プロセス/共有メモリが完全解放されず
# 累積して OOM を引き起こした (2026-04-24 incident)。改修ポイント:
#   1. ジョイントを配列化してループで実行
#   2. 各ラウンド後に failure_model_train.py に限定した pkill で残留を掃除
#   3. 自分が所有する /dev/shm の isaac/kit 関連ファイルも掃除
#   4. 各ラウンド前後にメモリ状況を記録して累積を早期検出できるようにする
#
# 旧版バックアップ: train_one_joint.sh.bak-20260424

set -u

source ~/.bash_functions

JOINTS=(
    right_hip_yaw
    left_hip_yaw
    right_hip_roll
    left_hip_roll
    right_hip_pitch
    left_hip_pitch
    right_knee
    left_knee
)

NUM_ENVS="${NUM_ENVS:-8192}"
DEVICE="${DEVICE:-cuda:0}"
TIMESTEPS="${TIMESTEPS:-24000}"
TASK="${TASK:-Parallel-failure-train-v0-one-joint}"

log() {
    echo "[$(date '+%F %T')] $*"
}

cleanup_round() {
    log "cleanup: terminating residual train_one processes"
    pkill -TERM -f 'train_one\.py' 2>/dev/null || true
    sleep 5
    pkill -KILL -f 'train_one\.py' 2>/dev/null || true

    # Isaac Sim / Kit が残した shared memory ファイルを自分のオーナーのみ削除
    find /dev/shm -maxdepth 1 -user "$USER" \
        \( -name 'isaac*' -o -name 'kit_*' -o -name 'omni*' -o -name 'carb*' \) \
        -delete 2>/dev/null || true

    sleep 2
}

log "=== train_one_joint.sh start (joints=${#JOINTS[@]}, num_envs=$NUM_ENVS, device=$DEVICE) ==="

for joint in "${JOINTS[@]}"; do
    log "---- training joint: $joint ----"
    free -h | awk 'NR<=2'

    _labpython train_one.py \
        --task "$TASK" \
        --num_envs "$NUM_ENVS" \
        --headless \
        --device "$DEVICE" \
        agent.trainer.timesteps="$TIMESTEPS" \
        env.events.change_random_joint_torque_with_delayed_notification.params.target_joint_cfg.joint_names="['$joint']"
    rc=$?
    log "joint=$joint finished (rc=$rc)"

    cleanup_round

    log "after cleanup:"
    free -h | awk 'NR<=2'
done

log "=== train_one_joint.sh done ==="
