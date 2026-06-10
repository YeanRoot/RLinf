#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ubuntu/users/angen.ye/gwp/RLinf"
DATA_ROOT="$REPO/examples/results/collect_piper_gigawa_success100_failure100_7/offline_collection/rank_0"
RUN_ROOT="${RUN_ROOT:-$REPO/examples/results/offline_rl_piper_success100_failure100_7_intervention_local_v3_$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="piper_intervention_actor_local_v3"
CONFIG_NAME="offline_rl_piper_success100_failure100_7_intervention_local"
NORM_JSON="/home/ubuntu/users/angen.ye/gwp/norm_stats_delta.json"
URDF="$REPO/examples/results/piper_local_assets_tmp/piper.urdf"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
STOP_STEP="${STOP_STEP:-600}"
MAX_FILES="${MAX_FILES:-2}"

cd "$REPO"
source switch_env gigaworld
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT/$EXP_NAME/logs"
printf "%s/%s\n" "$RUN_ROOT" "$EXP_NAME" > "$REPO/examples/results/latest_intervention_actor_local_path.txt"

tmux has-session -t piper_local_irl 2>/dev/null && tmux kill-session -t piper_local_irl || true
tmux has-session -t piper_local_iviz 2>/dev/null && tmux kill-session -t piper_local_iviz || true

tmux new-session -d -s piper_local_irl "bash -ic '
  source switch_env gigaworld &&
  cd \"$REPO/examples/embodiment\" &&
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\" &&
  python -u train_embodied_agent_gigawa_offline_rl_fast.py \
    --config-path ./config \
    --config-name \"$CONFIG_NAME\" \
    runner.logger.log_path=\"$RUN_ROOT\" \
    runner.logger.experiment_name=\"$EXP_NAME\" \
    runner.save_interval=\"$SAVE_INTERVAL\" \
    runner.save_initial_checkpoint=true \
    runner.save_replay_buffers_with_checkpoint=false \
    runner.resume_dir=null \
    runner.ckpt_path=null \
    algorithm.offline_rl_pretrain.steps_per_epoch=1 \
    algorithm.offline_rl_pretrain.val_steps_per_epoch=1 \
    algorithm.offline_rl_pretrain.class_eval_steps_per_epoch=1 \
    2>&1 | tee \"$RUN_ROOT/$EXP_NAME/logs/train.log\"
'"

tmux new-session -d -s piper_local_iviz "bash -ic '
  source switch_env gigaworld &&
  cd \"$REPO\" &&
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\" &&
  python -u examples/embodiment/watch_openloop_checkpoints.py \
    --repo \"$REPO\" \
    --experiment-dir \"$RUN_ROOT/$EXP_NAME\" \
    --success-pt \"$DATA_ROOT/success\" \
    --failure-pt \"$DATA_ROOT/failure\" \
    --norm-json \"$NORM_JSON\" \
    --urdf \"$URDF\" \
    --out-root \"$RUN_ROOT/$EXP_NAME/openloop_viz_exec_space\" \
    --max-files \"$MAX_FILES\" \
    --interval 30 \
    --stop-step \"$STOP_STEP\" \
    --min-step 0 \
    --step-multiple \"$SAVE_INTERVAL\" \
    --device cpu \
    --dtype float32 \
    2>&1 | tee \"$RUN_ROOT/$EXP_NAME/logs/watch_openloop.log\"
'"

echo "[launch] train session=piper_local_irl log=$RUN_ROOT/$EXP_NAME/logs/train.log"
echo "[launch] viz session=piper_local_iviz log=$RUN_ROOT/$EXP_NAME/logs/watch_openloop.log"
echo "[launch] latest path file=$REPO/examples/results/latest_intervention_actor_local_path.txt"
