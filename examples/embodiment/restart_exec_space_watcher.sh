#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ubuntu/users/angen.ye/gwp/RLinf"
DATA_ROOT="$REPO/examples/results/collect_piper_gigawa_success100_failure100_7/offline_collection/rank_0"
NORM_JSON="/home/ubuntu/users/angen.ye/gwp/rollout1/norm_stats_delta.json"
URDF="$REPO/examples/results/piper_local_assets_tmp/piper.urdf"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

source switch_env gigaworld
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RUN_DIR="$(cat "$REPO/examples/results/latest_zero_actor_local_path.txt")"
mkdir -p "$RUN_DIR/logs"

tmux has-session -t piper_local_zviz 2>/dev/null && tmux kill-session -t piper_local_zviz || true

tmux new-session -d -s piper_local_zviz "bash -ic '
  source switch_env gigaworld &&
  cd \"$REPO\" &&
  export PYTHONPATH=\"$REPO:\${PYTHONPATH:-}\" &&
  python -u examples/embodiment/watch_openloop_checkpoints.py \
    --repo \"$REPO\" \
    --experiment-dir \"$RUN_DIR\" \
    --success-pt \"$DATA_ROOT/success\" \
    --failure-pt \"$DATA_ROOT/failure\" \
    --norm-json \"$NORM_JSON\" \
    --urdf \"$URDF\" \
    --out-root \"$RUN_DIR/openloop_viz_exec_space\" \
    --max-files 3 \
    --interval 30 \
    --stop-step 600 \
    --min-step 0 \
    --step-multiple \"$SAVE_INTERVAL\" \
    --device cpu \
    --dtype float32 \
    2>&1 | tee \"$RUN_DIR/logs/watch_openloop_exec_space.log\"
'"

echo "[watcher] session=piper_local_zviz"
echo "[watcher] log=$RUN_DIR/logs/watch_openloop_exec_space.log"
echo "[watcher] out=$RUN_DIR/openloop_viz_exec_space"
