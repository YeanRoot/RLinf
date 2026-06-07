#!/usr/bin/env bash
set -euo pipefail

REPO_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RLinf
RESULT_ROOT=${REPO_PATH}/examples/results/collect_new_model_500
LOG_ROOT=${RESULT_ROOT}/_tmux_logs
TARGET=500
INTERVAL=300

TASKS=(
  beat_block_hammer
  click_bell
  place_empty_cup
  lift_pot
  move_can_pot
  place_container_plate
  place_shoe
)

while true; do
  all_done=1
  for task in "${TASKS[@]}"; do
    base="${RESULT_ROOT}/${task}/offline_collection/rank_0/all"
    if [[ -d "${base}" ]]; then
      saved=$(find "${base}" -maxdepth 1 -name 'rank*.pt' | wc -l | tr -d ' ')
    else
      saved=0
    fi
    session="gwp_collect_${task}"

    if (( saved < TARGET )); then
      all_done=0
      if ! tmux has-session -t "${session}" 2>/dev/null; then
        echo "[$(date '+%F %T')] restarting ${task}, saved=${saved}/${TARGET}"
        echo "" >> "${LOG_ROOT}/${task}.log"
        echo "===== watchdog restart $(date) saved=${saved}/${TARGET} =====" >> "${LOG_ROOT}/${task}.log"
        tmux new-session -d -s "${session}" \
          "bash '${LOG_ROOT}/run_${task}.sh' >> '${LOG_ROOT}/${task}.log' 2>&1"
      fi
    fi
  done

  if (( all_done == 1 )); then
    echo "[$(date '+%F %T')] all tasks reached ${TARGET}, watchdog exiting"
    exit 0
  fi

  sleep "${INTERVAL}"
done
