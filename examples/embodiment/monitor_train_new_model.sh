#!/usr/bin/env bash
set -euo pipefail

REPO_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RLinf
RESULT_ROOT=${REPO_PATH}/examples/results/train_new_model_7tasks
LOG_ROOT=${RESULT_ROOT}/_tmux_logs

TASKS=(
  beat_block_hammer
  click_bell
  place_container_plate
  place_empty_cup
  lift_pot
  move_can_pot
  place_shoe
)

printf "%-24s %-8s %s\n" "task" "tmux" "last_signal"
for task in "${TASKS[@]}"; do
  session="gwp_train_${task}"
  log_file="${LOG_ROOT}/${task}.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    state=running
  else
    state=stopped
  fi
  last="-"
  if [[ -f "${log_file}" ]]; then
    last=$(grep -E 'success_once|success_at_end|episode_reward|global_step|train/|eval/|Saving checkpoint|Error executing job|Traceback' "${log_file}" | tail -n 1 || true)
    [[ -n "${last}" ]] || last=$(tail -n 1 "${log_file}" || true)
    [[ -n "${last}" ]] || last="-"
  fi
  printf "%-24s %-8s %s\n" "${task}" "${state}" "${last}"
done

echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
