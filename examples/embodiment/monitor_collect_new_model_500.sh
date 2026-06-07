#!/usr/bin/env bash
set -euo pipefail

REPO_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RLinf
RESULT_ROOT=${REPO_PATH}/examples/results/collect_new_model_500

TASKS=(
  beat_block_hammer
  click_bell
  place_empty_cup
  lift_pot
  move_can_pot
  place_container_plate
  place_shoe
)

printf "%-24s %8s %8s %8s %8s %8s %-8s %s\n" "task" "saved" "summary" "success" "failure" "videos" "tmux" "last_log"
for task in "${TASKS[@]}"; do
  base="${RESULT_ROOT}/${task}"
  summary="${base}/offline_collection/rank_0/trajectory_summaries.jsonl"
  log_file="${RESULT_ROOT}/_tmux_logs/${task}.log"
  saved=$(find "${base}/offline_collection/rank_0/all" -maxdepth 1 -name 'rank*.pt' 2>/dev/null | wc -l | tr -d ' ')
  if [[ -f "${summary}" ]]; then
    all=$(wc -l < "${summary}" | tr -d ' ')
    success=$(grep -c '"is_success": true' "${summary}" || true)
    failure=$(grep -c '"is_success": false' "${summary}" || true)
  else
    all=0
    success=0
    failure=0
  fi
  if [[ -d "${base}/video/train/seed_0" ]]; then
    videos=$(find "${base}/video/train/seed_0" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  else
    videos=0
  fi
  if tmux has-session -t "gwp_collect_${task}" 2>/dev/null; then
    tmux_state=running
  else
    tmux_state=stopped
  fi
  last_log="-"
  if [[ -f "${log_file}" ]]; then
    last_log=$(grep '\[collect\]' "${log_file}" | tail -n 1 | sed -E 's/^.*\[collect\] //')
    [[ -n "${last_log}" ]] || last_log="-"
  fi
  printf "%-24s %8s %8s %8s %8s %8s %-8s %s\n" "${task}" "${saved}" "${all}" "${success}" "${failure}" "${videos}" "${tmux_state}" "${last_log}"
done
