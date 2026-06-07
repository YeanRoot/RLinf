#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=/mnt/pfs/users/angen.ye/myconda/conda/etc/profile.d/conda.sh
ENV_NAME=pi-rl
REPO_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RLinf
ROBOTWIN_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-RLinf_support
EMBODIMENT_DIR=${REPO_PATH}/examples/embodiment
RESULT_ROOT=${REPO_PATH}/examples/results/collect_new_model_500
LOG_ROOT=${RESULT_ROOT}/_tmux_logs
RAY_TMP=/tmp/ray_gwp_collect_new_model_500

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

mkdir -p "${LOG_ROOT}"
cd "${EMBODIMENT_DIR}"

export REPO_PATH
export ROBOTWIN_PATH
export PYTHONPATH="${ROBOTWIN_PATH}:${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus=8 --disable-usage-stats --temp-dir="${RAY_TMP}" \
    >"${LOG_ROOT}/ray_start.log" 2>&1
fi

TASKS=(
  "beat_block_hammer collect_robotwin_beat_block_hammer_new_model_500 0"
  "click_bell collect_robotwin_click_bell_new_model_500 1"
  "place_empty_cup collect_robotwin_place_empty_cup_new_model_500 2"
  "lift_pot collect_robotwin_lift_pot_new_model_500 3"
  "move_can_pot collect_robotwin_move_can_pot_new_model_500 5"
  "place_container_plate collect_robotwin_place_container_plate_new_model_500 6"
  "place_shoe collect_robotwin_place_shoe_new_model_500 7"
)

for item in "${TASKS[@]}"; do
  read -r task cfg gpu <<<"${item}"
  session="gwp_collect_${task}"
  log_file="${LOG_ROOT}/${task}.log"
  run_script="${LOG_ROOT}/run_${task}.sh"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session ${session} already exists, skipping"
    continue
  fi

  cat >"${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
cd "${EMBODIMENT_DIR}"
export REPO_PATH="${REPO_PATH}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH}"
export PYTHONPATH="${ROBOTWIN_PATH}:${REPO_PATH}:\${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
python -u collect_embodied_agent_gigawa.py \\
  --config-path ./config \\
  --config-name "${cfg}"
EOF
  chmod +x "${run_script}"

  echo "starting ${session} on configured GPU ${gpu}"
  tmux new-session -d -s "${session}" "bash '${run_script}' >> '${log_file}' 2>&1"
  sleep 20
done

tmux list-sessions | grep '^gwp_collect_' || true
