#!/usr/bin/env bash
set -euo pipefail

CONDA_SH=/mnt/pfs/users/angen.ye/myconda/conda/etc/profile.d/conda.sh
ENV_NAME=pi-rl
REPO_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RLinf
ROBOTWIN_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-RLinf_support
EMBODIMENT_DIR=${REPO_PATH}/examples/embodiment
RESULT_ROOT=${REPO_PATH}/examples/results/train_new_model_7tasks
LOG_ROOT=${RESULT_ROOT}/_tmux_logs
RAY_TMP=/tmp/ray_gwp_train_new_model_7tasks

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
  "beat_block_hammer online_rl_beat_block_hammer_new_model 0 1"
  "click_bell online_rl_click_bell_new_model 2 3"
  "place_container_plate online_rl_place_container_plate_new_model 5 6"
)

for item in "${TASKS[@]}"; do
  read -r task cfg actor_gpu rollout_gpu <<<"${item}"
  session="gwp_train_${task}"
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
python -u train_embodied_agent_gigawa.py \\
  --config-path ./config \\
  --config-name "${cfg}"
EOF
  chmod +x "${run_script}"

  echo "starting ${session}: actor GPU ${actor_gpu}, rollout/env GPU ${rollout_gpu}"
  echo "" >> "${log_file}"
  echo "===== start $(date) cfg=${cfg} actor_gpu=${actor_gpu} rollout_gpu=${rollout_gpu} =====" >> "${log_file}"
  tmux new-session -d -s "${session}" "bash '${run_script}' >> '${log_file}' 2>&1"
  sleep 30
done

tmux list-sessions | grep '^gwp_train_' || true
