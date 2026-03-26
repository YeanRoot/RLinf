#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_eval.sh /path/to/RLinf-main /path/to/RoboTwin /path/to/robotwin_assets

REPO_PATH=${1:-$(pwd)}
ROBOTWIN_PATH=${2:-/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-main}
ROBOTWIN_ASSETS=${3:-/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-main}

EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export EMBODIED_PATH
export REPO_PATH
export ROBOTWIN_PATH
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

cd "${EMBODIED_PATH}"
python eval_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name robotwin_place_empty_cup_eval_giga_world_policy \
  env.train.assets_path="${ROBOTWIN_ASSETS}" \
  env.eval.assets_path="${ROBOTWIN_ASSETS}" \
  runner.logger.log_path="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')"
