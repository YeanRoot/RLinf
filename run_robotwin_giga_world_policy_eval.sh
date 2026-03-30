#!/usr/bin/env bash
set -euo pipefail
set -x

# ===== user-editable paths =====
REPO_ROOT="/shared_disk/users/angen.ye/code/world_module_rollout/RLinf"
ROBOTWIN_ROOT="/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-main"
WAN_MODEL_ID="/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
TRANSFORMER_CKPT="/shared_disk/users/xinyu.zhou/experiments/wa_single/robotwin/models/checkpoint_epoch_2_step_100000/transformer"
NORM_JSON="/shared_disk/users/chaojun.ni/peking-human/paper/RoboTwin2.0/pi0_1/norm_stats_delta.json"
LOG_DIR="/shared_disk/users/angen.ye/code/world_module_rollout/results/rlinf_giga_world_policy_eval"
CONFIG_NAME="robotwin_place_empty_cup_eval_giga_world_policy"
GPU_ID="${GPU_ID:-0}"
ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"
ROBOTYPE="${ROBOTYPE:-aloha}"

# ===== derived =====
EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
RUN_LOG="${LOG_DIR}/eval_embodied_agent.log"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export EMBODIED_PATH
export REPO_PATH="${REPO_ROOT}"
export ROBOTWIN_PATH="${ROBOTWIN_ROOT}"
export ROBOT_PLATFORM
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${REPO_ROOT}:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  env.train.assets_path="${ROBOTWIN_ROOT}" \
  env.eval.assets_path="${ROBOTWIN_ROOT}" \
  env.train.total_num_envs=1 \
  env.eval.total_num_envs=10 \
  env.train.max_episode_steps=192 \
  env.eval.max_episode_steps=192 \
  env.train.max_steps_per_rollout_epoch=1920 \
  env.eval.max_steps_per_rollout_epoch=1920 \
  env.eval.video_cfg.save_video=True \
  env.eval.video_cfg.video_base_dir="${LOG_DIR}/video/eval" \
  actor.model.model_path="${TRANSFORMER_CKPT}" \
  actor.model.wan_model_id="${WAN_MODEL_ID}" \
  actor.model.norm_json="${NORM_JSON}" \
  actor.model.robotype="${ROBOTYPE}" \
  rollout.model.model_path="${TRANSFORMER_CKPT}" \
  rollout.model.precision=bf16 \
  2>&1 | tee "${RUN_LOG}"
