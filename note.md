apt-get update

apt-get install -y \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    mesa-utils

source /mnt/pfs/users/angen.ye/myconda/conda/etc/profile.d/conda.sh
conda activate pi-rl
cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf
export REPO_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RLinf
export ROBOTWIN_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-main
export PYTHONPATH=$ROBOTWIN_PATH:$REPO_PATH:$PYTHONPATH

cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment

conda activate pi-rl-h20


python collect_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name collect_bell_data

train
cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment
python train_embodied_agent_gigawa.py   --config-path ./config   --config-name online_exp_bc_guard_weighted  ++actor.fsdp_config.use_orig_params=true

eval:
python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name bell_eval_fix \
  ++actor.fsdp_config.use_orig_params=true

# original all sliding 
python reshard_offline_collection.py \
  --input-root /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/data_bell_fix_422/offline_collection \
  --bucket all \
  --data-mode original \
  --output-root /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/data_bell_fix_422/mergeall_original \
  --target-world-size 4 \
  --shuffle \
  --source-cache-size 2048


python train_embodied_agent_gigawa_offline_bc.py \
  --config-path ./config \
  --config-name offline_bc_pretrain


python train_embodied_agent_gigawa_offline_critic_fast.py \
  --config-path ./config \
  --config-name offline_critic_pretrain2

python train_embodied_agent_gigawa_offline_rl_fast.py \
  --config-path ./config \
  --config-name offline_rl_pretrain.yaml

tensorboard --logdir /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/offline_rl_pretrain_mergeall_422_2/robotwin_train_giga_world_policy/tensorboard \
  --host 0.0.0.0 \
  --port 6006

python repair_pre_earlystop_buffer.py   --input-root /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/gigawa_offline_collect4_12chunk_fix/mergeall2   --output-root /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/gigawa_offline_collect4_12chunk_fix/mergeall_repaired3



CUDA_VISIBLE_DEVICES=5 python analyze_gigawa_pt_qsa.py \
  --config /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment/config/analysis.yaml \
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/data_bell_422/mergeall/rank_1/rank0_9_slide6_rerank1.pt \
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/data_bell_422/mergeall/rank_1/rank0_12_rerank1.pt \
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/data_bell_422/offline_collection/rank_0/success/rank0_30.pt \
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/data_bell_422/offline_collection/rank_0/failure/rank0_30.pt\
  --checkpoint /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/offline_rl_pretrain_mergeall_422_3/robotwin_train_giga_world_policy/checkpoints/global_step_400 \
  --output-dir /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/qsa_debug_rl3 \
  --device cuda