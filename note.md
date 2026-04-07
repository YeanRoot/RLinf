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

eval_test.py
cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_embodied_agent.py   --config-path ./config   --config-name robotwin_place_empty_cup_eval_giga_world_policy_eval   env.eval.total_num_envs=4   algorithm.eval_rollout_epoch=5

train
cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment
python train_embodied_agent_gigawa.py   --config-path ./config   --config-name zero  ++actor.fsdp_config.use_orig_params=true

cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment
python train_embodied_agent_gigawa.py   --config-path ./config   --config-name delete  ++actor.fsdp_config.use_orig_params=true

tensorboard --logdir /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/test403_1_rl \
  --host 0.0.0.0 \
  --port 6006