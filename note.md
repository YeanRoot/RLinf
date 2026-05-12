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
export ROBOTWIN_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-RLinf_support
export PYTHONPATH=$ROBOTWIN_PATH:$REPO_PATH:$PYTHONPATH

cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment

conda activate pi-rl-h20
export ROBOTWIN_PATH=/shared_disk/users/angen.ye/code/world_module_rollout/RoboTwin-main

python collect_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name collect_bell_data_fix

train

source switch_env gigaworld

cd /home/ubuntu/users/angen.ye/gwp/RLinf
export PYTHONPATH=$PWD:$PYTHONPATH

cd /home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment
python train_embodied_agent_gigawa.py   --config-path ./config   --config-name online_rl_piper_gigawa

eval:
python eval_embodied_agent.py \
  --config-path ./config \
  --config-name eval_piper_gigawa_wa_only

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
  --config-name offline_critic_pretrain

python train_embodied_agent_gigawa_offline_rl_fast.py \
  --config-path ./config \
  --config-name offline_rl_pretrain

tensorboard --logdir /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/online_rl_cup_504_test/tensorboard \
  --host 0.0.0.0 \
  --port 6006

python repair_pre_earlystop_buffer.py   --input-root /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/gigawa_offline_collect4_12chunk_fix/mergeall2   --output-root /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/gigawa_offline_collect4_12chunk_fix/mergeall_repaired3



CUDA_VISIBLE_DEVICES=3 python analyze_gigawa_pt_qsa.py \
  --config /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment/config/analysis.yaml \
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/online_rl_425_cup_pick/robotwin_train_giga_world_policy/checkpoints/global_step_5000/actor/gigawa_components/replay_buffer/rank_0/trajectory_4999_cc920c6d-71ca-5a14-9155-db0fecdeb1b8.pt \
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/online_rl_425_cup_pick/robotwin_train_giga_world_policy/checkpoints/global_step_5000/actor/gigawa_components/replay_buffer/rank_0/trajectory_4993_5f777bbb-7da3-5139-a443-bb9f6d81d830.pt\
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/online_rl_425_cup_pick/robotwin_train_giga_world_policy/checkpoints/global_step_5000/actor/gigawa_components/replay_buffer/rank_0/trajectory_4981_0dd6a234-530b-5dc4-95a7-70c3a592d9a6.pt \
  --pt /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/online_rl_425_cup_pick/robotwin_train_giga_world_policy/checkpoints/global_step_5000/actor/gigawa_components/replay_buffer/rank_0/trajectory_4979_b913acad-78da-5e31-a86d-9329e7d3080e.pt \
  --checkpoint /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/online_rl_425_cup_pick/robotwin_train_giga_world_policy/checkpoints/global_step_7800 \
  --output-dir /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/qsa_debug_rl_429_online_cup_step_0.9836_4 \
  --device cuda

sudo chmod -R a+rwX /home/ubuntu/users/angen.ye/gwp/RLinf

sudo docker run -it \
  --name gwp_piper \
  -v /home/ubuntu/users/angen.ye:/home/ubuntu/users/angen.ye \
  -w /home/ubuntu/users/angen.ye \
  --entrypoint /bin/bash \
  giga-rlinf:gwp_piper

#在宿主机的终端中
export DISPLAY=:1
xhost +local:
#启动docker容器
 sudo docker run -it --gpus all \
    --privileged \
    --network host \
    --shm-size="24g" \
    -v /home/ubuntu/users/angen.ye:/home/ubuntu/users/angen.ye \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/input:/dev/input \
    --device /dev/uinput \
    --name piper \
    giga-rlinf:gwp_piper /bin/bash
  sudo docker start -ai gwp_piper

  sudo docker exec -it piper /bin/bash



 cd ~/cobot_magic/Piper_ros_private-ros-noetic-interrupt/
 bash can_config-4arms.sh

 export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
 echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc

 export RLINF_SKIP_ROS_CLEANUP=1
source /opt/ros/noetic/setup.bash
source /opt/venv/piper_ws/setup_piper_ros.sh