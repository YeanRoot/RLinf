cd /home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment
python collect_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name collect_piper_gigawa_realworld_success50_failure50

python collect_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name collect_piper_gigawa_realworld_actor_takeover_test


train

纯BC
python train_embodied_agent_gigawa_offline_bc.py \
--config-path ./config \
--config-name offline_piper_actor_bc_warmup

TD3+BC
python train_embodied_agent_gigawa_offline_rl_fast.py \
  --config-path ./config \
  --config-name offline_rl_pretrain

离线compare
cd /home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment
python compare_wa_actor_on_pt.py \
  --config-path ./config \
  --config-name offline_piper_actor_bc_warmup \
  --pt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/collect_piper_gigawa_intervention100/offline_collection/rank_0/all \
  --actor-ckpt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_kwj/piper_lerobot_actor_bc_warmup/checkpoints/global_step_400 \
  --out /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/action_compare_intervention_all \
  --batch-size 8
重新计算pt norm_state
cd /home/ubuntu/users/angen.ye/gwp/rollout1
python compute_norm_stats_from_pt.py

cd /home/ubuntu/users/angen.ye/gwp/RLinf
export PYTHONPATH=$PWD:$PYTHONPATH

cd /home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment
python train_embodied_agent_gigawa.py   --config-path ./config   --config-name online_rl_piper_gigawa

python analyze_realworld_pt.py \
  /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/collect_piper_gigawa_wa_only/offline_collection_raw/rank_0/all \
  --action-dim 14

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




python analyze_gigawa_pt_qsa.py \
  --config /home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment/config/analysis.yaml \
  --pt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_warmup/piper_lerobot_actor_bc_warmup/checkpoints/global_step_200/actor/gigawa_components/demo_buffer/rank_0/trajectory_0_lerobot_piper_bc.pt \
  --pt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_warmup/piper_lerobot_actor_bc_warmup/checkpoints/global_step_200/actor/gigawa_components/demo_buffer/rank_0/trajectory_1_lerobot_piper_bc.pt \
  --pt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_warmup/piper_lerobot_actor_bc_warmup/checkpoints/global_step_200/actor/gigawa_components/demo_buffer/rank_0/trajectory_3_lerobot_piper_bc.pt \
  --pt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_warmup/piper_lerobot_actor_bc_warmup/checkpoints/global_step_200/actor/gigawa_components/demo_buffer/rank_0/trajectory_7_lerobot_piper_bc.pt \
  --checkpoint /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_warmup/piper_lerobot_actor_bc_warmup/checkpoints/global_step_200 \
  --output-dir /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_warmup/piper_lerobot_actor_bc_warmup/checkpoints/global_step_200 \
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

启动docker
docker start -ai piper
source switch_env gigaworld

重新docker
docker exec -it piper /bin/bash
source switch_env gigaworld

容器外设置波特率
cd ~/cobot_magic/Piper_ros_private-ros-noetic-interrupt/
bash can_config-4arms.sh

export DISPLAY=:1
xhost +local:

gpu报错用这个
 export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
 echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc


启动ros节点
export RLINF_SKIP_ROS_CLEANUP=1
source /opt/ros/noetic/setup.bash
source /opt/venv/piper_ws/setup_piper_ros.sh

上使能
roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch auto_enable:=1

启动相机
roslaunch realsense2_camera multi_camera.launch

git端口
mkdir -p ~/.ssh

cat > ~/.ssh/config <<'EOF'
Host github.com
  Hostname ssh.github.com
  Port 443
  User git
EOF

chmod 600 ~/.ssh/config

按 c：
  reward = +1
  done / terminated = True
  表示成功

按 a：
  reward = -1
  done / terminated = True
  表示失败

按 b：
  reward = 0
  done / terminated = False
  表示中性标记，不结束

清理gpu
apt-get install -y psmisc
fuser -k -9 /dev/nvidia*

清理ros节点
rosnode kill -a
rostopic list 


python convert_lerobot_piper_to_gigawa_buffer.py \
  --project-root /home/ubuntu/users/angen.ye/gwp/RLinf \
  --dataset-root /home/ubuntu/users/angen.ye/gwp/260423160824_7c5a \
  --output-root /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/lerobot_actor_warmup_buffer \
  --config-path ./config \
  --config-name online_rl_piper_gigawa \
  --chunk-size 12 \
  --stride 12 \
  --use-episode-t5 \
  --skip-missing \
  --video-backend ffmpeg


python train_embodied_agent_gigawa_offline_bc.py \
--config-path ./config \
--config-name offline_piper_actor_bc_warmup





cd /home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment

export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

python collect_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name collect_piper_gigawa_realworld \
  runner.resume_dir=/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_warmup/piper_lerobot_actor_bc_warmup/checkpoints/global_step_200 \
  runner.resume_load_optimizer_and_scheduler_state=false \
  runner.logger.log_path=/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/collect_piper_actor_warmup_test \
  runner.logger.experiment_name=collect_piper_actor_warmup_test \
  actor.model.giga_world_policy.use_rl_head_for_rollout=true \
  algorithm.warmup_steps=0 \
  algorithm.rollout_actor_after_warmup=true \
  algorithm.rollout_actor_min_actor_updates=0 \
  algorithm.offline_collection.target_num_trajectories=5 \
  env.train.break_chunk_on_intervention=true \
  env.train.latch_intervention_until_chunk_end=false \
  env.train.collect_intervention_until_release=true \
  env.train.replan_on_intervention_release=true \
  env.train.pad_interrupted_chunks=true \
  env.train.force_disable_teleop_on_chunk_end=false \
  env.train.force_disable_teleop_on_terminal=true \
  env.train.force_disable_teleop_on_timeout=true \
  env.train.debug_intervention_chunks=true



cd /home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment

python compare_wa_actor_on_pt.py \
  --config-path ./config \
  --config-name offline_piper_actor_bc_warmup \
  --pt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/collect_piper_gigawa_intervention100/offline_collection/rank_0/success \
  --actor-ckpt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc_kwj/piper_lerobot_actor_bc_warmup/checkpoints/global_step_40 \
  --out /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/action_compare_success \
  --batch-size 8 \
  --max-files 10

#lerobotdataset check
python replay_episode_lerobot.py --episode 0 --dry-run
python replay_episode_lerobot.py --dataset-path /home/ubuntu/users/angen.ye/gwp/repaly/dianyuan/260524155410_8c85 --episode 10 --hz 30 --noise-reduce-after-step 120
#success seed
#wangxian 006
#jimu     025
#dianyuan 004 007 012 013 
#mukuai   014