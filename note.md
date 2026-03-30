source /mnt/pfs/users/angen.ye/myconda/conda/etc/profile.d/conda.sh
conda activate pi-rl
cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf

cd /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/embodiment
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_embodied_agent.py   --config-path ./config   --config-name robotwin_place_empty_cup_eval_giga_world_policy   env.eval.total_num_envs=4   algorithm.eval_rollout_epoch=5