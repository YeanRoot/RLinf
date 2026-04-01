# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedGigaWAFSDPPolicy(EmbodiedFSDPActor):
    """TD3-style off-policy worker for frozen Giga World Policy + small RL heads."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.replay_buffer = None
        self.demo_buffer = None
        self.update_step = 0
        self.rollout_rl_head_enabled = False

    # ---------------------------------------------------------------------
    # Init / setup
    # ---------------------------------------------------------------------
    def init_worker(self):
        self.setup_model_and_optimizer()
        self.setup_gigawa_components()

        policy = self._unwrap_policy(self.model)
        policy.soft_update_targets(tau=1.0)
        policy.set_use_rl_head_for_rollout(False)

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")

    def setup_model_and_optimizer(self) -> None:
        module = self.model_provider_func()
        self.model = self._strategy.wrap_model(model=module, device_mesh=self._device_mesh)
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        param_filters = {"critic": ["critic"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        self.lr_scheduler = self.build_lr_scheduler(self.optimizer, self.cfg.actor.optim)
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        self.grad_scaler = self.build_grad_scaler(self.cfg.actor.fsdp_config.grad_scaler)

    def setup_gigawa_components(self):
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"gigawa_replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")

        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get("trajectory_format", "pt"),
        )

        self.demo_buffer = None
        min_demo_buffer_size = 0
        demo_cfg = self.cfg.algorithm.get("demo_buffer", None)
        if demo_cfg is not None and demo_cfg.get("enable", False):
            auto_save_path = demo_cfg.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"gigawa_demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=demo_cfg.enable_cache,
                cache_size=demo_cfg.cache_size,
                sample_window_size=demo_cfg.sample_window_size,
                auto_save=demo_cfg.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format=demo_cfg.get("trajectory_format", "pt"),
            )
            min_demo_buffer_size = demo_cfg.min_buffer_size
            if demo_cfg.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    demo_cfg.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        buffer_dataset_cls = (
            PreloadReplayBufferDataset
            if self.cfg.algorithm.replay_buffer.get("enable_preload", False)
            else ReplayBufferDataset
        )
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = int(self.cfg.algorithm.get("critic_actor_ratio", 2))
        self.discount = float(self.cfg.algorithm.gamma) ** int(
            self.cfg.actor.model.get("num_action_chunks", 1)
        )
        self.bc_coef = float(self.cfg.algorithm.get("bc_coef", 1.0))
        self.target_update_freq = int(self.cfg.algorithm.get("target_update_freq", 1))
        self.target_tau = float(self.cfg.algorithm.get("tau", 0.005))
        self.ref_action_dropout_p = float(self.cfg.algorithm.get("ref_action_dropout_p", 0.5))
        self.target_policy_noise = float(self.cfg.algorithm.get("target_policy_noise", 0.2))
        self.target_noise_clip = float(self.cfg.algorithm.get("target_noise_clip", 0.5))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _unwrap_policy(model):
        return model.module if hasattr(model, "module") else model

    def _maybe_enable_rollout_rl_head(self):
        if self.rollout_rl_head_enabled:
            return
        policy = self._unwrap_policy(self.model)
        policy.set_use_rl_head_for_rollout(True)
        self.rollout_rl_head_enabled = True

    def _slice_obs_at_step(self, obs: dict[str, Any], t: int) -> dict[str, Any]:
        step_obs = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                step_obs[key] = value[t]
            elif isinstance(value, list):
                if len(value) == 0:
                    step_obs[key] = value
                elif len(value) > t and isinstance(value[t], list):
                    step_obs[key] = value[t]
                else:
                    step_obs[key] = value
            else:
                step_obs[key] = value
        return step_obs

    @torch.no_grad()
    def _extract_step_features(self, obs_step: dict[str, Any]) -> dict[str, torch.Tensor]:
        policy = self._unwrap_policy(self.model)
        feat = policy.extract_frozen_backbone_batch(obs_step)
        return {
            "visual_latent": feat["visual_latent"].cpu().contiguous(),
            "robot_state": feat["robot_state"].cpu().contiguous(),
            "ref_action": feat["ref_action"].cpu().contiguous(),
        }

    @torch.no_grad()
    def _convert_trajectory_for_gigawa(self, traj: Trajectory) -> Trajectory:
        assert traj.curr_obs and traj.next_obs, "Trajectory must contain curr_obs and next_obs."
        traj_len = int(traj.actions.shape[0])

        curr_visual_latents = []
        curr_robot_states = []
        curr_ref_actions = []
        next_visual_latents = []
        next_robot_states = []
        next_ref_actions = []

        for t in range(traj_len):
            curr_step_obs = self._slice_obs_at_step(traj.curr_obs, t)
            next_step_obs = self._slice_obs_at_step(traj.next_obs, t)

            curr_feat = self._extract_step_features(curr_step_obs)
            next_feat = self._extract_step_features(next_step_obs)

            curr_visual_latents.append(curr_feat["visual_latent"])
            curr_robot_states.append(curr_feat["robot_state"])
            curr_ref_actions.append(curr_feat["ref_action"])
            next_visual_latents.append(next_feat["visual_latent"])
            next_robot_states.append(next_feat["robot_state"])
            next_ref_actions.append(next_feat["ref_action"])

        return Trajectory(
            max_episode_length=traj.max_episode_length,
            model_weights_id=traj.model_weights_id,
            actions=traj.actions.cpu().contiguous() if traj.actions is not None else None,
            intervene_flags=traj.intervene_flags.cpu().contiguous() if traj.intervene_flags is not None else None,
            rewards=traj.rewards.cpu().contiguous() if traj.rewards is not None else None,
            terminations=traj.terminations.cpu().contiguous() if traj.terminations is not None else None,
            truncations=traj.truncations.cpu().contiguous() if traj.truncations is not None else None,
            dones=traj.dones.cpu().contiguous() if traj.dones is not None else None,
            prev_logprobs=traj.prev_logprobs.cpu().contiguous() if traj.prev_logprobs is not None else None,
            prev_values=traj.prev_values.cpu().contiguous() if traj.prev_values is not None else None,
            versions=traj.versions.cpu().contiguous() if traj.versions is not None else None,
            forward_inputs=traj.forward_inputs,
            curr_obs={
                "visual_latent": torch.stack(curr_visual_latents, dim=0),
                "robot_state": torch.stack(curr_robot_states, dim=0),
                "ref_action": torch.stack(curr_ref_actions, dim=0),
            },
            next_obs={
                "visual_latent": torch.stack(next_visual_latents, dim=0),
                "robot_state": torch.stack(next_robot_states, dim=0),
                "ref_action": torch.stack(next_ref_actions, dim=0),
            },
        )

    # ---------------------------------------------------------------------
    # Replay ingestion
    # ---------------------------------------------------------------------
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        gigawa_list = [self._convert_trajectory_for_gigawa(traj) for traj in recv_list]
        self.replay_buffer.add_trajectories(gigawa_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in gigawa_list:
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)
            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    # ---------------------------------------------------------------------
    # Losses
    # ---------------------------------------------------------------------
    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        policy = self._unwrap_policy(self.model)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"].to(self.device, dtype=self.torch_dtype)
        rewards = batch["rewards"]
        terminations = batch["terminations"]

        rewards_for_bootstrap = rewards.sum(dim=-1, keepdim=True).to(self.torch_dtype)
        done_mask = terminations.any(dim=-1, keepdim=True).to(self.torch_dtype)

        with torch.no_grad():
            curr_visual_feat = self.model(
                forward_type=ForwardType.DEFAULT,
                mode="encode_visual",
                visual_latent=curr_obs["visual_latent"].to(self.device, dtype=self.torch_dtype),
            )
            _, curr_actor_aux = self.model(
                forward_type=ForwardType.DEFAULT,
                mode="actor",
                visual_feat=curr_visual_feat.detach(),
                robot_state=curr_obs["robot_state"].to(self.device, dtype=self.torch_dtype),
                ref_action=curr_obs["ref_action"].to(self.device, dtype=self.torch_dtype),
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            curr_rl_state = curr_actor_aux["rl_state"].detach()

            next_visual_feat = self.model(
                forward_type=ForwardType.DEFAULT,
                mode="encode_visual",
                visual_latent=next_obs["visual_latent"].to(self.device, dtype=self.torch_dtype),
            )
            next_actions, next_actor_aux = policy.target_actor_forward(
                visual_feat=next_visual_feat.detach(),
                robot_state=next_obs["robot_state"].to(self.device, dtype=self.torch_dtype),
                ref_action=next_obs["ref_action"].to(self.device, dtype=self.torch_dtype),
            )
            if self.target_policy_noise > 0.0:
                noise = torch.randn_like(next_actions) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = next_actions + noise
            next_rl_state = next_actor_aux["rl_state"]
            target_q1, target_q2 = policy.target_critic_forward(
                rl_state=next_rl_state,
                action=next_actions,
            )
            target_q = torch.minimum(target_q1, target_q2)
            target_q_values = rewards_for_bootstrap + (1.0 - done_mask) * self.discount * target_q

        q1, q2 = self.model(
            forward_type=ForwardType.DEFAULT,
            mode="critic",
            rl_state=curr_rl_state,
            action=actions,
            use_target=False,
        )
        target_q_values = target_q_values.to(dtype=q1.dtype)
        critic_loss = F.mse_loss(q1, target_q_values) + F.mse_loss(q2, target_q_values)
        return critic_loss, {
            "q1_data": q1.mean().item(),
            "q2_data": q2.mean().item(),
            "target_q": target_q_values.mean().item(),
        }

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        policy = self._unwrap_policy(self.model)
        curr_obs = batch["curr_obs"]
        ref_action = curr_obs["ref_action"].to(self.device, dtype=self.torch_dtype)

        visual_feat = self.model(
            forward_type=ForwardType.DEFAULT,
            mode="encode_visual",
            visual_latent=curr_obs["visual_latent"].to(self.device, dtype=self.torch_dtype),
        )
        pi, actor_aux = self.model(
            forward_type=ForwardType.DEFAULT,
            mode="actor",
            visual_feat=visual_feat,
            robot_state=curr_obs["robot_state"].to(self.device, dtype=self.torch_dtype),
            ref_action=ref_action,
            ref_action_dropout_p=self.ref_action_dropout_p,
            use_target=False,
        )
        rl_state = actor_aux["rl_state"]
        q1_pi, q2_pi = self.model(
            forward_type=ForwardType.DEFAULT,
            mode="critic",
            rl_state=rl_state,
            action=pi,
            use_target=False,
        )
        q_pi = torch.minimum(q1_pi, q2_pi)

        bc_loss = policy.compute_bc_loss(pi, ref_action)
        actor_loss = (-q_pi).mean() + self.bc_coef * bc_loss
        return actor_loss, {
            "q_pi": q_pi.mean().item(),
            "bc_loss": bc_loss.item(),
            "ref_action_dropout_p": self.ref_action_dropout_p,
        }

    # ---------------------------------------------------------------------
    # Update loop
    # ---------------------------------------------------------------------
    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        train_micro_batch_list = [
            put_tensor_device(batch, device=self.device) for batch in train_micro_batch_list
        ]

        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{k}": np.mean(v) for k, v in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )
        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "gigawa/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        actor_updated = False
        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, actor_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                append_to_dict(all_actor_metrics, actor_metrics)
            all_actor_metrics = {f"actor/{k}": np.mean(v) for k, v in all_actor_metrics.items()}
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            actor_updated = True
            metrics_data.update(
                {
                    "gigawa/actor_loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    **all_actor_metrics,
                }
            )

        if actor_updated and self.update_step % self.target_update_freq == 0:
            policy = self._unwrap_policy(self.model)
            policy.soft_update_targets(tau=self.target_tau)
            self._maybe_enable_rollout_rl_head()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)

        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    @Worker.timer("run_training")
    def run_training(self):
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard" if self.cfg.actor.fsdp_config.use_orig_params else "dcp",
        )

        buffer_save_path = os.path.join(
            save_base_path, f"gigawa_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

        if self.demo_buffer is not None:
            demo_buffer_save_path = os.path.join(
                save_base_path, f"gigawa_components/demo_buffer/rank_{self._rank}"
            )
            self.demo_buffer.save_checkpoint(demo_buffer_save_path)

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard" if self.cfg.actor.fsdp_config.use_orig_params else "dcp",
        )

        buffer_load_path = os.path.join(
            load_base_path, f"gigawa_components/replay_buffer/rank_{self._rank}"
        )
        if os.path.isdir(buffer_load_path):
            self.replay_buffer.load_checkpoint(buffer_load_path)

        if self.demo_buffer is not None:
            demo_buffer_load_path = os.path.join(
                load_base_path, f"gigawa_components/demo_buffer/rank_{self._rank}"
            )
            if os.path.isdir(demo_buffer_load_path):
                self.demo_buffer.load_checkpoint(demo_buffer_load_path)

        policy = self._unwrap_policy(self.model)
        self.rollout_rl_head_enabled = policy.get_use_rl_head_for_rollout()
