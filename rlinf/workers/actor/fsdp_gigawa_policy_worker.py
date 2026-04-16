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

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        self.actor_update_step = 0
        self.rollout_rl_head_enabled = False
        self.offline_collection_enable = False
        self.offline_collection_buffers = {}
        self.offline_collection_output_dir = None
        self.offline_collection_summary_path = None
        self.offline_collection_success_threshold = 0.5
        self.offline_collection_quality_mode = "reward_max"
        self.offline_collection_summary_flush_every = 100
        self.offline_collection_summary_buffer = []
        self.offline_collection_num_received = 0
        self.offline_bc_pretrain_enable = False
        self.offline_bc_train_buffer = None
        self.offline_bc_val_buffer = None
        self.offline_bc_steps_per_epoch = 0
        self.offline_bc_val_steps = 0
        self.offline_bc_global_batch_per_rank = 0
        self.offline_critic_pretrain_enable = False
        self.offline_critic_train_buffer = None
        self.offline_critic_val_buffer = None
        self.offline_critic_train_success_buffer = None
        self.offline_critic_train_failure_buffer = None
        self.offline_critic_val_success_buffer = None
        self.offline_critic_val_failure_buffer = None
        self.offline_critic_steps_per_epoch = 0
        self.offline_critic_val_steps = 0
        self.offline_critic_eval_steps = 0
        self.offline_critic_global_batch_per_rank = 0
        self.offline_critic_quality_mode = "reward_max"
        self.offline_critic_success_threshold = 0.5
        self.offline_rl_pretrain_enable = False
        self.offline_rl_train_buffer = None
        self.offline_rl_val_buffer = None
        self.offline_rl_train_success_buffer = None
        self.offline_rl_train_failure_buffer = None
        self.offline_rl_val_success_buffer = None
        self.offline_rl_val_failure_buffer = None
        self.offline_rl_steps_per_epoch = 0
        self.offline_rl_val_steps = 0
        self.offline_rl_eval_steps = 0
        self.offline_rl_global_batch_per_rank = 0
        self.offline_rl_quality_mode = "reward_max"
        self.offline_rl_success_threshold = 0.5
        self.offline_rl_actor_updates_per_step = 1
        self.offline_rl_critic_updates_per_step = 1

    # ---------------------------------------------------------------------
    # Init / setup
    # ---------------------------------------------------------------------
    def _apply_rollout_flag_from_config(self, context: str) -> bool:
        rollout_flag = bool(
            self.cfg.actor.model.giga_world_policy.get("use_rl_head_for_rollout", False)
        )
        policy = self._unwrap_policy(self.model)
        policy.set_use_rl_head_for_rollout(rollout_flag)
        self.rollout_rl_head_enabled = rollout_flag
        self.log_on_first_rank(
            f"[{context}] Applied rollout flag from config: use_rl_head_for_rollout={rollout_flag}."
        )
        return rollout_flag

    def init_worker(self):
        self.setup_model_and_optimizer()
        self.setup_gigawa_components()

        policy = self._unwrap_policy(self.model)
        policy.soft_update_targets(tau=1.0)
        self._apply_rollout_flag_from_config("init_worker")

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
        replay_cfg = self.cfg.algorithm.replay_buffer
        auto_save_path = replay_cfg.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"gigawa_replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")

        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=replay_cfg.enable_cache,
            cache_size=replay_cfg.cache_size,
            sample_window_size=replay_cfg.sample_window_size,
            auto_save=replay_cfg.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=replay_cfg.get("trajectory_format", "pt"),
        )

        self.store_online_replay = bool(replay_cfg.get("store_online_data", True))
        self.replay_min_buffer_size = int(replay_cfg.get("min_buffer_size", 0))
        self.replay_train_actor_steps = int(
            replay_cfg.get(
                "train_actor_steps",
                self.cfg.algorithm.get("train_actor_steps", 0),
            )
        )
        self.replay_required_for_training = bool(
            replay_cfg.get("require_for_training", True)
        )

        self.demo_buffer = None
        self.demo_min_buffer_size = 0
        self.demo_train_actor_steps = 0
        self.demo_sample_ratio = 0.0
        self.allow_demo_only_fallback = True
        self.allow_replay_only_fallback = True
        self.allow_train_on_demo_only = False
        self.store_online_demo_interventions = True

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
            self.demo_min_buffer_size = int(demo_cfg.get("min_buffer_size", 0))
            self.demo_train_actor_steps = int(
                demo_cfg.get(
                    "train_actor_steps",
                    self.cfg.algorithm.get("train_actor_steps", 0),
                )
            )
            self.demo_sample_ratio = float(demo_cfg.get("sample_ratio", 0.5))
            self.demo_sample_ratio = min(max(self.demo_sample_ratio, 0.0), 1.0)
            self.allow_demo_only_fallback = bool(
                demo_cfg.get("allow_demo_only_fallback", True)
            )
            self.allow_replay_only_fallback = bool(
                demo_cfg.get("allow_replay_only_fallback", True)
            )
            self.allow_train_on_demo_only = bool(
                demo_cfg.get("allow_train_on_demo_only", self.demo_sample_ratio >= 1.0)
            )
            self.store_online_demo_interventions = bool(
                demo_cfg.get("store_online_interventions", True)
            )
            if demo_cfg.get("load_path", None) is not None:
                demo_load_path = demo_cfg.load_path
                rank_shard_path = os.path.join(demo_load_path, f"rank_{self._rank}")
                if os.path.exists(os.path.join(demo_load_path, "metadata.json")):
                    # Single-shard checkpoint root.
                    self.demo_buffer.load_checkpoint(
                        demo_load_path,
                        is_distributed=False,
                    )
                elif os.path.exists(os.path.join(rank_shard_path, "metadata.json")):
                    # Resharded distributed layout: root/rank_i/metadata.json
                    self.demo_buffer.load_checkpoint(
                        rank_shard_path,
                        is_distributed=False,
                    )
                else:
                    # Fall back to original behavior for older layouts.
                    self.demo_buffer.load_checkpoint(
                        demo_load_path,
                        is_distributed=True,
                        local_rank=self._rank,
                        world_size=self._world_size,
                    )
        else:
            self.replay_required_for_training = True

        buffer_dataset_cls = (
            PreloadReplayBufferDataset
            if replay_cfg.get("enable_preload", False)
            else ReplayBufferDataset
        )
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.replay_min_buffer_size,
            min_demo_buffer_size=self.demo_min_buffer_size,
            demo_sample_ratio=self.demo_sample_ratio,
            allow_demo_only_fallback=self.allow_demo_only_fallback,
            allow_replay_only_fallback=self.allow_replay_only_fallback,
            prefetch_size=replay_cfg.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.target_replay_batch_size = int(getattr(self.buffer_dataset, "replay_batch_size_target", 0))
        self.target_demo_batch_size = int(getattr(self.buffer_dataset, "demo_batch_size_target", 0))
        self.training_uses_demo = self.demo_buffer is not None and self.target_demo_batch_size > 0
        self.training_uses_replay = self.target_replay_batch_size > 0

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
        self.warmup_steps = int(self.cfg.algorithm.get("warmup_steps", 0))
        self.rollout_actor_after_warmup = bool(
            self.cfg.algorithm.get("rollout_actor_after_warmup", True)
        )
        self.rollout_actor_min_actor_updates = int(
            self.cfg.algorithm.get("rollout_actor_min_actor_updates", 50)
        )
        self.utd_ratio = int(
            self.cfg.algorithm.get(
                "utd_ratio", self.cfg.algorithm.get("update_epoch", 1)
            )
        )

        self.training_stage = str(
            self.cfg.algorithm.get("training_stage", "full_rl")
        ).lower()
        valid_stages = {"bc_actor_pretrain", "critic_warmup", "full_rl"}
        if self.training_stage not in valid_stages:
            raise ValueError(
                f"Unsupported training_stage={self.training_stage}, expected one of {sorted(valid_stages)}"
            )

        self.stage_actor_bc_only = self.training_stage == "bc_actor_pretrain"
        self.stage_freeze_actor = self.training_stage == "critic_warmup"
        self.stage_full_rl = self.training_stage == "full_rl"
        self.allow_rollout_actor_handoff = not self.stage_freeze_actor

        sliding_cfg = self.cfg.algorithm.get("gigawa_sliding_window", None)
        self.sliding_window_enable = bool(
            sliding_cfg is not None and sliding_cfg.get("enable", False)
        )
        self.sliding_window_offset_stride = int(
            sliding_cfg.get("offset_stride", 1) if sliding_cfg is not None else 1
        )
        self.sliding_window_max_offsets = (
            int(sliding_cfg.get("max_offsets"))
            if sliding_cfg is not None and sliding_cfg.get("max_offsets", None) is not None
            else None
        )

        diag_cfg = self.cfg.algorithm.get("action_diagnostics", None)
        if diag_cfg is None:
            diag_cfg = {}
        self.action_diag_enable = bool(diag_cfg.get("enable", False))
        self.action_diag_every_actor_updates = int(diag_cfg.get("every_actor_updates", 50))
        self.action_diag_max_save_samples = int(diag_cfg.get("max_save_samples", 4))
        self.action_diag_visual_feat_plot_dims = int(diag_cfg.get("visual_feat_plot_dims", 256))
        self.action_diag_last_captured_actor_update = -1
        self.action_diag_dir = Path(
            diag_cfg.get(
                "output_dir",
                os.path.join(self.cfg.runner.logger.log_path, "action_diagnostics"),
            )
        )
        if self.action_diag_enable and self._rank == 0:
            self.action_diag_dir.mkdir(parents=True, exist_ok=True)

        self._setup_offline_collection(seed=seed)
        self._setup_offline_bc_pretrain(seed=seed)
        self._setup_offline_critic_pretrain(seed=seed)
        self._setup_offline_rl_pretrain(seed=seed)

    def _setup_offline_collection(self, seed: int) -> None:
        collect_cfg = self.cfg.algorithm.get("offline_collection", None)
        if collect_cfg is None:
            return
        self.offline_collection_enable = bool(collect_cfg.get("enable", False))
        if not self.offline_collection_enable:
            return

        base_dir = collect_cfg.get(
            "output_dir",
            os.path.join(self.cfg.runner.logger.log_path, "offline_collection"),
        )
        self.offline_collection_output_dir = Path(base_dir) / f"rank_{self._rank}"
        self.offline_collection_output_dir.mkdir(parents=True, exist_ok=True)
        self.offline_collection_quality_mode = str(collect_cfg.get("quality_mode", "reward_max")).lower()
        self.offline_collection_success_threshold = float(collect_cfg.get("success_threshold", 0.5))
        self.offline_collection_summary_flush_every = int(collect_cfg.get("summary_flush_every", 100))
        self.offline_collection_summary_path = self.offline_collection_output_dir / "trajectory_summaries.jsonl"

        buffer_common = {
            "seed": seed,
            "enable_cache": bool(collect_cfg.get("enable_cache", False)),
            "cache_size": int(collect_cfg.get("cache_size", 64)),
            "sample_window_size": int(collect_cfg.get("sample_window_size", 1024)),
            "auto_save": True,
            "trajectory_format": str(collect_cfg.get("trajectory_format", "pt")),
        }
        resume_existing = bool(collect_cfg.get("resume_existing", True))

        buffer_specs = {
            "all": bool(collect_cfg.get("save_all", True)),
            "success": bool(collect_cfg.get("save_success", True)),
            "failure": bool(collect_cfg.get("save_failure", True)),
        }
        for name, enabled in buffer_specs.items():
            if not enabled:
                continue
            save_dir = self.offline_collection_output_dir / name
            save_dir.mkdir(parents=True, exist_ok=True)
            buffer = TrajectoryReplayBuffer(
                auto_save_path=str(save_dir),
                **buffer_common,
            )
            metadata_path = save_dir / "metadata.json"
            if resume_existing and metadata_path.exists():
                buffer.load_checkpoint(str(save_dir), is_distributed=False)
            self.offline_collection_buffers[name] = buffer

    def _compute_offline_collection_quality(self, traj: Trajectory) -> dict:
        rewards = traj.rewards.float().cpu() if traj.rewards is not None else torch.zeros(1)
        terminations = traj.terminations.bool().cpu() if traj.terminations is not None else None
        dones = traj.dones.bool().cpu() if traj.dones is not None else None
        reward_sum = float(rewards.sum().item())
        reward_max = float(rewards.max().item()) if rewards.numel() > 0 else 0.0
        terminal_any = bool(terminations.any().item()) if terminations is not None else False
        done_any = bool(dones.any().item()) if dones is not None else False

        mode = self.offline_collection_quality_mode
        threshold = self.offline_collection_success_threshold
        if mode == "reward_sum":
            is_success = reward_sum >= threshold
        elif mode == "any_positive":
            is_success = reward_max > 0.0
        elif mode == "terminal_and_reward":
            is_success = terminal_any and reward_max >= threshold
        else:
            is_success = reward_max >= threshold

        num_samples = int(traj.actions.shape[0]) if traj.actions is not None else 0
        return {
            "reward_sum": reward_sum,
            "reward_max": reward_max,
            "terminal_any": terminal_any,
            "done_any": done_any,
            "is_success": is_success,
            "num_samples": num_samples,
        }

    def _flush_offline_collection_summary(self) -> None:
        if not self.offline_collection_enable or self.offline_collection_summary_path is None:
            return
        if len(self.offline_collection_summary_buffer) == 0:
            return
        with open(self.offline_collection_summary_path, "a", encoding="utf-8") as f:
            for row in self.offline_collection_summary_buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.offline_collection_summary_buffer.clear()

    def _store_offline_collection_trajectories(self, trajs: list[Trajectory]) -> None:
        if not self.offline_collection_enable or len(trajs) == 0:
            return

        grouped = {"all": [], "success": [], "failure": []}
        summary_rows = []
        for traj in trajs:
            quality = self._compute_offline_collection_quality(traj)
            self.offline_collection_num_received += 1
            row = {
                "collection_index": int(self.offline_collection_num_received),
                "rank": int(self._rank),
                "model_weights_id": str(traj.model_weights_id),
                **quality,
            }
            summary_rows.append(row)
            if "all" in self.offline_collection_buffers:
                grouped["all"].append(traj)
            key = "success" if quality["is_success"] else "failure"
            if key in self.offline_collection_buffers:
                grouped[key].append(traj)

        for key, group in grouped.items():
            if key in self.offline_collection_buffers and len(group) > 0:
                self.offline_collection_buffers[key].add_trajectories(group)

        self.offline_collection_summary_buffer.extend(summary_rows)
        if len(self.offline_collection_summary_buffer) >= self.offline_collection_summary_flush_every:
            self._flush_offline_collection_summary()

    def get_offline_collection_stats(self) -> dict[str, Any]:
        stats = {
            "enabled": self.offline_collection_enable,
            "quality_mode": self.offline_collection_quality_mode,
            "success_threshold": self.offline_collection_success_threshold,
        }
        for name, buffer in self.offline_collection_buffers.items():
            buf_stats = buffer.get_stats()
            stats[name] = {
                "num_trajectories": int(buf_stats.get("num_trajectories", 0)),
                "total_samples": int(buf_stats.get("total_samples", 0)),
            }
        return stats

    def finalize_offline_collection(self) -> dict[str, Any]:
        self._flush_offline_collection_summary()
        for buffer in self.offline_collection_buffers.values():
            buffer.close(wait=True)
        return self.get_offline_collection_stats()

    def _make_subset_buffer(
        self,
        source_buffer: TrajectoryReplayBuffer,
        selected_ids: list[int],
        seed: int,
    ) -> TrajectoryReplayBuffer:
        subset = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=True,
            cache_size=max(1, int(self.cfg.algorithm.demo_buffer.get("cache_size", 2000))),
            sample_window_size=0,
            auto_save=False,
            trajectory_format=source_buffer.trajectory_format,
        )
        subset._trajectory_id_list = list(selected_ids)
        subset._trajectory_index = {
            int(tid): dict(source_buffer._trajectory_index[int(tid)])
            for tid in subset._trajectory_id_list
        }
        subset._trajectory_file_path = {
            int(tid): source_buffer._trajectory_file_path[int(tid)]
            for tid in subset._trajectory_id_list
        }
        subset.size = len(subset._trajectory_id_list)
        subset._total_samples = sum(
            info.get("num_samples", 0) for info in subset._trajectory_index.values()
        )
        subset._trajectory_counter = max(subset._trajectory_id_list, default=-1) + 1
        subset._index_version += 1
        return subset

    def _setup_offline_bc_pretrain(self, seed: int) -> None:
        offline_bc_cfg = self.cfg.algorithm.get("offline_bc_pretrain", None)
        if offline_bc_cfg is None:
            return
        self.offline_bc_pretrain_enable = bool(offline_bc_cfg.get("enable", False))
        if not self.offline_bc_pretrain_enable:
            return
        if self.demo_buffer is None:
            raise RuntimeError("offline_bc_pretrain requires algorithm.demo_buffer.enable=true and a valid load_path")

        all_ids = list(self.demo_buffer._trajectory_id_list)
        if len(all_ids) < 2:
            raise RuntimeError(f"offline_bc_pretrain needs at least 2 trajectories, got {len(all_ids)}")

        split_seed = int(offline_bc_cfg.get("split_seed", seed))
        val_ratio = float(offline_bc_cfg.get("val_ratio", 0.2))
        val_ratio = min(max(val_ratio, 0.0), 0.9)
        gen = torch.Generator().manual_seed(split_seed + self._rank)
        perm = torch.randperm(len(all_ids), generator=gen).tolist()
        shuffled_ids = [all_ids[i] for i in perm]
        val_count = max(1, int(round(len(shuffled_ids) * val_ratio)))
        if len(shuffled_ids) - val_count < 1:
            val_count = max(1, len(shuffled_ids) - 1)
        train_ids = shuffled_ids[:-val_count]
        val_ids = shuffled_ids[-val_count:]
        self.offline_bc_train_buffer = self._make_subset_buffer(self.demo_buffer, train_ids, seed + 101)
        self.offline_bc_val_buffer = self._make_subset_buffer(self.demo_buffer, val_ids, seed + 202)

        self.offline_bc_global_batch_per_rank = self.cfg.actor.global_batch_size // self._world_size
        default_steps = max(1, self.offline_bc_train_buffer.total_samples // max(1, self.offline_bc_global_batch_per_rank))
        self.offline_bc_steps_per_epoch = int(offline_bc_cfg.get("steps_per_epoch", default_steps))
        default_val_steps = max(1, min(50, self.offline_bc_val_buffer.total_samples // max(1, self.offline_bc_global_batch_per_rank)))
        self.offline_bc_val_steps = int(offline_bc_cfg.get("val_steps_per_epoch", default_val_steps))

        self.log_on_first_rank(
            f"[offline_bc_pretrain] enabled | train_traj={self.offline_bc_train_buffer.size} | "
            f"val_traj={self.offline_bc_val_buffer.size} | train_samples={self.offline_bc_train_buffer.total_samples} | "
            f"val_samples={self.offline_bc_val_buffer.total_samples} | steps_per_epoch={self.offline_bc_steps_per_epoch} | "
            f"val_steps={self.offline_bc_val_steps}"
        )

    def _setup_offline_critic_pretrain(self, seed: int) -> None:
        offline_critic_cfg = self.cfg.algorithm.get("offline_critic_pretrain", None)
        if offline_critic_cfg is None:
            return
        self.offline_critic_pretrain_enable = bool(offline_critic_cfg.get("enable", False))
        if not self.offline_critic_pretrain_enable:
            return
        if self.demo_buffer is None:
            raise RuntimeError(
                "offline_critic_pretrain requires algorithm.demo_buffer.enable=true and a valid load_path"
            )

        all_ids = list(self.demo_buffer._trajectory_id_list)
        if len(all_ids) < 2:
            raise RuntimeError(
                f"offline_critic_pretrain needs at least 2 trajectories, got {len(all_ids)}"
            )

        split_seed = int(offline_critic_cfg.get("split_seed", seed))
        val_ratio = float(offline_critic_cfg.get("val_ratio", 0.2))
        val_ratio = min(max(val_ratio, 0.0), 0.9)
        self.offline_critic_quality_mode = str(
            offline_critic_cfg.get("quality_mode", self.offline_collection_quality_mode)
        ).lower()
        self.offline_critic_success_threshold = float(
            offline_critic_cfg.get(
                "success_threshold", self.offline_collection_success_threshold
            )
        )

        success_ids = []
        failure_ids = []
        for trajectory_id in all_ids:
            info = self.demo_buffer._trajectory_index[int(trajectory_id)]
            model_weights_id = str(info.get("model_weights_id", ""))
            traj = self.demo_buffer._load_trajectory(int(trajectory_id), model_weights_id)
            quality = self._compute_offline_collection_quality(traj)
            if quality["is_success"]:
                success_ids.append(int(trajectory_id))
            else:
                failure_ids.append(int(trajectory_id))

        def _split_ids(ids: list[int], rng_seed: int) -> tuple[list[int], list[int]]:
            if len(ids) == 0:
                return [], []
            gen = torch.Generator().manual_seed(rng_seed)
            perm = torch.randperm(len(ids), generator=gen).tolist()
            shuffled = [ids[i] for i in perm]
            val_count = int(round(len(shuffled) * val_ratio))
            if len(shuffled) >= 2:
                val_count = min(max(val_count, 1), len(shuffled) - 1)
            else:
                val_count = 0
            if val_count == 0:
                return shuffled, []
            return shuffled[:-val_count], shuffled[-val_count:]

        train_success_ids, val_success_ids = _split_ids(success_ids, split_seed + 11 + self._rank)
        train_failure_ids, val_failure_ids = _split_ids(failure_ids, split_seed + 29 + self._rank)
        train_ids = train_success_ids + train_failure_ids
        val_ids = val_success_ids + val_failure_ids
        if len(train_ids) < 1:
            raise RuntimeError("offline_critic_pretrain requires at least one training trajectory")
        if len(val_ids) < 1:
            raise RuntimeError("offline_critic_pretrain requires at least one validation trajectory")

        self.offline_critic_train_buffer = self._make_subset_buffer(
            self.demo_buffer, train_ids, seed + 301
        )
        self.offline_critic_val_buffer = self._make_subset_buffer(
            self.demo_buffer, val_ids, seed + 302
        )
        self.offline_critic_train_success_buffer = (
            self._make_subset_buffer(self.demo_buffer, train_success_ids, seed + 303)
            if len(train_success_ids) > 0
            else None
        )
        self.offline_critic_train_failure_buffer = (
            self._make_subset_buffer(self.demo_buffer, train_failure_ids, seed + 304)
            if len(train_failure_ids) > 0
            else None
        )
        self.offline_critic_val_success_buffer = (
            self._make_subset_buffer(self.demo_buffer, val_success_ids, seed + 305)
            if len(val_success_ids) > 0
            else None
        )
        self.offline_critic_val_failure_buffer = (
            self._make_subset_buffer(self.demo_buffer, val_failure_ids, seed + 306)
            if len(val_failure_ids) > 0
            else None
        )

        self.offline_critic_global_batch_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        default_steps = max(
            1,
            self.offline_critic_train_buffer.total_samples
            // max(1, self.offline_critic_global_batch_per_rank),
        )
        self.offline_critic_steps_per_epoch = int(
            offline_critic_cfg.get("steps_per_epoch", default_steps)
        )
        default_val_steps = max(
            1,
            min(
                50,
                self.offline_critic_val_buffer.total_samples
                // max(1, self.offline_critic_global_batch_per_rank),
            ),
        )
        self.offline_critic_val_steps = int(
            offline_critic_cfg.get("val_steps_per_epoch", default_val_steps)
        )
        self.offline_critic_eval_steps = int(
            offline_critic_cfg.get(
                "class_eval_steps_per_epoch",
                min(50, max(1, self.offline_critic_val_steps)),
            )
        )

        self.log_on_first_rank(
            f"[offline_critic_pretrain] enabled | train_traj={self.offline_critic_train_buffer.size} | "
            f"val_traj={self.offline_critic_val_buffer.size} | train_success={len(train_success_ids)} | "
            f"train_failure={len(train_failure_ids)} | val_success={len(val_success_ids)} | "
            f"val_failure={len(val_failure_ids)} | train_samples={self.offline_critic_train_buffer.total_samples} | "
            f"val_samples={self.offline_critic_val_buffer.total_samples} | steps_per_epoch={self.offline_critic_steps_per_epoch} | "
            f"val_steps={self.offline_critic_val_steps} | class_eval_steps={self.offline_critic_eval_steps}"
        )

    def _setup_offline_rl_pretrain(self, seed: int) -> None:
        offline_rl_cfg = self.cfg.algorithm.get("offline_rl_pretrain", None)
        if offline_rl_cfg is None:
            return
        self.offline_rl_pretrain_enable = bool(offline_rl_cfg.get("enable", False))
        if not self.offline_rl_pretrain_enable:
            return
        if self.demo_buffer is None:
            raise RuntimeError(
                "offline_rl_pretrain requires algorithm.demo_buffer.enable=true and a valid load_path"
            )

        all_ids = list(self.demo_buffer._trajectory_id_list)
        if len(all_ids) < 2:
            raise RuntimeError(
                f"offline_rl_pretrain needs at least 2 trajectories, got {len(all_ids)}"
            )

        split_seed = int(offline_rl_cfg.get("split_seed", seed))
        val_ratio = float(offline_rl_cfg.get("val_ratio", 0.2))
        val_ratio = min(max(val_ratio, 0.0), 0.9)
        self.offline_rl_quality_mode = str(
            offline_rl_cfg.get("quality_mode", self.offline_collection_quality_mode)
        ).lower()
        self.offline_rl_success_threshold = float(
            offline_rl_cfg.get(
                "success_threshold", self.offline_collection_success_threshold
            )
        )

        # Reuse quality function by temporarily swapping mode/threshold.
        orig_mode = self.offline_collection_quality_mode
        orig_thr = self.offline_collection_success_threshold
        self.offline_collection_quality_mode = self.offline_rl_quality_mode
        self.offline_collection_success_threshold = self.offline_rl_success_threshold
        success_ids = []
        failure_ids = []
        for trajectory_id in all_ids:
            info = self.demo_buffer._trajectory_index[int(trajectory_id)]
            model_weights_id = str(info.get("model_weights_id", ""))
            traj = self.demo_buffer._load_trajectory(int(trajectory_id), model_weights_id)
            quality = self._compute_offline_collection_quality(traj)
            if quality["is_success"]:
                success_ids.append(int(trajectory_id))
            else:
                failure_ids.append(int(trajectory_id))
        self.offline_collection_quality_mode = orig_mode
        self.offline_collection_success_threshold = orig_thr

        def _split_ids(ids: list[int], rng_seed: int) -> tuple[list[int], list[int]]:
            if len(ids) == 0:
                return [], []
            gen = torch.Generator().manual_seed(rng_seed)
            perm = torch.randperm(len(ids), generator=gen).tolist()
            shuffled = [ids[i] for i in perm]
            val_count = int(round(len(shuffled) * val_ratio))
            if len(shuffled) >= 2:
                val_count = min(max(val_count, 1), len(shuffled) - 1)
            else:
                val_count = 0
            if val_count == 0:
                return shuffled, []
            return shuffled[:-val_count], shuffled[-val_count:]

        train_success_ids, val_success_ids = _split_ids(success_ids, split_seed + 101 + self._rank)
        train_failure_ids, val_failure_ids = _split_ids(failure_ids, split_seed + 211 + self._rank)
        train_ids = train_success_ids + train_failure_ids
        val_ids = val_success_ids + val_failure_ids
        if len(train_ids) < 1:
            raise RuntimeError("offline_rl_pretrain requires at least one training trajectory")
        if len(val_ids) < 1:
            raise RuntimeError("offline_rl_pretrain requires at least one validation trajectory")

        self.offline_rl_train_buffer = self._make_subset_buffer(self.demo_buffer, train_ids, seed + 401)
        self.offline_rl_val_buffer = self._make_subset_buffer(self.demo_buffer, val_ids, seed + 402)
        self.offline_rl_train_success_buffer = (
            self._make_subset_buffer(self.demo_buffer, train_success_ids, seed + 403)
            if len(train_success_ids) > 0 else None
        )
        self.offline_rl_train_failure_buffer = (
            self._make_subset_buffer(self.demo_buffer, train_failure_ids, seed + 404)
            if len(train_failure_ids) > 0 else None
        )
        self.offline_rl_val_success_buffer = (
            self._make_subset_buffer(self.demo_buffer, val_success_ids, seed + 405)
            if len(val_success_ids) > 0 else None
        )
        self.offline_rl_val_failure_buffer = (
            self._make_subset_buffer(self.demo_buffer, val_failure_ids, seed + 406)
            if len(val_failure_ids) > 0 else None
        )

        self.offline_rl_global_batch_per_rank = self.cfg.actor.global_batch_size // self._world_size
        default_steps = max(1, self.offline_rl_train_buffer.total_samples // max(1, self.offline_rl_global_batch_per_rank))
        self.offline_rl_steps_per_epoch = int(offline_rl_cfg.get("steps_per_epoch", default_steps))
        default_val_steps = max(1, min(50, self.offline_rl_val_buffer.total_samples // max(1, self.offline_rl_global_batch_per_rank)))
        self.offline_rl_val_steps = int(offline_rl_cfg.get("val_steps_per_epoch", default_val_steps))
        self.offline_rl_eval_steps = int(
            offline_rl_cfg.get("class_eval_steps_per_epoch", min(50, max(1, self.offline_rl_val_steps)))
        )
        self.offline_rl_actor_updates_per_step = int(offline_rl_cfg.get("actor_updates_per_step", 1))
        self.offline_rl_critic_updates_per_step = int(
            offline_rl_cfg.get("critic_updates_per_step", max(1, int(self.cfg.algorithm.get("critic_actor_ratio", 1))))
        )

        self.log_on_first_rank(
            f"[offline_rl_pretrain] enabled | train_traj={self.offline_rl_train_buffer.size} | "
            f"val_traj={self.offline_rl_val_buffer.size} | train_success={len(train_success_ids)} | "
            f"train_failure={len(train_failure_ids)} | val_success={len(val_success_ids)} | "
            f"val_failure={len(val_failure_ids)} | train_samples={self.offline_rl_train_buffer.total_samples} | "
            f"val_samples={self.offline_rl_val_buffer.total_samples} | steps_per_epoch={self.offline_rl_steps_per_epoch} | "
            f"val_steps={self.offline_rl_val_steps} | class_eval_steps={self.offline_rl_eval_steps} | "
            f"critic_updates_per_step={self.offline_rl_critic_updates_per_step} | actor_updates_per_step={self.offline_rl_actor_updates_per_step}"
        )

    def init_offline_critic_worker(self):
        self.setup_model_and_optimizer()
        self.setup_gigawa_components()
        policy = self._unwrap_policy(self.model)
        policy.soft_update_targets(tau=1.0)
        policy.set_use_rl_head_for_rollout(False)
        self.rollout_rl_head_enabled = False
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")

    def init_offline_rl_worker(self):
        self.setup_model_and_optimizer()
        self.setup_gigawa_components()
        policy = self._unwrap_policy(self.model)
        policy.soft_update_targets(tau=1.0)
        policy.set_use_rl_head_for_rollout(False)
        self.rollout_rl_head_enabled = False
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")

    def init_offline_bc_worker(self):
        self.setup_model_and_optimizer()
        self.setup_gigawa_components()
        policy = self._unwrap_policy(self.model)
        policy.soft_update_targets(tau=1.0)
        policy.set_use_rl_head_for_rollout(False)
        self.rollout_rl_head_enabled = False
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")

    def _offline_sample_batch_from_buffer(
        self, buffer: TrajectoryReplayBuffer | None, global_batch_per_rank: int
    ) -> dict[str, torch.Tensor]:
        assert buffer is not None, "offline pretrain buffers are not initialized"
        batch = buffer.sample(max(1, global_batch_per_rank))
        return put_tensor_device(batch, device=self.device)

    def _offline_bc_sample_batch(self, train: bool = True) -> dict[str, torch.Tensor]:
        buffer = self.offline_bc_train_buffer if train else self.offline_bc_val_buffer
        return self._offline_sample_batch_from_buffer(
            buffer, self.offline_bc_global_batch_per_rank
        )

    def _offline_critic_sample_batch(self, train: bool = True) -> dict[str, torch.Tensor]:
        buffer = self.offline_critic_train_buffer if train else self.offline_critic_val_buffer
        return self._offline_sample_batch_from_buffer(
            buffer, self.offline_critic_global_batch_per_rank
        )

    def _offline_critic_microbatches(self, global_batch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return split_dict_to_chunk(
            global_batch,
            max(1, self.offline_critic_global_batch_per_rank // self.cfg.actor.micro_batch_size),
        )

    def _offline_rl_sample_batch(self, train: bool = True) -> dict[str, torch.Tensor]:
        buffer = self.offline_rl_train_buffer if train else self.offline_rl_val_buffer
        return self._offline_sample_batch_from_buffer(
            buffer, self.offline_rl_global_batch_per_rank
        )

    def _offline_rl_microbatches(self, global_batch: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return split_dict_to_chunk(
            global_batch,
            max(1, self.offline_rl_global_batch_per_rank // self.cfg.actor.micro_batch_size),
        )

    def _offline_critic_collect_buffer_metrics(
        self,
        buffer: TrajectoryReplayBuffer | None,
        num_steps: int,
        prefix: str,
    ) -> dict[str, float]:
        if buffer is None or buffer.size <= 0 or num_steps <= 0:
            return {
                f"{prefix}/count": 0.0,
                f"{prefix}/critic_loss": 0.0,
                f"{prefix}/q1_mean": 0.0,
                f"{prefix}/q2_mean": 0.0,
                f"{prefix}/q_logged_mean": 0.0,
                f"{prefix}/target_q_mean": 0.0,
            }

        local = {
            f"{prefix}/count": 0.0,
            f"{prefix}/critic_loss_sum": 0.0,
            f"{prefix}/q1_sum": 0.0,
            f"{prefix}/q2_sum": 0.0,
            f"{prefix}/q_logged_sum": 0.0,
            f"{prefix}/target_q_sum": 0.0,
        }
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                global_batch = self._offline_sample_batch_from_buffer(
                    buffer, self.offline_critic_global_batch_per_rank
                )
                for batch in self._offline_critic_microbatches(global_batch):
                    critic_loss, q1, q2, target_q_values = self._compute_critic_outputs(batch)
                    q_logged = torch.minimum(q1, q2)
                    local[f"{prefix}/count"] += float(q_logged.numel())
                    local[f"{prefix}/critic_loss_sum"] += float(critic_loss.item()) * float(q_logged.numel())
                    local[f"{prefix}/q1_sum"] += float(q1.sum().item())
                    local[f"{prefix}/q2_sum"] += float(q2.sum().item())
                    local[f"{prefix}/q_logged_sum"] += float(q_logged.sum().item())
                    local[f"{prefix}/target_q_sum"] += float(target_q_values.sum().item())
        if was_training:
            self.model.train()

        reduced = all_reduce_dict(local, op=torch.distributed.ReduceOp.SUM)
        count = max(reduced[f"{prefix}/count"], 1.0)
        return {
            f"{prefix}/count": float(reduced[f"{prefix}/count"]),
            f"{prefix}/critic_loss": float(reduced[f"{prefix}/critic_loss_sum"] / count),
            f"{prefix}/q1_mean": float(reduced[f"{prefix}/q1_sum"] / count),
            f"{prefix}/q2_mean": float(reduced[f"{prefix}/q2_sum"] / count),
            f"{prefix}/q_logged_mean": float(reduced[f"{prefix}/q_logged_sum"] / count),
            f"{prefix}/target_q_mean": float(reduced[f"{prefix}/target_q_sum"] / count),
        }

    def _offline_rl_collect_buffer_critic_metrics(
        self,
        buffer: TrajectoryReplayBuffer | None,
        num_steps: int,
        prefix: str,
    ) -> dict[str, float]:
        if buffer is None or buffer.size <= 0 or num_steps <= 0:
            return {
                f"{prefix}/count": 0.0,
                f"{prefix}/critic_loss": 0.0,
                f"{prefix}/q1_mean": 0.0,
                f"{prefix}/q2_mean": 0.0,
                f"{prefix}/q_logged_mean": 0.0,
                f"{prefix}/target_q_mean": 0.0,
            }

        local = {
            f"{prefix}/count": 0.0,
            f"{prefix}/critic_loss_sum": 0.0,
            f"{prefix}/q1_sum": 0.0,
            f"{prefix}/q2_sum": 0.0,
            f"{prefix}/q_logged_sum": 0.0,
            f"{prefix}/target_q_sum": 0.0,
        }
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                global_batch = self._offline_sample_batch_from_buffer(
                    buffer, self.offline_rl_global_batch_per_rank
                )
                for batch in self._offline_rl_microbatches(global_batch):
                    critic_loss, q1, q2, target_q_values = self._compute_critic_outputs(batch)
                    q_logged = torch.minimum(q1, q2)
                    local[f"{prefix}/count"] += float(q_logged.numel())
                    local[f"{prefix}/critic_loss_sum"] += float(critic_loss.item()) * float(q_logged.numel())
                    local[f"{prefix}/q1_sum"] += float(q1.sum().item())
                    local[f"{prefix}/q2_sum"] += float(q2.sum().item())
                    local[f"{prefix}/q_logged_sum"] += float(q_logged.sum().item())
                    local[f"{prefix}/target_q_sum"] += float(target_q_values.sum().item())
        if was_training:
            self.model.train()

        reduced = all_reduce_dict(local, op=torch.distributed.ReduceOp.SUM)
        count = max(reduced[f"{prefix}/count"], 1e-12)
        return {
            f"{prefix}/count": reduced[f"{prefix}/count"],
            f"{prefix}/critic_loss": reduced[f"{prefix}/critic_loss_sum"] / count,
            f"{prefix}/q1_mean": reduced[f"{prefix}/q1_sum"] / count,
            f"{prefix}/q2_mean": reduced[f"{prefix}/q2_sum"] / count,
            f"{prefix}/q_logged_mean": reduced[f"{prefix}/q_logged_sum"] / count,
            f"{prefix}/target_q_mean": reduced[f"{prefix}/target_q_sum"] / count,
        }

    def _offline_rl_collect_buffer_actor_metrics(
        self,
        buffer: TrajectoryReplayBuffer | None,
        num_steps: int,
        prefix: str,
    ) -> dict[str, float]:
        if buffer is None or buffer.size <= 0 or num_steps <= 0:
            return {
                f"{prefix}/count": 0.0,
                f"{prefix}/actor_loss": 0.0,
                f"{prefix}/bc_loss": 0.0,
                f"{prefix}/q_pi_mean": 0.0,
            }

        local = {
            f"{prefix}/count": 0.0,
            f"{prefix}/actor_loss_sum": 0.0,
            f"{prefix}/bc_loss_sum": 0.0,
            f"{prefix}/q_pi_sum": 0.0,
        }
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                global_batch = self._offline_sample_batch_from_buffer(
                    buffer, self.offline_rl_global_batch_per_rank
                )
                for batch in self._offline_rl_microbatches(global_batch):
                    actor_loss, actor_metrics, _ = self.forward_actor(batch, capture_diagnostics=False)
                    batch_count = float(batch["actions"].shape[0])
                    local[f"{prefix}/count"] += batch_count
                    local[f"{prefix}/actor_loss_sum"] += float(actor_loss.item()) * batch_count
                    local[f"{prefix}/bc_loss_sum"] += float(actor_metrics.get("bc_loss", 0.0)) * batch_count
                    local[f"{prefix}/q_pi_sum"] += float(actor_metrics.get("q_pi", 0.0)) * batch_count
        if was_training:
            self.model.train()

        reduced = all_reduce_dict(local, op=torch.distributed.ReduceOp.SUM)
        count = max(reduced[f"{prefix}/count"], 1e-12)
        return {
            f"{prefix}/count": reduced[f"{prefix}/count"],
            f"{prefix}/actor_loss": reduced[f"{prefix}/actor_loss_sum"] / count,
            f"{prefix}/bc_loss": reduced[f"{prefix}/bc_loss_sum"] / count,
            f"{prefix}/q_pi_mean": reduced[f"{prefix}/q_pi_sum"] / count,
        }

    @Worker.timer("run_offline_critic_epoch")
    def run_offline_critic_epoch(self):
        if not self.offline_critic_pretrain_enable:
            raise RuntimeError("offline_critic_pretrain is not enabled in config")
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

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
        train_losses = []
        grad_norms = []
        q1_means = []
        q2_means = []
        target_q_means = []
        q_logged_means = []
        for _ in range(self.offline_critic_steps_per_epoch):
            global_batch = self._offline_critic_sample_batch(train=True)
            train_micro_batch_list = self._offline_critic_microbatches(global_batch)
            self.qf_optimizer.zero_grad()
            step_losses = []
            step_q1 = []
            step_q2 = []
            step_target_q = []
            step_q_logged = []
            for batch in train_micro_batch_list:
                critic_loss, q1, q2, target_q_values = self._compute_critic_outputs(batch)
                (critic_loss / self.gradient_accumulation).backward()
                q_logged = torch.minimum(q1.detach(), q2.detach())
                step_losses.append(float(critic_loss.detach().item()))
                step_q1.append(float(q1.detach().mean().item()))
                step_q2.append(float(q2.detach().mean().item()))
                step_target_q.append(float(target_q_values.detach().mean().item()))
                step_q_logged.append(float(q_logged.mean().item()))
            critic_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.critic_optim.clip_grad
            )
            self.qf_optimizer.step()
            self.qf_lr_scheduler.step()
            self.update_step += 1
            train_losses.append(float(np.mean(step_losses)))
            grad_norms.append(float(critic_grad_norm))
            q1_means.append(float(np.mean(step_q1)))
            q2_means.append(float(np.mean(step_q2)))
            target_q_means.append(float(np.mean(step_target_q)))
            q_logged_means.append(float(np.mean(step_q_logged)))

        self.model.eval()
        val_losses = []
        val_q1_means = []
        val_q2_means = []
        val_target_q_means = []
        val_q_logged_means = []
        with torch.no_grad():
            for _ in range(self.offline_critic_val_steps):
                global_batch = self._offline_critic_sample_batch(train=False)
                for batch in self._offline_critic_microbatches(global_batch):
                    critic_loss, q1, q2, target_q_values = self._compute_critic_outputs(batch)
                    q_logged = torch.minimum(q1, q2)
                    val_losses.append(float(critic_loss.item()))
                    val_q1_means.append(float(q1.mean().item()))
                    val_q2_means.append(float(q2.mean().item()))
                    val_target_q_means.append(float(target_q_values.mean().item()))
                    val_q_logged_means.append(float(q_logged.mean().item()))

        metrics = {
            "offline_critic/train_critic_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "offline_critic/val_critic_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "offline_critic/overfit_gap": (
                float(np.mean(val_losses)) - float(np.mean(train_losses))
            ) if train_losses and val_losses else 0.0,
            "offline_critic/train_q1_mean": float(np.mean(q1_means)) if q1_means else 0.0,
            "offline_critic/train_q2_mean": float(np.mean(q2_means)) if q2_means else 0.0,
            "offline_critic/train_q_logged_mean": float(np.mean(q_logged_means)) if q_logged_means else 0.0,
            "offline_critic/train_target_q_mean": float(np.mean(target_q_means)) if target_q_means else 0.0,
            "offline_critic/val_q1_mean": float(np.mean(val_q1_means)) if val_q1_means else 0.0,
            "offline_critic/val_q2_mean": float(np.mean(val_q2_means)) if val_q2_means else 0.0,
            "offline_critic/val_q_logged_mean": float(np.mean(val_q_logged_means)) if val_q_logged_means else 0.0,
            "offline_critic/val_target_q_mean": float(np.mean(val_target_q_means)) if val_target_q_means else 0.0,
            "offline_critic/train_num_trajectories": float(self.offline_critic_train_buffer.size),
            "offline_critic/val_num_trajectories": float(self.offline_critic_val_buffer.size),
            "offline_critic/train_success_num_trajectories": float(self.offline_critic_train_success_buffer.size if self.offline_critic_train_success_buffer is not None else 0),
            "offline_critic/train_failure_num_trajectories": float(self.offline_critic_train_failure_buffer.size if self.offline_critic_train_failure_buffer is not None else 0),
            "offline_critic/val_success_num_trajectories": float(self.offline_critic_val_success_buffer.size if self.offline_critic_val_success_buffer is not None else 0),
            "offline_critic/val_failure_num_trajectories": float(self.offline_critic_val_failure_buffer.size if self.offline_critic_val_failure_buffer is not None else 0),
            "offline_critic/train_total_samples": float(self.offline_critic_train_buffer.total_samples),
            "offline_critic/val_total_samples": float(self.offline_critic_val_buffer.total_samples),
            "offline_critic/steps_per_epoch": float(self.offline_critic_steps_per_epoch),
            "offline_critic/val_steps": float(self.offline_critic_val_steps),
            "offline_critic/class_eval_steps": float(self.offline_critic_eval_steps),
            "critic/lr": float(self.qf_optimizer.param_groups[0]["lr"]),
            "critic/grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        }
        metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
        metrics.update(
            self._offline_critic_collect_buffer_metrics(
                self.offline_critic_train_success_buffer,
                self.offline_critic_eval_steps,
                "offline_critic/train_success",
            )
        )
        metrics.update(
            self._offline_critic_collect_buffer_metrics(
                self.offline_critic_train_failure_buffer,
                self.offline_critic_eval_steps,
                "offline_critic/train_failure",
            )
        )
        metrics.update(
            self._offline_critic_collect_buffer_metrics(
                self.offline_critic_val_success_buffer,
                self.offline_critic_eval_steps,
                "offline_critic/val_success",
            )
        )
        metrics.update(
            self._offline_critic_collect_buffer_metrics(
                self.offline_critic_val_failure_buffer,
                self.offline_critic_eval_steps,
                "offline_critic/val_failure",
            )
        )
        metrics["offline_critic/train_success_failure_q_gap"] = (
            metrics.get("offline_critic/train_success/q_logged_mean", 0.0)
            - metrics.get("offline_critic/train_failure/q_logged_mean", 0.0)
        )
        metrics["offline_critic/val_success_failure_q_gap"] = (
            metrics.get("offline_critic/val_success/q_logged_mean", 0.0)
            - metrics.get("offline_critic/val_failure/q_logged_mean", 0.0)
        )
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        return metrics

    @Worker.timer("run_offline_rl_epoch")
    def run_offline_rl_epoch(self):
        if not self.offline_rl_pretrain_enable:
            raise RuntimeError("offline_rl_pretrain is not enabled in config")
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

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
        critic_train_losses = []
        critic_grad_norms = []
        critic_q1_means = []
        critic_q2_means = []
        critic_target_q_means = []
        critic_q_logged_means = []
        actor_train_losses = []
        actor_bc_losses = []
        actor_q_pi_means = []
        actor_grad_norms = []

        policy = self._unwrap_policy(self.model)
        tau = float(self.target_tau)

        for _ in range(self.offline_rl_steps_per_epoch):
            self.update_step += 1
            for _ in range(self.offline_rl_critic_updates_per_step):
                global_batch = self._offline_rl_sample_batch(train=True)
                train_micro_batch_list = self._offline_rl_microbatches(global_batch)
                self.qf_optimizer.zero_grad()
                step_losses = []
                step_q1 = []
                step_q2 = []
                step_target_q = []
                step_q_logged = []
                for batch in train_micro_batch_list:
                    critic_loss, q1, q2, target_q_values = self._compute_critic_outputs(batch)
                    (critic_loss / self.gradient_accumulation).backward()
                    q_logged = torch.minimum(q1.detach(), q2.detach())
                    step_losses.append(float(critic_loss.detach().item()))
                    step_q1.append(float(q1.detach().mean().item()))
                    step_q2.append(float(q2.detach().mean().item()))
                    step_target_q.append(float(target_q_values.detach().mean().item()))
                    step_q_logged.append(float(q_logged.mean().item()))
                critic_grad_norm = self.model.clip_grad_norm_(max_norm=self.cfg.actor.critic_optim.clip_grad)
                self.qf_optimizer.step()
                self.qf_lr_scheduler.step()
                policy.soft_update_targets(tau=tau)
                critic_train_losses.append(float(np.mean(step_losses)))
                critic_grad_norms.append(float(critic_grad_norm))
                critic_q1_means.append(float(np.mean(step_q1)))
                critic_q2_means.append(float(np.mean(step_q2)))
                critic_target_q_means.append(float(np.mean(step_target_q)))
                critic_q_logged_means.append(float(np.mean(step_q_logged)))

            for _ in range(self.offline_rl_actor_updates_per_step):
                global_batch = self._offline_rl_sample_batch(train=True)
                train_micro_batch_list = self._offline_rl_microbatches(global_batch)
                self.optimizer.zero_grad()
                step_actor_losses = []
                step_bc_losses = []
                step_q_pi = []
                for batch in train_micro_batch_list:
                    actor_loss, actor_metrics, _ = self.forward_actor(batch, capture_diagnostics=False)
                    (actor_loss / self.gradient_accumulation).backward()
                    step_actor_losses.append(float(actor_loss.detach().item()))
                    step_bc_losses.append(float(actor_metrics.get("bc_loss", 0.0)))
                    step_q_pi.append(float(actor_metrics.get("q_pi", 0.0)))
                actor_grad_norm = self.model.clip_grad_norm_(max_norm=self.cfg.actor.optim.clip_grad)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.actor_update_step += 1
                policy.soft_update_targets(tau=tau)
                actor_train_losses.append(float(np.mean(step_actor_losses)))
                actor_bc_losses.append(float(np.mean(step_bc_losses)))
                actor_q_pi_means.append(float(np.mean(step_q_pi)))
                actor_grad_norms.append(float(actor_grad_norm))

        metrics = {
            "offline_rl/train_critic_loss": float(np.mean(critic_train_losses)) if critic_train_losses else 0.0,
            "offline_rl/train_q1_mean": float(np.mean(critic_q1_means)) if critic_q1_means else 0.0,
            "offline_rl/train_q2_mean": float(np.mean(critic_q2_means)) if critic_q2_means else 0.0,
            "offline_rl/train_q_logged_mean": float(np.mean(critic_q_logged_means)) if critic_q_logged_means else 0.0,
            "offline_rl/train_target_q_mean": float(np.mean(critic_target_q_means)) if critic_target_q_means else 0.0,
            "offline_rl/train_actor_loss": float(np.mean(actor_train_losses)) if actor_train_losses else 0.0,
            "offline_rl/train_bc_loss": float(np.mean(actor_bc_losses)) if actor_bc_losses else 0.0,
            "offline_rl/train_q_pi_mean": float(np.mean(actor_q_pi_means)) if actor_q_pi_means else 0.0,
            "offline_rl/critic_grad_norm": float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0,
            "offline_rl/actor_grad_norm": float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
            "actor/lr": float(self.lr_scheduler.get_last_lr()[0]),
            "critic/lr": float(self.qf_lr_scheduler.get_last_lr()[0]),
        }

        metrics.update(
            self._offline_rl_collect_buffer_critic_metrics(
                self.offline_rl_val_buffer, self.offline_rl_val_steps, "offline_rl/val"
            )
        )
        metrics.update(
            self._offline_rl_collect_buffer_actor_metrics(
                self.offline_rl_val_buffer, self.offline_rl_val_steps, "offline_rl/val_actor"
            )
        )
        metrics.update(
            self._offline_rl_collect_buffer_critic_metrics(
                self.offline_rl_train_success_buffer,
                self.offline_rl_eval_steps,
                "offline_rl/train_success",
            )
        )
        metrics.update(
            self._offline_rl_collect_buffer_critic_metrics(
                self.offline_rl_train_failure_buffer,
                self.offline_rl_eval_steps,
                "offline_rl/train_failure",
            )
        )
        metrics.update(
            self._offline_rl_collect_buffer_critic_metrics(
                self.offline_rl_val_success_buffer,
                self.offline_rl_eval_steps,
                "offline_rl/val_success",
            )
        )
        metrics.update(
            self._offline_rl_collect_buffer_critic_metrics(
                self.offline_rl_val_failure_buffer,
                self.offline_rl_eval_steps,
                "offline_rl/val_failure",
            )
        )
        metrics["offline_rl/critic_overfit_gap"] = (
            metrics.get("offline_rl/val/critic_loss", 0.0)
            - metrics.get("offline_rl/train_critic_loss", 0.0)
        )
        metrics["offline_rl/train_success_failure_q_gap"] = (
            metrics.get("offline_rl/train_success/q_logged_mean", 0.0)
            - metrics.get("offline_rl/train_failure/q_logged_mean", 0.0)
        )
        metrics["offline_rl/val_success_failure_q_gap"] = (
            metrics.get("offline_rl/val_success/q_logged_mean", 0.0)
            - metrics.get("offline_rl/val_failure/q_logged_mean", 0.0)
        )
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        return metrics

    @Worker.timer("run_offline_bc_epoch")
    def run_offline_bc_epoch(self):
        if not self.offline_bc_pretrain_enable:
            raise RuntimeError("offline_bc_pretrain is not enabled in config")
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

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
        train_losses = []
        grad_norms = []
        for _ in range(self.offline_bc_steps_per_epoch):
            global_batch = self._offline_bc_sample_batch(train=True)
            train_micro_batch_list = split_dict_to_chunk(
                global_batch,
                max(1, self.offline_bc_global_batch_per_rank // self.cfg.actor.micro_batch_size),
            )
            self.optimizer.zero_grad()
            step_losses = []
            for batch in train_micro_batch_list:
                actor_loss, actor_metrics, _ = self.forward_actor(batch, capture_diagnostics=False)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                step_losses.append(float(actor_metrics["bc_loss"]))
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.actor_update_step += 1
            self.update_step += 1
            train_losses.append(float(np.mean(step_losses)))
            grad_norms.append(float(actor_grad_norm))

        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(self.offline_bc_val_steps):
                global_batch = self._offline_bc_sample_batch(train=False)
                val_micro_batch_list = split_dict_to_chunk(
                    global_batch,
                    max(1, self.offline_bc_global_batch_per_rank // self.cfg.actor.micro_batch_size),
                )
                for batch in val_micro_batch_list:
                    actor_loss, actor_metrics, _ = self.forward_actor(batch, capture_diagnostics=False)
                    val_losses.append(float(actor_metrics["bc_loss"]))

        metrics = {
            "offline_bc/train_bc_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "offline_bc/val_bc_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "offline_bc/overfit_gap": (float(np.mean(val_losses)) - float(np.mean(train_losses))) if train_losses and val_losses else 0.0,
            "offline_bc/train_num_trajectories": float(self.offline_bc_train_buffer.size),
            "offline_bc/val_num_trajectories": float(self.offline_bc_val_buffer.size),
            "offline_bc/train_total_samples": float(self.offline_bc_train_buffer.total_samples),
            "offline_bc/val_total_samples": float(self.offline_bc_val_buffer.total_samples),
            "offline_bc/steps_per_epoch": float(self.offline_bc_steps_per_epoch),
            "offline_bc/val_steps": float(self.offline_bc_val_steps),
            "actor/lr": float(self.optimizer.param_groups[0]["lr"]),
            "actor/grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        }
        metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        return metrics

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _unwrap_policy(model):
        return model.module if hasattr(model, "module") else model

    def _visual_mode(self) -> str:
        policy = self._unwrap_policy(self.model)
        return str(getattr(policy, "visual_input_mode", "normal")).lower()

    def _build_visual_feat_for_actor(self, visual_latent: torch.Tensor) -> torch.Tensor:
        policy = self._unwrap_policy(self.model)
        mode = self._visual_mode()
        if mode == "normal":
            return self.model(
                forward_type=ForwardType.DEFAULT,
                mode="encode_visual",
                visual_latent=visual_latent.to(self.device, dtype=self.torch_dtype),
            )

        batch_size = int(visual_latent.shape[0])
        return torch.zeros(
            batch_size,
            int(policy.visual_feature_dim),
            device=self.device,
            dtype=self.torch_dtype,
        )

    def _should_capture_action_diagnostics(self) -> bool:
        if not self.action_diag_enable:
            return False
        if self._rank != 0:
            return False
        interval = max(1, self.action_diag_every_actor_updates)
        upcoming_actor_update = self.actor_update_step + 1
        if upcoming_actor_update == self.action_diag_last_captured_actor_update:
            return False
        return (upcoming_actor_update % interval) == 0

    @staticmethod
    def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().cpu().numpy()

    def _save_action_diagnostics(self, debug_bundle: dict[str, Any]) -> dict[str, float]:
        policy = self._unwrap_policy(self.model)
        tag = debug_bundle["tag"]
        out_dir = self.action_diag_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_action = debug_bundle["ref_action"].detach().float()
        pred_action = debug_bundle["pred_action"].detach().float()
        robot_state = debug_bundle["robot_state"].detach().float()
        visual_feat = debug_bundle["visual_feat"].detach().float()
        visual_feat_for_state = debug_bundle["visual_feat_for_state"].detach().float()
        robot_state_for_state = debug_bundle["robot_state_for_state"].detach().float()
        ref_action_flat_for_state = debug_bundle["ref_action_flat_for_state"].detach().float()

        with torch.no_grad():
            ref_exec = policy.postprocess_action_model_batch(ref_action, robot_state)
            pred_exec = policy.postprocess_action_model_batch(pred_action, robot_state)

        model_diff = pred_action - ref_action
        exec_diff = pred_exec - ref_exec

        model_mse = float((model_diff.pow(2)).mean().item())
        exec_mse = float((exec_diff.pow(2)).mean().item())
        model_mae = float(model_diff.abs().mean().item())
        exec_mae = float(exec_diff.abs().mean().item())
        model_max_abs = float(model_diff.abs().max().item())
        exec_max_abs = float(exec_diff.abs().max().item())

        model_per_step_mse = model_diff.pow(2).mean(dim=(0, 2)).cpu().numpy()
        exec_per_step_mse = exec_diff.pow(2).mean(dim=(0, 2)).cpu().numpy()
        model_per_dim_mse = model_diff.pow(2).mean(dim=(0, 1)).cpu().numpy()
        exec_per_dim_mse = exec_diff.pow(2).mean(dim=(0, 1)).cpu().numpy()

        save_n = min(self.action_diag_max_save_samples, ref_action.shape[0])
        np.savez_compressed(
            out_dir / "action_diag_samples.npz",
            ref_action_model=self._tensor_to_numpy(ref_action[:save_n]),
            actor_action_model=self._tensor_to_numpy(pred_action[:save_n]),
            diff_action_model=self._tensor_to_numpy(model_diff[:save_n]),
            ref_action_exec=self._tensor_to_numpy(ref_exec[:save_n]),
            actor_action_exec=self._tensor_to_numpy(pred_exec[:save_n]),
            diff_action_exec=self._tensor_to_numpy(exec_diff[:save_n]),
            robot_state=self._tensor_to_numpy(robot_state[:save_n]),
            visual_feat=self._tensor_to_numpy(visual_feat[:save_n]),
            visual_feat_for_state=self._tensor_to_numpy(visual_feat_for_state[:save_n]),
            robot_state_for_state=self._tensor_to_numpy(robot_state_for_state[:save_n]),
            ref_action_flat_for_state=self._tensor_to_numpy(ref_action_flat_for_state[:save_n]),
            model_per_step_mse=model_per_step_mse,
            exec_per_step_mse=exec_per_step_mse,
            model_per_dim_mse=model_per_dim_mse,
            exec_per_dim_mse=exec_per_dim_mse,
        )

        summary = {
            "tag": tag,
            "update_step": int(debug_bundle["update_step"]),
            "actor_update_step": int(debug_bundle["actor_update_step"]),
            "batch_size": int(ref_action.shape[0]),
            "save_n": int(save_n),
            "model_mse": model_mse,
            "exec_mse": exec_mse,
            "model_mae": model_mae,
            "exec_mae": exec_mae,
            "model_max_abs": model_max_abs,
            "exec_max_abs": exec_max_abs,
            "visual_feat_norm": float(visual_feat.norm(dim=-1).mean().item()),
            "robot_state_norm": float(robot_state.norm(dim=-1).mean().item()),
            "model_per_step_mse": model_per_step_mse.tolist(),
            "exec_per_step_mse": exec_per_step_mse.tolist(),
            "model_per_dim_mse": model_per_dim_mse.tolist(),
            "exec_per_dim_mse": exec_per_dim_mse.tolist(),
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        sample_idx = 0
        ref0 = self._tensor_to_numpy(ref_action[sample_idx])
        pred0 = self._tensor_to_numpy(pred_action[sample_idx])
        diff0 = self._tensor_to_numpy(model_diff[sample_idx])
        ref_exec0 = self._tensor_to_numpy(ref_exec[sample_idx])
        pred_exec0 = self._tensor_to_numpy(pred_exec[sample_idx])
        diff_exec0 = self._tensor_to_numpy(exec_diff[sample_idx])
        robot0 = self._tensor_to_numpy(robot_state[sample_idx])
        visual0 = self._tensor_to_numpy(visual_feat_for_state[sample_idx])
        plot_visual_dims = min(self.action_diag_visual_feat_plot_dims, visual0.shape[0])

        fig, axes = plt.subplots(3, 4, figsize=(24, 14))
        ax = axes.reshape(-1)
        ims = []
        ims.append(ax[0].imshow(ref0, aspect="auto")); ax[0].set_title("WA ref action (model)")
        ims.append(ax[1].imshow(pred0, aspect="auto")); ax[1].set_title("Actor action (model)")
        ims.append(ax[2].imshow(diff0, aspect="auto")); ax[2].set_title("Diff (model)")
        ims.append(ax[3].imshow(ref_exec0, aspect="auto")); ax[3].set_title("WA ref action (exec)")
        ims.append(ax[4].imshow(pred_exec0, aspect="auto")); ax[4].set_title("Actor action (exec)")
        ims.append(ax[5].imshow(diff_exec0, aspect="auto")); ax[5].set_title("Diff (exec)")
        for i in range(6):
            fig.colorbar(ims[i], ax=ax[i], fraction=0.046, pad=0.04)
            ax[i].set_xlabel("action dim")
            ax[i].set_ylabel("chunk step")

        ax[6].plot(model_per_step_mse, label="model")
        ax[6].plot(exec_per_step_mse, label="exec")
        ax[6].set_title("Per-step MSE")
        ax[6].set_xlabel("chunk step")
        ax[6].legend()

        ax[7].plot(model_per_dim_mse, label="model")
        ax[7].plot(exec_per_dim_mse, label="exec")
        ax[7].set_title("Per-dim MSE")
        ax[7].set_xlabel("action dim")
        ax[7].legend()

        ax[8].plot(robot0)
        ax[8].set_title("Actor input robot_state (sample0)")
        ax[8].set_xlabel("state dim")

        ax[9].plot(visual0[:plot_visual_dims])
        ax[9].set_title(f"Actor input visual feat first {plot_visual_dims} dims")
        ax[9].set_xlabel("visual dim")

        ax[10].hist(diff0.reshape(-1), bins=50)
        ax[10].set_title("Model diff histogram (sample0)")

        ax[11].axis("off")
        ax[11].text(0.0, 1.0,
            f"tag: {tag}\nmodel_mse: {model_mse:.6e}\nexec_mse: {exec_mse:.6e}\nmodel_mae: {model_mae:.6e}\nexec_mae: {exec_mae:.6e}\nmodel_max_abs: {model_max_abs:.6e}\nexec_max_abs: {exec_max_abs:.6e}\nvisual_feat_norm: {summary['visual_feat_norm']:.6e}\nrobot_state_norm: {summary['robot_state_norm']:.6e}",
            va="top", family="monospace")

        fig.tight_layout()
        fig.savefig(out_dir / "action_diag_overview.png", dpi=180)
        plt.close(fig)

        self.action_diag_last_captured_actor_update = int(debug_bundle["actor_update_step"])
        self.log_on_first_rank(
            f"[action_diagnostics] saved to {out_dir} | model_mse={model_mse:.6e} | exec_mse={exec_mse:.6e}"
        )

        return {
            "actor_debug/model_mse": model_mse,
            "actor_debug/exec_mse": exec_mse,
            "actor_debug/model_mae": model_mae,
            "actor_debug/exec_mae": exec_mae,
            "actor_debug/model_max_abs": model_max_abs,
            "actor_debug/exec_max_abs": exec_max_abs,
        }

    def _maybe_enable_rollout_rl_head(self):
        if self.rollout_rl_head_enabled:
            return
        if not self.allow_rollout_actor_handoff:
            return
        if not self.rollout_actor_after_warmup:
            return
        if len(self.replay_buffer) < self.warmup_steps:
            return
        if self.actor_update_step < self.rollout_actor_min_actor_updates:
            return
        policy = self._unwrap_policy(self.model)
        policy.set_use_rl_head_for_rollout(True)
        self.rollout_rl_head_enabled = True

    def _resolve_buffer_usage(self, replay_min_size: int, demo_min_size: int):
        replay_min_size = int(max(0, replay_min_size))
        demo_min_size = int(max(0, demo_min_size))
        wants_replay = self.training_uses_replay
        wants_demo = self.training_uses_demo
        replay_ready = wants_replay and self.replay_buffer.is_ready(replay_min_size)
        demo_ready = wants_demo and self.demo_buffer is not None and self.demo_buffer.is_ready(demo_min_size)

        use_replay = False
        use_demo = False
        if wants_replay and wants_demo:
            if replay_ready and demo_ready:
                use_replay = True
                use_demo = True
            elif demo_ready and self.allow_demo_only_fallback:
                use_demo = True
            elif replay_ready and self.allow_replay_only_fallback:
                use_replay = True
        elif wants_replay:
            use_replay = replay_ready
        elif wants_demo:
            use_demo = demo_ready

        return {
            "wants_replay": wants_replay,
            "wants_demo": wants_demo,
            "replay_ready": replay_ready,
            "demo_ready": demo_ready,
            "use_replay": use_replay,
            "use_demo": use_demo,
        }

    def _get_training_readiness(self):
        replay_start_size = max(self.replay_min_buffer_size, self.warmup_steps) if self.training_uses_replay else 0
        demo_start_size = self.demo_min_buffer_size if self.training_uses_demo else 0
        start_status = self._resolve_buffer_usage(
            replay_min_size=replay_start_size,
            demo_min_size=demo_start_size,
        )

        train_actor_steps_cfg = int(self.cfg.algorithm.get("train_actor_steps", 0))
        replay_actor_threshold = 0
        demo_actor_threshold = 0
        if self.training_uses_replay:
            replay_actor_threshold = max(
                replay_start_size,
                train_actor_steps_cfg,
                self.replay_train_actor_steps,
            )
        if self.training_uses_demo:
            demo_actor_threshold = max(
                demo_start_size,
                train_actor_steps_cfg,
                self.demo_train_actor_steps,
            )
        actor_status = self._resolve_buffer_usage(
            replay_min_size=replay_actor_threshold,
            demo_min_size=demo_actor_threshold,
        )

        can_train = start_status["use_replay"] or start_status["use_demo"]
        if not can_train and self.allow_train_on_demo_only and start_status["demo_ready"]:
            can_train = True
            start_status["use_demo"] = True

        train_actor = actor_status["use_replay"] or actor_status["use_demo"]
        if not train_actor and self.allow_train_on_demo_only and actor_status["demo_ready"]:
            train_actor = True
            actor_status["use_demo"] = True

        if self.replay_required_for_training and self.training_uses_replay and not start_status["replay_ready"]:
            can_train = False
            train_actor = False

        return {
            "can_train": can_train,
            "train_actor": train_actor,
            "start_status": start_status,
            "actor_status": actor_status,
            "replay_start_size": replay_start_size,
            "demo_start_size": demo_start_size,
            "replay_actor_threshold": replay_actor_threshold,
            "demo_actor_threshold": demo_actor_threshold,
        }

    def _maybe_update_targets(self, actor_updated: bool, critic_updated: bool):
        if self.update_step % self.target_update_freq != 0:
            return

        should_update = False
        tau = self.target_tau
        if self.stage_actor_bc_only:
            should_update = actor_updated
            tau = 1.0
        elif self.stage_freeze_actor:
            should_update = critic_updated
        else:
            should_update = actor_updated or critic_updated

        if not should_update:
            return

        policy = self._unwrap_policy(self.model)
        policy.soft_update_targets(tau=tau)
        self._maybe_enable_rollout_rl_head()

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

    def _get_source_actions(self, traj: Trajectory) -> torch.Tensor:
        if traj.forward_inputs and "model_action" in traj.forward_inputs:
            source_actions = traj.forward_inputs["model_action"]
        else:
            source_actions = traj.actions
        assert source_actions is not None, "Trajectory must contain actions or model_action."
        return source_actions

    def _has_chunk_step_obs_seq(self, traj: Trajectory) -> bool:
        return bool(
            traj.curr_obs
            and isinstance(traj.curr_obs, dict)
            and "_chunk_step_states_seq" in traj.curr_obs
        )

    def _flatten_chunk_obs_sequence(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        # seq_tensor: [traj_len, batch, chunk+1, ...] -> [traj_len*chunk+1, batch, ...]
        traj_len = int(seq_tensor.shape[0])
        chunk_plus_one = int(seq_tensor.shape[2])
        chunk = chunk_plus_one - 1
        frames = [seq_tensor[0, :, 0].contiguous()]
        for t in range(traj_len):
            current_seq = seq_tensor[t]
            for k in range(1, chunk + 1):
                frames.append(current_seq[:, k].contiguous())
        return torch.stack(frames, dim=0).contiguous()

    def _build_primitive_obs_dict(self, traj: Trajectory, traj_len: int | None = None) -> dict[str, torch.Tensor]:
        obs = {}
        key_map = {
            "_chunk_step_main_images_seq": "main_images",
            "_chunk_step_wrist_images_seq": "wrist_images",
            "_chunk_step_extra_view_images_seq": "extra_view_images",
            "_chunk_step_states_seq": "states",
        }
        for seq_key, obs_key in key_map.items():
            if seq_key in traj.curr_obs and isinstance(traj.curr_obs[seq_key], torch.Tensor):
                seq_tensor = traj.curr_obs[seq_key]
                if traj_len is not None:
                    seq_tensor = seq_tensor[:traj_len]
                obs[obs_key] = self._flatten_chunk_obs_sequence(seq_tensor)
        return obs

    def _get_effective_traj_len(self, traj: Trajectory, source_actions: torch.Tensor) -> int:
        candidate_lengths = [int(source_actions.shape[0])]

        for tensor in [traj.rewards, traj.terminations, traj.truncations, traj.dones, traj.prev_logprobs, traj.prev_values, traj.versions]:
            if isinstance(tensor, torch.Tensor):
                candidate_lengths.append(int(tensor.shape[0]))

        if isinstance(traj.curr_obs, dict):
            for value in traj.curr_obs.values():
                if isinstance(value, torch.Tensor):
                    candidate_lengths.append(int(value.shape[0]))
        if isinstance(traj.next_obs, dict):
            for value in traj.next_obs.values():
                if isinstance(value, torch.Tensor):
                    candidate_lengths.append(int(value.shape[0]))

        effective_len = min(candidate_lengths)
        if effective_len <= 0:
            raise RuntimeError(f"Invalid effective trajectory length: {effective_len}, candidates={candidate_lengths}")
        if effective_len != candidate_lengths[0]:
            self.log_on_first_rank(
                f"[gigawa_sliding_window] cropping inconsistent trajectory lengths to {effective_len}; candidates={candidate_lengths}"
            )
        return effective_len

    def _slice_obs_dict(self, obs: dict[str, Any], traj_len: int) -> dict[str, Any]:
        out = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[:traj_len]
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def _convert_standard_trajectory_for_gigawa(self, traj: Trajectory) -> Trajectory:
        assert traj.curr_obs and traj.next_obs, "Trajectory must contain curr_obs and next_obs."
        source_actions = self._get_source_actions(traj)
        traj_len = self._get_effective_traj_len(traj, source_actions)
        source_actions = source_actions[:traj_len]
        curr_obs = self._slice_obs_dict(traj.curr_obs, traj_len)
        next_obs = self._slice_obs_dict(traj.next_obs, traj_len)

        curr_visual_latents = []
        curr_robot_states = []
        curr_ref_actions = []
        next_visual_latents = []
        next_robot_states = []
        next_ref_actions = []

        for t in range(traj_len):
            curr_step_obs = self._slice_obs_at_step(curr_obs, t)
            next_step_obs = self._slice_obs_at_step(next_obs, t)

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
            actions=source_actions.cpu().contiguous(),
            intervene_flags=traj.intervene_flags[:traj_len].cpu().contiguous() if traj.intervene_flags is not None else None,
            rewards=traj.rewards[:traj_len].cpu().contiguous() if traj.rewards is not None else None,
            terminations=traj.terminations[:traj_len].cpu().contiguous() if traj.terminations is not None else None,
            truncations=traj.truncations[:traj_len].cpu().contiguous() if traj.truncations is not None else None,
            dones=traj.dones[:traj_len].cpu().contiguous() if traj.dones is not None else None,
            prev_logprobs=traj.prev_logprobs[:traj_len].cpu().contiguous() if traj.prev_logprobs is not None else None,
            prev_values=traj.prev_values[:traj_len].cpu().contiguous() if traj.prev_values is not None else None,
            versions=traj.versions[:traj_len].cpu().contiguous() if traj.versions is not None else None,
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

    def _pad_primitive_action_chunk(
        self,
        primitive_actions: torch.Tensor,
        start_idx: int,
        valid_len: int,
        action_chunk: int,
    ) -> torch.Tensor:
        valid_actions = primitive_actions[start_idx : start_idx + valid_len]  # [valid_len, B, A]
        assert valid_actions.shape[0] > 0, "valid_len must be positive"
        if valid_len < action_chunk:
            pad_count = action_chunk - valid_len
            pad_actions = valid_actions[-1:].expand(pad_count, -1, -1)
            valid_actions = torch.cat([valid_actions, pad_actions], dim=0)
        return valid_actions.permute(1, 0, 2).reshape(valid_actions.shape[1], -1).cpu().contiguous()

    def _pad_primitive_scalar_chunk(
        self,
        primitive_tensor: torch.Tensor,
        start_idx: int,
        valid_len: int,
        action_chunk: int,
        fill_value: int | float | bool = 0,
    ) -> torch.Tensor:
        valid_values = primitive_tensor[start_idx : start_idx + valid_len]  # [valid_len, B]
        assert valid_values.shape[0] > 0, "valid_len must be positive"
        if valid_len < action_chunk:
            pad_count = action_chunk - valid_len
            pad_values = torch.full(
                (pad_count, valid_values.shape[1]),
                fill_value=fill_value,
                dtype=valid_values.dtype,
                device=valid_values.device,
            )
            valid_values = torch.cat([valid_values, pad_values], dim=0)
        return valid_values.permute(1, 0).cpu().contiguous()

    @torch.no_grad()
    def _convert_sliding_window_trajectories_for_gigawa(self, traj: Trajectory) -> list[Trajectory]:
        source_actions = self._get_source_actions(traj).cpu().contiguous()
        assert traj.curr_obs and traj.next_obs, "Trajectory must contain curr_obs and next_obs."
        traj_len = self._get_effective_traj_len(traj, source_actions)
        source_actions = source_actions[:traj_len].cpu().contiguous()
        rewards = traj.rewards[:traj_len] if traj.rewards is not None else None
        terminations = traj.terminations[:traj_len] if traj.terminations is not None else None
        truncations = traj.truncations[:traj_len] if traj.truncations is not None else None
        dones = traj.dones[:traj_len] if traj.dones is not None else None
        prev_logprobs = traj.prev_logprobs[:traj_len] if traj.prev_logprobs is not None else None
        prev_values = traj.prev_values[:traj_len] if traj.prev_values is not None else None
        versions = traj.versions[:traj_len] if traj.versions is not None else None
        intervene_flags = traj.intervene_flags[:traj_len] if traj.intervene_flags is not None else None
        primitive_obs = self._build_primitive_obs_dict(traj, traj_len=traj_len)
        if "states" not in primitive_obs:
            return [self._convert_standard_trajectory_for_gigawa(traj)]

        traj_len, batch_size, action_flat_dim = source_actions.shape
        action_chunk = int(self.cfg.actor.model.get("num_action_chunks", 1))
        action_dim = action_flat_dim // action_chunk
        primitive_actions = source_actions.view(
            traj_len, batch_size, action_chunk, action_dim
        ).permute(0, 2, 1, 3).reshape(
            traj_len * action_chunk, batch_size, action_dim
        )
        primitive_rewards = rewards.permute(0, 2, 1).reshape(
            traj_len * action_chunk, batch_size
        ).cpu().contiguous() if rewards is not None else None
        primitive_terminations = terminations.permute(0, 2, 1).reshape(
            traj_len * action_chunk, batch_size
        ).to(torch.bool).cpu().contiguous() if terminations is not None else None
        primitive_truncations = truncations.permute(0, 2, 1).reshape(
            traj_len * action_chunk, batch_size
        ).to(torch.bool).cpu().contiguous() if truncations is not None else None
        primitive_dones = dones.permute(0, 2, 1).reshape(
            traj_len * action_chunk, batch_size
        ).to(torch.bool).cpu().contiguous() if dones is not None else None

        primitive_intervene = None
        if intervene_flags is not None:
            primitive_intervene = intervene_flags.view(
                traj_len, batch_size, action_chunk, action_dim
            ).permute(0, 2, 1, 3).reshape(
                traj_len * action_chunk, batch_size, action_dim
            ).to(torch.bool).cpu().contiguous()

        primitive_logprobs = None
        if prev_logprobs is not None:
            primitive_logprobs = prev_logprobs.permute(0, 2, 1).reshape(
                traj_len * action_chunk, batch_size
            ).cpu().contiguous()

        total_primitive_steps = int(primitive_actions.shape[0])
        max_offsets = action_chunk
        if self.sliding_window_max_offsets is not None:
            max_offsets = min(max_offsets, int(self.sliding_window_max_offsets))

        converted_trajectories = []
        for offset in range(0, max_offsets, max(1, self.sliding_window_offset_stride)):
            if offset >= total_primitive_steps:
                continue

            start_indices = list(range(offset, total_primitive_steps, action_chunk))
            if len(start_indices) == 0:
                continue

            curr_visual_latents = []
            curr_robot_states = []
            curr_ref_actions = []
            next_visual_latents = []
            next_robot_states = []
            next_ref_actions = []

            window_actions = []
            window_rewards = []
            window_terminations = []
            window_truncations = []
            window_dones = []
            window_intervene = [] if primitive_intervene is not None else None
            window_logprobs = [] if primitive_logprobs is not None else None
            source_chunk_indices = []

            for start_idx in start_indices:
                valid_len = min(action_chunk, total_primitive_steps - start_idx)
                if valid_len <= 0:
                    continue
                next_idx = start_idx + valid_len

                curr_step_obs = {key: value[start_idx] for key, value in primitive_obs.items()}
                next_step_obs = {key: value[next_idx] for key, value in primitive_obs.items()}
                curr_feat = self._extract_step_features(curr_step_obs)
                next_feat = self._extract_step_features(next_step_obs)
                curr_visual_latents.append(curr_feat["visual_latent"])
                curr_robot_states.append(curr_feat["robot_state"])
                curr_ref_actions.append(curr_feat["ref_action"])
                next_visual_latents.append(next_feat["visual_latent"])
                next_robot_states.append(next_feat["robot_state"])
                next_ref_actions.append(next_feat["ref_action"])

                window_actions.append(
                    self._pad_primitive_action_chunk(
                        primitive_actions, start_idx, valid_len, action_chunk
                    )
                )
                if primitive_rewards is not None:
                    window_rewards.append(
                        self._pad_primitive_scalar_chunk(
                            primitive_rewards, start_idx, valid_len, action_chunk, fill_value=0.0
                        )
                    )
                if primitive_terminations is not None:
                    window_terminations.append(
                        self._pad_primitive_scalar_chunk(
                            primitive_terminations, start_idx, valid_len, action_chunk, fill_value=False
                        )
                    )
                if primitive_truncations is not None:
                    window_truncations.append(
                        self._pad_primitive_scalar_chunk(
                            primitive_truncations, start_idx, valid_len, action_chunk, fill_value=False
                        )
                    )
                if primitive_dones is not None:
                    window_dones.append(
                        self._pad_primitive_scalar_chunk(
                            primitive_dones, start_idx, valid_len, action_chunk, fill_value=False
                        )
                    )
                if primitive_intervene is not None:
                    padded_intervene = primitive_intervene[start_idx : start_idx + valid_len]
                    if valid_len < action_chunk:
                        pad_count = action_chunk - valid_len
                        pad_intervene = torch.zeros(
                            (pad_count, padded_intervene.shape[1], padded_intervene.shape[2]),
                            dtype=padded_intervene.dtype,
                            device=padded_intervene.device,
                        )
                        padded_intervene = torch.cat([padded_intervene, pad_intervene], dim=0)
                    window_intervene.append(
                        padded_intervene.permute(1, 0, 2).reshape(batch_size, -1).cpu().contiguous()
                    )
                if primitive_logprobs is not None:
                    window_logprobs.append(
                        self._pad_primitive_scalar_chunk(
                            primitive_logprobs, start_idx, valid_len, action_chunk, fill_value=0.0
                        )
                    )
                source_chunk_indices.append(start_idx // action_chunk)

            if len(window_actions) == 0:
                continue

            source_chunk_index_tensor = torch.as_tensor(source_chunk_indices, dtype=torch.long)
            version_source = versions[source_chunk_index_tensor].cpu().contiguous() if versions is not None else None
            value_source = prev_values[source_chunk_index_tensor].cpu().contiguous() if prev_values is not None else None

            converted_trajectories.append(
                Trajectory(
                    max_episode_length=traj.max_episode_length,
                    model_weights_id=traj.model_weights_id,
                    actions=torch.stack(window_actions, dim=0),
                    intervene_flags=torch.stack(window_intervene, dim=0) if window_intervene is not None else None,
                    rewards=torch.stack(window_rewards, dim=0) if primitive_rewards is not None else None,
                    terminations=torch.stack(window_terminations, dim=0) if primitive_terminations is not None else None,
                    truncations=torch.stack(window_truncations, dim=0) if primitive_truncations is not None else None,
                    dones=torch.stack(window_dones, dim=0) if primitive_dones is not None else None,
                    prev_logprobs=torch.stack(window_logprobs, dim=0) if window_logprobs is not None else None,
                    prev_values=value_source,
                    versions=version_source,
                    forward_inputs={},
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
            )

        return converted_trajectories if converted_trajectories else [self._convert_standard_trajectory_for_gigawa(traj)]

    @torch.no_grad()
    def _convert_trajectory_for_gigawa(self, traj: Trajectory) -> list[Trajectory]:
        if self.sliding_window_enable and self._has_chunk_step_obs_seq(traj):
            return self._convert_sliding_window_trajectories_for_gigawa(traj)
        return [self._convert_standard_trajectory_for_gigawa(traj)]

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

        gigawa_list = []
        for traj in recv_list:
            gigawa_list.extend(self._convert_trajectory_for_gigawa(traj))
        if self.store_online_replay:
            self.replay_buffer.add_trajectories(gigawa_list)

        self._store_offline_collection_trajectories(gigawa_list)

        if self.demo_buffer is not None and self.store_online_demo_interventions:
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
    def _compute_critic_outputs(self, batch):
        policy = self._unwrap_policy(self.model)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"].to(self.device, dtype=self.torch_dtype)
        rewards = batch["rewards"]
        terminations = batch["terminations"]

        rewards_for_bootstrap = rewards.sum(dim=-1, keepdim=True).to(self.torch_dtype)
        done_mask = terminations.any(dim=-1, keepdim=True).to(self.torch_dtype)

        with torch.no_grad():
            curr_visual_feat = self._build_visual_feat_for_actor(curr_obs["visual_latent"])
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

            next_visual_feat = self._build_visual_feat_for_actor(next_obs["visual_latent"])
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
        return critic_loss, q1, q2, target_q_values

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        critic_loss, q1, q2, target_q_values = self._compute_critic_outputs(batch)
        return critic_loss, {
            "q1_data": q1.mean().item(),
            "q2_data": q2.mean().item(),
            "target_q": target_q_values.mean().item(),
        }

    @Worker.timer("forward_actor")
    def forward_actor(self, batch, capture_diagnostics: bool = False):
        policy = self._unwrap_policy(self.model)
        curr_obs = batch["curr_obs"]
        robot_state = curr_obs["robot_state"].to(self.device, dtype=self.torch_dtype)
        ref_action = curr_obs["ref_action"].to(self.device, dtype=self.torch_dtype)

        visual_feat = self._build_visual_feat_for_actor(
            curr_obs["visual_latent"]
        )
        pi, actor_aux = self.model(
            forward_type=ForwardType.DEFAULT,
            mode="actor",
            visual_feat=visual_feat,
            robot_state=robot_state,
            ref_action=ref_action,
            ref_action_dropout_p=self.ref_action_dropout_p,
            use_target=False,
        )

        bc_loss = policy.compute_bc_loss(pi, ref_action)

        metrics = {
            "bc_loss": bc_loss.item(),
            "ref_action_dropout_p": self.ref_action_dropout_p,
            "training_stage": float(
                0 if self.stage_actor_bc_only else 1 if self.stage_freeze_actor else 2
            ),
        }

        debug_bundle = None
        if capture_diagnostics:
            debug_bundle = {
                "tag": f"update_{self.update_step + 1:07d}_actor_{self.actor_update_step + 1:07d}",
                "update_step": self.update_step + 1,
                "actor_update_step": self.actor_update_step + 1,
                "ref_action": ref_action.detach(),
                "pred_action": pi.detach(),
                "robot_state": robot_state.detach(),
                "visual_feat": visual_feat.detach(),
                "visual_feat_for_state": actor_aux.get("visual_feat_for_state", visual_feat).detach(),
                "robot_state_for_state": actor_aux.get("robot_state_for_state", robot_state).detach(),
                "ref_action_flat_for_state": actor_aux.get(
                    "ref_action_flat_for_state", ref_action.reshape(ref_action.shape[0], -1)
                ).detach(),
            }

        if self.stage_actor_bc_only:
            actor_loss = self.bc_coef * bc_loss
            metrics["q_pi"] = 0.0
            return actor_loss, metrics, debug_bundle

        rl_state = actor_aux["rl_state"]
        q1_pi, q2_pi = self.model(
            forward_type=ForwardType.DEFAULT,
            mode="critic",
            rl_state=rl_state,
            action=pi,
            use_target=False,
        )
        q_pi = torch.minimum(q1_pi, q2_pi)

        actor_loss = (-q_pi).mean() + self.bc_coef * bc_loss
        metrics["q_pi"] = q_pi.mean().item()
        return actor_loss, metrics, debug_bundle

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

        metrics_data = {
            "stage/is_bc_actor_pretrain": float(self.stage_actor_bc_only),
            "stage/is_critic_warmup": float(self.stage_freeze_actor),
            "stage/is_full_rl": float(self.stage_full_rl),
        }

        critic_updated = False
        if not self.stage_actor_bc_only:
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
            critic_updated = True

            metrics_data.update(
                {
                    "gigawa/critic_loss": np.mean(gbs_critic_loss),
                    "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
                    "critic/grad_norm": qf_grad_norm,
                    **all_critic_metrics,
                }
            )
        else:
            metrics_data.update({"gigawa/critic_skipped": 1.0})

        actor_updated = False
        actor_update_due = self.stage_actor_bc_only or (self.update_step % self.critic_actor_ratio == 0)
        actor_train_enabled = train_actor and (not self.stage_freeze_actor)

        if actor_update_due and actor_train_enabled:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            all_actor_metrics = {}
            capture_debug_this_update = self._should_capture_action_diagnostics()
            debug_bundle_to_save = None
            for batch_idx, batch in enumerate(train_micro_batch_list):
                actor_loss, actor_metrics, debug_bundle = self.forward_actor(
                    batch,
                    capture_diagnostics=(capture_debug_this_update and batch_idx == 0),
                )
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                append_to_dict(all_actor_metrics, actor_metrics)
                if debug_bundle is not None:
                    debug_bundle_to_save = debug_bundle
            all_actor_metrics = {f"actor/{k}": np.mean(v) for k, v in all_actor_metrics.items()}
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            actor_updated = True
            self.actor_update_step += 1
            if debug_bundle_to_save is not None:
                metrics_data.update(self._save_action_diagnostics(debug_bundle_to_save))
            metrics_data.update(
                {
                    "gigawa/actor_loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    **all_actor_metrics,
                }
            )
        elif self.stage_freeze_actor:
            metrics_data.update({"gigawa/actor_frozen": 1.0})

        self._maybe_update_targets(actor_updated=actor_updated, critic_updated=critic_updated)

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

        append_to_dict(
            metrics,
            {
                "buffer_mix/demo_sample_ratio_target": self.demo_sample_ratio,
                "buffer_mix/replay_batch_size_target": float(self.target_replay_batch_size),
                "buffer_mix/demo_batch_size_target": float(self.target_demo_batch_size),
                "buffer_mix/store_online_replay": float(self.store_online_replay),
                "buffer_mix/store_online_demo_interventions": float(self.store_online_demo_interventions),
            },
        )

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

        readiness = self._get_training_readiness()
        if not readiness["can_train"]:
            start_status = readiness["start_status"]
            self.log_on_first_rank(
                "Skipping training because buffers are not ready: "
                f"replay_ready={start_status['replay_ready']} "
                f"(need {readiness['replay_start_size']}, have {len(self.replay_buffer)}), "
                f"demo_ready={start_status['demo_ready']} "
                f"(need {readiness['demo_start_size']}, have {len(self.demo_buffer) if self.demo_buffer is not None else 0})"
            )
            return {}

        train_actor = readiness["train_actor"]

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
        metrics = {
            "buffer_mix/use_replay_this_step": float(readiness["start_status"]["use_replay"]),
            "buffer_mix/use_demo_this_step": float(readiness["start_status"]["use_demo"]),
            "buffer_mix/train_actor": float(train_actor),
            "buffer_mix/replay_ready": float(readiness["start_status"]["replay_ready"]),
            "buffer_mix/demo_ready": float(readiness["start_status"]["demo_ready"]),
            "buffer_mix/replay_required_for_training": float(self.replay_required_for_training),
            "buffer_mix/allow_train_on_demo_only": float(self.allow_train_on_demo_only),
        }

        update_epoch = self.utd_ratio
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
        load_optimizer_and_scheduler_state = bool(
            self.cfg.runner.get("resume_load_optimizer_and_scheduler_state", True)
        )
        if load_optimizer_and_scheduler_state:
            optimizers = [self.optimizer, self.qf_optimizer]
            lr_schedulers = [self.lr_scheduler, self.qf_lr_scheduler]
        else:
            # RLinf Checkpoint(...) expects iterables here and will call tuple(...).
            # Use empty lists for model-only resume instead of None.
            optimizers = []
            lr_schedulers = []
            self.log_on_first_rank(
                "Resuming actor model weights only; optimizer and lr_scheduler states are not restored."
            )

        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
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
        if self.stage_freeze_actor:
            policy.soft_update_targets(tau=1.0)

        checkpoint_rollout_flag = policy.get_use_rl_head_for_rollout()
        self.log_on_first_rank(
            "Checkpoint restored rollout flag before config override: "
            f"use_rl_head_for_rollout={checkpoint_rollout_flag}."
        )
        self._apply_rollout_flag_from_config("load_checkpoint")
