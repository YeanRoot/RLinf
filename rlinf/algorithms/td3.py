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

import torch
import torch.nn.functional as F


class TD3Algorithm:
    """TD3 algorithm logic, decoupled from worker infrastructure.

    Handles critic loss, actor loss composition, and target network updates.
    The worker is responsible for gradient accumulation, FSDP context, and
    replay buffer management.
    """

    def __init__(self, algorithm_cfg, policy_head_cfg):
        self.critic_actor_ratio = int(algorithm_cfg.get("critic_actor_ratio", 2))
        self.target_update_freq = int(algorithm_cfg.get("target_update_freq", 1))
        self.target_tau = float(algorithm_cfg.get("tau", 0.005))
        self.target_policy_noise = float(algorithm_cfg.get("target_policy_noise", 0.2))
        self.target_noise_clip = float(algorithm_cfg.get("target_noise_clip", 0.5))
        self.bc_coef = float(algorithm_cfg.get("bc_coef", 1.0))

        self.enable_critic_q_upper_bound = bool(
            policy_head_cfg.get("enable_critic_q_upper_bound", True)
        )
        self.critic_q_upper_bound = float(
            policy_head_cfg.get("critic_q_upper_bound", 1.0)
        )
        self.actor_q_loss_clamp_to_upper_bound = bool(
            policy_head_cfg.get(
                "actor_q_loss_clamp_to_upper_bound",
                self.enable_critic_q_upper_bound,
            )
        )
        self.critic_target_clamp_to_upper_bound = bool(
            policy_head_cfg.get(
                "critic_target_clamp_to_upper_bound",
                self.enable_critic_q_upper_bound,
            )
        )
        self.critic_overshoot_penalty_coef = float(
            policy_head_cfg.get("critic_overshoot_penalty_coef", 1.0)
        )

        bc_guard_cfg = algorithm_cfg.get("actor_bc_guard", None) or {}
        self.actor_bc_guard_mode = str(bc_guard_cfg.get("mode", "none")).lower()
        self.actor_bc_guard_threshold = float(bc_guard_cfg.get("threshold", 0.004))
        self.actor_bc_weighted_coef = float(bc_guard_cfg.get("weighted_bc_coef", 50.0))
        self.actor_bc_penalty_coef = float(bc_guard_cfg.get("hard_penalty_coef", 5000.0))
        self.actor_bc_penalty_power = float(bc_guard_cfg.get("hard_penalty_power", 1.0))

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    def compute_critic_loss(
        self,
        policy,
        model,
        batch,
        build_visual_feat_fn,
        reshape_action_fn,
        device,
        dtype,
        forward_type,
    ):
        """Compute TD3 critic loss.

        Args:
            policy: unwrapped policy (for target_actor_forward / target_critic_forward)
            model: FSDP-wrapped model (for current Q forward)
            batch: training batch dict
            build_visual_feat_fn: callable(visual_latent) -> visual_feat
            reshape_action_fn: callable(tensor, tensor_name) -> reshaped tensor
            device: torch device
            dtype: torch dtype
            forward_type: ForwardType.DEFAULT or ForwardType.TD3 depending on model

        Returns:
            (critic_loss, q1, q2, target_q_values, aux_dict)
        """
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = reshape_action_fn(
            batch["actions"].to(device, dtype=dtype),
            "batch.actions",
        )
        rewards = batch["rewards"]
        terminations = batch["terminations"]

        rewards_for_bootstrap = rewards.sum(dim=-1, keepdim=True).to(dtype)
        done_mask = terminations.any(dim=-1, keepdim=True).to(dtype)

        with torch.no_grad():
            curr_visual_feat = build_visual_feat_fn(curr_obs["visual_latent"])
            _, curr_actor_aux = model(
                forward_type=forward_type,
                mode="actor",
                visual_feat=curr_visual_feat.detach(),
                robot_state=curr_obs["robot_state"].to(device, dtype=dtype),
                ref_action=reshape_action_fn(
                    curr_obs["ref_action"].to(device, dtype=dtype),
                    "curr_obs.ref_action",
                ),
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            curr_rl_state = curr_actor_aux["rl_state"].detach()
            curr_critic_visual_tokens = curr_actor_aux.get("critic_visual_tokens", None)
            if curr_critic_visual_tokens is not None:
                curr_critic_visual_tokens = curr_critic_visual_tokens.detach()
            curr_critic_robot_state = curr_actor_aux.get("critic_robot_state", None)
            if curr_critic_robot_state is not None:
                curr_critic_robot_state = curr_critic_robot_state.detach()
            curr_critic_ref_action = curr_actor_aux.get("critic_ref_action", None)
            if curr_critic_ref_action is not None:
                curr_critic_ref_action = curr_critic_ref_action.detach()

            next_visual_feat = build_visual_feat_fn(next_obs["visual_latent"])
            next_actions, next_actor_aux = policy.target_actor_forward(
                visual_feat=next_visual_feat.detach(),
                robot_state=next_obs["robot_state"].to(device, dtype=dtype),
                ref_action=reshape_action_fn(
                    next_obs["ref_action"].to(device, dtype=dtype),
                    "next_obs.ref_action",
                ),
            )
            if self.target_policy_noise > 0.0:
                noise = torch.randn_like(next_actions) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = next_actions + noise
            next_rl_state = next_actor_aux["rl_state"]
            target_q1, target_q2 = policy.target_critic_forward(
                rl_state=next_rl_state,
                action=next_actions,
                critic_visual_tokens=next_actor_aux.get("critic_visual_tokens", None),
                critic_robot_state=next_actor_aux.get("critic_robot_state", None),
                critic_ref_action=next_actor_aux.get("critic_ref_action", None),
            )
            target_q = torch.minimum(target_q1, target_q2)
            if self.enable_critic_q_upper_bound and self.critic_target_clamp_to_upper_bound:
                target_q = torch.clamp(target_q, max=self.critic_q_upper_bound)
            target_q_values = rewards_for_bootstrap + (1.0 - done_mask) * self._discount * target_q
            if self.enable_critic_q_upper_bound and self.critic_target_clamp_to_upper_bound:
                target_q_values = torch.clamp(target_q_values, max=self.critic_q_upper_bound)

        q1, q2 = model(
            forward_type=forward_type,
            mode="critic",
            rl_state=curr_rl_state,
            action=actions,
            use_target=False,
            critic_visual_tokens=curr_critic_visual_tokens,
            critic_robot_state=curr_critic_robot_state,
            critic_ref_action=curr_critic_ref_action,
        )
        target_q_values = target_q_values.to(dtype=q1.dtype)
        critic_loss = F.mse_loss(q1, target_q_values) + F.mse_loss(q2, target_q_values)
        q1_overshoot = torch.clamp(q1 - self.critic_q_upper_bound, min=0.0)
        q2_overshoot = torch.clamp(q2 - self.critic_q_upper_bound, min=0.0)
        critic_overshoot_penalty = q1.new_zeros(())
        if self.enable_critic_q_upper_bound and self.critic_overshoot_penalty_coef > 0.0:
            critic_overshoot_penalty = self.critic_overshoot_penalty_coef * (
                (q1_overshoot ** 2).mean() + (q2_overshoot ** 2).mean()
            )
            critic_loss = critic_loss + critic_overshoot_penalty
        aux = {
            "critic_overshoot_penalty": float(critic_overshoot_penalty.detach().item()),
            "q1_overshoot": float(q1_overshoot.mean().detach().item()),
            "q2_overshoot": float(q2_overshoot.mean().detach().item()),
            "q_upper_bound": float(self.critic_q_upper_bound),
        }
        return critic_loss, q1, q2, target_q_values, aux

    # ------------------------------------------------------------------
    # Actor
    # ------------------------------------------------------------------

    def compose_actor_loss(self, q_pi: torch.Tensor | None, bc_loss: torch.Tensor):
        """Compose actor loss from Q-value term and BC loss.

        trust_region mode is intentionally excluded — it requires parameter
        snapshot/restore logic that belongs in the worker.

        Returns:
            (actor_loss, metrics_dict)
        """
        q_term = bc_loss.new_zeros(())
        q_weight = 1.0
        effective_bc_coef = float(self.bc_coef)
        hard_penalty = bc_loss.new_zeros(())
        guard_active = 0.0
        q_pi_for_loss = None

        if q_pi is not None:
            q_pi_for_loss = q_pi
            if self.enable_critic_q_upper_bound and self.actor_q_loss_clamp_to_upper_bound:
                q_pi_for_loss = torch.clamp(q_pi_for_loss, max=self.critic_q_upper_bound)
            q_term = (-q_pi_for_loss).mean()

        if self.actor_bc_guard_mode == "weighted":
            effective_bc_coef = float(self.actor_bc_weighted_coef)
        elif self.actor_bc_guard_mode == "hard_penalty":
            exceed = torch.clamp(bc_loss - self.actor_bc_guard_threshold, min=0.0)
            if float(exceed.detach().item()) > 0.0:
                guard_active = 1.0
            hard_penalty = self.actor_bc_penalty_coef * (exceed ** self.actor_bc_penalty_power)

        actor_loss = q_weight * q_term + effective_bc_coef * bc_loss + hard_penalty
        metrics = {
            "bc_coef_effective": effective_bc_coef,
            "q_upper_bound_enabled": float(self.enable_critic_q_upper_bound),
            "q_upper_bound": float(self.critic_q_upper_bound),
            "q_loss_clamped": float(
                self.enable_critic_q_upper_bound
                and self.actor_q_loss_clamp_to_upper_bound
                and q_pi is not None
            ),
            "q_pi_used_for_loss": (
                float(q_pi_for_loss.mean().detach().item()) if q_pi_for_loss is not None else 0.0
            ),
            "bc_guard_mode": float(
                0 if self.actor_bc_guard_mode == "none"
                else 1 if self.actor_bc_guard_mode == "weighted"
                else 2 if self.actor_bc_guard_mode == "hard_penalty"
                else 3 if self.actor_bc_guard_mode == "trust_region"
                else -1
            ),
            "bc_guard_threshold": float(self.actor_bc_guard_threshold),
            "bc_guard_penalty": float(hard_penalty.detach().item()),
            "bc_guard_active": guard_active,
            "q_weight": q_weight,
        }
        return actor_loss, metrics

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def should_update_actor(self, update_step: int) -> bool:
        return update_step % self.critic_actor_ratio == 0

    def get_target_update_tau(
        self,
        update_step: int,
        stage_actor_bc_only: bool,
        stage_freeze_actor: bool,
        actor_updated: bool,
        critic_updated: bool,
    ) -> float | None:
        """Return tau for target update, or None if no update should happen."""
        if update_step % self.target_update_freq != 0:
            return None

        if stage_actor_bc_only:
            if not actor_updated:
                return None
            return 1.0
        elif stage_freeze_actor:
            if not critic_updated:
                return None
        else:
            if not (actor_updated or critic_updated):
                return None

        return self.target_tau

    def set_discount(self, discount: float):
        self._discount = discount
