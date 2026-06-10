#!/usr/bin/env python3
"""Visualize Piper open-loop trajectories from saved .pt episodes.

WA modes:
  --wa-source saved  : visualize the WA/ref action already saved in the .pt.
  --wa-source latent : rerun WA action diffusion from saved visual_latent/state.
  --wa-source rerun  : rerun full VAE+WA from raw images, for future .pt files
                       that include camera frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for visualization") from exc

from diagnose_piper_backfeed import (
    DEFAULT_CKPT,
    DEFAULT_NORM_JSON,
    DEFAULT_PT,
    DEFAULT_URDF,
    CrossAttentionBlock,
    MLP,
    SOURCE_NAME,
    UrdfChain,
    _torch_dtype,
    action_source,
    find_episode_files,
    load_actor,
    load_episode,
    reshape_action,
    squeeze_env_dim,
    valid_mask,
)


DEFAULT_OUT = (
    "/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/"
    "piper_openloop_visualization"
)
DEFAULT_CONFIG_PATH = "/home/ubuntu/users/angen.ye/gwp/RLinf/examples/embodiment/config"
DEFAULT_CONFIG_NAME = "offline_piper_actor_bc_warmup"

COLORS = {
    "gt": "#111111",
    "wa": "#2468b2",
    "actor": "#d43d2a",
}
LABELS = {
    "gt": "GT executed",
    "wa": "WA",
    "actor": "Actor",
}


class TwinCriticValue(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.q1 = MLP(
            input_dim=input_dim,
            hidden_dims=[2048, 1024, 512],
            output_dim=1,
            activate_final=False,
            layer_norm=False,
        )
        self.q2 = MLP(
            input_dim=input_dim,
            hidden_dims=[2048, 1024, 512],
            output_dim=1,
            activate_final=False,
            layer_norm=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


class LiteCrossAttentionCritic(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        robot_state_dim: int = 14,
        action_dim: int = 14,
    ):
        super().__init__()
        self.state_proj = nn.Linear(robot_state_dim, hidden_dim)
        self.ref_action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.value_head = TwinCriticValue(input_dim=hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor | None,
        robot_state: torch.Tensor | None,
        ref_action: torch.Tensor | None,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_tokens = []
        if robot_state is not None:
            query_tokens.append(self.state_proj(robot_state).unsqueeze(1))
        if ref_action is not None:
            query_tokens.append(self.ref_action_proj(ref_action))
        query_tokens.append(self.action_proj(action))
        query_tokens = torch.cat(query_tokens, dim=1)
        fused_tokens = self.cross_attn(query_tokens, visual_tokens)
        fused_state = fused_tokens.mean(dim=1)
        q1, q2 = self.value_head(fused_state)
        return q1, q2, fused_state


def _safe_stem(path: Path) -> str:
    return path.stem.replace("/", "_")


def _as_float_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    return squeeze_env_dim(x, name).float()


def get_chunk_start_raw_state(data: dict[str, Any], t: int, action_dim: int) -> torch.Tensor:
    curr = data.get("curr_obs", {})
    if isinstance(curr, dict):
        for key in ("raw_robot_state", "states"):
            value = curr.get(key)
            if torch.is_tensor(value):
                s = _as_float_tensor(value, f"curr_obs.{key}")
                if s.shape == (t, action_dim):
                    return s
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict):
        for key in ("raw_robot_state",):
            value = fi.get(key)
            if torch.is_tensor(value):
                s = _as_float_tensor(value, f"forward_inputs.{key}")
                if s.shape == (t, action_dim):
                    return s
        value = fi.get("raw_states_before_action")
        if torch.is_tensor(value):
            s = _as_float_tensor(value, "forward_inputs.raw_states_before_action")
            if s.ndim == 3 and s.shape[:2] == (t, 12):
                return s[:, 0, :action_dim].float()
    raise KeyError("Could not find chunk-start raw qpos in curr_obs/forward_inputs")


def get_robot_state_for_actor(data: dict[str, Any], t: int, action_dim: int) -> torch.Tensor:
    curr = data.get("curr_obs", {})
    if isinstance(curr, dict) and torch.is_tensor(curr.get("robot_state")):
        s = _as_float_tensor(curr["robot_state"], "curr_obs.robot_state")
        if s.shape == (t, action_dim):
            return s
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict) and torch.is_tensor(fi.get("robot_state")):
        s = _as_float_tensor(fi["robot_state"], "forward_inputs.robot_state")
        if s.shape == (t, action_dim):
            return s
    raise KeyError("Could not find actor robot_state")


def get_visual_latent(data: dict[str, Any], t: int) -> torch.Tensor:
    curr = data.get("curr_obs", {})
    for container_name, container in (("curr_obs", curr), ("forward_inputs", data.get("forward_inputs", {}))):
        if isinstance(container, dict) and torch.is_tensor(container.get("visual_latent")):
            x = squeeze_env_dim(container["visual_latent"], f"{container_name}.visual_latent").float()
            if x.ndim == 6 and x.shape[1] == 1:
                x = x.squeeze(1)
            if x.ndim == 5 and x.shape[0] == t:
                return x
    raise KeyError("Could not find saved visual_latent [T,Z,t,h,w]")


def get_saved_wa_model(data: dict[str, Any], action_dim: int, chunks: int) -> tuple[torch.Tensor, str]:
    curr = data.get("curr_obs", {})
    if isinstance(curr, dict) and torch.is_tensor(curr.get("ref_action")):
        return reshape_action(curr["ref_action"], action_dim, chunks, "curr_obs.ref_action"), "curr_obs.ref_action"
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict):
        for key in ("policy_action_model", "ref_action", "model_action"):
            value = fi.get(key)
            if torch.is_tensor(value):
                return reshape_action(value, action_dim, chunks, f"forward_inputs.{key}"), f"forward_inputs.{key}"
    raise KeyError("Could not find saved WA/ref model action")


def get_saved_wa_exec(data: dict[str, Any], action_dim: int, chunks: int) -> tuple[torch.Tensor, str]:
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict):
        for key in ("policy_action_exec", "wa_action_exec", "ref_action_exec"):
            value = fi.get(key)
            if torch.is_tensor(value):
                return reshape_action(value, action_dim, chunks, f"forward_inputs.{key}"), f"forward_inputs.{key}"
    raise KeyError("Could not find saved WA exec-space action")


def get_gt_exec(data: dict[str, Any], action_dim: int, chunks: int) -> tuple[torch.Tensor, str]:
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict):
        for key in ("action_exec", "action"):
            value = fi.get(key)
            if torch.is_tensor(value):
                return reshape_action(value, action_dim, chunks, f"forward_inputs.{key}"), f"forward_inputs.{key}"
    raise KeyError("Could not find executable GT action_exec/action")


def get_gt_model(data: dict[str, Any], action_dim: int, chunks: int) -> tuple[torch.Tensor, str]:
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict):
        for key in ("model_action", "action_model"):
            value = fi.get(key)
            if torch.is_tensor(value):
                return reshape_action(value, action_dim, chunks, f"forward_inputs.{key}"), f"forward_inputs.{key}"
    value = data.get("actions")
    if torch.is_tensor(value):
        return reshape_action(value, action_dim, chunks, "actions"), "actions"
    raise KeyError("Could not find model-space GT action")


def model_to_exec(
    policy,
    model_action: torch.Tensor,
    raw_state: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return policy.model_to_exec(
        model_action.to(device),
        raw_state.to(device),
        use_chunk_start_state=True,
    ).detach().float().cpu()


def load_critic(
    ckpt: str,
    policy,
    device: torch.device,
    dtype: torch.dtype,
) -> LiteCrossAttentionCritic | None:
    weight_file = find_full_weight_file(ckpt)
    if weight_file is None:
        return None
    sd = torch.load(weight_file, map_location="cpu", weights_only=False)
    critic_sd = {}
    for key, value in sd.items():
        if key.startswith("critic."):
            critic_sd[key[len("critic.") :]] = value
    if not critic_sd:
        print("[viz] no critic.* weights found; skip Q plots")
        return None
    critic = LiteCrossAttentionCritic(
        hidden_dim=int(getattr(policy, "hidden_dim", 512)),
        num_heads=8,
        dropout=0.0,
        robot_state_dim=int(getattr(policy, "action_dim", 14)),
        action_dim=int(getattr(policy, "action_dim", 14)),
    )
    missing, unexpected = critic.load_state_dict(critic_sd, strict=False)
    print(f"[viz] load_critic missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[viz] critic missing sample:", missing[:20])
    if unexpected:
        print("[viz] critic unexpected sample:", unexpected[:20])
    critic.to(device=device, dtype=dtype)
    critic.eval()
    return critic


def compute_q_values(
    policy,
    critic: LiteCrossAttentionCritic | None,
    data: dict[str, Any],
    candidates: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    action_dim: int,
    chunks: int,
) -> dict[str, torch.Tensor] | None:
    if critic is None:
        return None
    gt_exec, _ = get_gt_exec(data, action_dim, chunks)
    t = gt_exec.shape[0]
    visual_latent = get_visual_latent(data, t)
    robot_state = get_robot_state_for_actor(data, t, action_dim)
    wa_model, _ = get_saved_wa_model(data, action_dim, chunks)

    out = {key: [] for key in candidates}
    with torch.no_grad():
        for start in range(0, t, batch_size):
            end = min(t, start + batch_size)
            visual_tokens = policy.encode_visual(
                visual_latent[start:end].to(device=device, dtype=dtype)
            )
            robot = robot_state[start:end].to(device=device, dtype=dtype)
            ref = wa_model[start:end].to(device=device, dtype=dtype)
            for key, action in candidates.items():
                q1, q2, _ = critic(
                    visual_tokens=visual_tokens,
                    robot_state=robot,
                    ref_action=ref,
                    action=action[start:end].to(device=device, dtype=dtype),
                )
                out[key].append(torch.minimum(q1, q2).detach().float().cpu().squeeze(-1))
    return {key: torch.cat(parts, dim=0) for key, parts in out.items()}


def actor_saved_features(
    policy,
    data: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    action_dim: int,
    chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gt_exec, _ = get_gt_exec(data, action_dim, chunks)
    t = gt_exec.shape[0]
    visual_latent = get_visual_latent(data, t)
    robot_state = get_robot_state_for_actor(data, t, action_dim)
    wa_model, _ = get_saved_wa_model(data, action_dim, chunks)
    raw_state = get_chunk_start_raw_state(data, t, action_dim)

    outs = []
    with torch.no_grad():
        for start in range(0, t, batch_size):
            end = min(t, start + batch_size)
            pred = policy.actor_forward(
                visual_latent[start:end].to(device=device, dtype=dtype),
                robot_state[start:end].to(device=device, dtype=dtype),
                wa_model[start:end].to(device=device, dtype=dtype),
            )
            outs.append(pred.detach().float().cpu())
    actor_model = torch.cat(outs, dim=0)
    actor_exec = model_to_exec(policy, actor_model, raw_state, device)
    return actor_model, actor_exec


def _select_tensor_sequence(container: dict[str, Any], keys: Iterable[str], t: int) -> torch.Tensor | None:
    for key in keys:
        value = container.get(key)
        if not torch.is_tensor(value):
            continue
        value = squeeze_env_dim(value, key)
        if value.shape[0] == t:
            return value
    return None


def _stack_wrist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack([left, right], dim=1)


def raw_images_from_episode(data: dict[str, Any], t: int) -> tuple[Any, Any]:
    """Return (main_images, wrist_images) if raw images are present."""
    curr = data.get("curr_obs", {})
    if not isinstance(curr, dict):
        raise KeyError("curr_obs is not a dict")

    main = _select_tensor_sequence(
        curr,
        ("main_images", "cam_high", "image", "rgb", "front_image"),
        t,
    )
    wrist = _select_tensor_sequence(curr, ("wrist_images",), t)

    images = curr.get("images")
    if isinstance(images, dict):
        main = main if main is not None else _select_tensor_sequence(images, ("cam_high", "front", "main"), t)
        left = _select_tensor_sequence(images, ("cam_left_wrist", "left_wrist"), t)
        right = _select_tensor_sequence(images, ("cam_right_wrist", "right_wrist"), t)
        if wrist is None and left is not None and right is not None:
            wrist = _stack_wrist(left, right)

    if wrist is None:
        left = _select_tensor_sequence(curr, ("cam_left_wrist", "left_wrist_image"), t)
        right = _select_tensor_sequence(curr, ("cam_right_wrist", "right_wrist_image"), t)
        if left is not None and right is not None:
            wrist = _stack_wrist(left, right)

    if main is None:
        raise KeyError(
            "No raw camera images found in pt. Current compact episodes only store "
            "visual_latent; use --wa-source saved, or re-collect with raw image fields."
        )
    return main, wrist


def find_full_weight_file(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    candidates = [
        p / "model_state_dict" / "full_weights.pt",
        p / "actor" / "model_state_dict" / "full_weights.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(p.glob("**/model_state_dict/full_weights.pt"), key=lambda x: x.stat().st_mtime)
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"Could not find full_weights.pt under {p}")


def load_full_policy_for_rerun(args, device: torch.device, dtype: torch.dtype):
    import hydra

    from rlinf.models.embodiment.giga_world_policy.giga_world_policy import get_model

    with hydra.initialize_config_dir(config_dir=os.path.abspath(args.config_path), version_base=None):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    model_cfg = cfg.actor.model
    policy = get_model(model_cfg, torch_dtype=dtype)
    weight_file = find_full_weight_file(args.actor_ckpt)
    if weight_file is not None:
        sd = torch.load(weight_file, map_location="cpu", weights_only=False)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        print(f"[viz] full policy load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    policy.to(device)
    policy.eval()
    return policy


def _slice_sequence(x: Any, start: int, end: int) -> Any:
    if torch.is_tensor(x):
        return x[start:end]
    if isinstance(x, np.ndarray):
        return x[start:end]
    return x[start:end]


def rerun_wa_and_actor(
    full_policy,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int,
    action_dim: int,
    chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_exec, _ = get_gt_exec(data, action_dim, chunks)
    t = gt_exec.shape[0]
    raw_state = get_chunk_start_raw_state(data, t, action_dim)
    main_images, wrist_images = raw_images_from_episode(data, t)
    task_descriptions = data.get("task_descriptions", [""] * t)
    if isinstance(data.get("metadata"), dict) and "task_description" in data["metadata"]:
        task_descriptions = [str(data["metadata"]["task_description"])] * t

    wa_model_parts, wa_exec_parts = [], []
    actor_model_parts, actor_exec_parts = [], []
    with torch.no_grad():
        for start in range(0, t, batch_size):
            end = min(t, start + batch_size)
            env_obs = {
                "states": raw_state[start:end].to(device),
                "main_images": _slice_sequence(main_images, start, end),
                "task_descriptions": task_descriptions[start:end],
            }
            if wrist_images is not None:
                env_obs["wrist_images"] = _slice_sequence(wrist_images, start, end)
            backbone = full_policy.extract_frozen_backbone_batch(env_obs)
            wa_model = backbone["ref_action"].detach().float().cpu()
            wa_exec = backbone["ref_action_exec"].detach().float().cpu()
            visual_feat = full_policy.encode_visual(backbone["visual_latent"])
            actor_model, _ = full_policy.actor_forward(
                visual_feat=visual_feat,
                robot_state=backbone["robot_state"],
                ref_action=backbone["ref_action"],
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            actor_exec_state = backbone.get("raw_robot_state", raw_state[start:end].to(device))
            actor_exec = full_policy.postprocess_action_model_batch(actor_model, actor_exec_state)
            wa_model_parts.append(wa_model)
            wa_exec_parts.append(wa_exec)
            actor_model_parts.append(actor_model.detach().float().cpu())
            actor_exec_parts.append(actor_exec.detach().float().cpu())
    return (
        torch.cat(wa_model_parts, dim=0),
        torch.cat(wa_exec_parts, dim=0),
        torch.cat(actor_model_parts, dim=0),
        torch.cat(actor_exec_parts, dim=0),
    )


def rerun_wa_from_saved_latent(
    full_policy,
    visual_latent: torch.Tensor,
    norm_state: torch.Tensor,
    raw_state: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run WA action diffusion again while reusing saved latent_condition."""
    if not bool(full_policy.pipe.config.expand_timesteps):
        raise RuntimeError("--wa-source latent currently supports expand_timesteps=True only")

    pipe = full_policy.pipe
    original_prepare_latents = pipe.prepare_latents
    condition = visual_latent.unsqueeze(0).to(device=device, dtype=torch.float32)
    _, _, latent_t, latent_h, latent_w = condition.shape
    dummy_image = Image.fromarray(
        np.zeros((full_policy.full_image_size[1], full_policy.full_image_size[0], 3), dtype=np.uint8)
    )

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    def prepare_latents_from_saved(
        self,
        image,
        batch_size,
        num_channels_latents=16,
        height=480,
        width=832,
        num_frames=81,
        dtype=None,
        device=None,
        generator=None,
        latents=None,
        last_image=None,
        action_chunk=None,
        action_dim=14,
        return_latent_debug=False,
    ):
        del image, num_channels_latents, height, width, num_frames, last_image
        dtype = dtype or torch.float32
        device = device or condition.device
        latents_local = torch.randn(
            condition.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        ) if latents is None else latents.to(device=device, dtype=dtype)
        action = torch.randn(
            (batch_size, int(action_chunk), int(action_dim)),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        first_frame_mask = torch.ones(
            1,
            1,
            latent_t,
            latent_h,
            latent_w,
            dtype=dtype,
            device=device,
        )
        first_frame_mask[:, :, 0] = 0
        condition_local = condition.to(device=device, dtype=dtype)
        if return_latent_debug:
            debug_dict = {
                "latent_condition": condition_local.detach().cpu(),
                "sampled_video_noise": latents_local.detach().cpu(),
                "sampled_action_noise": action.detach().cpu(),
                "first_frame_mask": first_frame_mask.detach().cpu(),
            }
            return latents_local, condition_local, first_frame_mask, action, debug_dict
        return latents_local, condition_local, first_frame_mask, action

    pipe.prepare_latents = types.MethodType(prepare_latents_from_saved, pipe)
    try:
        kwargs = dict(
            image=dummy_image,
            height=full_policy.full_image_size[1],
            width=full_policy.full_image_size[0],
            action_chunk=full_policy.wa_action_chunk,
            state=norm_state.unsqueeze(0).to(device=device, dtype=torch.float32),
            num_frames=full_policy.num_frames,
            guidance_scale=full_policy.guidance_scale,
            num_inference_steps=full_policy.num_inference_steps,
            return_dict=False,
            generator=generator,
        )
        if full_policy.fixed_prompt_embeds is not None:
            kwargs["prompt_embeds"] = full_policy.fixed_prompt_embeds
        else:
            kwargs["prompt"] = ""
        _, pred_delta_norm = pipe(**kwargs)
    finally:
        pipe.prepare_latents = original_prepare_latents

    wa_model = pred_delta_norm[:, : full_policy.action_chunk].detach().float()
    wa_exec = full_policy._postprocess_pred_delta(
        wa_model,
        raw_state.to(device=device, dtype=torch.float32),
    ).detach().float()
    return wa_model[0].cpu(), wa_exec[0].cpu()


def latent_wa_and_actor(
    full_policy,
    lite_policy,
    data: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    action_dim: int,
    chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_exec, _ = get_gt_exec(data, action_dim, chunks)
    t = gt_exec.shape[0]
    visual_latent = get_visual_latent(data, t)
    robot_state = get_robot_state_for_actor(data, t, action_dim)
    raw_state = get_chunk_start_raw_state(data, t, action_dim)

    wa_model_parts, wa_exec_parts = [], []
    actor_model_parts, actor_exec_parts = [], []
    with torch.no_grad():
        for idx in range(t):
            sample_seed = -1 if seed < 0 else int(seed) + idx
            wa_model_i, wa_exec_i = rerun_wa_from_saved_latent(
                full_policy=full_policy,
                visual_latent=visual_latent[idx],
                norm_state=robot_state[idx],
                raw_state=raw_state[idx],
                device=device,
                seed=sample_seed,
            )
            actor_model_i = lite_policy.actor_forward(
                visual_latent[idx : idx + 1].to(device=device, dtype=dtype),
                robot_state[idx : idx + 1].to(device=device, dtype=dtype),
                wa_model_i.unsqueeze(0).to(device=device, dtype=dtype),
            ).detach().float().cpu()[0]
            actor_exec_i = model_to_exec(
                lite_policy,
                actor_model_i.unsqueeze(0),
                raw_state[idx : idx + 1],
                device,
            )[0]
            wa_model_parts.append(wa_model_i)
            wa_exec_parts.append(wa_exec_i)
            actor_model_parts.append(actor_model_i)
            actor_exec_parts.append(actor_exec_i)

    return (
        torch.stack(wa_model_parts, dim=0),
        torch.stack(wa_exec_parts, dim=0),
        torch.stack(actor_model_parts, dim=0),
        torch.stack(actor_exec_parts, dim=0),
    )


def flatten_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])[mask.reshape(-1)]


def flatten_source(source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return source.reshape(-1)[mask.reshape(-1)]


def chunk_source(source: torch.Tensor) -> torch.Tensor:
    if source.ndim == 1:
        return source.contiguous()
    out = torch.full((source.shape[0],), 3, dtype=source.dtype, device=source.device)
    out[(source == 0).any(dim=-1)] = 0
    out[(source == 2).any(dim=-1)] = 2
    out[(source == 1).any(dim=-1)] = 1
    return out.contiguous()


def equalize_3d_axes(ax, points: list[np.ndarray]) -> None:
    if not points:
        return
    p = np.concatenate(points, axis=0)
    mins = p.min(axis=0)
    maxs = p.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def save_tcp_3d(out: Path, episode: str, positions: dict[str, torch.Tensor]) -> None:
    fig = plt.figure(figsize=(14, 6))
    for arm_idx, arm_name in enumerate(("left", "right")):
        ax = fig.add_subplot(1, 2, arm_idx + 1, projection="3d")
        point_sets = []
        for key in ("gt", "wa", "actor"):
            p = positions[key][:, arm_idx, :].cpu().numpy()
            point_sets.append(p)
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=COLORS[key], label=LABELS[key], linewidth=1.8)
            ax.scatter(p[0, 0], p[0, 1], p[0, 2], color=COLORS[key], marker="o", s=28)
            ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], color=COLORS[key], marker="x", s=36)
        equalize_3d_axes(ax, point_sets)
        ax.set_title(f"{episode} {arm_name} TCP")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_tcp_xyz(out: Path, episode: str, positions: dict[str, torch.Tensor]) -> None:
    steps = np.arange(next(iter(positions.values())).shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    for arm_idx, arm_name in enumerate(("left", "right")):
        for dim_idx, dim_name in enumerate(("x", "y", "z")):
            ax = axes[arm_idx, dim_idx]
            for key in ("gt", "wa", "actor"):
                vals = positions[key][:, arm_idx, dim_idx].cpu().numpy()
                ax.plot(steps, vals, color=COLORS[key], label=LABELS[key], linewidth=1.3)
            ax.set_title(f"{arm_name} TCP {dim_name}")
            ax.set_ylabel("m")
            ax.grid(True, alpha=0.25)
            if arm_idx == 1:
                ax.set_xlabel("valid step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"{episode} TCP xyz")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_joint_plot(out: Path, episode: str, actions: dict[str, torch.Tensor]) -> None:
    steps = np.arange(next(iter(actions.values())).shape[0])
    fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
    for dim in range(14):
        row = dim % 7
        col = 0 if dim < 7 else 1
        ax = axes[row, col]
        for key in ("gt", "wa", "actor"):
            ax.plot(steps, actions[key][:, dim].cpu().numpy(), color=COLORS[key], label=LABELS[key], linewidth=1.1)
        ax.set_title(("left" if dim < 7 else "right") + f" joint {dim % 7}")
        ax.grid(True, alpha=0.25)
        if row == 6:
            ax.set_xlabel("valid step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"{episode} executable joint actions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_error_plot(
    out: Path,
    episode: str,
    actions: dict[str, torch.Tensor],
    positions: dict[str, torch.Tensor],
    source: torch.Tensor,
) -> None:
    steps = np.arange(actions["gt"].shape[0])
    joint_wa = torch.linalg.vector_norm(actions["gt"] - actions["wa"], dim=-1).cpu().numpy()
    joint_actor = torch.linalg.vector_norm(actions["gt"] - actions["actor"], dim=-1).cpu().numpy()
    tcp_wa = torch.linalg.vector_norm(positions["gt"] - positions["wa"], dim=-1).mean(dim=-1).cpu().numpy()
    tcp_actor = torch.linalg.vector_norm(positions["gt"] - positions["actor"], dim=-1).mean(dim=-1).cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.5, 2.5, 0.5]})
    axes[0].plot(steps, joint_wa, color=COLORS["wa"], label="GT vs WA")
    axes[0].plot(steps, joint_actor, color=COLORS["actor"], label="GT vs Actor")
    axes[0].set_ylabel("joint L2 [rad]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, tcp_wa, color=COLORS["wa"], label="GT vs WA")
    axes[1].plot(steps, tcp_actor, color=COLORS["actor"], label="GT vs Actor")
    axes[1].set_ylabel("mean TCP dist [m]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    src_np = source.cpu().numpy().reshape(1, -1)
    axes[2].imshow(src_np, aspect="auto", cmap="tab10", vmin=0, vmax=9)
    axes[2].set_yticks([])
    axes[2].set_xlabel("valid step")
    axes[2].set_title("source: " + ", ".join(f"{k}={v}" for k, v in SOURCE_NAME.items()))
    fig.suptitle(f"{episode} open-loop errors")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_q_plot(
    out: Path,
    episode: str,
    q_values: dict[str, torch.Tensor],
    source: torch.Tensor,
) -> None:
    steps = np.arange(next(iter(q_values.values())).shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 2.2, 0.5]})
    for key in ("gt", "wa", "actor"):
        if key in q_values:
            axes[0].plot(steps, q_values[key].cpu().numpy(), color=COLORS[key], label=LABELS[key], linewidth=1.4)
    axes[0].set_ylabel("Q")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if "actor" in q_values and "gt" in q_values:
        axes[1].plot(
            steps,
            (q_values["actor"] - q_values["gt"]).cpu().numpy(),
            color=COLORS["actor"],
            label="Q(actor)-Q(GT)",
            linewidth=1.4,
        )
    if "gt" in q_values and "wa" in q_values:
        axes[1].plot(
            steps,
            (q_values["gt"] - q_values["wa"]).cpu().numpy(),
            color=COLORS["wa"],
            label="Q(GT)-Q(WA)",
            linewidth=1.4,
        )
    axes[1].axhline(0.0, color="#888888", linewidth=0.9, linestyle="--")
    axes[1].set_ylabel("Q gap")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    src_np = source.cpu().numpy().reshape(1, -1)
    axes[2].imshow(src_np, aspect="auto", cmap="tab10", vmin=0, vmax=9)
    axes[2].set_yticks([])
    axes[2].set_xlabel("chunk start")
    axes[2].set_title("source: " + ", ".join(f"{k}={v}" for k, v in SOURCE_NAME.items()))
    fig.suptitle(f"{episode} critic Q over chunks")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def summarize(actions: dict[str, torch.Tensor], positions: dict[str, torch.Tensor], source: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {
        "valid_steps": int(actions["gt"].shape[0]),
    }
    for sid, name in SOURCE_NAME.items():
        out[f"source_{name}_steps"] = int((source == sid).sum().item())
    for key in ("wa", "actor"):
        diff = actions[key] - actions["gt"]
        l2 = torch.linalg.vector_norm(diff, dim=-1)
        tcp = torch.linalg.vector_norm(positions[key] - positions["gt"], dim=-1)
        out[f"gt_vs_{key}_joint_mae"] = float(diff.abs().mean().item())
        out[f"gt_vs_{key}_joint_l2_mean"] = float(l2.mean().item())
        out[f"gt_vs_{key}_joint_l2_max"] = float(l2.max().item())
        out[f"gt_vs_{key}_tcp_mean_m"] = float(tcp.mean().item())
        out[f"gt_vs_{key}_tcp_p95_m"] = float(torch.quantile(tcp.flatten(), 0.95).item())
        out[f"gt_vs_{key}_tcp_max_m"] = float(tcp.max().item())
    return out


def summarize_q(q_values: dict[str, torch.Tensor] | None, source: torch.Tensor) -> dict[str, float]:
    if not q_values:
        return {}
    out: dict[str, float] = {}
    for key, value in q_values.items():
        out[f"q_{key}_mean"] = float(value.mean().item())
        out[f"q_{key}_min"] = float(value.min().item())
        out[f"q_{key}_max"] = float(value.max().item())
    for lhs, rhs in (("actor", "gt"), ("gt", "wa"), ("actor", "wa")):
        if lhs in q_values and rhs in q_values:
            gap = q_values[lhs] - q_values[rhs]
            out[f"q_{lhs}_minus_{rhs}_mean"] = float(gap.mean().item())
            out[f"q_{lhs}_minus_{rhs}_positive_frac"] = float((gap > 0).float().mean().item())
            human = source == 1
            if bool(human.any().item()):
                out[f"q_{lhs}_minus_{rhs}_human_mean"] = float(gap[human].mean().item())
                out[f"q_{lhs}_minus_{rhs}_human_positive_frac"] = float((gap[human] > 0).float().mean().item())
    return out


def write_step_csv(path: Path, actions: dict[str, torch.Tensor], positions: dict[str, torch.Tensor], source: torch.Tensor) -> None:
    with path.open("w", newline="") as f:
        fieldnames = ["step", "source_id", "source"]
        for prefix in ("gt", "wa", "actor"):
            fieldnames += [f"{prefix}_a{i}" for i in range(14)]
            for arm in ("left", "right"):
                fieldnames += [f"{prefix}_{arm}_{axis}" for axis in ("x", "y", "z")]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(actions["gt"].shape[0]):
            sid = int(source[i].item())
            row = {"step": i, "source_id": sid, "source": SOURCE_NAME.get(sid, str(sid))}
            for prefix in ("gt", "wa", "actor"):
                row.update({f"{prefix}_a{j}": float(actions[prefix][i, j].item()) for j in range(14)})
                for arm_idx, arm in enumerate(("left", "right")):
                    for axis_idx, axis in enumerate(("x", "y", "z")):
                        row[f"{prefix}_{arm}_{axis}"] = float(positions[prefix][i, arm_idx, axis_idx].item())
            writer.writerow(row)


def write_q_csv(path: Path, q_values: dict[str, torch.Tensor], source: torch.Tensor) -> None:
    with path.open("w", newline="") as f:
        fieldnames = ["chunk", "source_id", "source"] + [f"q_{key}" for key in sorted(q_values)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        t = next(iter(q_values.values())).shape[0]
        for i in range(t):
            sid = int(source[i].item()) if i < source.shape[0] else -1
            row = {"chunk": i, "source_id": sid, "source": SOURCE_NAME.get(sid, str(sid))}
            for key in sorted(q_values):
                row[f"q_{key}"] = float(q_values[key][i].item())
            writer.writerow(row)


def write_index_html(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for s in summaries:
        ep = s["episode"]
        q_line = ""
        if "q_actor_minus_gt_mean" in s:
            q_line = (
                f", Q(actor-GT)={s['q_actor_minus_gt_mean']:.4f}, "
                f"Q(GT-WA)={s.get('q_gt_minus_wa_mean', 0.0):.4f}"
            )
        q_img = f"<img src='{ep}_q.png' width='900'><br>\n" if s.get("has_q_plot", 0.0) else ""
        rows.append(
            f"<h2>{ep}</h2>\n"
            f"<p>WA joint MAE={s['gt_vs_wa_joint_mae']:.6f}, "
            f"Actor joint MAE={s['gt_vs_actor_joint_mae']:.6f}, "
            f"Actor TCP max={s['gt_vs_actor_tcp_max_m']:.4f} m{q_line}</p>\n"
            f"<img src='{ep}_tcp_3d.png' width='900'><br>\n"
            f"{q_img}"
            f"<img src='{ep}_errors.png' width='900'><br>\n"
            f"<img src='{ep}_tcp_xyz.png' width='900'><br>\n"
            f"<img src='{ep}_joints.png' width='900'><hr>\n"
        )
    html = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Piper Open-loop Visualization</title>"
        "<body style='font-family: sans-serif; margin: 24px;'>"
        "<h1>Piper Open-loop Visualization</h1>"
        + "\n".join(rows)
        + "</body>"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def visualize_one(
    *,
    pt_path: Path,
    out_dir: Path,
    chain: UrdfChain,
    lite_policy,
    critic,
    full_policy,
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    data = load_episode(pt_path)
    action_dim = int(args.action_dim)
    chunks = int(args.num_chunks)
    gt_exec, gt_source_key = get_gt_exec(data, action_dim, chunks)
    gt_model, gt_model_source_key = get_gt_model(data, action_dim, chunks)
    t = gt_exec.shape[0]
    mask = valid_mask(data, t, chunks)
    src = action_source(data, t, chunks)
    raw_state = get_chunk_start_raw_state(data, t, action_dim)

    if args.wa_source == "saved":
        wa_model, wa_source_key = get_saved_wa_model(data, action_dim, chunks)
        try:
            wa_exec, wa_exec_source_key = get_saved_wa_exec(data, action_dim, chunks)
        except KeyError:
            wa_exec = model_to_exec(lite_policy, wa_model, raw_state, device)
            wa_exec_source_key = f"{wa_source_key}->model_to_exec"
        actor_model, actor_exec = actor_saved_features(
            lite_policy,
            data,
            device,
            dtype,
            args.batch_size,
            action_dim,
            chunks,
        )
    elif args.wa_source == "latent":
        if full_policy is None:
            raise RuntimeError("--wa-source latent requires full_policy")
        wa_model, wa_exec, actor_model, actor_exec = latent_wa_and_actor(
            full_policy=full_policy,
            lite_policy=lite_policy,
            data=data,
            device=device,
            dtype=dtype,
            seed=args.wa_seed,
            action_dim=action_dim,
            chunks=chunks,
        )
        wa_source_key = "rerun_from_saved_visual_latent"
        wa_exec_source_key = "rerun_from_saved_visual_latent"
    else:
        if full_policy is None:
            raise RuntimeError("--wa-source rerun requires full_policy")
        wa_model, wa_exec, actor_model, actor_exec = rerun_wa_and_actor(
            full_policy,
            data,
            device,
            args.batch_size,
            action_dim,
            chunks,
        )
        wa_source_key = "rerun_full_gigawa_policy"
        wa_exec_source_key = "rerun_full_gigawa_policy"

    valid_gt = flatten_valid(gt_exec, mask)
    valid_wa = flatten_valid(wa_exec, mask)
    valid_actor = flatten_valid(actor_exec, mask)
    valid_src = flatten_source(src, mask)
    q_source = chunk_source(src)

    actions = {
        "gt": valid_gt,
        "wa": valid_wa,
        "actor": valid_actor,
    }
    positions = {key: chain.positions(value) for key, value in actions.items()}
    q_values = compute_q_values(
        lite_policy,
        critic,
        data,
        candidates={"gt": gt_model, "wa": wa_model, "actor": actor_model},
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        action_dim=action_dim,
        chunks=chunks,
    )

    episode = _safe_stem(pt_path)
    ep_dir = out_dir if not args.per_episode_dirs else out_dir / episode
    ep_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.per_episode_dirs else f"{episode}_"

    save_tcp_3d(ep_dir / f"{prefix}tcp_3d.png", episode, positions)
    save_tcp_xyz(ep_dir / f"{prefix}tcp_xyz.png", episode, positions)
    save_joint_plot(ep_dir / f"{prefix}joints.png", episode, actions)
    save_error_plot(ep_dir / f"{prefix}errors.png", episode, actions, positions, valid_src)
    if q_values is not None:
        save_q_plot(ep_dir / f"{prefix}q.png", episode, q_values, q_source)
    write_step_csv(ep_dir / f"{prefix}steps.csv", actions, positions, valid_src)
    if q_values is not None:
        write_q_csv(ep_dir / f"{prefix}q.csv", q_values, q_source)
    npz_payload = {
        "gt_action": actions["gt"].cpu().numpy(),
        "wa_action": actions["wa"].cpu().numpy(),
        "actor_action": actions["actor"].cpu().numpy(),
        "gt_tcp": positions["gt"].cpu().numpy(),
        "wa_tcp": positions["wa"].cpu().numpy(),
        "actor_tcp": positions["actor"].cpu().numpy(),
        "source": valid_src.cpu().numpy(),
        "chunk_source": q_source.cpu().numpy(),
        "gt_model": gt_model.cpu().numpy(),
        "actor_model": flatten_valid(actor_model, mask).cpu().numpy(),
        "wa_model": flatten_valid(wa_model, mask).cpu().numpy(),
    }
    if q_values is not None:
        for key, value in q_values.items():
            npz_payload[f"q_{key}"] = value.cpu().numpy()
    np.savez_compressed(
        ep_dir / f"{prefix}trajectories.npz",
        **npz_payload,
    )

    summary = summarize(actions, positions, valid_src)
    summary.update(summarize_q(q_values, q_source))
    summary.update(
        {
            "episode": episode,
            "file": str(pt_path),
            "gt_source": gt_source_key,
            "gt_model_source": gt_model_source_key,
            "wa_source": wa_source_key,
            "wa_exec_source": wa_exec_source_key,
            "wa_mode": args.wa_source,
            "out_dir": str(ep_dir),
            "has_q_plot": float(q_values is not None),
        }
    )
    (ep_dir / f"{prefix}summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[viz] {episode} | valid={summary['valid_steps']} | "
        f"WA joint MAE={summary['gt_vs_wa_joint_mae']:.6f} | "
        f"Actor joint MAE={summary['gt_vs_actor_joint_mae']:.6f} | "
        f"Actor TCP max={summary['gt_vs_actor_tcp_max_m']:.4f}m"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", default=DEFAULT_PT, help="episode .pt file or directory")
    parser.add_argument("--actor-ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--norm-json", default=DEFAULT_NORM_JSON)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--num-chunks", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument(
        "--wa-source",
        choices=("saved", "latent", "rerun"),
        default="saved",
        help=(
            "saved: use pt WA/ref_action; latent: rerun WA action diffusion from "
            "saved visual_latent; rerun: full raw-image VAE+WA path"
        ),
    )
    parser.add_argument("--wa-seed", type=int, default=0, help="Seed for --wa-source latent action/noise initialization; <0 means random")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--per-episode-dirs", action="store_true")
    parser.add_argument("--enable-bound", action="store_true")
    parser.add_argument("--skip-q", action="store_true", help="Do not load critic weights or write Q plots")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides used only with --wa-source rerun")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(args.dtype)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] device={device} dtype={dtype} wa_source={args.wa_source}")

    chain = UrdfChain(args.urdf)
    lite_policy = load_actor(args.actor_ckpt, args.norm_json, device=device, dtype=dtype, enable_bound=args.enable_bound)
    critic = None if args.skip_q else load_critic(args.actor_ckpt, lite_policy, device=device, dtype=dtype)
    files = find_episode_files(args.pt, args.max_files)
    full_policy = None
    if args.wa_source == "rerun":
        for pt_path in files:
            data = load_episode(pt_path)
            gt_exec, _ = get_gt_exec(data, int(args.action_dim), int(args.num_chunks))
            raw_images_from_episode(data, gt_exec.shape[0])
    if args.wa_source in {"latent", "rerun"}:
        full_policy = load_full_policy_for_rerun(args, device=device, dtype=dtype)

    summaries = []
    for pt_path in files:
        summaries.append(
            visualize_one(
                pt_path=pt_path,
                out_dir=out_dir,
                chain=chain,
                lite_policy=lite_policy,
                critic=critic,
                full_policy=full_policy,
                args=args,
                device=device,
                dtype=dtype,
            )
        )

    keys = sorted({key for row in summaries for key in row})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    write_index_html(out_dir, summaries)
    print(f"[viz] wrote {out_dir / 'summary.csv'}")
    print(f"[viz] open {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
