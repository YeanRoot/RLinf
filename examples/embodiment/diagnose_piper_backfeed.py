#!/usr/bin/env python3
"""Offline Piper backfeed diagnostics for saved GigaWA features and actor outputs.

This script intentionally avoids importing GigaWorldPolicy because that imports the
full WA pipeline.  It only needs the saved .pt fields:
  curr_obs.visual_latent, curr_obs.robot_state, curr_obs.raw_robot_state,
  curr_obs.ref_action, forward_inputs.action_exec/policy_action_exec,
  forward_inputs.raw_states_before_action, and actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_PT = (
    "/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/"
    "collect_piper_gigawa_intervention100/offline_collection/rank_0/all"
)
DEFAULT_CKPT = (
    "/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/"
    "offline_piper_actor_bc_kwj/piper_lerobot_actor_bc_warmup/checkpoints/"
    "global_step_300/actor/model_state_dict/full_weights.pt"
)
DEFAULT_NORM_JSON = "/home/ubuntu/users/angen.ye/gwp/norm_stats_delta.json"
DEFAULT_URDF = (
    "/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/"
    "piper_local_assets_tmp/piper.urdf"
)
DEFAULT_OUT = (
    "/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/"
    "piper_backfeed_diagnosis"
)


SOURCE_NAME = {
    0: "policy",
    1: "human",
    2: "replan",
    3: "padding",
}


def _torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activate_final: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if (not is_last) or activate_final:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        norm_tokens = self.token_norm(tokens)
        attn_out, _ = self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        x = tokens + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        visual_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if visual_tokens is None or visual_tokens.numel() == 0:
            x = query_tokens
        else:
            q = self.query_norm(query_tokens)
            kv = self.kv_norm(visual_tokens)
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)
            x = query_tokens + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CrossAttentionActor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        robot_state_dim: int = 14,
        action_dim: int = 14,
        action_chunk: int = 12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self.state_proj = nn.Linear(robot_state_dim, hidden_dim)
        self.ref_action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_query_embed = nn.Parameter(torch.zeros(1, action_chunk, hidden_dim))
        nn.init.normal_(self.action_query_embed, mean=0.0, std=0.02)
        self.self_attn = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.cross_attn = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.output_head = MLP(
            input_dim=hidden_dim,
            hidden_dims=[1024, 512],
            output_dim=action_dim,
            activate_final=False,
            layer_norm=False,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor | None,
        robot_state: torch.Tensor | None,
        ref_action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = None
        if visual_tokens is not None:
            batch_size = visual_tokens.shape[0]
            device = visual_tokens.device
            dtype = visual_tokens.dtype
        elif ref_action is not None:
            batch_size = ref_action.shape[0]
            device = ref_action.device
            dtype = ref_action.dtype
        elif robot_state is not None:
            batch_size = robot_state.shape[0]
            device = robot_state.device
            dtype = robot_state.dtype
        else:
            raise RuntimeError("Actor needs at least one input branch.")

        action_tokens = self.action_query_embed.expand(batch_size, -1, -1).to(
            device=device,
            dtype=dtype,
        )
        if ref_action is not None:
            action_tokens = action_tokens + self.ref_action_proj(ref_action)

        query_tokens = []
        if robot_state is not None:
            query_tokens.append(self.state_proj(robot_state).unsqueeze(1))
        query_tokens.append(action_tokens)
        query = torch.cat(query_tokens, dim=1)

        query = self.self_attn(query)
        fused = self.cross_attn(query, visual_tokens)
        action_token_start = fused.shape[1] - self.action_chunk
        action = self.output_head(fused[:, action_token_start:])
        return action, fused.mean(dim=1)


class LiteActorPolicy(nn.Module):
    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
        action_q01_raw: torch.Tensor,
        action_q99_raw: torch.Tensor,
        delta_mask: torch.Tensor,
        action_chunk: int = 12,
        action_dim: int = 14,
        hidden_dim: int = 512,
        max_visual_tokens: int = 1024,
        enable_absolute_action_bound: bool = False,
    ):
        super().__init__()
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self.vae_z_dim = 48
        self.hidden_dim = hidden_dim
        self.max_visual_tokens = max_visual_tokens
        self.enable_absolute_action_bound = enable_absolute_action_bound

        self.register_buffer("state_mean", state_mean.float(), persistent=False)
        self.register_buffer("state_std", state_std.float(), persistent=False)
        self.register_buffer("delta_mean", delta_mean.float(), persistent=False)
        self.register_buffer("delta_std", delta_std.float(), persistent=False)
        self.register_buffer("delta_mask", delta_mask.bool(), persistent=False)
        self.register_buffer("action_q01_raw", action_q01_raw.float(), persistent=False)
        self.register_buffer("action_q99_raw", action_q99_raw.float(), persistent=False)

        safe_std = torch.where(
            self.delta_std.abs() < 1e-8,
            torch.ones_like(self.delta_std),
            self.delta_std,
        )
        action_q01 = (self.action_q01_raw - self.delta_mean) / safe_std
        action_q99 = (self.action_q99_raw - self.delta_mean) / safe_std
        self.register_buffer("action_q01", action_q01.float(), persistent=False)
        self.register_buffer("action_q99", action_q99.float(), persistent=False)
        self.register_buffer("action_bound_center", 0.5 * (action_q01 + action_q99), persistent=False)
        self.register_buffer("action_bound_half_range", 0.5 * (action_q99 - action_q01), persistent=False)

        self.visual_compressor = nn.Sequential(
            nn.Linear(self.vae_z_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_pos_embed = nn.Parameter(torch.zeros(1, max_visual_tokens, hidden_dim))
        nn.init.normal_(self.visual_pos_embed, mean=0.0, std=0.02)
        self.actor_head = CrossAttentionActor(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=0.0,
            robot_state_dim=action_dim,
            action_dim=action_dim,
            action_chunk=action_chunk,
        )

    def encode_visual(self, visual_latent: torch.Tensor) -> torch.Tensor:
        if visual_latent.ndim != 5:
            raise ValueError(f"Expected visual_latent [B,Z,T,H,W], got {tuple(visual_latent.shape)}")
        x = visual_latent.to(device=next(self.visual_compressor.parameters()).device, dtype=next(self.visual_compressor.parameters()).dtype)
        x = x[:, :, 0] if x.shape[2] == 1 else x.mean(dim=2)
        batch_size, _, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.vae_z_dim)
        if x.shape[1] > self.max_visual_tokens:
            raise RuntimeError(
                f"visual token count {x.shape[1]} exceeds max_visual_tokens={self.max_visual_tokens}"
            )
        x = self.visual_compressor(x)
        pos = self.visual_pos_embed[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)
        return x + pos

    def _bound_absolute_action_model(self, raw_action: torch.Tensor) -> torch.Tensor:
        center = self.action_bound_center.to(device=raw_action.device, dtype=raw_action.dtype)
        half = self.action_bound_half_range.to(device=raw_action.device, dtype=raw_action.dtype)
        low = self.action_q01.to(device=raw_action.device, dtype=raw_action.dtype)
        high = self.action_q99.to(device=raw_action.device, dtype=raw_action.dtype)
        bounded = center + half * torch.tanh(raw_action)
        return torch.maximum(torch.minimum(bounded, high), low)

    def actor_forward(
        self,
        visual_latent: torch.Tensor,
        robot_state: torch.Tensor,
        ref_action: torch.Tensor,
    ) -> torch.Tensor:
        visual_feat = self.encode_visual(visual_latent)
        learned, _ = self.actor_head(
            visual_tokens=visual_feat,
            robot_state=robot_state.to(device=visual_feat.device, dtype=visual_feat.dtype),
            ref_action=ref_action.to(device=visual_feat.device, dtype=visual_feat.dtype),
        )
        if self.enable_absolute_action_bound:
            learned = self._bound_absolute_action_model(learned)
        return learned

    def model_to_exec(
        self,
        model_action: torch.Tensor,
        raw_state: torch.Tensor,
        use_chunk_start_state: bool = True,
    ) -> torch.Tensor:
        model_view = model_action
        batch, chunk = model_view.shape[:2]
        state = raw_state.to(device=model_view.device, dtype=model_view.dtype)
        if state.dim() == 2:
            state = state[:, None, :].expand(batch, chunk, state.shape[-1])
        elif state.dim() == 3 and use_chunk_start_state:
            state = state[:, :1, :].expand(batch, chunk, state.shape[-1])
        if state.shape[-1] < self.action_dim:
            pad = torch.zeros(*state.shape[:-1], self.action_dim - state.shape[-1], device=state.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=-1)
        state = state[..., : self.action_dim]
        delta = model_view * self.delta_std.clamp_min(1e-8).to(model_view.device, model_view.dtype)
        delta = delta + self.delta_mean.to(model_view.device, model_view.dtype)
        out = delta.clone()
        mask = self.delta_mask.to(model_view.device)
        out[..., mask] += state[..., mask]
        return out[..., : self.action_dim].float()


def load_stats(path: str, action_dim: int) -> dict[str, torch.Tensor]:
    payload = json.load(open(path, "r", encoding="utf-8"))
    stats = payload.get("norm_stats", payload)

    def stat(key1: str, key2: str, default: float) -> torch.Tensor:
        vals = stats.get(key1, {}).get(key2, [default] * action_dim)
        x = torch.as_tensor(vals, dtype=torch.float32).flatten()
        if x.numel() >= action_dim:
            return x[:action_dim]
        return F.pad(x, (0, action_dim - x.numel()), value=float(default))

    return {
        "state_mean": stat("observation.state", "mean", 0.0),
        "state_std": stat("observation.state", "std", 1.0),
        "delta_mean": stat("action", "mean", 0.0),
        "delta_std": stat("action", "std", 1.0),
        "action_q01_raw": stat("action", "q01", -1.0),
        "action_q99_raw": stat("action", "q99", 1.0),
    }


def build_delta_mask(action_dim: int) -> torch.Tensor:
    base = np.array(
        [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
        dtype=bool,
    )
    if action_dim > len(base):
        base = np.pad(base, (0, action_dim - len(base)), constant_values=False)
    return torch.as_tensor(base[:action_dim], dtype=torch.bool)


def load_actor(ckpt: str, norm_json: str, device: torch.device, dtype: torch.dtype, enable_bound: bool) -> LiteActorPolicy:
    stats = load_stats(norm_json, action_dim=14)
    policy = LiteActorPolicy(
        **stats,
        delta_mask=build_delta_mask(14),
        enable_absolute_action_bound=enable_bound,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    actor_sd = {}
    for key, value in sd.items():
        if key.startswith("visual_compressor.") or key.startswith("visual_pos_embed") or key.startswith("actor_head."):
            actor_sd[key] = value
    missing, unexpected = policy.load_state_dict(actor_sd, strict=False)
    print(f"[diag] load_actor missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[diag] missing sample:", missing[:20])
    if unexpected:
        print("[diag] unexpected sample:", unexpected[:20])
    policy.to(device=device, dtype=dtype)
    policy.eval()
    return policy


def rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = np.asarray([float(v) for v in xyz], dtype=np.float64)
    return t


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    r = np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    return out


def axis_translation(axis: np.ndarray, distance: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(axis, dtype=np.float64) * float(distance)
    return out


class UrdfChain:
    def __init__(self, urdf_path: str):
        root = ET.parse(urdf_path).getroot()
        child_to_joint = {}
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            jtype = joint.attrib.get("type", "fixed")
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            axis = joint.find("axis")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin is not None:
                xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
            axis_xyz = [0.0, 0.0, 1.0]
            if axis is not None:
                axis_xyz = [float(v) for v in axis.attrib.get("xyz", "0 0 1").split()]
            child_to_joint[child] = {
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "origin": transform_from_xyz_rpy(xyz, rpy),
                "axis": np.asarray(axis_xyz, dtype=np.float64),
            }
        self.child_to_joint = child_to_joint
        self.left_chain = self._chain_to("gripper")

    def _chain_to(self, link: str) -> list[dict[str, Any]]:
        joints = []
        cur = link
        while cur in self.child_to_joint:
            joint = self.child_to_joint[cur]
            joints.append(joint)
            cur = joint["parent"]
        joints.reverse()
        return joints

    def fk_one_arm(self, q7: np.ndarray) -> np.ndarray:
        q7 = np.asarray(q7, dtype=np.float64)
        t = np.eye(4, dtype=np.float64)
        revolute_idx = 0
        prismatic_idx = 6
        for joint in self.left_chain:
            t = t @ joint["origin"]
            if joint["type"] in {"revolute", "continuous"}:
                q = q7[revolute_idx] if revolute_idx < min(6, q7.shape[0]) else 0.0
                t = t @ axis_angle(joint["axis"], float(q))
                revolute_idx += 1
            elif joint["type"] == "prismatic":
                q = q7[prismatic_idx] if prismatic_idx < q7.shape[0] else 0.0
                t = t @ axis_translation(joint["axis"], float(q))
        return t

    def positions(self, actions: torch.Tensor) -> torch.Tensor:
        """Return [N,2,3] left/right TCP positions for [N,14] actions."""
        a = actions.detach().cpu().float().numpy()
        out = []
        for row in a:
            left_t = self.fk_one_arm(row[:7])
            right_t = self.fk_one_arm(row[7:14])
            out.append(np.stack([left_t[:3, 3], right_t[:3, 3]], axis=0))
        return torch.as_tensor(np.stack(out, axis=0), dtype=torch.float32)


def squeeze_env_dim(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be tensor, got {type(x)}")
    if x.dim() >= 2 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def reshape_action(x: torch.Tensor, action_dim: int, num_chunks: int, name: str) -> torch.Tensor:
    x = squeeze_env_dim(x, name)
    if x.dim() == 3 and x.shape[-2:] == (num_chunks, action_dim):
        return x.float()
    if x.dim() == 2 and x.shape[-1] == num_chunks * action_dim:
        return x.view(x.shape[0], num_chunks, action_dim).float()
    if x.dim() == 3 and x.shape[-1] == num_chunks * action_dim:
        return x.squeeze(1).view(x.shape[0], num_chunks, action_dim).float()
    raise ValueError(f"Cannot reshape {name} with shape {tuple(x.shape)} into [T,{num_chunks},{action_dim}]")


def find_episode_files(pt: str, max_files: int) -> list[Path]:
    p = Path(pt)
    if p.is_file():
        return [p]
    files = sorted(p.glob("episode_*.pt"))
    if not files:
        files = sorted(p.glob("*.pt"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No .pt files found under {p}")
    return files


def load_episode(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict episode, got {type(obj)}")
    return obj


def valid_mask(data: dict[str, Any], t: int, c: int) -> torch.Tensor:
    mask = data.get("action_valid_mask")
    if torch.is_tensor(mask):
        mask = squeeze_env_dim(mask, "action_valid_mask").bool()
        if mask.shape == (t, c):
            return mask
    return torch.ones(t, c, dtype=torch.bool)


def action_source(data: dict[str, Any], t: int, c: int) -> torch.Tensor:
    fi = data.get("forward_inputs", {})
    src = fi.get("action_source") if isinstance(fi, dict) else None
    if torch.is_tensor(src):
        src = squeeze_env_dim(src, "action_source").long()
        if src.shape == (t, c):
            return src
    return torch.zeros(t, c, dtype=torch.long)


def flat_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask.unsqueeze(-1).expand_as(x)]


def pair_metrics(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    vals = flat_valid(a.float() - b.float(), mask)
    if vals.numel() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    return {
        "mae": float(vals.abs().mean().item()),
        "rmse": float(torch.sqrt((vals ** 2).mean()).item()),
        "max_abs": float(vals.abs().max().item()),
    }


def pos_metrics(a_pos: torch.Tensor, b_pos: torch.Tensor, mask_flat: torch.Tensor) -> dict[str, float]:
    diff = torch.linalg.vector_norm(a_pos - b_pos, dim=-1)
    vals = diff[mask_flat]
    if vals.numel() == 0:
        return {"tcp_mean_m": float("nan"), "tcp_p95_m": float("nan"), "tcp_max_m": float("nan")}
    return {
        "tcp_mean_m": float(vals.mean().item()),
        "tcp_p95_m": float(torch.quantile(vals, 0.95).item()),
        "tcp_max_m": float(vals.max().item()),
    }


def source_metrics(prefix: str, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    out = {}
    for sid in [0, 1, 2, 3]:
        m = mask & (source == sid)
        if m.any():
            met = pair_metrics(a, b, m)
            for key, val in met.items():
                out[f"{prefix}_{SOURCE_NAME.get(sid, sid)}_{key}"] = val
            out[f"{prefix}_{SOURCE_NAME.get(sid, sid)}_steps"] = int(m.sum().item())
    return out


def save_episode_plot(
    out: Path,
    title: str,
    gt_exec: torch.Tensor,
    wa_exec: torch.Tensor,
    actor_exec: torch.Tensor,
    actor_exec_bad_norm: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if plt is None:
        return
    x = np.arange(gt_exec.numel() // gt_exec.shape[-1])
    m = mask.reshape(-1).cpu().numpy().astype(bool)
    curves = {
        "GT vs WA exec": torch.linalg.vector_norm(gt_exec - wa_exec, dim=-1).reshape(-1).cpu().numpy(),
        "GT vs actor exec": torch.linalg.vector_norm(gt_exec - actor_exec, dim=-1).reshape(-1).cpu().numpy(),
        "GT vs actor BAD normalized-state exec": torch.linalg.vector_norm(gt_exec - actor_exec_bad_norm, dim=-1).reshape(-1).cpu().numpy(),
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, vals in curves.items():
        vals = vals.copy()
        vals[~m] = np.nan
        ax.plot(x, vals, label=label)
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel("joint-space L2 rad")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def compare_one(
    policy: LiteActorPolicy,
    chain: UrdfChain,
    path: Path,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[str, Any]:
    data = load_episode(path)
    curr = data.get("curr_obs", {})
    fi = data.get("forward_inputs", {})
    target_model = reshape_action(data["actions"], 14, 12, "actions")
    wa_model = reshape_action(curr["ref_action"], 14, 12, "curr_obs.ref_action")
    gt_exec = reshape_action(fi["action_exec"], 14, 12, "forward_inputs.action_exec")
    wa_exec = reshape_action(fi.get("policy_action_exec", fi["action_exec"]), 14, 12, "forward_inputs.policy_action_exec")
    raw_state = squeeze_env_dim(curr.get("raw_robot_state", fi["raw_robot_state"]), "raw_robot_state").float()
    norm_state = squeeze_env_dim(curr["robot_state"], "curr_obs.robot_state").float()
    raw_states_before = squeeze_env_dim(fi.get("raw_states_before_action"), "raw_states_before_action").float()
    visual_latent = squeeze_env_dim(curr["visual_latent"], "curr_obs.visual_latent").float()
    if visual_latent.ndim == 6 and visual_latent.shape[1] == 1:
        visual_latent = visual_latent.squeeze(1)

    actor_parts = []
    with torch.no_grad():
        for start in range(0, visual_latent.shape[0], batch_size):
            end = min(start + batch_size, visual_latent.shape[0])
            actor_parts.append(
                policy.actor_forward(
                    visual_latent[start:end].to(device=device, dtype=dtype),
                    norm_state[start:end].to(device=device, dtype=dtype),
                    wa_model[start:end].to(device=device, dtype=dtype),
                ).detach().float().cpu()
            )
    actor_model = torch.cat(actor_parts, dim=0)
    actor_exec = policy.model_to_exec(actor_model.to(device), raw_state.to(device), True).cpu()
    actor_exec_bad_norm = policy.model_to_exec(actor_model.to(device), norm_state.to(device), True).cpu()
    actor_exec_perstep = policy.model_to_exec(actor_model.to(device), raw_states_before.to(device), False).cpu()
    wa_exec_from_model = policy.model_to_exec(wa_model.to(device), raw_state.to(device), True).cpu()
    target_exec_from_model = policy.model_to_exec(target_model.to(device), raw_state.to(device), True).cpu()

    mask = valid_mask(data, target_model.shape[0], 12)
    src = action_source(data, target_model.shape[0], 12)
    mask_flat = mask.reshape(-1)

    gt_flat = gt_exec.reshape(-1, 14)
    wa_flat = wa_exec.reshape(-1, 14)
    actor_flat = actor_exec.reshape(-1, 14)
    actor_bad_flat = actor_exec_bad_norm.reshape(-1, 14)
    actor_perstep_flat = actor_exec_perstep.reshape(-1, 14)

    gt_pos = chain.positions(gt_flat)
    wa_pos = chain.positions(wa_flat)
    actor_pos = chain.positions(actor_flat)
    actor_bad_pos = chain.positions(actor_bad_flat)
    actor_perstep_pos = chain.positions(actor_perstep_flat)

    summary = {
        "file": str(path),
        "episode": path.stem,
        "chunks": int(target_model.shape[0]),
        "valid_steps": int(mask.sum().item()),
        "human_steps": int((mask & (src == 1)).sum().item()),
        "replan_steps": int((mask & (src == 2)).sum().item()),
    }
    pairs = [
        ("model_gt_vs_wa", target_model, wa_model),
        ("model_gt_vs_actor", target_model, actor_model),
        ("model_actor_vs_wa", actor_model, wa_model),
        ("exec_gt_vs_wa_saved", gt_exec, wa_exec),
        ("exec_gt_vs_wa_from_model", gt_exec, wa_exec_from_model),
        ("exec_gt_vs_target_from_model", gt_exec, target_exec_from_model),
        ("exec_gt_vs_actor", gt_exec, actor_exec),
        ("exec_gt_vs_actor_bad_norm_state", gt_exec, actor_exec_bad_norm),
        ("exec_gt_vs_actor_perstep_state", gt_exec, actor_exec_perstep),
    ]
    for name, a, b in pairs:
        for key, val in pair_metrics(a, b, mask).items():
            summary[f"{name}_{key}"] = val
        summary.update(source_metrics(name, a, b, mask, src))

    for name, pos in [
        ("tcp_gt_vs_wa_saved", wa_pos),
        ("tcp_gt_vs_actor", actor_pos),
        ("tcp_gt_vs_actor_bad_norm_state", actor_bad_pos),
        ("tcp_gt_vs_actor_perstep_state", actor_perstep_pos),
    ]:
        for key, val in pos_metrics(gt_pos, pos, mask_flat).items():
            summary[f"{name}_{key}"] = val

    out_dir.mkdir(parents=True, exist_ok=True)
    save_episode_plot(
        out_dir / f"{path.stem}_joint_l2.png",
        f"{path.stem}",
        gt_exec,
        wa_exec,
        actor_exec,
        actor_exec_bad_norm,
        mask,
    )
    torch.save(
        {
            "summary": summary,
            "target_model": target_model,
            "wa_model": wa_model,
            "actor_model": actor_model,
            "gt_exec": gt_exec,
            "wa_exec_saved": wa_exec,
            "actor_exec": actor_exec,
            "actor_exec_bad_norm_state": actor_exec_bad_norm,
            "actor_exec_perstep_state": actor_exec_perstep,
            "valid_mask": mask,
            "action_source": src,
        },
        out_dir / f"{path.stem}_diagnostic_tensors.pt",
    )
    print(
        f"[diag] {path.name} valid={summary['valid_steps']} | "
        f"GT-WA exec MAE={summary['exec_gt_vs_wa_saved_mae']:.6f} | "
        f"GT-actor exec MAE={summary['exec_gt_vs_actor_mae']:.6f} | "
        f"GT-actor BAD(norm state) MAE={summary['exec_gt_vs_actor_bad_norm_state_mae']:.6f} | "
        f"TCP BAD max={summary['tcp_gt_vs_actor_bad_norm_state_tcp_max_m']:.4f}m"
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--enable-bound", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(args.dtype)
    print(f"[diag] device={device} dtype={dtype}")
    print(f"[diag] pt={args.pt}")
    print(f"[diag] ckpt={args.actor_ckpt}")

    policy = load_actor(args.actor_ckpt, args.norm_json, device=device, dtype=dtype, enable_bound=args.enable_bound)
    chain = UrdfChain(args.urdf)
    files = find_episode_files(args.pt, args.max_files)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        compare_one(policy, chain, path, out_dir, device, dtype, batch_size=args.batch_size)
        for path in files
    ]
    keys = sorted({key for row in summaries for key in row})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"[diag] wrote {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
