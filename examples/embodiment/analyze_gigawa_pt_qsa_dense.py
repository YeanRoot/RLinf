#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def _add_repo_to_path(script_path: Path) -> None:
    candidates = [
        script_path.parent,
        script_path.parent.parent,
        script_path.parent.parent.parent,
        Path.cwd(),
    ]
    for cand in candidates:
        if (cand / 'rlinf').is_dir():
            sys.path.insert(0, str(cand))
            return
    raise RuntimeError('Could not locate repo root containing rlinf/. Please place this script under the RLinf repo.')


def _resolve_full_weights_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidates: list[Path] = []
    patterns = [
        'model_state_dict/full_weights.pt',
        'actor/model_state_dict/full_weights.pt',
        '**/model_state_dict/full_weights.pt',
    ]
    for pat in patterns:
        candidates.extend(path.glob(pat))
    candidates = sorted(set(p for p in candidates if p.is_file()))
    if not candidates:
        raise FileNotFoundError(
            f'Could not find full_weights.pt under checkpoint path: {path}. '
            'Please pass either the full_weights.pt file or a checkpoint dir containing model_state_dict/full_weights.pt.'
        )
    return candidates[0]


def _parse_model_default_from_defaults(defaults: list[Any]) -> str | None:
    for item in defaults:
        if isinstance(item, str):
            s = item.strip()
            if s.startswith('model/') and '@actor.model' in s:
                return s.split('model/', 1)[1].split('@actor.model', 1)[0]
        elif isinstance(item, dict):
            for k, v in item.items():
                if isinstance(k, str) and k.endswith('@actor.model'):
                    return str(v)
    return None


def _load_model_cfg_from_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    if OmegaConf.select(cfg, 'actor.model.model_type', default=None) is not None:
        return cfg, cfg.actor.model

    defaults = cfg.get('defaults', [])
    model_default = _parse_model_default_from_defaults(defaults)
    if model_default is None:
        raise KeyError(
            f'Config {config_path} does not contain actor.model.model_type and no model/*@actor.model default could be resolved.'
        )

    config_dir = config_path.parent
    model_default_path = config_dir / 'model' / f'{model_default}.yaml'
    if not model_default_path.is_file():
        raise FileNotFoundError(
            f'Could not resolve model default file: {model_default_path}. '
            'Please pass a fully composed yaml or place the script under examples/embodiment with the config tree intact.'
        )

    base_model_cfg = OmegaConf.load(model_default_path)
    merged_model_cfg = OmegaConf.merge(base_model_cfg, cfg.actor.model)
    cfg.actor.model = merged_model_cfg
    return cfg, cfg.actor.model


def _reshape_action_tensor(action: torch.Tensor, action_chunk: int, action_dim: int) -> torch.Tensor:
    if action.ndim == 3 and action.shape[-2:] == (action_chunk, action_dim):
        return action
    if action.ndim == 2 and action.shape[-1] == action_chunk * action_dim:
        return action.view(action.shape[0], action_chunk, action_dim)
    if action.ndim == 1 and action.shape[0] == action_chunk * action_dim:
        return action.view(1, action_chunk, action_dim)
    raise ValueError(f'Unsupported action shape {tuple(action.shape)} for action_chunk={action_chunk}, action_dim={action_dim}')


def _squeeze_batch_dim(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = _squeeze_batch_dim(v)
        elif torch.is_tensor(v):
            if v.ndim >= 2 and v.shape[1] == 1:
                out[k] = v[:, 0].contiguous()
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _load_pt_trajectory(pt_path: Path) -> dict[str, Any]:
    data = torch.load(pt_path, map_location='cpu')
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict trajectory file, got {type(data)}')
    return _squeeze_batch_dim(data)


def _build_step_mc_returns_at_chunk_start(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    step_gamma: float,
    action_valid_mask: torch.Tensor | None = None,
    dones: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Monte-Carlo Q targets with primitive-step discounting.

    Dense-reward trajectories are action-level: rewards has shape [T, H], where
    H is the number of primitive actions in one predicted chunk.  After an
    in-chunk terminal, later primitive actions are padding and should not affect
    critic targets.  Therefore this function supports `action_valid_mask` and
    computes MC targets from masked rewards only.

    Recurrence on flattened primitive steps:

        G[k] = r[k] + valid[k] * (1 - done[k]) * step_gamma * G[k + 1]

    Invalid padding steps get q_step=0 and are skipped by the recurrence.  The
    returned chunk target is G at the first primitive step of every chunk.
    """
    if rewards.shape != terminations.shape:
        raise ValueError(
            f'rewards and terminations must have the same shape, got '
            f'{tuple(rewards.shape)} vs {tuple(terminations.shape)}'
        )
    if dones is not None and dones.shape != rewards.shape:
        raise ValueError(f'dones must match rewards shape, got {tuple(dones.shape)} vs {tuple(rewards.shape)}')
    if action_valid_mask is not None and action_valid_mask.shape != rewards.shape:
        raise ValueError(
            f'action_valid_mask must match rewards shape, got '
            f'{tuple(action_valid_mask.shape)} vs {tuple(rewards.shape)}'
        )

    if rewards.ndim == 1:
        rewards_2d = rewards[:, None]
        terminations_2d = terminations[:, None]
        dones_2d = dones[:, None] if dones is not None else terminations_2d
        valid_2d = action_valid_mask[:, None] if action_valid_mask is not None else torch.ones_like(rewards_2d, dtype=torch.bool)
        squeeze_step_dim = True
    elif rewards.ndim == 2:
        rewards_2d = rewards
        terminations_2d = terminations
        dones_2d = dones if dones is not None else terminations_2d
        valid_2d = action_valid_mask if action_valid_mask is not None else torch.ones_like(rewards_2d, dtype=torch.bool)
        squeeze_step_dim = False
    else:
        raise ValueError(f'Expected rewards shape [T] or [T, H], got {tuple(rewards.shape)}')

    valid_2d = valid_2d.bool()
    flat_rewards = (rewards_2d.float() * valid_2d.float()).reshape(-1)
    flat_done = (dones_2d.bool() & valid_2d).reshape(-1)
    flat_valid = valid_2d.reshape(-1)
    q_step_flat = torch.zeros_like(flat_rewards, dtype=torch.float32)
    running = torch.zeros((), dtype=torch.float32, device=flat_rewards.device)

    for step_idx in reversed(range(flat_rewards.numel())):
        if not bool(flat_valid[step_idx].item()):
            q_step_flat[step_idx] = 0.0
            # Padding does not consume time and does not change the return.
            continue
        done = flat_done[step_idx].float()
        running = flat_rewards[step_idx] + (1.0 - done) * float(step_gamma) * running
        q_step_flat[step_idx] = running

    q_step = q_step_flat.view_as(rewards_2d)
    q_chunk_start = q_step[:, 0].contiguous()
    if squeeze_step_dim:
        q_step = q_step[:, 0]
    return q_chunk_start, q_step


def _coerce_like_bool_tensor(value: Any, like: torch.Tensor, default: bool, name: str) -> torch.Tensor:
    """Return a bool tensor with the same shape as `like`.

    The trajectory format changed during dense-reward integration, so this helper
    accepts either [T], [T, H], or absent fields and normalizes them to rewards'
    shape.  Absent `action_valid_mask` means every primitive action is valid.
    """
    if value is None:
        return torch.full_like(like, fill_value=default, dtype=torch.bool)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=like.device)
    if value.shape == like.shape:
        return value.bool()
    if value.ndim == 1 and like.ndim == 2 and value.shape[0] == like.shape[0]:
        return value[:, None].expand_as(like).bool()
    if value.ndim == 2 and like.ndim == 1 and value.shape[1] == 1 and value.shape[0] == like.shape[0]:
        return value[:, 0].bool()
    raise ValueError(f'{name} shape {tuple(value.shape)} cannot be aligned to rewards shape {tuple(like.shape)}')


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not bool(mask.any().item()):
        return torch.tensor(float('nan'), dtype=value.float().dtype, device=value.device)
    return value.float()[mask].mean()


def _masked_abs_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not bool(mask.any().item()):
        return torch.tensor(float('nan'), dtype=value.float().dtype, device=value.device)
    return value.float()[mask].abs().mean()


def _first_true_index_2d(mask: torch.Tensor) -> list[int] | None:
    idx = torch.nonzero(mask.bool(), as_tuple=False)
    if idx.numel() == 0:
        return None
    return [int(v) for v in idx[0].detach().cpu().tolist()]


def _reward_kind_stats(rewards: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, Any]:
    valid_rewards = rewards.float()[valid_mask.bool()]
    if valid_rewards.numel() == 0:
        return {
            'kind': 'empty',
            'num_valid_rewards': 0,
            'num_unique_rounded_6': 0,
            'min': None,
            'max': None,
            'mean': None,
            'sum': 0.0,
        }
    rounded = torch.round(valid_rewards.detach().cpu() * 1_000_000) / 1_000_000
    unique_vals = torch.unique(rounded)
    min_v = float(valid_rewards.min().item())
    max_v = float(valid_rewards.max().item())
    mean_v = float(valid_rewards.mean().item())
    sum_v = float(valid_rewards.sum().item())
    unique_list = [float(x.item()) for x in unique_vals[:20]]
    # Sparse success reward usually has only {0, 1}.  Relative dense reward often
    # has many unique values and may include negative progress penalties.
    is_binary_like = bool(torch.all((rounded == 0) | (rounded == 1)).item())
    kind = 'sparse_binary_like' if is_binary_like else 'dense_or_shaped'
    return {
        'kind': kind,
        'num_valid_rewards': int(valid_rewards.numel()),
        'num_unique_rounded_6': int(unique_vals.numel()),
        'unique_rounded_6_head': unique_list,
        'min': min_v,
        'max': max_v,
        'mean': mean_v,
        'sum': sum_v,
        'positive_count': int((valid_rewards > 0).sum().item()),
        'negative_count': int((valid_rewards < 0).sum().item()),
        'zero_count': int((valid_rewards == 0).sum().item()),
    }

def _tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _save_curve_plot(xs: np.ndarray, ys: list[tuple[str, np.ndarray]], title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for label, y in ys:
        plt.plot(xs, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def _save_dual_axis_curve_plot(
    xs: np.ndarray,
    left_ys: list[tuple[str, np.ndarray]],
    right_ys: list[tuple[str, np.ndarray]],
    title: str,
    xlabel: str,
    left_ylabel: str,
    right_ylabel: str,
    out_path: Path,
) -> None:
    """Save a two-axis curve plot.

    This is mainly for comparing value-like quantities, such as discounted
    remaining return, with reward-like quantities, such as per-chunk dense
    reward. They often have different scales, so drawing them on one y-axis can
    hide the reward variation.
    """
    fig, ax_left = plt.subplots(figsize=(10, 5))
    left_handles = []
    left_labels = []
    for label, y in left_ys:
        (line,) = ax_left.plot(xs, y, label=label)
        left_handles.append(line)
        left_labels.append(label)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(left_ylabel)
    ax_left.grid(True, alpha=0.3)

    ax_right = ax_left.twinx()
    right_handles = []
    right_labels = []
    for label, y in right_ys:
        (line,) = ax_right.plot(xs, y, linestyle='--', label=label)
        right_handles.append(line)
        right_labels.append(label)
    ax_right.set_ylabel(right_ylabel)

    ax_left.set_title(title)
    handles = left_handles + right_handles
    labels = left_labels + right_labels
    if handles:
        ax_left.legend(handles, labels, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def _save_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mat, aspect='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_chunk_matrix_figure(
    chunk_idx: int,
    true_model: np.ndarray,
    pred_model: np.ndarray,
    diff_model: np.ndarray,
    true_exec: np.ndarray,
    pred_exec: np.ndarray,
    diff_exec: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    mats = [
        (true_model, 'True action (model)'),
        (pred_model, 'Pred action (model)'),
        (diff_model, 'Pred-True (model)'),
        (true_exec, 'True action (exec)'),
        (pred_exec, 'Pred action (exec)'),
        (diff_exec, 'Pred-True (exec)'),
    ]
    for ax, (mat, ttl) in zip(axes.reshape(-1), mats):
        im = ax.imshow(mat, aspect='auto')
        ax.set_title(f'{ttl} | chunk={chunk_idx}')
        ax.set_xlabel('action dim')
        ax.set_ylabel('step in chunk')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _sanitize_name(s: str) -> str:
    safe = ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in s)
    return safe.strip('._') or 'trajectory'


def _trajectory_output_dir(pt_path: Path, root_out: Path, seen: dict[str, int]) -> Path:
    base = _sanitize_name(pt_path.stem)
    parent = _sanitize_name(pt_path.parent.name)
    rel = f'{parent}__{base}'
    count = seen.get(rel, 0)
    seen[rel] = count + 1
    if count > 0:
        rel = f'{rel}__{count}'
    out = root_out / rel
    out.mkdir(parents=True, exist_ok=True)
    (out / 'chunk_matrices').mkdir(exist_ok=True)
    return out


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v) != '']
    return [str(value)]


def _select_list(cfg, key: str) -> list[str]:
    value = OmegaConf.select(cfg, key, default=[])
    if value is None:
        return []
    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return _as_list(value)


def _select_optional_str(cfg, key: str, default: str | None = None) -> str | None:
    value = OmegaConf.select(cfg, key, default=default)
    if value is None:
        return None
    return str(value)


def _select_optional_int(cfg, key: str, default: int | None = None) -> int | None:
    value = OmegaConf.select(cfg, key, default=default)
    if value is None:
        return None
    return int(value)


def _resolve_pt_paths(
    pt: list[str] | None = None,
    pt_glob: list[str] | None = None,
    input_dir: list[str] | None = None,
    pt_list_file: str | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for p in pt or []:
        paths.append(Path(p))
    for pat in pt_glob or []:
        paths.extend(sorted(Path().glob(pat)))
    for d in input_dir or []:
        dpath = Path(d)
        if not dpath.is_dir():
            raise FileNotFoundError(f'input_dir not found: {dpath}')
        paths.extend(sorted(dpath.rglob('*.pt')))
    if pt_list_file:
        with open(pt_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    paths.append(Path(line))
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(Path(rp))
            seen.add(rp)
    if not uniq:
        raise ValueError(
            'No pt files resolved. Set analysis.pt, analysis.pt_glob, analysis.input_dir, '
            'or analysis.pt_list_file in the config, or pass --pt/--pt-glob/--input-dir/--pt-list-file.'
        )
    return uniq


def _resolve_discount_from_config(cfg, action_chunk: int) -> tuple[float, str, float, float]:
    input_gamma = OmegaConf.select(cfg, 'algorithm.gamma', default=None)
    if input_gamma is None:
        raise KeyError('Missing required config field: algorithm.gamma')
    input_gamma = float(input_gamma)
    if not (0.0 <= input_gamma <= 1.0):
        raise ValueError(f'algorithm.gamma should be in [0, 1], got {input_gamma}')

    gamma_mode = OmegaConf.select(cfg, 'algorithm.gamma_mode', default=None)
    if gamma_mode is None:
        raise KeyError(
            'Missing required config field: algorithm.gamma_mode. '
            'Use gamma_mode: chunk if algorithm.gamma is chunk-level, '
            'or gamma_mode: step if algorithm.gamma is primitive-step-level.'
        )
    gamma_mode = str(gamma_mode).lower().strip()
    if gamma_mode not in ('chunk', 'step'):
        raise ValueError(f'Unsupported algorithm.gamma_mode={gamma_mode!r}. Expected "chunk" or "step".')

    if gamma_mode == 'chunk':
        step_gamma = input_gamma ** (1.0 / float(action_chunk))
        effective_chunk_gamma = input_gamma
    else:
        step_gamma = input_gamma
        effective_chunk_gamma = input_gamma ** int(action_chunk)
    return input_gamma, gamma_mode, step_gamma, effective_chunk_gamma


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None or str(device_str).lower() == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(str(device_str))


def _analyze_single_trajectory(
    policy,
    step_gamma: float,
    input_gamma: float,
    gamma_mode: str,
    effective_chunk_gamma: float,
    pt_path: Path,
    out_dir: Path,
    action_chunk: int,
    action_dim: int,
    chunk_index: int | None,
    max_chunk_matrix_plots: int,
    device: torch.device,
    config_path: Path,
    weights_path: Path,
    missing: list[str],
    unexpected: list[str],
) -> dict[str, Any]:
    traj = _load_pt_trajectory(pt_path)

    curr_visual_latent = traj['curr_obs']['visual_latent'].to(device)
    curr_robot_state = traj['curr_obs']['robot_state'].to(device)
    curr_ref_action = traj['curr_obs']['ref_action'].to(device)
    true_action = _reshape_action_tensor(traj['actions'].to(device), action_chunk, action_dim)
    rewards = traj['rewards'].to(device).float()
    terminations = _coerce_like_bool_tensor(traj.get('terminations', None), rewards, default=False, name='terminations')
    truncations = _coerce_like_bool_tensor(traj.get('truncations', None), rewards, default=False, name='truncations')
    action_valid_mask = _coerce_like_bool_tensor(traj.get('action_valid_mask', None), rewards, default=True, name='action_valid_mask')
    dones = _coerce_like_bool_tensor(traj.get('dones', None), rewards, default=False, name='dones')
    if 'dones' not in traj:
        dones = terminations | truncations

    if rewards.ndim != 2:
        raise ValueError(f'Expected dense/action-level rewards shape [T, H] after batch squeeze, got {tuple(rewards.shape)}')
    if true_action.shape[:2] != rewards.shape:
        raise ValueError(
            f'action shape {tuple(true_action.shape)} is inconsistent with rewards shape {tuple(rewards.shape)}. '
            'Expected true_action [T, H, action_dim] and rewards [T, H].'
        )

    valid_step_mask = action_valid_mask.bool()
    valid_chunk_mask = valid_step_mask.any(dim=-1)
    rewards_masked = rewards * valid_step_mask.float()
    terminations_masked = terminations & valid_step_mask
    truncations_masked = truncations & valid_step_mask
    dones_masked = dones & valid_step_mask

    with torch.no_grad():
        visual_feat = policy.encode_visual(curr_visual_latent)
        pred_action, actor_aux = policy.actor_forward(
            visual_feat=visual_feat,
            robot_state=curr_robot_state,
            ref_action=curr_ref_action,
            ref_action_dropout_p=0.0,
            use_target=False,
        )
        rl_state = actor_aux['rl_state']
        q1_true, q2_true = policy.critic_forward(
            rl_state=rl_state,
            action=true_action,
            use_target=False,
            critic_visual_tokens=actor_aux.get('critic_visual_tokens', None),
            critic_robot_state=actor_aux.get('critic_robot_state', None),
            critic_ref_action=actor_aux.get('critic_ref_action', None),
        )
        q1_pred, q2_pred = policy.critic_forward(
            rl_state=rl_state,
            action=pred_action,
            use_target=False,
            critic_visual_tokens=actor_aux.get('critic_visual_tokens', None),
            critic_robot_state=actor_aux.get('critic_robot_state', None),
            critic_ref_action=actor_aux.get('critic_ref_action', None),
        )
        q_true_action = torch.minimum(q1_true, q2_true).squeeze(-1)
        q_pred_action = torch.minimum(q1_pred, q2_pred).squeeze(-1)
        true_exec = policy.postprocess_action_model_batch(true_action, curr_robot_state)
        pred_exec = policy.postprocess_action_model_batch(pred_action, curr_robot_state)

    q_mc_true, q_mc_step_all = _build_step_mc_returns_at_chunk_start(
        rewards=rewards,
        terminations=terminations,
        step_gamma=step_gamma,
        action_valid_mask=valid_step_mask,
        dones=dones,
    )
    # Undiscounted remaining episode return.  This is useful for checking whether
    # a successful relative-dense trajectory sums to about 1.0.  It is not the
    # critic target unless step_gamma == 1.0; q_mc_true above remains the
    # discounted Q target.
    q_mc_undiscounted, q_mc_undiscounted_step_all = _build_step_mc_returns_at_chunk_start(
        rewards=rewards,
        terminations=terminations,
        step_gamma=1.0,
        action_valid_mask=valid_step_mask,
        dones=dones,
    )
    q_mc_true = q_mc_true.to(device)
    q_mc_step_all = q_mc_step_all.to(device)
    q_mc_undiscounted = q_mc_undiscounted.to(device)
    q_mc_undiscounted_step_all = q_mc_undiscounted_step_all.to(device)

    model_diff = pred_action - true_action
    exec_diff = pred_exec - true_exec
    model_sqerr = model_diff.float().pow(2)
    exec_sqerr = exec_diff.float().pow(2)

    step_mask_f = valid_step_mask.float()
    step_mask_action_f = step_mask_f[..., None]
    denom_chunk = (step_mask_f.sum(dim=-1) * float(action_dim)).clamp_min(1.0)
    denom_dim = step_mask_f.sum(dim=-1).clamp_min(1.0)[:, None]
    denom_global = (step_mask_f.sum() * float(action_dim)).clamp_min(1.0)

    mse_per_chunk_model = (model_sqerr * step_mask_action_f).sum(dim=(1, 2)) / denom_chunk
    mse_per_chunk_exec = (exec_sqerr * step_mask_action_f).sum(dim=(1, 2)) / denom_chunk
    mse_per_step_model = model_sqerr.mean(dim=-1).masked_fill(~valid_step_mask, float('nan'))
    mse_per_step_exec = exec_sqerr.mean(dim=-1).masked_fill(~valid_step_mask, float('nan'))
    mse_per_dim_model = (model_sqerr * step_mask_action_f).sum(dim=1) / denom_dim
    mse_per_dim_exec = (exec_sqerr * step_mask_action_f).sum(dim=1) / denom_dim
    model_mse_mean = (model_sqerr * step_mask_action_f).sum() / denom_global
    exec_mse_mean = (exec_sqerr * step_mask_action_f).sum() / denom_global

    reward_chunk = rewards_masked.sum(dim=-1)
    cumulative_valid_return = torch.cumsum(reward_chunk, dim=0)
    episode_return_undiscounted = rewards_masked.sum()
    episode_return_raw = rewards.sum()
    valid_chunk_indices = torch.nonzero(valid_chunk_mask, as_tuple=False).flatten()
    if valid_chunk_indices.numel() > 0:
        first_valid_chunk_idx = int(valid_chunk_indices[0].item())
        episode_return_discounted_from_start = q_mc_true[first_valid_chunk_idx]
        episode_return_undiscounted_from_start = q_mc_undiscounted[first_valid_chunk_idx]
    else:
        first_valid_chunk_idx = None
        episode_return_discounted_from_start = torch.tensor(float('nan'), device=device)
        episode_return_undiscounted_from_start = torch.tensor(float('nan'), device=device)
    episode_return_total_curve = torch.full_like(
        cumulative_valid_return,
        fill_value=float(episode_return_undiscounted.detach().item()),
    )
    discounted_episode_return_total_curve = torch.full_like(
        cumulative_valid_return,
        fill_value=float(episode_return_discounted_from_start.detach().item()),
    )
    done_chunk = dones_masked.any(dim=-1)
    termination_chunk = terminations_masked.any(dim=-1)
    truncation_chunk = truncations_masked.any(dim=-1)

    xs = np.arange(true_action.shape[0])
    _save_curve_plot(
        xs=xs,
        ys=[
            ('q_mc_true(masked dense target)', _tensor_to_np(q_mc_true)),
            ('critic_q(true_action)', _tensor_to_np(q_true_action)),
            ('critic_q(pred_action)', _tensor_to_np(q_pred_action)),
        ],
        title='Q(s,a) curves by chunk (MC target uses masked primitive-step dense rewards)',
        xlabel='chunk index',
        ylabel='Q value',
        out_path=out_dir / 'q_curves.png',
    )
    _save_curve_plot(
        xs=xs,
        ys=[
            ('critic_q(true_action)-q_mc_true', _tensor_to_np(q_true_action - q_mc_true)),
            ('critic_q(pred_action)-q_mc_true', _tensor_to_np(q_pred_action - q_mc_true)),
        ],
        title='Q error vs masked step-discounted Monte Carlo target',
        xlabel='chunk index',
        ylabel='Q error',
        out_path=out_dir / 'q_error_curves.png',
    )
    _save_curve_plot(
        xs=xs,
        ys=[
            ('reward_sum_per_chunk(masked)', _tensor_to_np(reward_chunk)),
            ('cumulative_valid_return', _tensor_to_np(cumulative_valid_return)),
            ('done_chunk', _tensor_to_np(done_chunk.float())),
        ],
        title='Dense reward diagnostics by chunk',
        xlabel='chunk index',
        ylabel='reward / flag',
        out_path=out_dir / 'reward_curves.png',
    )
    _save_curve_plot(
        xs=xs,
        ys=[
            ('cumulative_episode_return(undiscounted)', _tensor_to_np(cumulative_valid_return)),
            ('remaining_episode_return(undiscounted)', _tensor_to_np(q_mc_undiscounted)),
            ('remaining_return_discounted(q_mc_true)', _tensor_to_np(q_mc_true)),
            ('total_episode_return(undiscounted)', _tensor_to_np(episode_return_total_curve)),
            ('discounted_episode_return_from_start', _tensor_to_np(discounted_episode_return_total_curve)),
        ],
        title='Episode return diagnostics by chunk',
        xlabel='chunk index',
        ylabel='return',
        out_path=out_dir / 'episode_return_curves.png',
    )

    # Direct comparison requested for dense reward debugging: remaining discounted
    # return is the critic target at chunk start, while reward_sum_per_chunk is the
    # immediate shaped reward collected by that chunk.  They should not be expected
    # to have the same shape, so we draw them with two y-axes.
    _save_dual_axis_curve_plot(
        xs=xs,
        left_ys=[('remaining_return_discounted(q_mc_true)', _tensor_to_np(q_mc_true))],
        right_ys=[('reward_sum_per_chunk(masked)', _tensor_to_np(reward_chunk))],
        title='Remaining discounted return vs immediate reward by chunk',
        xlabel='chunk index',
        left_ylabel='remaining discounted return / Q target',
        right_ylabel='masked reward sum in current chunk',
        out_path=out_dir / 'remaining_discounted_return_vs_reward_by_chunk.png',
    )

    flat_valid_step_mask = valid_step_mask.reshape(-1)
    reward_step_flat = rewards_masked.reshape(-1).masked_fill(~flat_valid_step_mask, float('nan'))
    remaining_discounted_step_flat = q_mc_step_all.reshape(-1).masked_fill(~flat_valid_step_mask, float('nan'))
    primitive_xs = np.arange(int(reward_step_flat.numel()))
    _save_dual_axis_curve_plot(
        xs=primitive_xs,
        left_ys=[('remaining_return_discounted_per_step', _tensor_to_np(remaining_discounted_step_flat))],
        right_ys=[('reward_per_primitive_step(masked)', _tensor_to_np(reward_step_flat))],
        title='Remaining discounted return vs reward by primitive step',
        xlabel='primitive step index',
        left_ylabel='remaining discounted return / step MC target',
        right_ylabel='masked primitive-step reward',
        out_path=out_dir / 'remaining_discounted_return_vs_reward_by_step.png',
    )
    _save_curve_plot(
        xs=xs,
        ys=[
            ('model_mse_per_chunk(masked)', _tensor_to_np(mse_per_chunk_model)),
            ('exec_mse_per_chunk(masked)', _tensor_to_np(mse_per_chunk_exec)),
        ],
        title='Action MSE by chunk (masked by action_valid_mask)',
        xlabel='chunk index',
        ylabel='MSE',
        out_path=out_dir / 'action_mse_curves.png',
    )
    _save_heatmap(_tensor_to_np(rewards_masked), 'Masked dense reward per primitive step', 'step in chunk', 'chunk index', out_dir / 'reward_heatmap.png')
    _save_heatmap(_tensor_to_np(valid_step_mask.float()), 'action_valid_mask', 'step in chunk', 'chunk index', out_dir / 'action_valid_mask_heatmap.png')
    _save_heatmap(_tensor_to_np(q_mc_step_all), 'MC return per primitive step', 'step in chunk', 'chunk index', out_dir / 'q_mc_step_heatmap.png')
    _save_heatmap(_tensor_to_np(mse_per_step_model), 'Model-space MSE per valid step in chunk', 'step in chunk', 'chunk index', out_dir / 'mse_step_heatmap_model.png')
    _save_heatmap(_tensor_to_np(mse_per_step_exec), 'Exec-space MSE per valid step in chunk', 'step in chunk', 'chunk index', out_dir / 'mse_step_heatmap_exec.png')
    _save_heatmap(_tensor_to_np(mse_per_dim_model), 'Model-space MSE per action dim, valid steps only', 'action dim', 'chunk index', out_dir / 'mse_dim_heatmap_model.png')
    _save_heatmap(_tensor_to_np(mse_per_dim_exec), 'Exec-space MSE per action dim, valid steps only', 'action dim', 'chunk index', out_dir / 'mse_dim_heatmap_exec.png')

    if chunk_index is not None:
        chunk_indices = [int(chunk_index)]
    else:
        total = int(true_action.shape[0])
        limit = min(total, int(max_chunk_matrix_plots))
        if limit <= 0:
            chunk_indices = []
        elif limit == total:
            chunk_indices = list(range(total))
        else:
            chunk_indices = sorted(set(np.linspace(0, total - 1, limit, dtype=int).tolist()))

    for idx in chunk_indices:
        _save_chunk_matrix_figure(
            chunk_idx=idx,
            true_model=_tensor_to_np(true_action[idx]),
            pred_model=_tensor_to_np(pred_action[idx]),
            diff_model=_tensor_to_np(model_diff[idx]),
            true_exec=_tensor_to_np(true_exec[idx]),
            pred_exec=_tensor_to_np(pred_exec[idx]),
            diff_exec=_tensor_to_np(exec_diff[idx]),
            out_path=out_dir / 'chunk_matrices' / f'chunk_{idx:03d}.png',
        )

    invalid_mask = ~valid_step_mask
    invalid_reward_abs_sum = float(rewards.float()[invalid_mask].abs().sum().item()) if bool(invalid_mask.any().item()) else 0.0
    invalid_nonzero_reward_count = int((rewards.float()[invalid_mask].abs() > 1e-8).sum().item()) if bool(invalid_mask.any().item()) else 0
    invalid_done_count = int(dones[invalid_mask].sum().item()) if bool(invalid_mask.any().item()) else 0
    invalid_terminal_count = int(terminations[invalid_mask].sum().item()) if bool(invalid_mask.any().item()) else 0
    valid_step_count = int(valid_step_mask.sum().item())
    total_step_count = int(valid_step_mask.numel())
    first_done_index = _first_true_index_2d(dones_masked)
    first_terminal_index = _first_true_index_2d(terminations_masked)
    first_truncation_index = _first_true_index_2d(truncations_masked)
    reward_stats = _reward_kind_stats(rewards, valid_step_mask)

    q_true_err = q_true_action - q_mc_true
    q_pred_err = q_pred_action - q_mc_true
    summary = {
        'pt_path': str(pt_path.resolve()),
        'weights_path': str(weights_path.resolve()),
        'config_path': str(config_path.resolve()),
        'device': str(device),
        'input_gamma': input_gamma,
        'gamma_mode': gamma_mode,
        'step_gamma': step_gamma,
        'effective_chunk_gamma': effective_chunk_gamma,
        'num_chunks': int(true_action.shape[0]),
        'action_chunk': action_chunk,
        'action_dim': action_dim,
        'trajectory_metadata': traj.get('metadata', {}),
        'missing_state_dict_keys': list(missing),
        'unexpected_state_dict_keys': list(unexpected),
        'reward_diagnostics': {
            **reward_stats,
            'valid_step_count': valid_step_count,
            'total_step_count': total_step_count,
            'valid_step_ratio': float(valid_step_count / max(total_step_count, 1)),
            'reward_sum_masked': float(rewards_masked.sum().item()),
            'reward_sum_raw': float(rewards.sum().item()),
            'episode_return_undiscounted_masked': float(episode_return_undiscounted.item()),
            'episode_return_raw': float(episode_return_raw.item()),
            'episode_return_discounted_from_start': float(episode_return_discounted_from_start.item()),
            'episode_return_undiscounted_from_start': float(episode_return_undiscounted_from_start.item()),
            'first_valid_chunk_index': first_valid_chunk_idx,
            'termination_count_valid': int(terminations_masked.sum().item()),
            'done_count_valid': int(dones_masked.sum().item()),
            'truncation_count_valid': int(truncations_masked.sum().item()),
            'first_done_index': first_done_index,
            'first_terminal_index': first_terminal_index,
            'first_truncation_index': first_truncation_index,
            'invalid_reward_abs_sum': invalid_reward_abs_sum,
            'invalid_nonzero_reward_count': invalid_nonzero_reward_count,
            'invalid_done_count': invalid_done_count,
            'invalid_terminal_count': invalid_terminal_count,
            'dones_equal_termination_or_truncation_on_valid_steps': bool(torch.equal(dones_masked, (terminations_masked | truncations_masked))),
        },
        'global_metrics': {
            # Backward-compatible aggregate keys, now computed with action_valid_mask.
            'model_mse_mean': float(model_mse_mean.item()),
            'exec_mse_mean': float(exec_mse_mean.item()),
            'q_true_action_abs_err_mean': float(_masked_abs_mean(q_true_err, valid_chunk_mask).item()),
            'q_pred_action_abs_err_mean': float(_masked_abs_mean(q_pred_err, valid_chunk_mask).item()),
            'q_true_action_mean': float(_masked_mean(q_true_action, valid_chunk_mask).item()),
            'q_pred_action_mean': float(_masked_mean(q_pred_action, valid_chunk_mask).item()),
            'q_mc_true_mean': float(_masked_mean(q_mc_true, valid_chunk_mask).item()),
            # Explicit dense/mask-aware aliases for readability.
            'model_mse_mean_masked': float(model_mse_mean.item()),
            'exec_mse_mean_masked': float(exec_mse_mean.item()),
            'q_true_action_abs_err_mean_valid_chunks': float(_masked_abs_mean(q_true_err, valid_chunk_mask).item()),
            'q_pred_action_abs_err_mean_valid_chunks': float(_masked_abs_mean(q_pred_err, valid_chunk_mask).item()),
            'q_true_action_mean_valid_chunks': float(_masked_mean(q_true_action, valid_chunk_mask).item()),
            'q_pred_action_mean_valid_chunks': float(_masked_mean(q_pred_action, valid_chunk_mask).item()),
            'q_mc_true_mean_valid_chunks': float(_masked_mean(q_mc_true, valid_chunk_mask).item()),
        },
        'per_chunk': [],
    }
    for i in range(true_action.shape[0]):
        summary['per_chunk'].append({
            'chunk_index': int(i),
            'valid_step_count': int(valid_step_mask[i].sum().item()),
            'reward_sum': float(reward_chunk[i].item()),
            'cumulative_valid_return': float(cumulative_valid_return[i].item()),
            'episode_return_total_undiscounted': float(episode_return_undiscounted.item()),
            'remaining_return_undiscounted': float(q_mc_undiscounted[i].item()),
            'remaining_return_discounted': float(q_mc_true[i].item()),
            'done_chunk': bool(done_chunk[i].item()),
            'termination_chunk': bool(termination_chunk[i].item()),
            'truncation_chunk': bool(truncation_chunk[i].item()),
            'mc_q_true': float(q_mc_true[i].item()),
            'critic_q_true_action': float(q_true_action[i].item()),
            'critic_q_pred_action': float(q_pred_action[i].item()),
            'critic_q_true_action_err': float((q_true_action[i] - q_mc_true[i]).item()),
            'critic_q_pred_action_err': float((q_pred_action[i] - q_mc_true[i]).item()),
            'model_mse_masked': float(mse_per_chunk_model[i].item()),
            'exec_mse_masked': float(mse_per_chunk_exec[i].item()),
        })

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        out_dir / 'arrays.npz',
        true_action_model=_tensor_to_np(true_action),
        pred_action_model=_tensor_to_np(pred_action),
        diff_action_model=_tensor_to_np(model_diff),
        true_action_exec=_tensor_to_np(true_exec),
        pred_action_exec=_tensor_to_np(pred_exec),
        diff_action_exec=_tensor_to_np(exec_diff),
        rewards_raw=_tensor_to_np(rewards),
        rewards_masked=_tensor_to_np(rewards_masked),
        action_valid_mask=_tensor_to_np(valid_step_mask.float()),
        terminations=_tensor_to_np(terminations.float()),
        truncations=_tensor_to_np(truncations.float()),
        dones=_tensor_to_np(dones.float()),
        q_mc_true=_tensor_to_np(q_mc_true),
        q_mc_step_all=_tensor_to_np(q_mc_step_all),
        q_mc_undiscounted=_tensor_to_np(q_mc_undiscounted),
        q_mc_undiscounted_step_all=_tensor_to_np(q_mc_undiscounted_step_all),
        episode_return_total_curve=_tensor_to_np(episode_return_total_curve),
        discounted_episode_return_total_curve=_tensor_to_np(discounted_episode_return_total_curve),
        q_true_action=_tensor_to_np(q_true_action),
        q_pred_action=_tensor_to_np(q_pred_action),
        mse_per_chunk_model=_tensor_to_np(mse_per_chunk_model),
        mse_per_chunk_exec=_tensor_to_np(mse_per_chunk_exec),
        mse_per_step_model=_tensor_to_np(mse_per_step_model),
        mse_per_step_exec=_tensor_to_np(mse_per_step_exec),
        mse_per_dim_model=_tensor_to_np(mse_per_dim_model),
        mse_per_dim_exec=_tensor_to_np(mse_per_dim_exec),
        reward_chunk=_tensor_to_np(reward_chunk),
        cumulative_valid_return=_tensor_to_np(cumulative_valid_return),
        remaining_return_undiscounted=_tensor_to_np(q_mc_undiscounted),
        remaining_return_discounted=_tensor_to_np(q_mc_true),
        reward_step=_tensor_to_np(reward_step_flat),
        remaining_return_discounted_step=_tensor_to_np(remaining_discounted_step_flat),
        primitive_step_index=primitive_xs,
        chunk_index=xs,
        done_chunk=_tensor_to_np(done_chunk.float()),
        termination_chunk=_tensor_to_np(termination_chunk.float()),
        truncation_chunk=_tensor_to_np(truncation_chunk.float()),
    )

    lines = [
        f'pt_path: {pt_path.resolve()}',
        f'weights_path: {weights_path.resolve()}',
        f'config_path: {config_path.resolve()}',
        f'device: {device}',
        f'input_gamma: {input_gamma}',
        f'gamma_mode: {gamma_mode}',
        f'step_gamma: {step_gamma}',
        f'effective_chunk_gamma: {effective_chunk_gamma}',
        f'num_chunks: {true_action.shape[0]} | action_chunk: {action_chunk} | action_dim: {action_dim}',
        f'trajectory_metadata: {json.dumps(traj.get("metadata", {}), ensure_ascii=False)}',
        '',
        'New reward-return plots:',
        '  remaining_discounted_return_vs_reward_by_chunk.png',
        '  remaining_discounted_return_vs_reward_by_step.png',
        '',
        'Reward diagnostics:',
    ]
    for k, v in summary['reward_diagnostics'].items():
        lines.append(f'  {k}: {v}')
    lines.extend(['', 'Global metrics:'])
    for k, v in summary['global_metrics'].items():
        lines.append(f'  {k}: {v:.8f}')
    lines.extend(['', 'Per-chunk summary:'])
    header = 'chunk | valid | reward_sum | cum_return | rem_return | done | term | trunc | mc_q_true | critic_q(true) | critic_q(pred) | err_true | err_pred | model_mse | exec_mse'
    lines.append(header)
    lines.append('-' * len(header))
    for row in summary['per_chunk']:
        lines.append(
            f"{row['chunk_index']:5d} | {row['valid_step_count']:5d} | {row['reward_sum']:10.4f} | {row['cumulative_valid_return']:10.4f} | "
            f"{row['remaining_return_undiscounted']:10.4f} | "
            f"{int(row['done_chunk']):4d} | {int(row['termination_chunk']):4d} | {int(row['truncation_chunk']):5d} | "
            f"{row['mc_q_true']:9.6f} | {row['critic_q_true_action']:14.6f} | {row['critic_q_pred_action']:14.6f} | "
            f"{row['critic_q_true_action_err']:8.6f} | {row['critic_q_pred_action_err']:8.6f} | "
            f"{row['model_mse_masked']:9.6f} | {row['exec_mse_masked']:9.6f}"
        )
    with open(out_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return {
        'name': out_dir.name,
        'pt_path': str(pt_path.resolve()),
        'num_chunks': int(true_action.shape[0]),
        **summary['reward_diagnostics'],
        **summary['global_metrics'],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze one or more GigaWA trajectory pt files against actor/critic checkpoints.')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config. Discount is read from algorithm.gamma and algorithm.gamma_mode.')
    parser.add_argument('--pt', type=str, action='append', default=[], help='Optional runtime override: trajectory .pt file. Repeat for multiple files.')
    parser.add_argument('--pt-glob', type=str, action='append', default=[], help='Optional runtime override: glob pattern for .pt files. Repeatable.')
    parser.add_argument('--input-dir', type=str, action='append', default=[], help='Optional runtime override: directory to recursively search for .pt files. Repeatable.')
    parser.add_argument('--pt-list-file', type=str, default=None, help='Optional runtime override: text file with one .pt path per line.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional runtime override: full_weights.pt or checkpoint dir. Default reads analysis.checkpoint.')
    parser.add_argument('--output-dir', type=str, default=None, help='Optional runtime override: output directory. Default reads analysis.output_dir.')
    parser.add_argument('--device', type=str, default=None, help='Optional runtime override: device. Default reads analysis.device or auto.')
    parser.add_argument('--chunk-index', type=int, default=None, help='Optional runtime override: chunk index to render detailed action matrices.')
    parser.add_argument('--max-chunk-matrix-plots', type=int, default=None, help='Optional runtime override: number of chunk matrix plots. Default reads analysis.max_chunk_matrix_plots or 8.')
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    _add_repo_to_path(script_path)

    from rlinf.models import get_model

    config_path = Path(args.config)
    cfg, model_cfg = _load_model_cfg_from_config(config_path)

    action_chunk = int(model_cfg.num_action_chunks)
    action_dim = int(model_cfg.action_dim)
    input_gamma, gamma_mode, step_gamma, effective_chunk_gamma = _resolve_discount_from_config(cfg, action_chunk)

    checkpoint = args.checkpoint or _select_optional_str(cfg, 'analysis.checkpoint')
    if checkpoint is None:
        raise KeyError('Missing checkpoint. Set analysis.checkpoint in config or pass --checkpoint.')
    weights_path = _resolve_full_weights_path(checkpoint)

    output_dir = args.output_dir or _select_optional_str(cfg, 'analysis.output_dir')
    if output_dir is None:
        raise KeyError('Missing output_dir. Set analysis.output_dir in config or pass --output-dir.')
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pt = args.pt if args.pt else _select_list(cfg, 'analysis.pt')
    pt_glob = args.pt_glob if args.pt_glob else _select_list(cfg, 'analysis.pt_glob')
    input_dir = args.input_dir if args.input_dir else _select_list(cfg, 'analysis.input_dir')
    pt_list_file = args.pt_list_file or _select_optional_str(cfg, 'analysis.pt_list_file')
    pt_paths = _resolve_pt_paths(pt=pt, pt_glob=pt_glob, input_dir=input_dir, pt_list_file=pt_list_file)

    device = _resolve_device(args.device or _select_optional_str(cfg, 'analysis.device', default='auto'))
    chunk_index = args.chunk_index
    if chunk_index is None:
        chunk_index = _select_optional_int(cfg, 'analysis.chunk_index', default=None)
    max_chunk_matrix_plots = args.max_chunk_matrix_plots
    if max_chunk_matrix_plots is None:
        max_chunk_matrix_plots = _select_optional_int(cfg, 'analysis.max_chunk_matrix_plots', default=8)
    max_chunk_matrix_plots = int(max_chunk_matrix_plots)

    print('[analyze_gigawa_pt_qsa_batch] discount config:')
    print(f'  algorithm.gamma={input_gamma}')
    print(f'  algorithm.gamma_mode={gamma_mode}')
    print(f'  step_gamma={step_gamma}')
    print(f'  effective_chunk_gamma={effective_chunk_gamma}')

    policy = get_model(model_cfg)
    state_dict = torch.load(weights_path, map_location='cpu')
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    policy = policy.to(device)
    policy.eval()

    seen: dict[str, int] = {}
    aggregate: list[dict[str, Any]] = []
    for idx, pt_path in enumerate(pt_paths, start=1):
        traj_out = _trajectory_output_dir(pt_path, out_root, seen)
        print(f'[analyze_gigawa_pt_qsa_batch] [{idx}/{len(pt_paths)}] analyzing {pt_path} -> {traj_out}')
        aggregate.append(
            _analyze_single_trajectory(
                policy=policy,
                step_gamma=step_gamma,
                input_gamma=input_gamma,
                gamma_mode=gamma_mode,
                effective_chunk_gamma=effective_chunk_gamma,
                pt_path=pt_path,
                out_dir=traj_out,
                action_chunk=action_chunk,
                action_dim=action_dim,
                chunk_index=chunk_index,
                max_chunk_matrix_plots=max_chunk_matrix_plots,
                device=device,
                config_path=config_path,
                weights_path=weights_path,
                missing=list(missing),
                unexpected=list(unexpected),
            )
        )

    with open(out_root / 'aggregate_summary.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'num_trajectories': len(aggregate),
                'weights_path': str(weights_path.resolve()),
                'config_path': str(config_path.resolve()),
                'input_gamma': input_gamma,
                'gamma_mode': gamma_mode,
                'step_gamma': step_gamma,
                'effective_chunk_gamma': effective_chunk_gamma,
                'items': aggregate,
            },
            f,
            indent=2,
        )

    lines = [
        f'num_trajectories: {len(aggregate)}',
        f'weights_path: {weights_path.resolve()}',
        f'config_path: {config_path.resolve()}',
        f'input_gamma: {input_gamma}',
        f'gamma_mode: {gamma_mode}',
        f'step_gamma: {step_gamma}',
        f'effective_chunk_gamma: {effective_chunk_gamma}',
        '',
    ]
    header = 'name | num_chunks | episode_return | discounted_return_start | reward_kind | model_mse_mean | exec_mse_mean | q_true_abs_err_mean | q_pred_abs_err_mean | pt_path'
    lines.append(header)
    lines.append('-' * len(header))
    for item in aggregate:
        lines.append(
            f"{item['name']} | {item['num_chunks']} | "
            f"{float(item.get('episode_return_undiscounted_masked', float('nan'))):.8f} | "
            f"{float(item.get('episode_return_discounted_from_start', float('nan'))):.8f} | "
            f"{item.get('kind', 'unknown')} | "
            f"{item['model_mse_mean']:.8f} | {item['exec_mse_mean']:.8f} | "
            f"{item['q_true_action_abs_err_mean']:.8f} | {item['q_pred_action_abs_err_mean']:.8f} | {item['pt_path']}"
        )
    with open(out_root / 'aggregate_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[analyze_gigawa_pt_qsa_batch] analyzed {len(aggregate)} trajectories')
    print(f'[analyze_gigawa_pt_qsa_batch] outputs saved under: {out_root}')


if __name__ == '__main__':
    main()
