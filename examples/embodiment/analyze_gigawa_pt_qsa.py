#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import glob
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
    candidates = []
    patterns = [
        'model_state_dict/full_weights.pt',
        'actor/model_state_dict/full_weights.pt',
        '**/model_state_dict/full_weights.pt',
    ]
    for pat in patterns:
        candidates.extend(path.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f'Could not find full_weights.pt under checkpoint path: {path}. '
            'Please pass either the full_weights.pt file or a checkpoint dir containing model_state_dict/full_weights.pt.'
        )
    candidates = sorted(set(candidates))
    return candidates[0]




def _load_model_cfg_from_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.actor.model

    if OmegaConf.select(cfg, 'actor.model.model_type', default=None) is not None:
        return cfg, model_cfg

    defaults = cfg.get('defaults', [])
    model_default = None
    for item in defaults:
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            if k.endswith('@actor.model'):
                model_default = v
                break
        if model_default is not None:
            break

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
    merged_model_cfg = OmegaConf.merge(base_model_cfg, model_cfg)
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



def _build_chunk_mc_returns(rewards: torch.Tensor, terminations: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    rewards: [T, C]
    terminations: [T, C] bool
    Return chunk-level Monte Carlo return matching current critic target semantics:
        R_t = sum(rewards[t])
        done_t = any(terminations[t])
        Q_t = R_t + (1-done_t) * gamma * Q_{t+1}
    """
    reward_chunk = rewards.sum(dim=-1).float()          # [T]
    done_chunk = terminations.any(dim=-1).float()       # [T]
    q_true = torch.zeros_like(reward_chunk)
    running = torch.zeros((), dtype=torch.float32)
    for t in reversed(range(reward_chunk.shape[0])):
        running = reward_chunk[t] + (1.0 - done_chunk[t]) * gamma * running
        q_true[t] = running
    return q_true



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



def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze one GigaWA trajectory pt file against actor/critic checkpoints.')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config used to construct the GigaWorldPolicy model.')
    parser.add_argument('--pt', type=str, required=True, help='Path to a saved trajectory .pt file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to full_weights.pt or checkpoint dir containing model_state_dict/full_weights.pt.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save plots/json/npz summary.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk-index', type=int, default=None, help='Chunk index to render detailed action matrices. Default renders all chunks.')
    parser.add_argument('--max-chunk-matrix-plots', type=int, default=8, help='If --chunk-index is not set, render up to this many chunk matrix plots.')
    parser.add_argument('--gamma', type=float, default=None, help='Override gamma. Default uses algorithm.gamma from config.')
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    _add_repo_to_path(script_path)

    from rlinf.models import get_model

    cfg, model_cfg = _load_model_cfg_from_config(Path(args.config))
    gamma = float(args.gamma if args.gamma is not None else cfg.algorithm.gamma)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'chunk_matrices').mkdir(exist_ok=True)

    weights_path = _resolve_full_weights_path(args.checkpoint)
    policy = get_model(model_cfg)
    state_dict = torch.load(weights_path, map_location='cpu')
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    device = torch.device(args.device)
    policy = policy.to(device)
    policy.eval()

    traj = _load_pt_trajectory(Path(args.pt))
    action_chunk = int(model_cfg.num_action_chunks)
    action_dim = int(model_cfg.action_dim)

    curr_visual_latent = traj['curr_obs']['visual_latent'].to(device)
    curr_robot_state = traj['curr_obs']['robot_state'].to(device)
    curr_ref_action = traj['curr_obs']['ref_action'].to(device)
    true_action = _reshape_action_tensor(traj['actions'].to(device), action_chunk, action_dim)
    rewards = traj['rewards'].to(device).float()
    terminations = traj['terminations'].to(device).bool()
    dones = traj['dones'].to(device).bool()

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
        q1_true, q2_true = policy.critic_forward(rl_state=rl_state, action=true_action, use_target=False)
        q1_pred, q2_pred = policy.critic_forward(rl_state=rl_state, action=pred_action, use_target=False)
        q_true_action = torch.minimum(q1_true, q2_true).squeeze(-1)
        q_pred_action = torch.minimum(q1_pred, q2_pred).squeeze(-1)
        true_exec = policy.postprocess_action_model_batch(true_action, curr_robot_state)
        pred_exec = policy.postprocess_action_model_batch(pred_action, curr_robot_state)

    q_mc_true = _build_chunk_mc_returns(rewards=rewards, terminations=terminations, gamma=gamma).to(device)

    model_diff = pred_action - true_action
    exec_diff = pred_exec - true_exec
    model_sqerr = model_diff.float().pow(2)
    exec_sqerr = exec_diff.float().pow(2)
    mse_per_chunk_model = model_sqerr.mean(dim=(1, 2))
    mse_per_chunk_exec = exec_sqerr.mean(dim=(1, 2))
    mse_per_step_model = model_sqerr.mean(dim=-1)    # [T, C]
    mse_per_step_exec = exec_sqerr.mean(dim=-1)      # [T, C]
    mse_per_dim_model = model_sqerr.mean(dim=1)      # [T, A]
    mse_per_dim_exec = exec_sqerr.mean(dim=1)        # [T, A]

    xs = np.arange(true_action.shape[0])
    _save_curve_plot(
        xs=xs,
        ys=[
            ('q_mc_true', _tensor_to_np(q_mc_true)),
            ('critic_q(true_action)', _tensor_to_np(q_true_action)),
            ('critic_q(pred_action)', _tensor_to_np(q_pred_action)),
        ],
        title='Q(s,a) curves by chunk',
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
        title='Q error vs Monte Carlo ground truth',
        xlabel='chunk index',
        ylabel='Q error',
        out_path=out_dir / 'q_error_curves.png',
    )
    _save_curve_plot(
        xs=xs,
        ys=[
            ('model_mse_per_chunk', _tensor_to_np(mse_per_chunk_model)),
            ('exec_mse_per_chunk', _tensor_to_np(mse_per_chunk_exec)),
        ],
        title='Action MSE by chunk',
        xlabel='chunk index',
        ylabel='MSE',
        out_path=out_dir / 'action_mse_curves.png',
    )
    _save_heatmap(_tensor_to_np(mse_per_step_model), 'Model-space MSE per step in chunk', 'step in chunk', 'chunk index', out_dir / 'mse_step_heatmap_model.png')
    _save_heatmap(_tensor_to_np(mse_per_step_exec), 'Exec-space MSE per step in chunk', 'step in chunk', 'chunk index', out_dir / 'mse_step_heatmap_exec.png')
    _save_heatmap(_tensor_to_np(mse_per_dim_model), 'Model-space MSE per action dim', 'action dim', 'chunk index', out_dir / 'mse_dim_heatmap_model.png')
    _save_heatmap(_tensor_to_np(mse_per_dim_exec), 'Exec-space MSE per action dim', 'action dim', 'chunk index', out_dir / 'mse_dim_heatmap_exec.png')

    if args.chunk_index is not None:
        chunk_indices = [int(args.chunk_index)]
    else:
        total = int(true_action.shape[0])
        limit = min(total, int(args.max_chunk_matrix_plots))
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

    # Save summary artifacts.
    summary = {
        'pt_path': str(Path(args.pt).resolve()),
        'weights_path': str(weights_path.resolve()),
        'config_path': str(Path(args.config).resolve()),
        'device': str(device),
        'gamma': gamma,
        'num_chunks': int(true_action.shape[0]),
        'action_chunk': action_chunk,
        'action_dim': action_dim,
        'trajectory_metadata': traj.get('metadata', {}),
        'missing_state_dict_keys': list(missing),
        'unexpected_state_dict_keys': list(unexpected),
        'global_metrics': {
            'model_mse_mean': float(model_sqerr.mean().item()),
            'exec_mse_mean': float(exec_sqerr.mean().item()),
            'q_true_action_abs_err_mean': float((q_true_action - q_mc_true).abs().mean().item()),
            'q_pred_action_abs_err_mean': float((q_pred_action - q_mc_true).abs().mean().item()),
            'q_true_action_mean': float(q_true_action.mean().item()),
            'q_pred_action_mean': float(q_pred_action.mean().item()),
            'q_mc_true_mean': float(q_mc_true.mean().item()),
        },
        'per_chunk': [],
    }
    reward_chunk = rewards.sum(dim=-1)
    done_chunk = terminations.any(dim=-1)
    for i in range(true_action.shape[0]):
        summary['per_chunk'].append({
            'chunk_index': int(i),
            'reward_sum': float(reward_chunk[i].item()),
            'done_chunk': bool(done_chunk[i].item()),
            'mc_q_true': float(q_mc_true[i].item()),
            'critic_q_true_action': float(q_true_action[i].item()),
            'critic_q_pred_action': float(q_pred_action[i].item()),
            'critic_q_true_action_err': float((q_true_action[i] - q_mc_true[i]).item()),
            'critic_q_pred_action_err': float((q_pred_action[i] - q_mc_true[i]).item()),
            'model_mse': float(mse_per_chunk_model[i].item()),
            'exec_mse': float(mse_per_chunk_exec[i].item()),
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
        q_mc_true=_tensor_to_np(q_mc_true),
        q_true_action=_tensor_to_np(q_true_action),
        q_pred_action=_tensor_to_np(q_pred_action),
        mse_per_chunk_model=_tensor_to_np(mse_per_chunk_model),
        mse_per_chunk_exec=_tensor_to_np(mse_per_chunk_exec),
        mse_per_step_model=_tensor_to_np(mse_per_step_model),
        mse_per_step_exec=_tensor_to_np(mse_per_step_exec),
        mse_per_dim_model=_tensor_to_np(mse_per_dim_model),
        mse_per_dim_exec=_tensor_to_np(mse_per_dim_exec),
        reward_chunk=_tensor_to_np(reward_chunk),
        done_chunk=_tensor_to_np(done_chunk.float()),
    )

    # Human-readable text summary.
    lines = []
    lines.append(f'pt_path: {Path(args.pt).resolve()}')
    lines.append(f'weights_path: {weights_path.resolve()}')
    lines.append(f'config_path: {Path(args.config).resolve()}')
    lines.append(f'device: {device}')
    lines.append(f'gamma: {gamma}')
    lines.append(f'num_chunks: {true_action.shape[0]} | action_chunk: {action_chunk} | action_dim: {action_dim}')
    lines.append(f'trajectory_metadata: {json.dumps(traj.get("metadata", {}), ensure_ascii=False)}')
    lines.append('')
    lines.append('Global metrics:')
    for k, v in summary['global_metrics'].items():
        lines.append(f'  {k}: {v:.8f}')
    lines.append('')
    lines.append('Per-chunk summary:')
    header = 'chunk | reward_sum | done | mc_q_true | critic_q(true) | critic_q(pred) | err_true | err_pred | model_mse | exec_mse'
    lines.append(header)
    lines.append('-' * len(header))
    for row in summary['per_chunk']:
        lines.append(
            f"{row['chunk_index']:5d} | {row['reward_sum']:10.4f} | {int(row['done_chunk']):4d} | "
            f"{row['mc_q_true']:9.6f} | {row['critic_q_true_action']:14.6f} | {row['critic_q_pred_action']:14.6f} | "
            f"{row['critic_q_true_action_err']:8.6f} | {row['critic_q_pred_action_err']:8.6f} | {row['model_mse']:9.6f} | {row['exec_mse']:9.6f}"
        )
    with open(out_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[analyze_gigawa_pt_qsa] Saved outputs to: {out_dir}')
    print(f'[analyze_gigawa_pt_qsa] weights_path={weights_path}')
    print(f'[analyze_gigawa_pt_qsa] model_mse_mean={summary["global_metrics"]["model_mse_mean"]:.8f} | exec_mse_mean={summary["global_metrics"]["exec_mse_mean"]:.8f}')
    print(f'[analyze_gigawa_pt_qsa] q_true_abs_err_mean={summary["global_metrics"]["q_true_action_abs_err_mean"]:.8f} | q_pred_abs_err_mean={summary["global_metrics"]["q_pred_action_abs_err_mean"]:.8f}')


if __name__ == '__main__':
    main()
