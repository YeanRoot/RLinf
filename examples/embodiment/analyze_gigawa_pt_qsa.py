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


def _build_chunk_mc_returns(rewards: torch.Tensor, terminations: torch.Tensor, gamma: float) -> torch.Tensor:
    reward_chunk = rewards.sum(dim=-1).float()
    done_chunk = terminations.any(dim=-1).float()
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


def _resolve_pt_paths(args) -> list[Path]:
    paths: list[Path] = []
    for p in args.pt or []:
        paths.append(Path(p))
    for pat in args.pt_glob or []:
        paths.extend(sorted(Path().glob(pat)))
    for d in args.input_dir or []:
        dpath = Path(d)
        if not dpath.is_dir():
            raise FileNotFoundError(f'input_dir not found: {dpath}')
        paths.extend(sorted(dpath.rglob('*.pt')))
    if args.pt_list_file:
        with open(args.pt_list_file, 'r', encoding='utf-8') as f:
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
        raise ValueError('No pt files resolved. Use --pt, --pt-glob, --input-dir, or --pt-list-file.')
    return uniq


def _analyze_single_trajectory(
    policy,
    gamma: float,
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
    terminations = traj['terminations'].to(device).bool()

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
    mse_per_step_model = model_sqerr.mean(dim=-1)
    mse_per_step_exec = exec_sqerr.mean(dim=-1)
    mse_per_dim_model = model_sqerr.mean(dim=1)
    mse_per_dim_exec = exec_sqerr.mean(dim=1)

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

    summary = {
        'pt_path': str(pt_path.resolve()),
        'weights_path': str(weights_path.resolve()),
        'config_path': str(config_path.resolve()),
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

    lines = [
        f'pt_path: {pt_path.resolve()}',
        f'weights_path: {weights_path.resolve()}',
        f'config_path: {config_path.resolve()}',
        f'device: {device}',
        f'gamma: {gamma}',
        f'num_chunks: {true_action.shape[0]} | action_chunk: {action_chunk} | action_dim: {action_dim}',
        f'trajectory_metadata: {json.dumps(traj.get("metadata", {}), ensure_ascii=False)}',
        '',
        'Global metrics:',
    ]
    for k, v in summary['global_metrics'].items():
        lines.append(f'  {k}: {v:.8f}')
    lines.extend(['', 'Per-chunk summary:'])
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

    return {
        'name': out_dir.name,
        'pt_path': str(pt_path.resolve()),
        'num_chunks': int(true_action.shape[0]),
        **summary['global_metrics'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze one or more GigaWA trajectory pt files against actor/critic checkpoints.')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config used to construct the GigaWorldPolicy model.')
    parser.add_argument('--pt', type=str, action='append', default=[], help='Path to a saved trajectory .pt file. Repeat for multiple files.')
    parser.add_argument('--pt-glob', type=str, action='append', default=[], help='Glob pattern for .pt files. Repeatable.')
    parser.add_argument('--input-dir', type=str, action='append', default=[], help='Directory to recursively search for .pt files. Repeatable.')
    parser.add_argument('--pt-list-file', type=str, default=None, help='Text file with one .pt path per line.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to full_weights.pt or checkpoint dir containing model_state_dict/full_weights.pt.')
    parser.add_argument('--output-dir', type=str, required=True, help='Root directory to save per-trajectory results.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk-index', type=int, default=None, help='Chunk index to render detailed action matrices. Default renders selected chunks.')
    parser.add_argument('--max-chunk-matrix-plots', type=int, default=8, help='If --chunk-index is not set, render up to this many chunk matrix plots per trajectory.')
    parser.add_argument('--gamma', type=float, default=None, help='Override gamma. Default uses algorithm.gamma from config.')
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    _add_repo_to_path(script_path)

    from rlinf.models import get_model

    cfg, model_cfg = _load_model_cfg_from_config(Path(args.config))
    gamma = float(args.gamma if args.gamma is not None else cfg.algorithm.gamma)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pt_paths = _resolve_pt_paths(args)
    weights_path = _resolve_full_weights_path(args.checkpoint)

    policy = get_model(model_cfg)
    state_dict = torch.load(weights_path, map_location='cpu')
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    device = torch.device(args.device)
    policy = policy.to(device)
    policy.eval()

    action_chunk = int(model_cfg.num_action_chunks)
    action_dim = int(model_cfg.action_dim)

    seen: dict[str, int] = {}
    aggregate: list[dict[str, Any]] = []
    for idx, pt_path in enumerate(pt_paths, start=1):
        traj_out = _trajectory_output_dir(pt_path, out_root, seen)
        print(f'[analyze_gigawa_pt_qsa_batch] [{idx}/{len(pt_paths)}] analyzing {pt_path} -> {traj_out}')
        aggregate.append(
            _analyze_single_trajectory(
                policy=policy,
                gamma=gamma,
                pt_path=pt_path,
                out_dir=traj_out,
                action_chunk=action_chunk,
                action_dim=action_dim,
                chunk_index=args.chunk_index,
                max_chunk_matrix_plots=args.max_chunk_matrix_plots,
                device=device,
                config_path=Path(args.config),
                weights_path=weights_path,
                missing=list(missing),
                unexpected=list(unexpected),
            )
        )

    with open(out_root / 'aggregate_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'num_trajectories': len(aggregate), 'items': aggregate}, f, indent=2)

    lines = [f'num_trajectories: {len(aggregate)}', f'weights_path: {weights_path.resolve()}', '']
    header = 'name | num_chunks | model_mse_mean | exec_mse_mean | q_true_abs_err_mean | q_pred_abs_err_mean | pt_path'
    lines.append(header)
    lines.append('-' * len(header))
    for item in aggregate:
        lines.append(
            f"{item['name']} | {item['num_chunks']} | {item['model_mse_mean']:.8f} | {item['exec_mse_mean']:.8f} | "
            f"{item['q_true_action_abs_err_mean']:.8f} | {item['q_pred_action_abs_err_mean']:.8f} | {item['pt_path']}"
        )
    with open(out_root / 'aggregate_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[analyze_gigawa_pt_qsa_batch] analyzed {len(aggregate)} trajectories')
    print(f'[analyze_gigawa_pt_qsa_batch] outputs saved under: {out_root}')


if __name__ == '__main__':
    main()
