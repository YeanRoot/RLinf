from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy

from .inference_utils import build_wa_pipeline, run_single_observation, _build_delta_mask, _load_norm_stats


class GigaWorldPolicy(nn.Module, BasePolicy):
    """
    Eval-only RoboTwin policy wrapper for Wan Casual World-Action model.

    This wrapper only implements predict_action_batch(), which is sufficient for
    RLinf's eval rollout path. Training-related forward methods are intentionally
    unsupported for now.
    """

    def __init__(self, cfg, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.torch_dtype = torch_dtype

        self.action_dim = int(getattr(cfg, "action_dim", 14))
        self.num_action_chunks = int(getattr(cfg, "num_action_chunks", 48))
        self.num_frames = int(getattr(cfg, "num_frames", 5))
        self.dst_size = tuple(getattr(cfg, "dst_size", [768, 192]))
        self.max_text_length = int(getattr(cfg, "max_text_length", 60))
        self.guidance_scale = float(getattr(cfg, "guidance_scale", 0.0))
        self.num_inference_steps = int(getattr(cfg, "num_inference_steps", 10))
        self.state_dim = int(getattr(cfg, "state_dim", 14))
        self.robotype = str(getattr(cfg, "robotype", "aloha"))
        self.norm_json = getattr(cfg, "norm_json", None)

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = build_wa_pipeline(
            cfg,
            device=self.device_str,
            torch_dtype=torch_dtype,
        )

        self.model_action_dim = int(self.pipe.transformer.action_encoder[0].in_features)
        self.norm_stats = _load_norm_stats(cfg, self.model_action_dim, self.device_str)
        self.delta_mask = _build_delta_mask(self.robotype, self.model_action_dim, self.device_str)

        # Register the actual submodules to make eval()/state_dict()/to() more predictable.
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder

    def to(self, device: Optional[str | torch.device] = None):
        if device is None:
            return self
        super().to(device)
        self.device_str = str(device)
        self.pipe.to(device)
        return self

    def eval(self):
        super().eval()
        self.pipe.transformer.eval()
        self.pipe.vae.eval()
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder.eval()
        return self

    def train(self, mode: bool = True):
        # Keep PyTorch semantics, although this policy is intended for eval-only.
        super().train(mode)
        self.pipe.transformer.train(mode)
        self.pipe.vae.train(mode)
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder.train(mode)
        return self

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "GigaWorldPolicy is currently eval-only inside RLinf. "
            "Use predict_action_batch() for RoboTwin rollout inference."
        )

    @torch.no_grad()
    def predict_action_batch(self, env_obs: Optional[dict[str, Any]] = None, **kwargs):
        if env_obs is None:
            raise ValueError("env_obs must be provided for GigaWorldPolicy.predict_action_batch")

        main_images = env_obs["main_images"]
        wrist_images = env_obs.get("wrist_images", None)
        states = env_obs["states"]
        instructions = env_obs["task_descriptions"]

        if isinstance(main_images, np.ndarray):
            main_images = torch.from_numpy(main_images)
        if wrist_images is not None and isinstance(wrist_images, np.ndarray):
            wrist_images = torch.from_numpy(wrist_images)
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)

        batch_actions = []
        batch_size = int(states.shape[0])

        for i in range(batch_size):
            single_wrist = None if wrist_images is None else wrist_images[i]
            single_instruction = instructions[i]
            if isinstance(single_instruction, bytes):
                single_instruction = single_instruction.decode("utf-8")

            action_chunk = run_single_observation(
                pipe=self.pipe,
                main_image=main_images[i],
                wrist_images=single_wrist,
                state=states[i],
                instruction=str(single_instruction),
                dst_size=self.dst_size,
                num_frames=self.num_frames,
                action_chunk=self.num_action_chunks,
                state_dim=self.state_dim,
                max_text_length=self.max_text_length,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                device=self.device_str,
                norm_stats=self.norm_stats,
                delta_mask=self.delta_mask,
                env_action_dim=self.action_dim,
            )
            batch_actions.append(action_chunk)

        actions = torch.stack(batch_actions, dim=0)  # [B, horizon, action_dim]
        return actions, {}
