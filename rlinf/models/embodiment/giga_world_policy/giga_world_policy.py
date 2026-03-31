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

import copy
import importlib.util
import json
import os
import sys
import types
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from rlinf.models.embodiment.base_policy import BasePolicy


def _load_module_from_file(
    module_name: str,
    file_path: str,
    package_dir: Optional[str] = None,
):
    if module_name in sys.modules:
        return sys.modules[module_name]
    if package_dir is None:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
    else:
        spec = importlib.util.spec_from_file_location(
            module_name,
            file_path,
            submodule_search_locations=[package_dir],
        )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _setup_wa_paths(wa_root: str, diffusers_src: Optional[str] = None):
    wa_root = os.path.abspath(wa_root)
    if diffusers_src:
        diffusers_src = os.path.abspath(diffusers_src)
        if diffusers_src not in sys.path:
            sys.path.insert(0, diffusers_src)

    extra_paths = [
        os.path.join(wa_root, "giga-train", "projects", "diffusion", "world_action_model"),
        os.path.join(wa_root, "giga-train", "projects", "diffusion", "world_action_model", "scripts"),
        os.path.join(wa_root, "giga-models"),
        os.path.join(wa_root, "giga-datasets"),
        os.path.join(wa_root, "giga-train"),
    ]
    for path in extra_paths:
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.insert(0, path)


def _preload_wa_runtime(wa_root: str):
    """Load only the inference-time WA modules without training-time imports."""
    wa_root = os.path.abspath(wa_root)
    world_dir = os.path.join(
        wa_root, "giga-train", "projects", "diffusion", "world_action_model"
    )
    wa_pkg_dir = os.path.join(world_dir, "wa")
    infer_file = os.path.join(world_dir, "scripts", "inference_openloop_action_only.py")
    transformer_file = os.path.join(wa_pkg_dir, "transformer_wa_casual.py")

    if not os.path.isdir(world_dir):
        raise FileNotFoundError(f"world_action_model dir not found under wa_root: {world_dir}")
    if not os.path.isfile(transformer_file):
        raise FileNotFoundError(f"transformer_wa_casual.py not found: {transformer_file}")
    if not os.path.isfile(infer_file):
        raise FileNotFoundError(
            f"inference_openloop_action_only.py not found: {infer_file}"
        )

    if "wa" not in sys.modules:
        wa_pkg = types.ModuleType("wa")
        wa_pkg.__path__ = [wa_pkg_dir]
        sys.modules["wa"] = wa_pkg

    transformer_mod = _load_module_from_file(
        "wa.transformer_wa_casual", transformer_file
    )

    if "giga_models" not in sys.modules:
        gm_pkg = types.ModuleType("giga_models")
        gm_pkg.__path__ = [os.path.join(wa_root, "giga-models", "giga_models")]
        sys.modules["giga_models"] = gm_pkg
    if "giga_models.sockets" not in sys.modules:
        sockets_mod = types.ModuleType("giga_models.sockets")

        class _DummyRobotInferenceServer:
            pass

        class _DummyRobotInferenceClient:
            pass

        sockets_mod.RobotInferenceServer = _DummyRobotInferenceServer
        sockets_mod.RobotInferenceClient = _DummyRobotInferenceClient
        sys.modules["giga_models.sockets"] = sockets_mod

    infer_module_name = "wa_inference_openloop_action_only_min"
    if infer_module_name in sys.modules:
        infer_mod = sys.modules[infer_module_name]
    else:
        with open(infer_file, "r", encoding="utf-8") as f:
            infer_src = f.read()
        sentinel = "\nfrom giga_datasets import image_utils, video_utils"
        cut = infer_src.find(sentinel)
        if cut == -1:
            raise RuntimeError(
                "Could not find dataset-import sentinel in inference_openloop_action_only.py"
            )
        infer_src = infer_src[:cut]
        infer_mod = types.ModuleType(infer_module_name)
        infer_mod.__file__ = infer_file
        infer_mod.__package__ = ""
        sys.modules[infer_module_name] = infer_mod
        exec(compile(infer_src, infer_file, "exec"), infer_mod.__dict__)

    return transformer_mod.CasualWorldActionTransformer, infer_mod.WAPipeline


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


class VisualCompressor2D(nn.Module):
    """
    Input:  [B, 48, 12, 48]
    Output: [B, 2048]
    """
    def __init__(self, in_channels: int = 48, out_dim: int = 2048):
        super().__init__()
        if out_dim != 2048:
            raise ValueError("This first version assumes out_dim=2048.")
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),  # 12x48 -> 6x24
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),          # 6x24 -> 3x12
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),          # 3x12 -> 3x12
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 4)),                                      # -> 2x4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.flatten(1)  # [B, 256*2*4] = [B, 2048]


class TwinCritic(nn.Module):
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

    def forward(self, x: torch.Tensor, action_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qa_input = torch.cat([x, action_flat], dim=-1)
        return self.q1(qa_input), self.q2(qa_input)


class GigaWorldPolicy(BasePolicy, nn.Module):
    """
    RLinf Giga World Action policy wrapper.

    This version keeps the frozen WA backbone for reference-action generation
    and VAE-latent extraction, and adds trainable RL heads:
      - visual compressor: latent -> 2048
      - actor head: (visual feat + robot state + reference action) -> final action
      - twin critics + target networks
    """

    def __init__(self, cfg: DictConfig, torch_dtype: Optional[torch.dtype] = None):
        super().__init__()

        policy_cfg = cfg.giga_world_policy
        _setup_wa_paths(
            wa_root=policy_cfg.wa_root,
            diffusers_src=policy_cfg.get("diffusers_src", None),
        )

        from diffusers.models import AutoencoderKLWan

        CasualWorldActionTransformer, WAPipeline = _preload_wa_runtime(
            policy_cfg.wa_root
        )

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        self.cfg = cfg
        self.dtype = torch_dtype
        self.action_chunk = int(cfg.num_action_chunks)
        self.env_action_dim = int(cfg.action_dim)
        self.num_inference_steps = int(policy_cfg.num_inference_steps)
        self.guidance_scale = float(policy_cfg.guidance_scale)
        self.num_frames = int(policy_cfg.get("num_frames", 5))
        self.prompt_override = policy_cfg.get("prompt", None)
        self.robotype = str(policy_cfg.robotype)
        self.single_view_size = (
            int(policy_cfg.get("single_view_width", 256)),
            int(policy_cfg.get("single_view_height", 192)),
        )
        self.full_image_size = (
            self.single_view_size[0] * 3,
            self.single_view_size[1],
        )
        self.device_ref = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transformer_ckpt = cfg.model_path
        base_model_dir = policy_cfg.base_model_dir
        norm_json = policy_cfg.norm_json

        self.enable_latent_debug = bool(policy_cfg.get("enable_latent_debug", True))
        self.latent_debug_dir = str(
            policy_cfg.get(
                "latent_debug_dir",
                "/shared_disk/users/angen.ye/code/world_module_rollout/results/latent_debug",
            )
        )
        self._latent_debug_dumped = False
        self._latent_debug_warned = False

        # rollout switch: keep base WA by default until RL worker is ready
        self.use_rl_head_for_rollout = bool(policy_cfg.get("use_rl_head_for_rollout", False))

        # RLT-style knobs to expose to the future worker
        self.visual_feature_dim = int(policy_cfg.get("visual_feature_dim", 2048))
        self.bc_coef = float(policy_cfg.get("bc_coef", 1.0))
        self.ref_action_dropout_p = float(policy_cfg.get("ref_action_dropout_p", 0.5))
        self.target_tau = float(policy_cfg.get("target_tau", 0.005))

        vae = AutoencoderKLWan.from_pretrained(
            base_model_dir,
            subfolder="vae",
            torch_dtype=torch_dtype,
        )
        transformer = CasualWorldActionTransformer.from_pretrained(
            transformer_ckpt,
            use_safetensors=bool(policy_cfg.get("use_safetensors", False)),
        ).to(torch_dtype)

        self.pipe = WAPipeline.from_pretrained(
            base_model_dir,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
        self.pipe.to(self.device_ref)
        self._freeze_pipe_modules()

        self.model_action_dim = int(self.pipe.transformer.action_encoder[0].in_features)
        self.vae_z_dim = int(self.pipe.vae.config.z_dim)

        with open(norm_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.stats = stats["norm_stats"] if "norm_stats" in stats else stats

        self.register_buffer(
            "state_mean",
            self._load_stat("observation.state", "mean", 0.0),
            persistent=False,
        )
        self.register_buffer(
            "state_std",
            self._load_stat("observation.state", "std", 1.0),
            persistent=False,
        )
        self.register_buffer(
            "delta_mean",
            self._load_stat("action", "mean", 0.0),
            persistent=False,
        )
        self.register_buffer(
            "delta_std",
            self._load_stat("action", "std", 1.0),
            persistent=False,
        )
        self.register_buffer(
            "delta_mask",
            self._build_delta_mask(self.robotype, self.model_action_dim),
            persistent=False,
        )

        # ----------------------------
        # Trainable RL heads
        # ----------------------------
        self.visual_compressor = VisualCompressor2D(
            in_channels=self.vae_z_dim,
            out_dim=self.visual_feature_dim,
        )
        self.ref_action_flat_dim = self.action_chunk * self.env_action_dim
        self.robot_state_dim = self.model_action_dim
        self.rl_state_dim = self.visual_feature_dim + self.robot_state_dim + self.ref_action_flat_dim

        # actor outputs FINAL action chunk directly
        self.actor_head = MLP(
            input_dim=self.rl_state_dim,
            hidden_dims=[2048, 1024],
            output_dim=self.ref_action_flat_dim,
            activate_final=False,
            layer_norm=False,
        )

        # twin critics
        critic_input_dim = self.rl_state_dim + self.ref_action_flat_dim
        self.critic = TwinCritic(input_dim=critic_input_dim)

        # target networks
        self.actor_target = copy.deepcopy(self.actor_head)
        self.critic_target = copy.deepcopy(self.critic)
        self._set_requires_grad(self.actor_target, False)
        self._set_requires_grad(self.critic_target, False)

        # small init for actor output to avoid too wild first actions
        self._init_actor_output_small()

    def _freeze_pipe_modules(self):
        """
        Freeze all trainable WA backbone submodules safely.
        self.pipe is a DiffusionPipeline container, not an nn.Module.
        We must freeze its registered nn.Module children individually.
        """
        module_names = [
            "vae",
            "text_encoder",
            "image_encoder",
            "transformer",
            "transformer_2",
        ]
        for name in module_names:
            module = getattr(self.pipe, name, None)
            if module is None:
                continue
            if hasattr(module, "eval"):
                module.eval()
            if hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad = False

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _init_actor_output_small(self):
        last_linear = None
        for m in self.actor_head.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.uniform_(last_linear.weight, -1e-3, 1e-3)
            nn.init.uniform_(last_linear.bias, -1e-3, 1e-3)

    def _load_stat(self, key1: str, key2: str, pad_value: float) -> torch.Tensor:
        x = torch.as_tensor(self.stats[key1][key2], dtype=torch.float32)
        if x.numel() >= self.model_action_dim:
            x = x[: self.model_action_dim]
        else:
            pad = torch.full(
                (self.model_action_dim - x.numel(),),
                float(pad_value),
                dtype=torch.float32,
            )
            x = torch.cat([x, pad], dim=0)
        return x

    @staticmethod
    def _robotype_to_embed_id(robotype: str) -> int:
        name = robotype.lower()
        if "agibot" in name:
            return 1
        return 0

    def _build_delta_mask(self, robotype: str, dim: int) -> torch.Tensor:
        embed_id = self._robotype_to_embed_id(robotype)
        templates = {
            0: np.array(
                [True, True, True, True, True, True, False,
                 True, True, True, True, True, True, False],
                dtype=bool,
            ),
            1: np.array(
                [True, True, True, True, True, True, True, False,
                 True, True, True, True, True, True, True, False],
                dtype=bool,
            ),
        }
        base = templates[embed_id]
        if dim > len(base):
            base = np.pad(base, (0, dim - len(base)), constant_values=False)
        else:
            base = base[:dim]
        return torch.as_tensor(base, dtype=torch.bool)

    @staticmethod
    def _to_pil(img: Any) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().float()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0).numpy()
            else:
                img = img.numpy()
        img = np.asarray(img)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

    @staticmethod
    def _resize_center_crop(img: Image.Image, dst_w: int, dst_h: int) -> Image.Image:
        w, h = img.size
        if float(dst_h) / h < float(dst_w) / w:
            new_h = int(round(float(dst_w) / w * h))
            new_w = dst_w
        else:
            new_h = dst_h
            new_w = int(round(float(dst_h) / h * w))
        img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        img = TF.center_crop(img, (dst_h, dst_w))
        return img

    def _blank_view(self) -> Image.Image:
        return Image.fromarray(
            np.zeros(
                (self.single_view_size[1], self.single_view_size[0], 3),
                dtype=np.uint8,
            )
        )

    def _extract_views(self, env_obs: dict[str, Any], index: int):
        main_image = env_obs["main_images"][index]
        wrist_images = env_obs.get("wrist_images", None)

        img_high = self._resize_center_crop(
            self._to_pil(main_image),
            self.single_view_size[0],
            self.single_view_size[1],
        )

        img_left = self._blank_view()
        img_right = self._blank_view()
        if wrist_images is not None:
            sample_wrist = wrist_images[index]
            n_views = int(sample_wrist.shape[0]) if hasattr(sample_wrist, "shape") else len(sample_wrist)
            if n_views >= 1:
                img_left = self._resize_center_crop(
                    self._to_pil(sample_wrist[0]),
                    self.single_view_size[0],
                    self.single_view_size[1],
                )
            if n_views >= 2:
                img_right = self._resize_center_crop(
                    self._to_pil(sample_wrist[1]),
                    self.single_view_size[0],
                    self.single_view_size[1],
                )
        return img_high, img_left, img_right

    def _build_ref_image(self, env_obs: dict[str, Any], index: int) -> Image.Image:
        img_high, img_left, img_right = self._extract_views(env_obs, index)
        cat = np.concatenate(
            [np.asarray(img_high), np.asarray(img_left), np.asarray(img_right)],
            axis=1,
        )
        return Image.fromarray(cat)

    def _normalize_state(self, state_raw: torch.Tensor):
        state_raw = torch.as_tensor(state_raw, dtype=torch.float32).flatten()
        if state_raw.numel() >= self.model_action_dim:
            state = state_raw[: self.model_action_dim]
        else:
            pad = torch.zeros(self.model_action_dim - state_raw.numel(), dtype=torch.float32)
            state = torch.cat([state_raw, pad], dim=0)
        state = state.to(self.device_ref)
        eps = 1e-8
        norm_state = (state - self.state_mean) / self.state_std.clamp_min(eps)
        return norm_state.unsqueeze(0), state

    def _select_prompt(self, env_obs: dict[str, Any], index: int) -> str:
        if self.prompt_override:
            return str(self.prompt_override)
        task_descriptions = env_obs.get("task_descriptions", None)
        if task_descriptions is None:
            return ""
        return str(task_descriptions[index])

    @staticmethod
    def _tensor_summary(x: torch.Tensor) -> str:
        x = x.detach().float().cpu()
        return (
            f"shape={tuple(x.shape)}, dtype={x.dtype}, "
            f"mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
            f"min={x.min().item():.6f}, max={x.max().item():.6f}"
        )

    def _dump_latent_debug(self, debug_dict: dict[str, Any], prompt: str):
        if debug_dict is None or self._latent_debug_dumped:
            return

        os.makedirs(self.latent_debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()

        pt_path = os.path.join(
            self.latent_debug_dir,
            f"giga_world_policy_latent_debug_pid{pid}_{timestamp}.pt",
        )
        txt_path = os.path.join(
            self.latent_debug_dir,
            f"giga_world_policy_latent_debug_pid{pid}_{timestamp}.txt",
        )

        safe_debug = {}
        summary_lines = [
            f"prompt: {prompt}",
            f"pid: {pid}",
            f"device_ref: {self.device_ref}",
        ]

        for k, v in debug_dict.items():
            if torch.is_tensor(v):
                safe_debug[k] = v.detach().cpu()
                summary_lines.append(f"{k}: {self._tensor_summary(v)}")
            else:
                safe_debug[k] = v
                summary_lines.append(f"{k}: type={type(v)}")

        torch.save(safe_debug, pt_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        for line in summary_lines:
            print(f"[giga_world_policy] {line}", flush=True)
        print(f"[giga_world_policy] latent debug saved to: {pt_path}", flush=True)
        print(f"[giga_world_policy] latent summary saved to: {txt_path}", flush=True)

        self._latent_debug_dumped = True

    @torch.no_grad()
    def _run_pipe(
        self,
        ref_image: Image.Image,
        norm_state: torch.Tensor,
        prompt: str,
    ) -> tuple[torch.Tensor, Optional[dict[str, Any]]]:
        """
        Debug path used only once to dump latent stats.
        """
        want_debug = self.enable_latent_debug and (not self._latent_debug_dumped)

        common_kwargs = dict(
            height=self.full_image_size[1],
            width=self.full_image_size[0],
            action_chunk=self.action_chunk,
            state=norm_state,
            num_frames=self.num_frames,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            image=ref_image,
            prompt=prompt,
            return_dict=False,
        )

        if want_debug:
            try:
                outputs = self.pipe(
                    **common_kwargs,
                    return_latent_debug=True,
                )
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    _, pred_delta_norm, debug_dict = outputs[:3]
                    return pred_delta_norm, debug_dict

                if not self._latent_debug_warned:
                    print(
                        "[giga_world_policy] warning: pipe accepted debug path but did not "
                        "return a 3-tuple; falling back to normal inference.",
                        flush=True,
                    )
                    self._latent_debug_warned = True
            except TypeError as e:
                if not self._latent_debug_warned:
                    print(
                        "[giga_world_policy] warning: pipe does not yet support "
                        "`return_latent_debug`; falling back to normal inference. "
                        f"TypeError: {e}",
                        flush=True,
                    )
                    self._latent_debug_warned = True

        _, pred_delta_norm = self.pipe(**common_kwargs)
        return pred_delta_norm, None

    @torch.no_grad()
    def _extract_visual_latent_from_ref(
        self,
        ref_image: Image.Image,
        norm_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get on-device visual latent from the frozen WA VAE path without using
        the debug dict, so this is suitable for training.
        Returns latent_condition with shape [1, z_dim, T_lat, H_lat, W_lat].
        """
        image = self.pipe.video_processor.preprocess(
            ref_image,
            height=self.full_image_size[1],
            width=self.full_image_size[0],
        ).to(self.device_ref, dtype=torch.float32)

        latents_outputs = self.pipe.prepare_latents(
            image=image,
            batch_size=1,
            num_channels_latents=self.pipe.vae.config.z_dim,
            height=self.full_image_size[1],
            width=self.full_image_size[0],
            num_frames=self.num_frames,
            dtype=torch.float32,
            device=self.device_ref,
            generator=None,
            latents=None,
            last_image=None,
            action_chunk=self.action_chunk,
            action_dim=self.model_action_dim,
            return_latent_debug=False,
        )

        if self.pipe.config.expand_timesteps:
            # returns: latents, latent_condition, first_frame_mask, action
            _, latent_condition, _, _ = latents_outputs
        else:
            # returns: latents, packed_condition, action
            _, packed_condition, _ = latents_outputs
            latent_condition = packed_condition[:, -self.vae_z_dim:]

        return latent_condition.float()

    def _postprocess_pred_delta(
        self,
        pred_delta_norm: torch.Tensor,
        state_pad: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        pred_delta = pred_delta_norm * self.delta_std.clamp_min(eps) + self.delta_mean
        pred_action = pred_delta.clone()
        pred_action[:, :, self.delta_mask] += state_pad[self.delta_mask]
        return pred_action[:, :, : self.env_action_dim].float()

    @torch.no_grad()
    def _extract_frozen_backbone_single(
        self,
        env_obs: dict[str, Any],
        index: int,
    ) -> dict[str, torch.Tensor]:
        ref_image = self._build_ref_image(env_obs, index)
        norm_state, state_pad = self._normalize_state(env_obs["states"][index])
        prompt = self._select_prompt(env_obs, index)

        pred_delta_norm, debug_dict = self._run_pipe(
            ref_image=ref_image,
            norm_state=norm_state,
            prompt=prompt,
        )
        if debug_dict is not None:
            self._dump_latent_debug(debug_dict, prompt=prompt)

        ref_action = self._postprocess_pred_delta(pred_delta_norm, state_pad)[0]  # [C, A]
        visual_latent = self._extract_visual_latent_from_ref(
            ref_image=ref_image,
            norm_state=norm_state,
        )[0]  # [Z, T, H, W]

        return {
            "visual_latent": visual_latent,         # [Z, T, H, W]
            "robot_state": state_pad.float(),       # [state_dim]
            "ref_action": ref_action,               # [C, A]
        }

    @torch.no_grad()
    def extract_frozen_backbone_batch(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Frozen backbone extraction for a batch of RobotWin observations.

        Returns:
            visual_latent: [B, Z, T, H, W]
            robot_state:   [B, state_dim]
            ref_action:    [B, C, A]
        """
        batch_size = int(env_obs["states"].shape[0])
        outs = [self._extract_frozen_backbone_single(env_obs, i) for i in range(batch_size)]
        visual_latent = torch.stack([o["visual_latent"] for o in outs], dim=0).to(self.device_ref)
        robot_state = torch.stack([o["robot_state"] for o in outs], dim=0).to(self.device_ref)
        ref_action = torch.stack([o["ref_action"] for o in outs], dim=0).to(self.device_ref)
        return {
            "visual_latent": visual_latent,
            "robot_state": robot_state,
            "ref_action": ref_action,
        }

    def encode_visual(self, visual_latent: torch.Tensor) -> torch.Tensor:
        """
        visual_latent: [B, Z, T, H, W]
        compress to    [B, 2048]
        """
        if visual_latent.ndim != 5:
            raise ValueError(f"Expected visual_latent [B,Z,T,H,W], got {tuple(visual_latent.shape)}")
        x = visual_latent.float()
        if x.shape[2] == 1:
            x = x[:, :, 0]        # [B, Z, H, W]
        else:
            x = x.mean(dim=2)     # first version: temporal average if T>1
        return self.visual_compressor(x)

    def build_rl_state(
        self,
        visual_feat: torch.Tensor,
        robot_state: torch.Tensor,
        ref_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        RL state x = [visual_feat, robot_state, ref_action_flat]
        """
        ref_action_flat = ref_action.reshape(ref_action.shape[0], -1).float()
        robot_state = robot_state.float()
        visual_feat = visual_feat.float()
        return torch.cat([visual_feat, robot_state, ref_action_flat], dim=-1)

    def _apply_ref_action_dropout(
        self,
        ref_action: torch.Tensor,
        p: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Batch-wise dropout on reference action conditioning.
        If dropped, the whole reference chunk for that sample is set to zero.
        """
        if (not self.training) or p <= 0.0:
            return ref_action, None
        batch_size = ref_action.shape[0]
        keep = (torch.rand(batch_size, 1, 1, device=ref_action.device) > p).float()
        dropped = ref_action * keep
        return dropped, keep

    def actor_forward(
        self,
        visual_feat: torch.Tensor,
        robot_state: torch.Tensor,
        ref_action: torch.Tensor,
        ref_action_dropout_p: Optional[float] = None,
        use_target: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Actor outputs FINAL action chunk directly.

        Inputs:
            visual_feat: [B, 2048]
            robot_state: [B, state_dim]
            ref_action:  [B, C, A]
        Returns:
            action: [B, C, A]
            aux: dict with rl_state / dropped_ref_action / dropout_mask
        """
        if ref_action_dropout_p is None:
            ref_action_dropout_p = self.ref_action_dropout_p

        cond_ref_action, dropout_mask = self._apply_ref_action_dropout(
            ref_action, p=ref_action_dropout_p
        )
        rl_state = self.build_rl_state(
            visual_feat=visual_feat,
            robot_state=robot_state,
            ref_action=cond_ref_action,
        )

        head = self.actor_target if use_target else self.actor_head
        action_flat = head(rl_state)
        action = action_flat.view(-1, self.action_chunk, self.env_action_dim)

        aux = {
            "rl_state": rl_state,
            "cond_ref_action": cond_ref_action,
        }
        if dropout_mask is not None:
            aux["ref_dropout_mask"] = dropout_mask
        return action, aux

    def critic_forward(
        self,
        rl_state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_flat = action.reshape(action.shape[0], -1).float()
        critic = self.critic_target if use_target else self.critic
        return critic(rl_state.float(), action_flat)

    def target_actor_forward(
        self,
        visual_feat: torch.Tensor,
        robot_state: torch.Tensor,
        ref_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.actor_forward(
            visual_feat=visual_feat,
            robot_state=robot_state,
            ref_action=ref_action,
            ref_action_dropout_p=0.0,
            use_target=True,
        )

    def target_critic_forward(
        self,
        rl_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic_forward(
            rl_state=rl_state,
            action=action,
            use_target=True,
        )

    def compute_bc_loss(
        self,
        pred_action: torch.Tensor,
        ref_action: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Reserve BC loss exactly for later worker-side actor objective:
            ||a - a_ref||^2
        """
        diff = (pred_action.float() - ref_action.float()) ** 2  # [B, C, A]
        if valid_mask is not None:
            # valid_mask can be [B,C] or [B,C,1]
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(-1)
            diff = diff * valid_mask.float()

        if reduction == "mean":
            return diff.mean()
        if reduction == "sum":
            return diff.sum()
        if reduction == "none":
            return diff
        raise ValueError(f"Unknown reduction: {reduction}")

    def soft_update_targets(self, tau: Optional[float] = None):
        if tau is None:
            tau = self.target_tau

        with torch.no_grad():
            for p_tgt, p in zip(self.actor_target.parameters(), self.actor_head.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

            for p_tgt, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

    def build_training_batch(
        self,
        env_obs: dict[str, Any],
        ref_action_dropout_p: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Convenience API for future worker code.

        Returns a dict containing:
            visual_latent
            visual_feat
            robot_state
            ref_action
            rl_state
            actor_action
            q1
            q2
            bc_loss_to_ref
        """
        backbone = self.extract_frozen_backbone_batch(env_obs)
        visual_latent = backbone["visual_latent"]
        robot_state = backbone["robot_state"]
        ref_action = backbone["ref_action"]

        visual_feat = self.encode_visual(visual_latent)
        actor_action, actor_aux = self.actor_forward(
            visual_feat=visual_feat,
            robot_state=robot_state,
            ref_action=ref_action,
            ref_action_dropout_p=ref_action_dropout_p,
            use_target=False,
        )
        rl_state = actor_aux["rl_state"]
        q1, q2 = self.critic_forward(rl_state=rl_state, action=actor_action, use_target=False)
        bc_loss = self.compute_bc_loss(actor_action, ref_action)

        out = {
            "visual_latent": visual_latent,
            "visual_feat": visual_feat,
            "robot_state": robot_state,
            "ref_action": ref_action,
            "rl_state": rl_state,
            "actor_action": actor_action,
            "q1": q1,
            "q2": q2,
            "bc_loss_to_ref": bc_loss,
        }
        out.update(actor_aux)
        return out

    def default_forward(self, **kwargs):
        """
        Temporary training-capable entry point.

        Supported usage:
            default_forward(env_obs=..., mode="build_training_batch")
            default_forward(visual_feat=..., robot_state=..., ref_action=..., mode="actor")
            default_forward(rl_state=..., action=..., mode="critic")
        """
        mode = kwargs.pop("mode", "build_training_batch")

        if mode == "build_training_batch":
            env_obs = kwargs.pop("env_obs")
            ref_action_dropout_p = kwargs.pop("ref_action_dropout_p", None)
            return self.build_training_batch(
                env_obs=env_obs,
                ref_action_dropout_p=ref_action_dropout_p,
            )

        if mode == "actor":
            return self.actor_forward(
                visual_feat=kwargs["visual_feat"],
                robot_state=kwargs["robot_state"],
                ref_action=kwargs["ref_action"],
                ref_action_dropout_p=kwargs.get("ref_action_dropout_p", None),
                use_target=kwargs.get("use_target", False),
            )

        if mode == "critic":
            return self.critic_forward(
                rl_state=kwargs["rl_state"],
                action=kwargs["action"],
                use_target=kwargs.get("use_target", False),
            )

        if mode == "encode_visual":
            return self.encode_visual(kwargs["visual_latent"])

        raise ValueError(f"Unsupported mode for default_forward: {mode}")

    @torch.no_grad()
    def _plan_single(self, env_obs: dict[str, Any], index: int) -> torch.Tensor:
        """
        Base WA reference action for rollout / warmup.
        """
        ref_image = self._build_ref_image(env_obs, index)
        norm_state, state_pad = self._normalize_state(env_obs["states"][index])
        prompt = self._select_prompt(env_obs, index)

        pred_delta_norm, debug_dict = self._run_pipe(
            ref_image=ref_image,
            norm_state=norm_state,
            prompt=prompt,
        )

        if debug_dict is not None:
            self._dump_latent_debug(debug_dict, prompt=prompt)

        pred_action = self._postprocess_pred_delta(pred_delta_norm, state_pad)
        return pred_action[0]

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = False,
        **kwargs,
    ):
        del mode, compute_values, kwargs
        batch_size = int(env_obs["states"].shape[0])

        if not self.use_rl_head_for_rollout:
            action_chunks = [self._plan_single(env_obs, idx) for idx in range(batch_size)]
            actions = torch.stack(action_chunks, dim=0).to(self.device_ref)
        else:
            backbone = self.extract_frozen_backbone_batch(env_obs)
            visual_feat = self.encode_visual(backbone["visual_latent"])
            actions, _ = self.actor_forward(
                visual_feat=visual_feat,
                robot_state=backbone["robot_state"],
                ref_action=backbone["ref_action"],
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            actions = actions.to(self.device_ref)

        result = {
            "prev_logprobs": torch.zeros(
                batch_size,
                self.action_chunk,
                device=actions.device,
                dtype=torch.float32,
            ),
            "prev_values": torch.zeros(
                batch_size,
                1,
                device=actions.device,
                dtype=torch.float32,
            ),
            "forward_inputs": {
                "action": actions.reshape(batch_size, -1).contiguous(),
            },
        }
        return actions, result

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if device is not None:
            self.device_ref = torch.device(device)
        self.pipe.to(self.device_ref)
        self._freeze_pipe_modules()   # 双保险，防止后续 device 迁移后忘记冻结检查

        self.visual_compressor.to(self.device_ref)
        self.actor_head.to(self.device_ref)
        self.critic.to(self.device_ref)
        self.actor_target.to(self.device_ref)
        self.critic_target.to(self.device_ref)
        return self


def get_model(cfg: DictConfig, torch_dtype: Optional[torch.dtype] = None):
    return GigaWorldPolicy(cfg, torch_dtype=torch_dtype)