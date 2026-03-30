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

import importlib.util
import json
import os
import sys
import types
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
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


class GigaWorldPolicy(BasePolicy, nn.Module):
    """
    RLinf inference-only wrapper for the RobotWin Giga World Action policy.

    It consumes RLinf RobotWin observations directly and mirrors the user's
    validated standalone WA inference logic.
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

        self.model_action_dim = int(self.pipe.transformer.action_encoder[0].in_features)

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

    @torch.no_grad()
    def _plan_single(self, env_obs: dict[str, Any], index: int) -> torch.Tensor:
        ref_image = self._build_ref_image(env_obs, index)
        norm_state, state_pad = self._normalize_state(env_obs["states"][index])
        prompt = self._select_prompt(env_obs, index)

        _, pred_delta_norm = self.pipe(
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

        eps = 1e-8
        pred_delta = pred_delta_norm * self.delta_std.clamp_min(eps) + self.delta_mean
        pred_action = pred_delta.clone()
        pred_action[:, :, self.delta_mask] += state_pad[self.delta_mask]
        return pred_action[0, :, : self.env_action_dim].float()

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "GigaWorldPolicy is integrated for RobotWin inference only. Training forward is not implemented."
        )

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
        action_chunks = [self._plan_single(env_obs, idx) for idx in range(batch_size)]
        actions = torch.stack(action_chunks, dim=0).to(self.device_ref)

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
        return self


def get_model(cfg: DictConfig, torch_dtype: Optional[torch.dtype] = None):
    return GigaWorldPolicy(cfg, torch_dtype=torch_dtype)
