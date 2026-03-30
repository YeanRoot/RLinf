from __future__ import annotations

import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import regex as re
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from .transformer_wa_casual import CasualWorldActionTransformer


logger = logging.get_logger(__name__)

if is_ftfy_available():
    import ftfy
else:
    ftfy = None


def basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class WAPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    """
    Minimal inference-only Wan World-Action pipeline for RoboTwin.

    Compared with the original script, this version keeps only the parts needed
    by RLinf eval rollout: text embedding, first-frame conditioning, and action
    chunk denoising.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: CasualWorldActionTransformer,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.action_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.scheduler.config
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 60,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        if dtype is None:
            dtype = getattr(self.text_encoder, "dtype", torch.float32)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = text_outputs.last_hidden_state.to(device=device, dtype=dtype)
        prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
        return prompt_embeds

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        action_chunk: int,
        action_dim: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 5,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        latent_shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        if latents is None:
            latents = randn_tensor(
                latent_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        action = randn_tensor(
            (batch_size, action_chunk, action_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        image = image.unsqueeze(2)  # [B, C, 1, H, W]
        video_condition = torch.cat(
            [
                image,
                image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width),
            ],
            dim=2,
        )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device=latents.device, dtype=latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device=latents.device, dtype=latents.dtype)

        latent_condition = retrieve_latents(
            self.vae.encode(video_condition), sample_mode="argmax"
        )
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        latent_condition = latent_condition.to(dtype=dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width, device=device)
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
        )
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            self.vae_scale_factor_temporal,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device, dtype=dtype)

        return latents, torch.cat([mask_lat_size, latent_condition], dim=1), action

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        action_chunk: int,
        state: torch.Tensor,
        height: int,
        width: int,
        num_frames: int = 5,
        num_inference_steps: int = 10,
        guidance_scale: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["action"],
        max_sequence_length: int = 60,
    ):
        if image is None:
            raise ValueError("image must be provided")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"height and width must be divisible by 16, got {(height, width)}")

        device = self._execution_device

        # state -> [B, 1, D]
        state = state.to(device=device, dtype=self.transformer.dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0).unsqueeze(1)
        elif state.ndim == 2:
            state = state.unsqueeze(1)
        elif state.ndim != 3:
            raise ValueError(f"state must be 1D/2D/3D tensor, got shape={tuple(state.shape)}")

        batch_size = state.shape[0]

        # prompt embeds
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("Either prompt_embeds or prompt must be provided")
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=self.transformer.dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=self.transformer.dtype)

        # 这里只保留 action scheduler
        self.action_scheduler.set_timesteps(num_inference_steps, device=device)
        action_timesteps = self.action_scheduler.timesteps

        # image preprocess
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device=device, dtype=torch.float32
        )

        # prepare latents / condition / action noise
        num_channels_latents = self.vae.config.z_dim
        latents, condition, action = self.prepare_latents(
            image=image,
            batch_size=batch_size,
            action_chunk=action_chunk,
            action_dim=state.shape[-1],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        action = action.to(device=device, dtype=self.transformer.dtype)

        # action-only: state tokens + clean ref-latent tokens + action tokens
        frame_per_tokens = condition.shape[-1] * condition.shape[-2] // 4
        num_ref_latent_tokens = frame_per_tokens
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t_action in enumerate(action_timesteps):
                latent_model_input = torch.cat([latents, condition], dim=1).to(self.transformer.dtype)

                # 只覆盖 [state tokens | clean ref-latent tokens | action tokens]
                timestep = torch.zeros(
                    batch_size,
                    num_state_tokens + num_ref_latent_tokens + num_action_tokens,
                    device=latent_model_input.device,
                    dtype=latent_model_input.dtype,
                )

                noise_t = t_action.reshape(1, 1).to(
                    device=latent_model_input.device,
                    dtype=latent_model_input.dtype,
                )

                # 仅 action tokens 标当前 diffusion timestep
                timestep[:, num_state_tokens + num_ref_latent_tokens :] = noise_t

                with self.transformer.cache_context("cond"):
                    action_pred = self.transformer(
                        ref_latents=latent_model_input[:, :, :1],
                        noisy_latents=latent_model_input[:, :, 1:],
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                        action=action,
                        state=state,
                        action_only=True,
                    )

                # 只更新 action
                action = self.action_scheduler.step(
                    action_pred, t_action, action, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {
                        name: locals()[name]
                        for name in callback_on_step_end_tensor_inputs
                        if name in locals()
                    }
                    callback_outputs = callback_on_step_end(self, i, t_action, callback_kwargs)
                    action = callback_outputs.pop("action", action)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                progress_bar.update()

        if return_dict:
            return {"images": None, "actions": action}
        return None, action

def build_wa_pipeline(cfg, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> WAPipeline:
    model_id = getattr(cfg, "wan_model_id")
    transformer_model_path = getattr(cfg, "model_path")

    if model_id is None:
        raise ValueError("cfg.wan_model_id must be provided")
    if transformer_model_path is None:
        raise ValueError("cfg.model_path must be provided")

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch_dtype,
    )
    transformer = CasualWorldActionTransformer.from_pretrained(
        transformer_model_path,
        torch_dtype=torch_dtype,
        use_safetensors=False,
    )
    pipe = WAPipeline.from_pretrained(
        model_id,
        vae=vae,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    return pipe


def _as_chw_float01(image: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={tuple(image.shape)}")

    if image.shape[0] == 3:
        tensor = image.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    # assume HWC uint8/float image
    tensor = image.permute(2, 0, 1).contiguous().float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _resize_center_crop(image: Image.Image, dst_size: Tuple[int, int]) -> Image.Image:
    dst_w, dst_h = dst_size
    src_w, src_h = image.size
    scale = max(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    image = TF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    left = max((new_w - dst_w) // 2, 0)
    top = max((new_h - dst_h) // 2, 0)
    image = TF.crop(image, top, left, dst_h, dst_w)
    return image


def build_ref_image(
    main_image: torch.Tensor | np.ndarray,
    wrist_images: Optional[torch.Tensor | np.ndarray],
    dst_size: Tuple[int, int],
) -> Image.Image:
    dst_w, dst_h = dst_size
    single_w = dst_w // 3

    main = _as_chw_float01(main_image)
    main = _resize_center_crop(
        Image.fromarray((main.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)),
        (single_w, dst_h),
    )

    if wrist_images is None:
        left = main
        right = main
    else:
        if isinstance(wrist_images, np.ndarray):
            wrist_images = torch.from_numpy(wrist_images)
        if wrist_images.ndim != 4:
            raise ValueError(
                f"Expected wrist_images with 4 dims [N,H,W,C] or [N,C,H,W], got {tuple(wrist_images.shape)}"
            )
        left_tensor = _as_chw_float01(wrist_images[0])
        right_tensor = _as_chw_float01(wrist_images[1]) if wrist_images.shape[0] > 1 else _as_chw_float01(wrist_images[0])
        left = _resize_center_crop(
            Image.fromarray((left_tensor.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)),
            (single_w, dst_h),
        )
        right = _resize_center_crop(
            Image.fromarray((right_tensor.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)),
            (single_w, dst_h),
        )

    # Match the standalone working script exactly: head | left_wrist | right_wrist
    ref = np.concatenate([np.asarray(main), np.asarray(left), np.asarray(right)], axis=1)
    return Image.fromarray(ref)


def _load_norm_stats(cfg, model_action_dim: int, device: str):
    norm_json = getattr(cfg, "norm_json", None)
    if norm_json is None:
        raise ValueError("cfg.norm_json must be provided for giga_world_policy inference")

    import json

    with open(norm_json, "r", encoding="utf-8") as f:
        stats = json.load(f)
    stats = stats["norm_stats"] if "norm_stats" in stats else stats

    def _load_stat(key1: str, key2: str, pad_value: float):
        x = torch.as_tensor(stats[key1][key2], dtype=torch.float32)
        if x.numel() >= model_action_dim:
            x = x[:model_action_dim]
        else:
            pad = torch.full((model_action_dim - x.numel(),), float(pad_value), dtype=torch.float32)
            x = torch.cat([x, pad], dim=0)
        return x.to(device)

    return {
        "state_mean": _load_stat("observation.state", "mean", 0.0),
        "state_std": _load_stat("observation.state", "std", 1.0),
        "delta_mean": _load_stat("action", "mean", 0.0),
        "delta_std": _load_stat("action", "std", 1.0),
    }


def _build_delta_mask(robotype: str, dim: int, device: str):
    name = robotype.lower()
    embed_id = 1 if "agibot" in name else 0
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
    return torch.as_tensor(base, dtype=torch.bool, device=device)


@torch.no_grad()
def run_single_observation(
    pipe: WAPipeline,
    main_image: torch.Tensor | np.ndarray,
    wrist_images: Optional[torch.Tensor | np.ndarray],
    state: torch.Tensor | np.ndarray,
    instruction: str,
    dst_size: Tuple[int, int],
    num_frames: int,
    action_chunk: int,
    state_dim: int,
    max_text_length: int,
    guidance_scale: float,
    num_inference_steps: int,
    device: str = "cuda",
    norm_stats: Optional[Dict[str, torch.Tensor]] = None,
    delta_mask: Optional[torch.Tensor] = None,
    env_action_dim: int = 14,
) -> torch.Tensor:
    ref_image = build_ref_image(main_image, wrist_images, dst_size)

    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    state_raw = state.to(device=device, dtype=torch.float32).flatten()

    model_action_dim = int(pipe.transformer.action_encoder[0].in_features)
    if state_raw.numel() >= model_action_dim:
        state_pad = state_raw[:model_action_dim]
    else:
        pad = torch.zeros(model_action_dim - state_raw.numel(), dtype=torch.float32, device=state_raw.device)
        state_pad = torch.cat([state_raw, pad], dim=0)

    if norm_stats is None:
        state_in = state_pad.unsqueeze(0)
    else:
        eps = 1e-8
        state_in = ((state_pad - norm_stats["state_mean"]) / norm_stats["state_std"].clamp_min(eps)).unsqueeze(0)

    prompt_embeds = pipe._get_t5_prompt_embeds(
        prompt=[instruction],
        max_sequence_length=max_text_length,
        device=torch.device(device),
        dtype=pipe.transformer.dtype,
    )

    _pred_imgs, pred_action = pipe(
        image=ref_image,
        action_chunk=action_chunk,
        state=state_in,
        height=dst_size[1],
        width=dst_size[0],
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        prompt_embeds=prompt_embeds,
        return_dict=False,
        max_sequence_length=max_text_length,
    )

    pred_action = pred_action[0].float()
    if norm_stats is not None:
        eps = 1e-8
        pred_delta = pred_action * norm_stats["delta_std"].clamp_min(eps) + norm_stats["delta_mean"]
        pred_action = pred_delta.clone()
        if delta_mask is not None:
            pred_action[:, delta_mask] += state_pad[delta_mask]

    return pred_action[:, :env_action_dim].cpu()
