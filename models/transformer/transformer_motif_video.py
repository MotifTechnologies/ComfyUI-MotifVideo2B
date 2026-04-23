# Copyright 2025 Motif Technologies. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    Timesteps,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

try:
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
except ImportError:
    TransformerBlockRegistry = None
    TransformerBlockMetadata = None

from .attention import MotifVideoAttention
from .tread_mixin import is_tread_end, is_tread_start
from .ops_primitives import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    FeedForward,
    PixArtAlphaTextProjection as LocalPixArtAlphaTextProjection,
    TimestepEmbedding as LocalTimestepEmbedding,
    _get_default_ops,
)

# Apply FSDP2 patches for activation checkpointing.
# Please checkout models.transformers.accelerate_patch for more details.
from .accelerate_patch import apply_fsdp_patches


apply_fsdp_patches()


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

NUM_TRAIN_TIMESTEPS = 1000


class MotifVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        ops = operations or _get_default_ops()

        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = ops.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class MotifVideoAdaNorm(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        ops = operations or _get_default_ops()

        out_features = out_features or 2 * in_features
        self.linear = ops.Linear(in_features, out_features, dtype=dtype, device=device)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class MotifVideoConditionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int | None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = LocalTimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim,
            dtype=dtype, device=device, operations=ops,
        )

        if isinstance(pooled_projection_dim, int):
            self.text_embedder = LocalPixArtAlphaTextProjection(
                pooled_projection_dim, embedding_dim, act_fn="silu",
                dtype=dtype, device=device, operations=ops,
            )

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timesteps_proj = self.time_proj(timestep)
        # Input dtype contract for timestep_embedder.
        # diffusers.Timesteps always emits float32 sinusoidal embeddings, so the
        # downstream Linear must receive a dtype matching our compute path.
        # - bf16 weight storage: cast input to the weight dtype (= bf16).
        # - fp8 weight storage (comfy.ops.fp8_ops.Linear): cast input to bf16 as well.
        #   fp8_ops.Linear defers weight cast to cast_bias_weight (dtype derived
        #   from input). scaled_mm supports (bf16 in, bf16 out, bias) but rejects
        #   (float32 in, float32 out, bias); passing float32 here would force
        #   fallback to F.linear and let float32 bleed into downstream
        #   activations (speed + VRAM regression, see issue #25).
        timestep_embedder_dtype = next(self.timestep_embedder.parameters()).dtype
        if timestep_embedder_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            conditioning = self.timestep_embedder(timesteps_proj.to(torch.bfloat16))
        else:
            conditioning = self.timestep_embedder(timesteps_proj.to(timestep_embedder_dtype))  # (N, D)
        if pooled_projection is not None:
            conditioning = conditioning + self.text_embedder(pooled_projection)

        token_replace_emb = None

        return conditioning, token_replace_emb


# Copied from https://github.com/guyyariv/DyPE/blob/5dd4fab99b479ee487754140d717bfb888a6afa2/flux/transformer_flux.py#L485-L486
def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    dtype = num_rotations.dtype if isinstance(num_rotations, torch.Tensor) else torch.float32
    max_pos_tensor = torch.as_tensor(max_position_embeddings, dtype=dtype)
    return (dim * torch.log(max_pos_tensor / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )  # Inverse dim formula to find number of rotations


# Copied from https://github.com/guyyariv/DyPE/blob/5dd4fab99b479ee487754140d717bfb888a6afa2/flux/transformer_flux.py#L489-L495
def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    """
    Find the correction range for NTK-by-parts interpolation.
    """
    low = torch.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = torch.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    low = torch.clamp(low, min=0)
    high = torch.clamp(high, max=dim - 1)
    return low, high  # Clamp values just in case


# Copied from https://github.com/guyyariv/DyPE/blob/5dd4fab99b479ee487754140d717bfb888a6afa2/flux/transformer_flux.py#L498-L504
def linear_ramp_mask(min_val, max_val, num_dim):
    if isinstance(min_val, torch.Tensor):
        if (min_val == max_val).all():
            max_val = max_val + 0.001
    elif min_val == max_val:
        max_val += 0.001

    linear_func = (torch.arange(num_dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


# Copied from https://github.com/guyyariv/DyPE/blob/5dd4fab99b479ee487754140d717bfb888a6afa2/flux/transformer_flux.py#L507-L511
def find_newbase_ntk(dim, base, scale):
    """
    Calculate the new base for NTK-aware scaling.
    """
    # Avoid division by zero when dim == 2 (or invalid smaller values).
    # In these degenerate cases, fall back to the original base (no NTK adjustment).
    if dim <= 2:
        return base
    return base * (scale ** (dim / (dim - 2)))


# Copied from https://github.com/guyyariv/DyPE/blob/5dd4fab99b479ee487754140d717bfb888a6afa2/flux/transformer_flux.py#L514-L652
def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
    yarn=False,
    max_pe_len=None,
    ori_max_pe_len=64,
    dype=False,
    current_timestep=1.0,
):
    """
    Precompute the frequency tensor for complex exponentials with RoPE.
    Supports YARN interpolation for vision transformers.

    Args:
        dim (`int`):
            Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`):
            Position indices for the frequency tensor. [S] or scalar.
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation.
        use_real (`bool`, *optional*, defaults to False):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for linear interpolation.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for NTK-Aware RoPE.
        repeat_interleave_real (`bool`, *optional*, defaults to True):
            If True and use_real, real and imaginary parts are interleaved with themselves to reach dim.
            Otherwise, they are concatenated.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            Data type of the frequency tensor.
        yarn (`bool`, *optional*, defaults to False):
            If True, use YARN interpolation combining NTK, linear, and base methods.
        max_pe_len (`int`, *optional*):
            Maximum position encoding length (current patches for vision models).
        ori_max_pe_len (`int`, *optional*, defaults to 64):
            Original maximum position encoding length (base patches for vision models).
        dype (`bool`, *optional*, defaults to False):
            If True, enable DyPE (Dynamic Position Encoding) with timestep-aware scaling.
        current_timestep (`float`, *optional*, defaults to 1.0):
            Current timestep for DyPE, normalized to [0, 1] where 1 is pure noise.

    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
            If use_real=True, returns tuple of (cos, sin) tensors.
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    device = pos.device

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=device)

        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        beta_0 = 1.25
        beta_1 = 0.75
        gamma_0 = 16
        gamma_1 = 2

        freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))

        freqs_linear = 1.0 / torch.einsum(
            "..., f -> ... f",
            scale,
            (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)),
        )

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(new_base, (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        if dype:
            beta_0 = torch.pow(beta_0, 2.0 * torch.pow(current_timestep, 2.0))
            beta_1 = torch.pow(beta_1, 2.0 * torch.pow(current_timestep, 2.0))

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        high = torch.clamp(high, max=dim // 2)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        if dype:
            gamma_0 = torch.pow(gamma_0, 2.0 * torch.pow(current_timestep, 2.0))
            gamma_1 = torch.pow(gamma_1, 2.0 * torch.pow(current_timestep, 2.0))

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        high = torch.clamp(high, max=dim // 2)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask

    else:
        theta_ntk = theta * ntk_factor
        freqs = 1.0 / (theta_ntk ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)) / linear_factor

    freqs = torch.outer(pos, freqs)

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()

        if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
            mscale = torch.where(scale <= 1.0, 1.0, 0.1 * torch.log(scale) + 1.0).to(scale)
            freqs_cos = freqs_cos * mscale
            freqs_sin = freqs_sin * mscale

        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class MotifVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: List[int],
        theta: float = 256.0,
        base_latent_size: int | None = None,
    ):
        """
        Rotary Positional Embedding (RoPE) for video latents.

        Args:
            patch_size (`int`):
                Spatial patch size (e.g., 2).
            patch_size_t (`int`):
                Temporal patch size (e.g., 1).
            rope_dim (`List[int]`):
                Dimensions for RoPE across [Time, Height, Width] axes.
            theta (`float`, *optional*, defaults to 256.0):
                Base frequency for rotary embeddings.
            base_latent_size (`int`, *optional*):
                The maximum spatial dimension (in latent units) seen during training,
                i.e. `training_resolution / vae_scale_factor_spatial`.
                For example, for 1280x1280 training images and a VAE spatial downscale
                (`vae_scale_factor_spatial`) of 8, this would be 160; for a downscale
                of 16, it would be 80.
        """
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta
        self.base_latent_size = base_latent_size

    @lru_cache(maxsize=8)
    def _get_base_patch_grid_size(self, base_latent_size: Optional[int], patch_size: int) -> Optional[int]:
        return base_latent_size // patch_size if base_latent_size else None

    @lru_cache(maxsize=8)
    def _get_dynamic_interpolation_scale(self, h: int, w: int, base_grid_size: int) -> float:
        return math.sqrt(h * w / (base_grid_size**2))

    def forward(self, hidden_states: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training:
            assert self.base_latent_size is None, (
                "RoPE interpolation/extrapolation logic should only be enabled for inference. "
                f"During training, base_latent_size must be None, but got {self.base_latent_size!r}."
            )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(3):
            # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
            # original implementation creates it on CPU and then moves it to device. This results in numerical
            # differences in layerwise debugging outputs, but visually it is the same.
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        base_patch_grid_size = self._get_base_patch_grid_size(self.base_latent_size, self.patch_size)
        if base_patch_grid_size is not None:
            if base_patch_grid_size <= 0:
                raise ValueError(f"base_patch_grid_size must be a positive number, got {base_patch_grid_size}.")
            dynamic_interpolation_scale = self._get_dynamic_interpolation_scale(
                rope_sizes[1], rope_sizes[2], base_patch_grid_size
            )

        normalized_timestep = torch.tensor(1.0)
        if not self.training and timestep is not None:
            normalized_timestep = timestep[0] / NUM_TRAIN_TIMESTEPS

        freqs = []
        for i in range(3):
            common_kwargs = {
                "dim": self.rope_dim[i],
                "pos": grid[i].reshape(-1),
                "theta": self.theta,
                "use_real": True,
                "freqs_dtype": torch.float64,
            }

            # Apply scaling only to spatial dimensions (Height and Width, i=1 and i=2)
            if i > 0 and base_patch_grid_size is not None and dynamic_interpolation_scale > 1.0:
                # We project the training base to the current size using the uniform scale factor.
                # max_pe_len tells the RoPE logic the "new" maximum length it's dealing with.
                max_pe_len = torch.tensor(
                    base_patch_grid_size * dynamic_interpolation_scale,
                    dtype=torch.float64,
                    device=hidden_states.device,
                )

                freq = get_1d_rotary_pos_embed(
                    **common_kwargs,
                    yarn=True,  # Enable Yet Another RoPE extensioN (YARN) for extrapolation
                    max_pe_len=max_pe_len,
                    ori_max_pe_len=base_patch_grid_size,  # The original training scale
                    dype=True,  # Enable Dynamic Position Encoding (time-aware)
                    current_timestep=normalized_timestep,
                )
            else:
                # Time dimension OR within training bounds -> Standard RoPE
                freq = get_1d_rotary_pos_embed(**common_kwargs)

            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin


class MotifVideoImageProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, dtype=None, device=None, operations=None):
        super().__init__()
        ops = operations or _get_default_ops()
        self.norm_in = ops.LayerNorm(in_features, dtype=dtype, device=device)
        self.linear_1 = ops.Linear(in_features, in_features, dtype=dtype, device=device)
        self.act_fn = nn.GELU()
        self.linear_2 = ops.Linear(in_features, hidden_size, dtype=dtype, device=device)
        self.norm_out = ops.LayerNorm(hidden_size, dtype=dtype, device=device)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class MotifVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
        norm_type: str = "layer_norm",
        enable_text_cross_attention: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()

        ops = operations or _get_default_ops()
        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        # P3.1 (#18): swap diffusers.Attention for ops-aware MotifVideoAttention.
        # ops manages dtype/device, so the post-hoc `.to(dtype, device)` cast is gone.
        self.attn = MotifVideoAttention(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm=qk_norm,
            pre_only=True,
            added_kv=False,
            eps=1e-6,
            bias=True,
            dtype=dtype,
            device=device,
            operations=ops,
        )

        self.enable_text_cross_attention = enable_text_cross_attention
        if enable_text_cross_attention:
            self.cross_attn_query_proj = ops.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
            self.cross_attn_query_norm = ops.LayerNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
            self.cross_attn_out_proj = ops.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
            nn.init.zeros_(self.cross_attn_out_proj.weight)
            nn.init.zeros_(self.cross_attn_out_proj.bias)

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type=norm_type, dtype=dtype, device=device, operations=ops)
        self.proj_mlp = ops.Linear(hidden_size, mlp_dim, dtype=dtype, device=device)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = ops.Linear(hidden_size + mlp_dim, hidden_size, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        token_replace_emb: torch.Tensor | None = None,
        first_frame_num_tokens: int | None = None,
        image_embed_seq_len: int = 0,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # Text cross-attention: Q=proj(attn_output), K/V=normed text, reuse self.attn weights
        if self.enable_text_cross_attention:
            txt_kv = norm_encoder_hidden_states[:, image_embed_seq_len:, :]
            text_mask = None
            if encoder_attention_mask is not None:
                text_mask = encoder_attention_mask[:, image_embed_seq_len:]
                text_mask = text_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)  # [B, 1, 1, L_txt]
            cross_q = self.cross_attn_query_norm(self.cross_attn_query_proj(attn_output))
            cross_output, _ = self.attn(
                hidden_states=cross_q,
                query_input=cross_q,
                key_input=txt_kv,
                value_input=txt_kv,
                attention_mask=text_mask,
                image_rotary_emb=image_rotary_emb,
            )
            attn_output = attn_output + self.cross_attn_out_proj(cross_output)

        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class MotifVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
        norm_type: str = "layer_norm",
        enable_text_cross_attention: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()

        ops = operations or _get_default_ops()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type=norm_type, dtype=dtype, device=device, operations=ops)
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type=norm_type, dtype=dtype, device=device, operations=ops)

        # P3.2 (#18): swap diffusers.Attention for ops-aware MotifVideoAttention.
        # ops manages dtype/device; post-hoc `.to(dtype, device)` is gone.
        self.attn = MotifVideoAttention(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm=qk_norm,
            pre_only=False,
            added_kv=True,
            eps=1e-6,
            bias=True,
            dtype=dtype,
            device=device,
            operations=ops,
        )

        self.enable_text_cross_attention = enable_text_cross_attention
        if enable_text_cross_attention:
            self.cross_attn_query_proj = ops.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
            self.cross_attn_query_norm = ops.LayerNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
            self.cross_attn_out_proj = ops.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
            nn.init.zeros_(self.cross_attn_out_proj.weight)
            nn.init.zeros_(self.cross_attn_out_proj.bias)

        self.norm2 = ops.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm2_context = ops.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate", dtype=dtype, device=device, operations=ops)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate", dtype=dtype, device=device, operations=ops)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        token_replace_emb: torch.Tensor | None = None,
        first_frame_num_tokens: int | None = None,
        image_embed_seq_len: int = 0,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        # Text cross-attention: Q=proj(attn_output), K/V=normed text, reuse self.attn weights
        if self.enable_text_cross_attention:
            txt_kv = norm_encoder_hidden_states[:, image_embed_seq_len:, :]
            text_mask = None
            if encoder_attention_mask is not None:
                text_mask = encoder_attention_mask[:, image_embed_seq_len:]
                text_mask = text_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)  # [B, 1, 1, L_txt]
            cross_q = self.cross_attn_query_norm(self.cross_attn_query_proj(attn_output))
            cross_output, _ = self.attn(
                hidden_states=cross_q,
                query_input=cross_q,
                key_input=txt_kv,
                value_input=txt_kv,
                attention_mask=text_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + self.cross_attn_out_proj(cross_output)

        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


if TransformerBlockRegistry is not None:
    TransformerBlockRegistry.register(
        model_class=MotifVideoTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )
    TransformerBlockRegistry.register(
        model_class=MotifVideoSingleTransformerBlock,
        metadata=TransformerBlockMetadata(
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=1,
        ),
    )


class MotifVideoTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in [MotifVideo](https://huggingface.co/motif/motifvideo).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of dual-stream blocks to use.
        num_single_layers (`int`, defaults to `40`):
            The number of layers of single-stream blocks to use.

        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the temporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
        base_latent_size (`int`, *optional*):
            The maximum spatial dimension (in latent units) seen during training.
            For example, if trained on 1280x1280 with a VAE downscale of 16, this is 80.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "MotifVideoTransformerBlock",
        "MotifVideoSingleTransformerBlock",
        "MotifVideoPatchEmbed",
    ]
    # Non-serializable constructor args that must NOT be captured into self.config.
    # dtype: torch.dtype (non-JSON), device: str|torch.device, operations: class/module.
    # Handled by ComfyUI at runtime; never part of the diffusers checkpoint config.
    ignore_for_config = ["dtype", "device", "operations"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 33,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_decoder_layers: int = 0,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        norm_type: str = "layer_norm",
        text_embed_dim: int = 4096,
        image_embed_dim: int | None = None,
        pooled_projection_dim: int | None = None,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
        base_latent_size: int | None = None,
        enable_text_cross_attention_dual: bool = False,
        enable_text_cross_attention_single: bool = False,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,  # absorb unknown keys forwarded by BaseModel (e.g. image_model, disable_unet_model_creation)
    ) -> None:
        super().__init__()

        ops = operations or _get_default_ops()
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = MotifVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim,
            dtype=dtype, device=device, operations=ops,
        )
        self.context_embedder = LocalPixArtAlphaTextProjection(
            in_features=text_embed_dim, hidden_size=inner_dim,
            dtype=dtype, device=device, operations=ops,
        )

        # First frame conditioning: Image conditioning embedders
        self.image_embed_dim = image_embed_dim
        if image_embed_dim is not None:
            # Project image embeddings from vision encoder to transformer dim
            self.image_embedder = MotifVideoImageProjection(
                in_features=image_embed_dim, hidden_size=inner_dim,
                dtype=dtype, device=device, operations=ops,
            )

        self.time_text_embed = MotifVideoConditionEmbedding(
            inner_dim, pooled_projection_dim,
            dtype=dtype, device=device, operations=ops,
        )

        # 2. RoPE
        self.rope = MotifVideoRotaryPosEmbed(
            patch_size, patch_size_t, rope_axes_dim, rope_theta, base_latent_size=base_latent_size
        )

        # Cross-attention config
        self.enable_text_cross_attention_dual = enable_text_cross_attention_dual
        self.enable_text_cross_attention_single = enable_text_cross_attention_single

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MotifVideoTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                    enable_text_cross_attention=enable_text_cross_attention_dual,
                    operations=operations,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        # Encoder blocks get cross-attention; decoder blocks do not (no text stream in decoder)
        num_encoder_single = num_single_layers - num_decoder_layers
        self.single_transformer_blocks = nn.ModuleList(
            [
                MotifVideoSingleTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                    enable_text_cross_attention=enable_text_cross_attention_single
                    if i < num_encoder_single
                    else False,
                    operations=operations,
                    dtype=dtype,
                    device=device,
                )
                for i in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6, norm_type=norm_type,
            dtype=dtype, device=device, operations=ops,
        )
        self.proj_out = ops.Linear(
            inner_dim, patch_size_t * patch_size * patch_size * out_channels,
            dtype=dtype, device=device,
        )

        # Verify cross-attention config matches actual block state.
        # Catches silent misconfiguration (e.g. checkpoint config with renamed keys).
        for i, block in enumerate(self.transformer_blocks):
            if block.enable_text_cross_attention != enable_text_cross_attention_dual:
                raise ValueError(
                    f"transformer_blocks[{i}].enable_text_cross_attention="
                    f"{block.enable_text_cross_attention}, expected {enable_text_cross_attention_dual}. "
                    f"Check checkpoint config.json key names match __init__ parameters."
                )
        num_encoder_single = num_single_layers - num_decoder_layers
        for i, block in enumerate(self.single_transformer_blocks):
            expected = enable_text_cross_attention_single if i < num_encoder_single else False
            if block.enable_text_cross_attention != expected:
                raise ValueError(
                    f"single_transformer_blocks[{i}].enable_text_cross_attention="
                    f"{block.enable_text_cross_attention}, expected {expected}. "
                    f"Check checkpoint config.json key names match __init__ parameters."
                )

        self.gradient_checkpointing = False
        self.num_decoder_layers = num_decoder_layers

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _maybe_gradient_checkpoint_block(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(block, *args)
        return block(*args)

    def _get_unwrapped_blocks(self, blocks):
        if hasattr(blocks, "_checkpoint_wrapped_module"):
            return blocks._checkpoint_wrapped_module
        elif hasattr(blocks, "module"):
            return blocks.module
        return blocks

    def _create_attention_mask(
        self,
        hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create attention mask of shape [B, 1, 1, N] where N = L + E,
        based on latent tokens (always valid) and the encoder mask.

        Args:
            hidden_states: [B, L, D]
            encoder_attention_mask: [B, E] (required)

        Returns:
            attention_mask: [B, 1, 1, N]
        """
        attention_mask = F.pad(
            encoder_attention_mask.to(torch.bool),
            (hidden_states.shape[1], 0),
            value=True,
        )
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L+E]
        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        tread_mixin: Optional[Any] = None,
        tread_disabled: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the MotifVideoTransformer3DModel.

        Args:
            hidden_states: Input latent tensor [B, C, F, H, W].
            timestep: Diffusion timesteps [B].
            encoder_hidden_states: Text conditioning [B, E, D].
            encoder_attention_mask: Mask for text conditioning [B, E].
            pooled_projections: Pooled text embeddings [B, D].
            image_embeds: Optional image embeddings from vision encoder [B, N, D].
            attention_kwargs: Additional arguments for attention processors.
            return_dict: Whether to return a Transformer2DModelOutput.
            tread_mixin: Optional TreadMixin instance for token reduction.
            tread_disabled: When True, force tread_mixin to None (dense pass).
                torch.compile specializes on this bool, producing separate graphs
                for dense vs routed without attribute toggling.

        Returns:
            Transformer2DModelOutput or tuple containing the predicted samples.
        """
        # tread_disabled=True forces a dense pass; torch.compile specializes on
        # this bool to produce separate graphs for dense vs routed without
        # attribute toggling.
        # Upstream motif-core additionally falls back to `self._inference_tread_mixin`
        # when `tread_mixin is None`; that hidden-instance-state path only makes
        # sense with motif-models' sparse-guidance pipeline, which ComfyUI does
        # not drive. We deliberately omit the fallback here to keep forward()
        # behavior a function of its explicit arguments only.
        if tread_disabled:
            tread_mixin = None

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        first_frame_num_tokens = 1 * post_patch_height * post_patch_width
        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states, timestep=timestep)
        # 2. Conditional embeddings
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # First frame conditioning: Image embeddings from vision encoder
        if image_embeds is not None:
            # image_embeds: [B, N, D_img] -> [B, N, D]
            image_embeds = self.image_embedder(image_embeds)
            encoder_hidden_states = torch.cat([image_embeds, encoder_hidden_states], dim=1)
            # Extend attention mask for image tokens
            if encoder_attention_mask is not None:
                image_mask = torch.ones(
                    image_embeds.shape[0],
                    image_embeds.shape[1],
                    device=encoder_attention_mask.device,
                    dtype=encoder_attention_mask.dtype,
                )
                encoder_attention_mask = torch.cat([image_mask, encoder_attention_mask], dim=1)

        # image_embed_seq_len: used by cross-attention blocks to slice text from encoder_hidden_states
        image_embed_seq_len = image_embeds.shape[1] if image_embeds is not None else 0

        decoder_hidden_states = hidden_states.clone()

        if encoder_attention_mask is not None:
            attention_mask = self._create_attention_mask(
                hidden_states=hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            attention_mask = None

        # TREAD state initialization: manage token reduction manually to support activation checkpointing
        tread_active = False
        current_route = None
        ids_keep = None
        x_full = None
        orig_mask = attention_mask
        orig_rope = image_rotary_emb
        latent_len = hidden_states.shape[1]

        # 4. Dual stream transformer blocks (Encoder)
        for i, block in enumerate(self.transformer_blocks):
            # Drop tokens if (1) TREAD is enabled, (2) current block is within the TREAD route.
            if is_tread_start(tread_mixin, tread_active, i):
                tread_active = True
                current_route = tread_mixin._tread_route
                # Reduce sequence length at the start of a TREAD route
                ids_keep = tread_mixin.keep_indices(hidden_states, current_route["sel"]).to(hidden_states.device)
                x_full = hidden_states.contiguous()
                hidden_states = tread_mixin.gather_tokens(hidden_states, ids_keep)
                attention_mask = tread_mixin.adjust_mask(orig_mask, latent_len, ids_keep)
                image_rotary_emb = tread_mixin.gather_rope(orig_rope, ids_keep)

            hidden_states, encoder_hidden_states = self._maybe_gradient_checkpoint_block(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                image_embed_seq_len,
                encoder_attention_mask,
            )

            if is_tread_end(tread_mixin, tread_active, i):
                # Restore full sequence length at the end of a TREAD route
                hidden_states = tread_mixin.scatter_tokens(hidden_states, ids_keep, x_full)
                tread_active = False
                current_route = None
                ids_keep = None
                x_full = None
                attention_mask = orig_mask
                image_rotary_emb = orig_rope

        # We need to unwrap the blocks because CheckpointWrapper does not support len(),
        # which is required for slicing the blocks into encoder and decoder parts.
        single_transformer_blocks = self.single_transformer_blocks

        # 5. Single stream transformer blocks (Encoder)
        num_dual = len(self.transformer_blocks)
        for i, block in enumerate(
            single_transformer_blocks[: len(single_transformer_blocks) - self.num_decoder_layers]
        ):
            # Drop tokens if (1) TREAD is enabled, (2) current block is within the TREAD route.
            abs_i = num_dual + i
            if is_tread_start(tread_mixin, tread_active, abs_i):
                tread_active = True
                current_route = tread_mixin._tread_route
                # Reduce sequence length at the start of a TREAD route
                ids_keep = tread_mixin.keep_indices(hidden_states, current_route["sel"]).to(hidden_states.device)
                x_full = hidden_states.contiguous()
                hidden_states = tread_mixin.gather_tokens(hidden_states, ids_keep)
                attention_mask = tread_mixin.adjust_mask(orig_mask, latent_len, ids_keep)
                image_rotary_emb = tread_mixin.gather_rope(orig_rope, ids_keep)

            hidden_states, encoder_hidden_states = self._maybe_gradient_checkpoint_block(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
                image_embed_seq_len,
                encoder_attention_mask,
            )

            if is_tread_end(tread_mixin, tread_active, abs_i):
                # Restore full sequence length at the end of a TREAD route
                hidden_states = tread_mixin.scatter_tokens(hidden_states, ids_keep, x_full)
                tread_active = False
                current_route = None
                ids_keep = None
                x_full = None
                attention_mask = orig_mask
                image_rotary_emb = orig_rope

        # 6. Single stream transformer blocks (Decoder)
        if self.num_decoder_layers > 0:
            encoder_hidden_states = hidden_states
            attention_mask = None

            num_single = len(single_transformer_blocks)

            for i, block in enumerate(single_transformer_blocks[-self.num_decoder_layers :]):
                abs_i = num_dual + (num_single - self.num_decoder_layers) + i
                if is_tread_start(tread_mixin, tread_active, abs_i):
                    tread_active = True
                    current_route = tread_mixin._tread_route
                    # Reduce sequence length at the start of a TREAD route
                    ids_keep = tread_mixin.keep_indices(decoder_hidden_states, current_route["sel"]).to(
                        decoder_hidden_states.device
                    )
                    x_full = encoder_hidden_states.contiguous()
                    x_t_full = decoder_hidden_states.contiguous()
                    decoder_hidden_states = tread_mixin.gather_tokens(decoder_hidden_states, ids_keep)
                    encoder_hidden_states = tread_mixin.gather_tokens(encoder_hidden_states, ids_keep)
                    attention_mask = tread_mixin.adjust_mask(orig_mask, latent_len, ids_keep)
                    image_rotary_emb = tread_mixin.gather_rope(orig_rope, ids_keep)

                decoder_hidden_states, encoder_hidden_states = self._maybe_gradient_checkpoint_block(
                    block,
                    decoder_hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

                if is_tread_end(tread_mixin, tread_active, abs_i):
                    # Restore full sequence length at the end of a TREAD route
                    decoder_hidden_states = tread_mixin.scatter_tokens(decoder_hidden_states, ids_keep, x_t_full)
                    encoder_hidden_states = tread_mixin.scatter_tokens(encoder_hidden_states, ids_keep, x_full)
                    tread_active = False
                    current_route = None
                    ids_keep = None
                    x_full = None
                    x_t_full = None
                    attention_mask = orig_mask
                    image_rotary_emb = orig_rope

            hidden_states = decoder_hidden_states

        # 7. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(
            sample=hidden_states,
        )
