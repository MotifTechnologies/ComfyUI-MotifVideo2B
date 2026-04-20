# ops_primitives.py
#
# Subset principle: this module is NOT a drop-in replica of diffusers.
# It is "the subset of diffusers primitives that MotifVideo transformer
# actually uses, re-implemented ops-aware."
#
# Rules that follow from this principle:
#   1. Only branches/options exercised by MotifVideo transformer are implemented.
#   2. Unused branches raise NotImplementedError with an explicit message that
#      names the upstream original and the reason for exclusion.
#   3. Default argument values are kept identical to diffusers originals so that
#      call-site drift is visible (a call without explicit args hits the guard
#      immediately rather than silently using a wrong default).
#   4. Attribute names mirror diffusers exactly for state_dict key parity.
#
# Supported configurations per class:
#   TimestepEmbedding      — all branches used by transformer are supported
#   PixArtAlphaTextProjection — gelu_tanh, silu
#   AdaLayerNormZero       — norm_type="layer_norm" only
#   AdaLayerNormZeroSingle — norm_type="layer_norm" only
#   AdaLayerNormContinuous — norm_type="layer_norm" only
#   FeedForward            — activation_fn="gelu-approximate" only
#
# Usage pattern:
#   from models.transformer.ops_primitives import (
#       TimestepEmbedding, PixArtAlphaTextProjection,
#       AdaLayerNormZero, AdaLayerNormZeroSingle, AdaLayerNormContinuous,
#       FeedForward,
#   )

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default ops fallback (lazy import to avoid CUDA init at module load time)
# ---------------------------------------------------------------------------
_DEFAULT_OPS = None


def _get_default_ops():
    global _DEFAULT_OPS
    if _DEFAULT_OPS is None:
        import comfy.ops
        _DEFAULT_OPS = comfy.ops.disable_weight_init
    return _DEFAULT_OPS


# ---------------------------------------------------------------------------
# 1. TimestepEmbedding
#    Original: diffusers.models.embeddings.TimestepEmbedding
#    Attributes: linear_1, act, linear_2  (+ optional cond_proj, post_act)
# ---------------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()

        self.linear_1 = ops.Linear(
            in_channels, time_embed_dim, sample_proj_bias,
            dtype=dtype, device=device,
        )

        if cond_proj_dim is not None:
            self.cond_proj = ops.Linear(
                cond_proj_dim, in_channels, bias=False,
                dtype=dtype, device=device,
            )
        else:
            self.cond_proj = None

        self.act = _get_activation(act_fn)

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = ops.Linear(
            time_embed_dim, time_embed_dim_out, sample_proj_bias,
            dtype=dtype, device=device,
        )

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = _get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


# ---------------------------------------------------------------------------
# 2. PixArtAlphaTextProjection
#    Original: diffusers.models.embeddings.PixArtAlphaTextProjection
#    Attributes: linear_1, act_1, linear_2
# ---------------------------------------------------------------------------
class PixArtAlphaTextProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()

        if out_features is None:
            out_features = hidden_size

        self.linear_1 = ops.Linear(
            in_features=in_features, out_features=hidden_size, bias=True,
            dtype=dtype, device=device,
        )

        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

        self.linear_2 = ops.Linear(
            in_features=hidden_size, out_features=out_features, bias=True,
            dtype=dtype, device=device,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# 3. AdaLayerNormZero
#    Original: diffusers.models.normalization.AdaLayerNormZero
#    Attributes: emb (None when num_embeddings is None), silu, linear, norm
#    Only the num_embeddings=None branch is needed in this plan scope.
# ---------------------------------------------------------------------------
class AdaLayerNormZero(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int | None = None,
        norm_type: str = "layer_norm",
        bias: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()

        if num_embeddings is not None:
            raise NotImplementedError(
                "AdaLayerNormZero with num_embeddings is not supported in ops_primitives "
                "(outside plan scope). Use diffusers original for that branch."
            )
        self.emb = None

        self.silu = nn.SiLU()
        self.linear = ops.Linear(
            embedding_dim, 6 * embedding_dim, bias=bias,
            dtype=dtype, device=device,
        )
        if norm_type == "layer_norm":
            self.norm = ops.LayerNorm(
                embedding_dim, elementwise_affine=False, eps=1e-6,
                dtype=dtype, device=device,
            )
        elif norm_type == "fp32_layer_norm":
            # Upstream diffusers uses a dedicated FP32LayerNorm class that accumulates
            # in float32 regardless of input dtype.  ops_primitives does not replicate
            # that class because MotifVideo transformer only calls AdaLayerNormZero with
            # norm_type="layer_norm".  Implementing fp32_layer_norm silently as plain
            # LayerNorm would introduce a precision regression.
            raise NotImplementedError(
                "AdaLayerNormZero: norm_type='fp32_layer_norm' is outside this subset's scope. "
                "Upstream diffusers uses FP32LayerNorm (fp32 accumulation); plain LayerNorm is "
                "not equivalent.  MotifVideo transformer uses norm_type='layer_norm' only."
            )
        else:
            raise ValueError(
                f"Unsupported norm_type ({norm_type!r}). This subset supports 'layer_norm' only."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ):
        # self.emb is always None in this implementation (num_embeddings=None branch only)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ---------------------------------------------------------------------------
# 4. AdaLayerNormZeroSingle
#    Original: diffusers.models.normalization.AdaLayerNormZeroSingle
#    Attributes: silu, linear, norm
# ---------------------------------------------------------------------------
class AdaLayerNormZeroSingle(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()

        self.silu = nn.SiLU()
        self.linear = ops.Linear(
            embedding_dim, 3 * embedding_dim, bias=bias,
            dtype=dtype, device=device,
        )
        if norm_type == "layer_norm":
            self.norm = ops.LayerNorm(
                embedding_dim, elementwise_affine=False, eps=1e-6,
                dtype=dtype, device=device,
            )
        else:
            raise ValueError(
                f"Unsupported norm_type ({norm_type}). Supported: 'layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


# ---------------------------------------------------------------------------
# 5. AdaLayerNormContinuous
#    Original: diffusers.models.normalization.AdaLayerNormContinuous
#    Attributes: silu, linear, norm
#    Note: norm_type="rms_norm" is out of plan scope (NotImplementedError).
# ---------------------------------------------------------------------------
class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()

        self.silu = nn.SiLU()
        self.linear = ops.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias,
            dtype=dtype, device=device,
        )
        if norm_type == "layer_norm":
            self.norm = ops.LayerNorm(
                embedding_dim, eps=eps, elementwise_affine=elementwise_affine,
                bias=bias,
                dtype=dtype, device=device,
            )
        elif norm_type == "rms_norm":
            # Upstream diffusers uses RMSNorm (no mean subtraction, no additive bias).
            # ops_primitives does not implement it because MotifVideo transformer's
            # norm_out layer uses norm_type="layer_norm" only.
            raise NotImplementedError(
                "AdaLayerNormContinuous: norm_type='rms_norm' is outside this subset's scope. "
                "Upstream diffusers uses RMSNorm; substituting LayerNorm would be semantically "
                "wrong.  MotifVideo transformer uses norm_type='layer_norm' only."
            )
        else:
            raise ValueError(f"unknown norm_type {norm_type!r}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ---------------------------------------------------------------------------
# 6. FeedForward
#    Original: diffusers.models.attention.FeedForward
#    Attributes: net (ModuleList: [activation_module, Dropout, Linear])
#    state_dict keys: net.0.proj.weight/bias, net.2.weight/bias
#
#    Subset scope: only activation_fn="gelu-approximate" is implemented.
#    All other values (including the upstream default "geglu") raise
#    NotImplementedError.  The default value is kept as diffusers original
#    ("geglu") so that any call site omitting activation_fn gets an explicit
#    error rather than silently running a wrong activation.
#
#    Rationale for not implementing "geglu":
#      GEGLU requires a separate ops-aware proj wrapper (dim_out*2 linear),
#      which is outside this subset's scope and risks unverified state_dict
#      key changes.  MotifVideo transformer always passes
#      activation_fn="gelu-approximate" explicitly.
# ---------------------------------------------------------------------------

class _GELUApproximate(nn.Module):
    """Local ops-aware GELU wrapper matching diffusers GELU attribute layout.

    diffusers GELU stores its projection as `self.proj`.
    state_dict key: net.0.proj.weight / net.0.proj.bias
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()
        self.proj = ops.Linear(dim_in, dim_out, bias=bias, dtype=dtype, device=device)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations or _get_default_ops()

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu-approximate":
            act_fn = _GELUApproximate(
                dim, inner_dim, approximate="tanh", bias=bias,
                dtype=dtype, device=device, operations=ops,
            )
        elif activation_fn == "geglu":
            # Upstream diffusers default.  GEGLU needs a dim_out*2 projection whose
            # state_dict keys differ from the gelu-approximate layout; implementing it
            # is outside this subset's scope and risks key-parity regressions.
            # MotifVideo transformer always passes activation_fn="gelu-approximate".
            raise NotImplementedError(
                "FeedForward: activation_fn='geglu' is outside this subset's scope. "
                "Upstream diffusers default — kept as default here so omitting the arg "
                "produces an explicit error rather than a silent wrong activation. "
                "MotifVideo transformer uses activation_fn='gelu-approximate' only."
            )
        else:
            raise NotImplementedError(
                f"FeedForward: activation_fn={activation_fn!r} is not supported. "
                "This subset only implements 'gelu-approximate'. "
                "Use diffusers FeedForward for other activation functions."
            )

        self.net = nn.ModuleList([])
        # net[0]: activation module (contains proj Linear)
        self.net.append(act_fn)
        # net[1]: dropout
        self.net.append(nn.Dropout(dropout))
        # net[2]: output projection
        self.net.append(ops.Linear(inner_dim, dim_out, bias=bias, dtype=dtype, device=device))

        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_activation(act_fn: str) -> nn.Module:
    """Minimal activation factory for the activations used by TimestepEmbedding."""
    if act_fn == "silu":
        return nn.SiLU()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "gelu_tanh":
        return nn.GELU(approximate="tanh")
    elif act_fn == "mish":
        return nn.Mish()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
