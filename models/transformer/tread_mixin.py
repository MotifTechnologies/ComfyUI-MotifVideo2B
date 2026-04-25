from __future__ import annotations

import logging
import weakref
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def is_tread_start(tread_mixin: TreadMixin, is_tread_activated: bool, layer_idx: int) -> bool:
    if tread_mixin is None:
        return False
    route = tread_mixin._tread_route
    return tread_mixin.should_run() and not is_tread_activated and route["start"] == layer_idx


def is_tread_end(tread_mixin: TreadMixin, is_tread_activated: bool, layer_idx: int) -> bool:
    if tread_mixin is None:
        return False
    route = tread_mixin._tread_route
    return is_tread_activated and layer_idx == route["end"]


class Router(nn.Module):
    """
    Implements token selection logic for TREAD (Token Reduction via Efficient Attention Dropping).
    """

    def __init__(self):
        super().__init__()

    def keep_indices(self, x: torch.Tensor, token_drop_ratio: float) -> torch.Tensor:
        """
        Randomly select a subset of token indices to keep based on the selection ratio.

        Args:
            x: Input tensor [B, L, D].
            token_drop_ratio: Fraction of tokens to drop (0.0 to 1.0).

        Returns:
            Sorted indices of kept tokens [B, L_keep].
        """
        B, L, _ = x.shape
        if token_drop_ratio <= 0.0:
            base = torch.arange(L, device=x.device)
            return base.unsqueeze(0).repeat(B, 1).contiguous()
        L_keep = max(1, int(round(L * (1.0 - token_drop_ratio))))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        idx = ids_shuffle[:, :L_keep].to(torch.long)
        idx = idx.sort(dim=1).values
        return idx.contiguous()


class TreadMixin(nn.Module):
    """
    TREAD implementation provider for MotifVideoTransformer3DModel.

    Paper: https://arxiv.org/abs/2501.04765

    TREAD reduces computation by processing only a subset of latent tokens through
    specified transformer layer spans. This simplified Mixin provides stateless utility
    methods to gather tokens, adjust masks, and scatter tokens back, ensuring compatibility
    with activation checkpointing.

    Logic is manually integrated into the transformer's forward loop instead of using hooks.
    """

    def __init__(self, config, transformer):
        super().__init__()
        self._tr_ref = weakref.ref(transformer)
        tread_cfg = getattr(config, "tread", None)
        self._router = Router()
        self._tread_enabled = bool(getattr(tread_cfg, "enabled", False))

        # Parse route (only one route is supported now)
        self._tread_route: Dict[str, int | float] | None = None
        if self._tread_enabled:
            routes = getattr(tread_cfg, "routes", [])
            if routes:
                # Use the first route if a list is provided
                r = routes[0] if isinstance(routes, (list, tuple)) else routes
                start = int(r["start_layer_idx"] if isinstance(r, dict) else getattr(r, "start_layer_idx"))
                end = int(r["end_layer_idx"] if isinstance(r, dict) else getattr(r, "end_layer_idx"))
                sel = float(r["selection_ratio"] if isinstance(r, dict) else getattr(r, "selection_ratio", 0.0))
                self._tread_route = {"start": start, "end": end, "sel": sel}

        if self._tread_enabled:
            assert self._tread_route is not None, "TREAD is enabled but no route was specified in the configuration."

        if self._tread_enabled and self._tread_route:
            logger.info(f"[TREAD] enabled=True, route={self._tread_route}")
        else:
            logger.info("[TREAD] disabled")

    def _tr(self) -> nn.Module:
        tr = self._tr_ref()
        if tr is None:
            raise RuntimeError("Transformer reference is missing.")
        return tr

    def should_run(self):
        """Check if TREAD logic should be applied in the current mode (train/eval)."""
        return self._tread_enabled and self.training and self._tread_route is not None

    def rebind_transformer(self, transformer: nn.Module):
        """Update the weak reference to the transformer model."""
        self._tr_ref = weakref.ref(transformer)

    def keep_indices(self, x: torch.Tensor, selection_ratio: float) -> torch.Tensor:
        """Select token indices to keep based on the current sequence and ratio."""
        return self._router.keep_indices(x, selection_ratio)

    @staticmethod
    def gather_tokens(x: torch.Tensor, ids_keep: torch.Tensor):
        """Gather tokens from a full sequence based on kept indices."""
        return x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1))).contiguous()

    @staticmethod
    def scatter_tokens(x_sub: torch.Tensor, ids_keep: torch.Tensor, base_full: torch.Tensor):
        """Restore gathered tokens back to their original positions in the full sequence."""
        return base_full.scatter(1, ids_keep.unsqueeze(-1).expand(-1, -1, base_full.size(-1)), x_sub).contiguous()

    @staticmethod
    def adjust_mask(attn_mask: torch.Tensor, latent_len: int, ids_keep: torch.Tensor):
        """
        Gather attention mask elements corresponding to the kept latent tokens.
        Preserves encoder tokens (cross-attention) while reducing latent dimensions.
        """
        if attn_mask is None:
            return None
        mask_latent = attn_mask[..., :latent_len]  # [B,1,1,L]
        mask_enc = attn_mask[..., latent_len:]  # [B,1,1,E]

        idx = ids_keep[:, None, None, :]
        kept_latent = torch.gather(mask_latent, dim=-1, index=idx)

        return torch.cat([kept_latent, mask_enc], dim=-1)

    @staticmethod
    def gather_rope(rope, ids_keep):
        """
        Gather RoPE embeddings to match the reduced latent sequence length.
        Supports both shared [L, Dh] and batched [B, 1, L, Dh] RoPE formats.
        """
        if rope is None:
            return None
        cos, sin = rope
        if cos.dim() == 2:  # [L,Dh] -> [B,1,Lk,Dh]
            B, Lk = ids_keep.shape
            L, Dh = cos.shape
            idx = ids_keep.unsqueeze(-1).expand(-1, -1, Dh)
            cos_sel = cos.unsqueeze(0).expand(B, -1, -1).gather(1, idx).contiguous()
            sin_sel = sin.unsqueeze(0).expand(B, -1, -1).gather(1, idx).contiguous()
            return (cos_sel.unsqueeze(1), sin_sel.unsqueeze(1))
        elif cos.dim() == 4:  # [B,1,L,Dh] -> [B,1,Lk,Dh]
            B, _, L, Dh = cos.shape
            idx = ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, Dh)
            return (cos.gather(2, idx).contiguous(), sin.gather(2, idx).contiguous())
        else:
            raise RuntimeError(f"Unexpected rope dims: cos.dim={cos.dim()}")
