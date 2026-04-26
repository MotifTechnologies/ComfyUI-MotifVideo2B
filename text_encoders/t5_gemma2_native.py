"""T5Gemma2 native implementation scaffold for ComfyUI partial offload.

This module hosts the native T5Gemma2 encoder. RMSNorm uses raw `nn.Parameter`
(small weight, no offload benefit). The Linear/Embedding layers added in
later items will receive `comfy.ops` injection for `comfy_cast_weights`
support — the `comfy.ops` import is wired here so subsequent additions
land without further bootstrap edits.

Derived from transformers.models.t5gemma2.modeling_t5gemma2 (Apache 2.0).
"""

import torch
import torch.nn as nn

import comfy.ops  # noqa: F401  # used by Linear/Embedding wrappers in later items


class T5Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # T5Gemma2 specific: (x * w).to(dtype) NOT x.to(dtype) * w
        # See: https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
