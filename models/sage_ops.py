"""SageAttention custom_op 이식 모듈.

포트 출처: motif-models compile_configs.py:52-126 의 4개 custom_op + dispatch_optimized_attention.
모듈 import 시점에 torch.library 에 op 를 등록 (side-effect). 네임스페이스는 충돌 방지를
위해 원본 'custom::' 에서 'motifvideo::' 로 변경.
sageattention 미설치 환경에서는 custom_op 등록을 skip (_SAGE_AVAILABLE = False).
호출측(apply_sage_attention)에서 _SAGE_AVAILABLE 플래그로 활성 여부를 결정한다.
"""

import logging

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.getLogger(__name__)

try:
    from sageattention import sageattn
    _SAGE_AVAILABLE = True
except ImportError:
    _SAGE_AVAILABLE = False
    sageattn = None

if not _SAGE_AVAILABLE:
    logger.info("[MotifVideo] sageattention not installed — dispatch_optimized_attention will not register custom ops")


if _SAGE_AVAILABLE:
    # 포트 출처: compile_configs.py:52-59
    @torch.library.custom_op("motifvideo::enforce_contiguous", mutates_args=())
    def enforce_contiguous(x: torch.Tensor) -> torch.Tensor:
        return x.clone(memory_format=torch.contiguous_format)

    @enforce_contiguous.register_fake
    def _(x):
        return x.new_empty(x.shape)

    # 포트 출처: compile_configs.py:62-81
    @torch.library.custom_op("motifvideo::sage_attention_forward_op", mutates_args=())
    def sage_attention_forward_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        active_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        active_len = active_sequence_length.item()
        q_active = query[:, :, :active_len, :].contiguous()
        k_active = key[:, :, :active_len, :].contiguous()
        v_active = value[:, :, :active_len, :].contiguous()
        attn_out = sageattn(q_active, k_active, v_active, tensor_layout="HND", is_causal=False)
        out = query.new_zeros(query.shape)
        out[:, :, :active_len, :] = attn_out
        return out

    @sage_attention_forward_op.register_fake
    def _(query, key, value, active_sequence_length):
        return query.new_empty(query.shape)

    # 포트 출처: compile_configs.py:84-93
    @torch.library.custom_op("motifvideo::safe_sdpa_fallback", mutates_args=())
    def safe_sdpa_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        return out.contiguous()

    @safe_sdpa_fallback.register_fake
    def _(q, k, v, mask):
        return q.new_empty(q.shape)


# 포트 출처: compile_configs.py:96-126
def dispatch_optimized_attention(query, key, value, attention_mask):
    """Route attention to SageAttention or SDPA fallback.

    ⚠️ Call-site contract (원본 motif-pipelines 설계와 동일). 본 dispatcher 는
    generic attention 이 아니라 MotifVideo 의 **self-attention joint path
    전용**이다. `xDiTMotifVideoAttnProcessor.__call__` 은 아래 조건에서만
    이 함수를 부른다 (원본 attention_processor.py:50-104):

    1. `query_input is None` → cross-attention query 경로가 아님.
    2. 직전에 `query = torch.cat([query, encoder_query], dim=2)` 로 latent+text
       를 합쳐 **query_len == key_len == value_len == L+E** 가 보장됨.
    3. `attention_mask` 는 `MotifVideoTransformer3DModel._create_attention_mask`
       가 만든 **joint mask**: shape `[B, 1, 1, L+E]`, dtype `torch.bool`, 앞쪽
       L 개의 latent token 은 강제 True, 뒤쪽 E 개는 `encoder_attention_mask`
       를 그대로 씀. 즉 SDPA 의 `[B, H, Q, K]` 에 정확히 broadcast 가능.

    위 세 조건하에서만:
    - `sum(dim=-1)` 가 total valid token count (L + text_valid) 로 동작
    - `query[:, :, :active_len, :]` 슬라이스가 query/key 양쪽에 모두 의미
    - fallback path 의 `F.scaled_dot_product_attention(..., attn_mask=mask)`
      는 `[B, 1, 1, L+E]` bool mask 를 `[B, H, L+E, L+E]` 로 broadcast 해 올
      바르게 keep/drop 처리.

    Generic cross-attention (query_len != key_len) 이나 `[B, S]` shape 의
    길이-전용 mask, additive/holey mask 에는 **부정확한 결과를 낸다**.
    MotifVideo 외 사용 금지. Codex reviewer HIGH/MEDIUM override —
    04_log 2026-04-20 P1.1 참조.
    """
    if not _SAGE_AVAILABLE:
        # sage 미설치 환경: custom op 들이 등록되어 있지 않으므로 pure SDPA
        # 경로로 degrade. 호출측(apply_sage_attention)이 애초에 processor 교체
        # 를 안 할 예정이지만, 실수로 호출되어도 crash 대신 정답 반환하도록
        # 보호막을 둔다 (Codex reviewer MEDIUM 수용).
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, is_causal=False
        ).contiguous()
    if attention_mask is None:
        query = torch.ops.motifvideo.enforce_contiguous(query)
        key = torch.ops.motifvideo.enforce_contiguous(key)
        value = torch.ops.motifvideo.enforce_contiguous(value)
        padding_idx = torch.tensor(query.shape[2], dtype=torch.long, device=query.device)
        return torch.ops.motifvideo.sage_attention_forward_op(query, key, value, padding_idx)

    padding_indices = attention_mask.sum(dim=-1).long().flatten()
    common_padding_index = padding_indices[0]
    is_padding_index_uniform = (padding_indices == common_padding_index).all()

    query = torch.ops.motifvideo.enforce_contiguous(query)
    key = torch.ops.motifvideo.enforce_contiguous(key)
    value = torch.ops.motifvideo.enforce_contiguous(value)
    attention_mask = torch.ops.motifvideo.enforce_contiguous(attention_mask)

    def execute_sage_attention(q, k, v, padding_idx, mask):
        return torch.ops.motifvideo.sage_attention_forward_op(q, k, v, padding_idx)

    def execute_sdpa_fallback(q, k, v, padding_idx, mask):
        return torch.ops.motifvideo.safe_sdpa_fallback(q, k, v, mask)

    return torch.cond(
        is_padding_index_uniform,
        execute_sage_attention,
        execute_sdpa_fallback,
        operands=[query, key, value, common_padding_index, attention_mask],
    )
