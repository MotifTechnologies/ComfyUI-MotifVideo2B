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


# ---------------------------------------------------------------------------
# RoPE helpers (HF modeling_t5gemma2.py:175-205, Apache 2.0)
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to q, k.

    cos/sin shape: (B, S, head_dim). After unsqueeze_dim=1: (B, 1, S, head_dim),
    broadcasts to q/k of shape (B, heads, S, head_dim).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# GQA helper (HF modeling_t5gemma2.py:208-217, Apache 2.0)
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """KV head repeat for GQA: (B, n_kv, S, D) → (B, n_q, S, D) where n_q = n_kv * n_rep."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class T5Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))

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


# ---------------------------------------------------------------------------
# Self-Attention (HF modeling_t5gemma2.py:255-330, Apache 2.0)
# ---------------------------------------------------------------------------

class T5Gemma2SelfAttention(nn.Module):
    """Self-attention block — encoder, GQA, q_norm/k_norm + RoPE.

    Mask + RoPE responsibility split:
      - `attention_mask` and `position_embeddings` are built externally by the
        caller (T5Gemma2TextEncoder, item 5). The caller dispatches per
        `layer_types[layer_idx]` and supplies the matching mask/RoPE table
        for "full_attention" or "sliding_attention".
      - `self.sliding_window` and `self.is_sliding` are stored only as a hint
        for the caller; this forward does not derive a mask from them. SDPA
        receives the prebuilt `attn_mask` as-is.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        operations = operations if operations is not None else comfy.ops.disable_weight_init

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # HF uses query_pre_attn_scalar (=256) rather than head_dim for the scale denominator;
        # numerically identical for this model but kept for interface parity.
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # encoder

        self.q_proj = operations.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias, dtype=dtype, device=device,
        )
        self.k_proj = operations.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, dtype=dtype, device=device,
        )
        self.v_proj = operations.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, dtype=dtype, device=device,
        )
        self.o_proj = operations.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias, dtype=dtype, device=device,
        )

        # attn_logit_softcapping: null in our model — stored for interface match.
        # If non-null, SDPA path cannot be used as-is; requires pre-softmax cap.
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.is_sliding = self.layer_type == "sliding_attention"

        # q_norm/k_norm: per-head RMSNorm over head_dim — Gemma2 specific.
        # Raw nn.Parameter inside; no comfy.ops injection needed (small weight).
        self.q_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (B, n_q, S, D)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # (B, n_kv, S, D)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (B, n_kv, S, D)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA: expand KV heads to match Q heads (explicit expand for portability over torch 2.5+).
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# MLP (HF modeling_t5gemma2.py:75-91, Apache 2.0)
# ---------------------------------------------------------------------------

class T5Gemma2MLP(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else comfy.ops.disable_weight_init

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = ops.Linear(
            self.hidden_size, self.intermediate_size,
            bias=False, dtype=dtype, device=device,
        )
        self.up_proj = ops.Linear(
            self.hidden_size, self.intermediate_size,
            bias=False, dtype=dtype, device=device,
        )
        self.down_proj = ops.Linear(
            self.intermediate_size, self.hidden_size,
            bias=False, dtype=dtype, device=device,
        )
        # config.hidden_activation = "gelu_pytorch_tanh" — F.gelu(..., approximate="tanh")
        self.act_fn = self._make_act_fn(config.hidden_activation)

        # dropout_rate = 0 in this model; stored for HF interface parity.
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _make_act_fn(name: str):
        if name == "gelu_pytorch_tanh":
            return lambda x: torch.nn.functional.gelu(x, approximate="tanh")
        raise NotImplementedError(
            f"T5Gemma2MLP: hidden_activation '{name}' not supported. "
            f"Only 'gelu_pytorch_tanh' is wired for the bundled config."
        )

    def forward(self, x):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        return self.down_proj(hidden_states)


# ---------------------------------------------------------------------------
# EncoderLayer (HF modeling_t5gemma2.py:454-503, Apache 2.0)
# ---------------------------------------------------------------------------

class T5Gemma2EncoderLayer(nn.Module):
    """Single encoder block: sandwich norm + self-attention + MLP."""

    def __init__(self, config, layer_idx: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        self.self_attn = T5Gemma2SelfAttention(
            config, layer_idx, dtype=dtype, device=device, operations=operations,
        )
        self.pre_self_attn_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device,
        )
        self.post_self_attn_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device,
        )
        self.mlp = T5Gemma2MLP(config, dtype=dtype, device=device, operations=operations)
        self.pre_feedforward_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device,
        )
        self.post_feedforward_layernorm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


# Checklist-5 verify alias (03_checklist.md verify uses T5Gemma2DecoderLayer).
T5Gemma2DecoderLayer = T5Gemma2EncoderLayer


# ---------------------------------------------------------------------------
# Scaled word embedding factory (HF modeling_t5gemma2.py:627-647, Apache 2.0)
# ---------------------------------------------------------------------------

def make_text_scaled_word_embedding_class(operations):
    """Return a T5Gemma2TextScaledWordEmbedding class backed by operations.Embedding.

    Dynamic subclassing ensures `weight` lives directly on the instance so
    that the state_dict key stays `<prefix>.weight` (not `<prefix>._embed.weight`),
    matching the HF checkpoint layout. comfy_cast_weights is inherited from the
    base (True for manual_cast, False for disable_weight_init).
    """
    base = operations.Embedding

    class T5Gemma2TextScaledWordEmbedding(base):
        def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
            embed_scale: float = 1.0,
            eoi_token_index: int = 256_000,
            dtype=None,
            device=None,
        ):
            super().__init__(
                num_embeddings, embedding_dim,
                padding_idx=padding_idx, dtype=dtype, device=device,
            )
            self.scalar_embed_scale = embed_scale
            self.register_buffer(
                "embed_scale",
                torch.tensor(embed_scale, dtype=torch.float32),
                persistent=False,
            )
            self.eoi_token_index = eoi_token_index
            self.eoi_embedding = nn.Parameter(
                torch.zeros(embedding_dim, dtype=dtype, device=device),
            )

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # Use the runtime output dtype/device — under `manual_cast` the
            # stored `self.weight` may be off-device / float32 while the
            # embedding forward executes on the active device with cast
            # weights. Aligning embed_scale and eoi_embedding to `out` keeps
            # the multiply and EOI assignment device/dtype consistent.
            out = super().forward(input_ids)
            out = out * self.embed_scale.to(dtype=out.dtype, device=out.device)
            eoi_pos = input_ids == self.eoi_token_index
            if eoi_pos.any():
                out[eoi_pos] = self.eoi_embedding.to(dtype=out.dtype, device=out.device)
            return out

    T5Gemma2TextScaledWordEmbedding.__name__ = "T5Gemma2TextScaledWordEmbedding"
    T5Gemma2TextScaledWordEmbedding.__qualname__ = "T5Gemma2TextScaledWordEmbedding"
    return T5Gemma2TextScaledWordEmbedding


# ---------------------------------------------------------------------------
# RoPE (HF modeling_t5gemma2.py:94-172, Apache 2.0)
# ---------------------------------------------------------------------------

class T5Gemma2RotaryEmbedding(nn.Module):
    """Per-layer-type RoPE buffers.

    Registers `<layer_type>_inv_freq` and `<layer_type>_original_inv_freq`
    (persistent=False) for each unique value in config.layer_types.
    forward(x, position_ids, layer_type) → (cos, sin) of shape (B, S, head_dim).
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        layer_types = list(dict.fromkeys(config.layer_types))  # stable-unique
        self.layer_types = layer_types
        self.rope_type = {}

        for layer_type in layer_types:
            rope_params = config.rope_parameters.get(layer_type)
            if rope_params is None:
                continue
            self.rope_type[layer_type] = rope_params["rope_type"]
            inv_freq, attn_scaling = self._compute_inv_freq(config, layer_type, device)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(
                f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False,
            )
            setattr(self, f"{layer_type}_attention_scaling", attn_scaling)

    @staticmethod
    def _compute_inv_freq(config, layer_type: str, device=None):
        rope_params = config.rope_parameters[layer_type]
        rope_type = rope_params["rope_type"]
        base = rope_params["rope_theta"]
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads,
        )
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float()
                / head_dim
            )
        )
        if rope_type == "default":
            return inv_freq, 1.0
        elif rope_type == "linear":
            factor = rope_params["factor"]
            return inv_freq / factor, 1.0
        else:
            raise NotImplementedError(
                f"T5Gemma2RotaryEmbedding: rope_type '{rope_type}' not supported. "
                "Supported: 'default', 'linear'."
            )

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type: str):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        # inv_freq: (D/2,), position_ids: (B, S)
        inv_freq_expanded = (
            inv_freq[None, :, None].float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        if device_type == "mps":
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Mask helpers (HF sliding_window_mask_function, bidirectional, Apache 2.0)
# ---------------------------------------------------------------------------

def _build_bidirectional_mask(
    seq_len: int,
    attention_mask=None,
    dtype=torch.float32,
    device=None,
):
    """(B, 1, S, S) additive bidirectional mask. Pad positions get -inf."""
    if attention_mask is None:
        return None  # SDPA handles None directly
    B = attention_mask.shape[0]
    mask = torch.zeros(B, 1, seq_len, seq_len, dtype=dtype, device=device)
    pad_mask = attention_mask == 0  # (B, S)
    if pad_mask.any():
        mask = mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))
    return mask


def _build_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    attention_mask=None,
    dtype=torch.float32,
    device=None,
):
    """(B, 1, S, S) additive sliding-window + padding mask (encoder, non-causal).

    left_window  = (sliding_window + 1) // 2
    right_window = sliding_window // 2 + 1
    """
    left_w = (sliding_window + 1) // 2
    right_w = sliding_window // 2 + 1

    q_idx = torch.arange(seq_len, device=device)[:, None]  # (S, 1)
    k_idx = torch.arange(seq_len, device=device)[None, :]  # (1, S)
    dist = q_idx - k_idx
    in_window = ((dist >= 0) & (dist < left_w)) | ((dist < 0) & (-dist < right_w))

    base = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    base = base.masked_fill(~in_window, float("-inf"))

    if attention_mask is None:
        return base[None, None, :, :].expand(1, 1, seq_len, seq_len)

    B = attention_mask.shape[0]
    mask = base[None, None, :, :].expand(B, 1, seq_len, seq_len).clone()
    pad_mask = attention_mask == 0
    if pad_mask.any():
        mask = mask.masked_fill(pad_mask[:, None, None, :], float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# Lightweight encoder output container
# ---------------------------------------------------------------------------

class _T5Gemma2EncoderOutput:
    """Minimal output container — only `.last_hidden_state` is exposed.

    This is NOT a HF `ModelOutput`: it does not support tuple indexing
    (`outputs[0]`), `.hidden_states`, `.attentions`, or `return_dict=False`
    semantics. The native encoder only targets the in-repo wrapper
    (`MotifVideoT5Gemma2Model`, see `text_encoders/t5_gemma2.py`), which
    accesses `outputs.last_hidden_state` exclusively. External callers that
    need full HF `ModelOutput` parity must use HF transformers directly.
    """

    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


# ---------------------------------------------------------------------------
# T5Gemma2TextEncoder — text_model body (HF modeling_t5gemma2.py:756-854, Apache 2.0)
# ---------------------------------------------------------------------------

class T5Gemma2TextEncoder(nn.Module):
    """Encoder body: embed_tokens + layers + norm + rotary_emb."""

    def __init__(
        self,
        config,
        eoi_token_index: int = 256_000,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        ops = operations if operations is not None else comfy.ops.disable_weight_init

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        EmbedClass = make_text_scaled_word_embedding_class(ops)
        self.embed_tokens = EmbedClass(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size ** 0.5,
            eoi_token_index=eoi_token_index,
            dtype=dtype,
            device=device,
        )
        self.norm = T5Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device,
        )
        self.layers = nn.ModuleList([
            T5Gemma2EncoderLayer(
                config, layer_idx, dtype=dtype, device=device, operations=operations,
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.dropout = nn.Dropout(config.dropout_rate)
        self.rotary_emb = T5Gemma2RotaryEmbedding(config, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Exactly one of input_ids or inputs_embeds must be specified."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        B, S = inputs_embeds.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(S, device=inputs_embeds.device).unsqueeze(0)
        else:
            # Caller-provided ids may live on a different device (e.g. CPU
            # while inputs_embeds is on CUDA). Move once before RoPE matmul.
            position_ids = position_ids.to(inputs_embeds.device)

        layer_types_set = set(self.config.layer_types)

        # Build per-layer-type additive attention masks.
        masks = {}
        if "full_attention" in layer_types_set:
            masks["full_attention"] = _build_bidirectional_mask(
                S, attention_mask,
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )
        if "sliding_attention" in layer_types_set:
            masks["sliding_attention"] = _build_sliding_window_mask(
                S, self.config.sliding_window, attention_mask,
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )

        # Build per-layer-type RoPE tables.
        position_embeddings = {
            lt: self.rotary_emb(inputs_embeds, position_ids, lt)
            for lt in layer_types_set
        }

        hidden_states = self.dropout(inputs_embeds)

        for i, layer in enumerate(self.layers):
            lt = self.config.layer_types[i]
            hidden_states = layer(hidden_states, position_embeddings[lt], masks[lt])

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return _T5Gemma2EncoderOutput(last_hidden_state=hidden_states)


# Checklist-5 verify alias (03_checklist.md verify uses T5Gemma2TextModel).
T5Gemma2TextModel = T5Gemma2TextEncoder


# ---------------------------------------------------------------------------
# T5Gemma2Encoder — outer wrapper (HF modeling_t5gemma2.py:857-958, Apache 2.0)
# ---------------------------------------------------------------------------

class T5Gemma2Encoder(nn.Module):
    """Outer encoder wrapper (text-only path) for the in-repo wrapper.

    Scope: this is NOT a drop-in replacement for HF `T5Gemma2Encoder`.
    It only supports the call shape used by `MotifVideoT5Gemma2Model`
    (see `text_encoders/t5_gemma2.py`). Specifically:
      - vision_tower / multi_modal_projector are omitted; non-None
        `pixel_values` raises NotImplementedError.
      - Standard HF kwargs that change return shape (`return_dict`,
        `output_hidden_states`, `output_attentions`, `use_cache`) are
        rejected with NotImplementedError so silent behaviour drift is
        impossible. The output is always a minimal container exposing
        only `.last_hidden_state`.
    External callers that need full HF parity must use HF transformers.
    """

    _UNSUPPORTED_HF_KWARGS = (
        "return_dict",
        "output_hidden_states",
        "output_attentions",
        "use_cache",
        "past_key_values",
        "head_mask",
        "cross_attn_head_mask",
        "encoder_outputs",
    )

    def __init__(
        self,
        config,
        eoi_token_index: int = 256_000,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.config = config
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.text_model = T5Gemma2TextEncoder(
            text_config,
            eoi_token_index=eoi_token_index,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def get_input_embeddings(self):
        return self.text_model.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise NotImplementedError(
                "Native T5Gemma2Encoder omits vision_tower; pixel_values must be None. "
                "Use HF T5Gemma2Encoder for the multimodal path."
            )
        # return_dict=True is compatible (we always return an object). Reject
        # False explicitly because the caller would expect a tuple and silently
        # receive an object instead.
        if kwargs.get("return_dict", None) is False:
            raise NotImplementedError(
                "Native T5Gemma2Encoder always returns an output object; "
                "return_dict=False (tuple form) is not supported."
            )
        for k in self._UNSUPPORTED_HF_KWARGS:
            if k == "return_dict":
                continue  # handled above
            v = kwargs.get(k, None)
            if v is None or v is False:
                continue  # default value, no behaviour change
            raise NotImplementedError(
                f"Native T5Gemma2Encoder does not support HF kwarg '{k}'. "
                f"This implementation targets the in-repo wrapper only; "
                f"use HF T5Gemma2Encoder if you need '{k}'."
            )
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
