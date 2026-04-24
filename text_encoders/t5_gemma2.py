"""T5Gemma2 text encoder integration for ComfyUI.

Wraps transformers T5Gemma2Encoder so it fits the ComfyUI CLIP system
(SD1ClipModel / SD1Tokenizer pattern).  Only the encoder half of the
encoder-decoder model is used for text-conditioning.

Key facts from inspection:
  - Checkpoint keys: encoder.{layers|embed_tokens|norm|vision_tower|...}
  - T5Gemma2Encoder state_dict keys: text_model.{layers|embed_tokens|norm|...}
    plus vision_tower.* and multi_modal_projector.*
  - Mapping: strip "encoder." prefix from checkpoint keys → T5Gemma2Encoder keys
  - hidden_size = 2560 (encoder.text_config.hidden_size)
  - vocab_size  = 262144
  - bos_token_id = 2, eos_token_id = 1, pad_token_id = 0
  - GemmaTokenizer (tokenizers backend, fast) lives at tokenizer/ dir
"""

import os
import logging

import torch
import torch.nn as nn

from comfy import sd1_clip
from comfy.sd1_clip import ClipTokenWeightEncoder

from .t5_gemma2_config import T5_GEMMA2_CONFIG


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class MotifVideoTokenizer(sd1_clip.SDTokenizer):
    """GemmaTokenizer wrapped as SDTokenizer for ComfyUI.

    Defaults to the bundled directory at ``text_encoders/tokenizer_assets/``
    (tokenizer.json + tokenizer_config.json shipped with the node). Callers
    may override with ``tokenizer_data["motifvideo_tokenizer_path"]`` to
    point to an alternate HuggingFace-layout directory.

    Special token mapping:
      bos_token_id = 2  → SDTokenizer start_token
      eos_token_id = 1  → SDTokenizer end_token
      pad_token_id = 0  → pad_token

    GemmaTokenizer.from_pretrained(path)("") returns [2] (bos only),
    so has_start_token=True and has_end_token=False — we set end_token=1
    explicitly.
    """

    _BUNDLED_TOKENIZER_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "tokenizer_assets",
    )

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = tokenizer_data.get(
            "motifvideo_tokenizer_path",
            self._BUNDLED_TOKENIZER_DIR,
        )

        from transformers import GemmaTokenizerFast

        super().__init__(
            tokenizer_path=tokenizer_path,
            max_length=99999999,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=2560,
            embedding_key="motifvideo",
            tokenizer_class=GemmaTokenizerFast,
            has_start_token=True,
            has_end_token=False,
            end_token=1,        # eos_token_id
            pad_token=0,        # pad_token_id
            pad_to_max_length=False,
            min_length=None,
            tokenizer_data=tokenizer_data,
        )

    def state_dict(self):
        return {}


class MotifVideoSD1Tokenizer(sd1_clip.SD1Tokenizer):
    """SD1Tokenizer wrapper using 'motifvideo' as clip_name."""

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            clip_name="motifvideo",
            tokenizer=MotifVideoTokenizer,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MotifVideoT5Gemma2Model(nn.Module, ClipTokenWeightEncoder):
    """T5Gemma2Encoder wrapped for ComfyUI CLIP system.

    Follows SD1ClipModel / SDClipModel interface:
      - encode_token_weights(token_weight_pairs) via ClipTokenWeightEncoder mixin
      - encode(tokens) → (last_hidden_state, None)
      - load_sd(sd) → load state_dict from checkpoint
      - dtypes set for ComfyUI dtype tracking

    The internal model is transformers T5Gemma2Encoder.  Only text inputs
    are used (pixel_values is None); the vision tower weights remain in
    memory but are inactive during text-only inference.
    """

    # For ClipTokenWeightEncoder compatibility
    special_tokens = {"end": 1, "pad": 0}

    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()

        from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2EncoderConfig
        from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Encoder

        encoder_config = T5Gemma2EncoderConfig(**T5_GEMMA2_CONFIG)

        if dtype is None:
            dtype = torch.bfloat16

        # Instantiate encoder on target device with chosen dtype.
        self.encoder = T5Gemma2Encoder(encoder_config).to(device=device, dtype=dtype)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.dtype = dtype
        self.dtypes = {dtype}

        # Needed by ClipTokenWeightEncoder
        self.layer = "last"
        self.layer_idx = None
        self.return_projected_pooled = False
        self.execution_device = None

        # num_layers shim (SDClipModel compatibility)
        self.num_layers = encoder_config.text_config.num_hidden_layers

        # DEBUG: include instance id so duplicate constructions are traceable.
        logging.info(
            "[MotifVideo] T5Gemma2Encoder initialised | device=%s dtype=%s hidden=%d layers=%d id=0x%x",
            device, dtype, encoder_config.text_config.hidden_size,
            self.num_layers, id(self),
        )

    def freeze(self):
        self.encoder.eval()
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)

    def reset_clip_options(self):
        self.execution_device = None
        self.return_projected_pooled = False

    def get_input_embeddings(self):
        """Required by SDClipModel.process_tokens for embedding lookup."""
        return self.encoder.text_model.embed_tokens

    def encode(self, tokens):
        """Forward pass over a batch of token id lists.

        Args:
            tokens: list of list[int]  — same format SDClipModel.encode() receives.

        Returns:
            (last_hidden_state, None) where last_hidden_state is
            shape (batch, seq_len, hidden_size).
        """
        if self.execution_device is not None:
            device = self.execution_device
        else:
            device = next(self.encoder.parameters()).device

        # Build padded tensor and attention mask for the encoder forward pass.
        max_len = max(len(t) for t in tokens)
        pad_id = self.special_tokens["pad"]
        input_ids = torch.tensor(
            [t + [pad_id] * (max_len - len(t)) for t in tokens],
            dtype=torch.long,
            device=device,
        )
        attention_mask = (input_ids != pad_id).long()

        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=None,
            )

        hidden_state = outputs.last_hidden_state.float()

        # Per user direction: don't pass attention_mask downstream — trim the
        # padded tail here instead. This follows the HF PR #8 pattern
        # (use_attention_mask=False + padding trim). Intent: the transformer's
        # cross-attention only sees seq_len = valid_len, so it doesn't need a
        # mask, and SDPA's mask=None branch lets cuDNN/Flash be selected
        # automatically.
        #
        # Batch safety (Codex HIGH defense): for non-uniform batches (e.g.
        # CFG where positive and negative differ in length), a simple max-
        # trim leaves padded residue inside shorter samples, which then
        # cross-attends and silently contaminates conditioning. In that case
        # we give up on trimming and return the original tensor. The standard
        # ComfyUI MotifVideo workflow runs CFG as batch=1 per forward
        # (measured q=(1, 12, 114233, 128)), so the uniform path is hit in
        # practice.
        if attention_mask is not None and attention_mask.shape[0] >= 1:
            valid_lens = attention_mask.sum(dim=1)  # [B]
            first_len = valid_lens[0]
            is_uniform = bool((valid_lens == first_len).all().item())
            if is_uniform:
                max_valid = int(first_len.item())
                if 0 < max_valid < hidden_state.shape[1]:
                    hidden_state = hidden_state[:, :max_valid, :].contiguous()
            # else: non-uniform batch — return the original tensor. Padded
            #       positions are still attended, but that is a safer choice
            #       than crashing. Violating the uniformity assumption means
            #       a non-standard workflow; a future extension could use
            #       per-sample packed attention.

        return hidden_state, None

    def encode_token_weights(self, token_weight_pairs):
        """ComfyUI CLIP interface.  Dispatches through ClipTokenWeightEncoder.

        token_weight_pairs comes from SD1ClipModel.encode_token_weights which
        already strips the clip_name key, so we receive the raw list here.
        """
        # ClipTokenWeightEncoder.encode_token_weights calls self.encode(tokens)
        # where tokens is a list-of-lists-of-ints.  We need to extract just
        # the token ids from (token, weight) pairs.
        to_encode = []
        max_token_len = 0
        has_weights = False

        for x in token_weight_pairs:
            tokens = [a[0] for a in x if isinstance(a[0], int)]
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(a[1] == 1.0 for a in x)
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            pad_id = self.special_tokens["pad"]
            to_encode.append([pad_id] * max_token_len)

        out, pooled = self.encode(to_encode)

        # pooled is None for T5-style encoders
        first_pooled = pooled

        import comfy.model_management as model_management
        intermed_device = model_management.intermediate_device()
        output = []
        for k in range(sections):
            z = out[k : k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if len(output) == 0:
            r = (out[-1:].to(device=intermed_device), first_pooled)
        else:
            r = (
                torch.cat(output, dim=-2).to(device=intermed_device),
                first_pooled,
            )

        return r

    def load_sd(self, sd):
        """Load state dict from checkpoint.

        Checkpoint key structure (after stripping 'encoder.' prefix):
          embed_tokens.*  →  T5Gemma2Encoder: text_model.embed_tokens.*
          layers.*        →  T5Gemma2Encoder: text_model.layers.*
          norm.*          →  T5Gemma2Encoder: text_model.norm.*
          vision_tower.*  →  T5Gemma2Encoder: vision_tower.*   (direct)
          multi_modal_projector.* → T5Gemma2Encoder: multi_modal_projector.* (direct)

        Text-model submodules (embed_tokens, layers, norm) are under
        T5Gemma2Encoder.text_model, so they need the 'text_model.' prefix
        added.  Vision/projector submodules live directly on the encoder.
        """
        # Submodules that live directly on T5Gemma2Encoder (not under text_model).
        _direct_submodules = {"vision_tower", "multi_modal_projector"}

        mapped = {}
        for k, v in sd.items():
            if not k.startswith("encoder."):
                # Pass through keys not belonging to encoder (e.g. decoder.*)
                continue
            inner = k[len("encoder."):]  # strip 'encoder.' prefix
            top = inner.split(".")[0]
            if top in _direct_submodules:
                # vision_tower.* / multi_modal_projector.*  → direct on encoder
                mapped[inner] = v
            else:
                # embed_tokens.* / layers.* / norm.*  → under text_model
                mapped["text_model." + inner] = v

        result = self.encoder.load_state_dict(mapped, strict=False, assign=False)
        return result


class MotifVideoSD1ClipModel(nn.Module):
    """SD1ClipModel-compatible wrapper for MotifVideoT5Gemma2Model.

    ComfyUI's sd.CLIP expects a model that:
      - has .dtypes attribute
      - has .encode_token_weights(token_weight_pairs[clip_name]) callable
      - has .load_sd(sd) callable
      - has .set_clip_options / .reset_clip_options

    Mirrors SD1ClipModel but uses 'motifvideo' as clip_name.
    """

    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()

        self.clip_name = "motifvideo"
        self.clip = "motifvideo"

        self.motifvideo = MotifVideoT5Gemma2Model(
            device=device, dtype=dtype, model_options=model_options
        )

        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)
        else:
            self.dtypes.update(self.motifvideo.dtypes)

    def set_clip_options(self, options):
        self.motifvideo.set_clip_options(options)

    def reset_clip_options(self):
        self.motifvideo.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        pairs = token_weight_pairs.get(self.clip_name, token_weight_pairs)
        return self.motifvideo.encode_token_weights(pairs)

    def load_sd(self, sd):
        return self.motifvideo.load_sd(sd)


# ---------------------------------------------------------------------------
# Factory function (mirrors WAN te())
# ---------------------------------------------------------------------------

def te(dtype_t5gemma2=None):
    """Factory function for clip_target().

    Returns a MotifVideoSD1ClipModel subclass that locks in the dtype.
    """

    class MotifVideoTEModel(MotifVideoSD1ClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_t5gemma2 is not None:
                dtype = dtype_t5gemma2
            super().__init__(device=device, dtype=dtype, model_options=model_options)

    return MotifVideoTEModel
