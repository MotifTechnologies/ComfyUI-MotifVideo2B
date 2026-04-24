# Register MotifVideo19B model config with ComfyUI at startup.
# config.py is a placeholder; replace with full implementation in checklist item 3.
try:
    import comfy.supported_models
    from .config import MotifVideo19B
    _before = len(comfy.supported_models.models)
    comfy.supported_models.models.append(MotifVideo19B)
    _after = len(comfy.supported_models.models)
    print(
        f"[ComfyUI-MotifVideo1.9B] MotifVideo19B registered "
        f"(supported_models.models: {_before} -> {_after}, "
        f"last={comfy.supported_models.models[-1].__name__})"
    )
except Exception as e:
    import traceback
    print(f"[ComfyUI-MotifVideo1.9B] WARNING: Failed to register model config: {e}")
    traceback.print_exc()

# Monkey-patch comfy.model_detection.detect_unet_config to support MotifVideo transformers.
# detect_unet_config() is called by comfy.sd.load_diffusion_model() via Load Diffusion Model node.
# MotifVideo is identified by the combination of context_embedder.linear_1.weight and
# image_embedder.linear_1.weight keys, which are unique to this architecture.
try:
    import comfy.model_detection

    _original_detect_unet_config = comfy.model_detection.detect_unet_config

    def _detect_unet_config_with_motif(state_dict, key_prefix, metadata=None):
        # MotifVideo detection: context_embedder + image_embedder combination is unique.
        if ('{}context_embedder.linear_1.weight'.format(key_prefix) in state_dict and
                '{}image_embedder.linear_1.weight'.format(key_prefix) in state_dict):

            context_w = state_dict['{}context_embedder.linear_1.weight'.format(key_prefix)]
            image_w = state_dict['{}image_embedder.linear_1.weight'.format(key_prefix)]
            attn_k_w = state_dict['{}single_transformer_blocks.0.attn.to_k.weight'.format(key_prefix)]

            # Dynamically detect architecture params from state_dict shapes.
            inner_dim = attn_k_w.shape[0]
            # Architecture constant: head_dim = sum(rope_axes_dim) = 16+56+56 = 128.
            # Not derivable from state_dict; tied to RoPE design.
            attention_head_dim = 128
            num_attention_heads = inner_dim // attention_head_dim
            text_embed_dim = context_w.shape[1]
            image_embed_dim = image_w.shape[1]

            # Count block types by iterating keys.
            num_layers = 0
            while '{}transformer_blocks.{}.norm1.linear.weight'.format(key_prefix, num_layers) in state_dict:
                num_layers += 1
            num_single_layers = 0
            while '{}single_transformer_blocks.{}.attn.to_k.weight'.format(key_prefix, num_single_layers) in state_dict:
                num_single_layers += 1
            # Architecture constant: decoder reuses last N single_transformer_blocks.
            # No separate keys in state_dict to distinguish encoder/decoder blocks.
            # Not derivable from weights; must match training config.
            num_decoder_layers = 8

            # Dynamically detect cross-attention presence from state_dict keys.
            # Keys are present only in cross-attn checkpoints; absence means False
            # (backward-compatible with base checkpoints).
            single_cross_attn_key = '{}single_transformer_blocks.0.cross_attn_query_proj.weight'.format(key_prefix)
            dual_cross_attn_key = '{}transformer_blocks.0.cross_attn_query_proj.weight'.format(key_prefix)
            single_cross_attn_w = state_dict.get(single_cross_attn_key)
            dual_cross_attn_w = state_dict.get(dual_cross_attn_key)
            enable_text_cross_attention_single = (
                single_cross_attn_w is not None and single_cross_attn_w.numel() > 0
            )
            enable_text_cross_attention_dual = (
                dual_cross_attn_w is not None and dual_cross_attn_w.numel() > 0
            )

            # in_channels from x_embedder
            x_embed_w = state_dict['{}x_embedder.proj.weight'.format(key_prefix)]
            in_channels = x_embed_w.shape[1]

            # patch_size / patch_size_t from x_embedder Conv3d kernel shape:
            # weight shape = [embed_dim, in_ch, patch_size_t, patch_size, patch_size]
            patch_size_t = x_embed_w.shape[2]
            patch_size = x_embed_w.shape[3]

            # out_channels: proj_out.weight shape [patch_size_t * patch_size^2 * out_channels, inner_dim]
            proj_out_w = state_dict['{}proj_out.weight'.format(key_prefix)]
            out_channels = proj_out_w.shape[0] // (patch_size_t * patch_size * patch_size)

            return {
                "image_model": "motif_video",
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_attention_heads": num_attention_heads,
                "attention_head_dim": attention_head_dim,
                "num_layers": num_layers,
                "num_single_layers": num_single_layers,
                "num_decoder_layers": num_decoder_layers,
                "text_embed_dim": text_embed_dim,
                "image_embed_dim": image_embed_dim,
                "patch_size": patch_size,
                "patch_size_t": patch_size_t,
                # Architecture constants: RoPE config is not stored in state_dict.
                # rope_axes_dim must sum to attention_head_dim (128).
                "rope_axes_dim": [16, 56, 56],
                "rope_theta": 10000.0,
                "enable_text_cross_attention_dual": enable_text_cross_attention_dual,
                "enable_text_cross_attention_single": enable_text_cross_attention_single,
            }

        # Not a MotifVideo model — fall through to original detection logic.
        return _original_detect_unet_config(state_dict, key_prefix, metadata=metadata)

    comfy.model_detection.detect_unet_config = _detect_unet_config_with_motif
    print("[ComfyUI-MotifVideo1.9B] detect_unet_config monkey-patch applied.")
except Exception as e:
    print(f"[ComfyUI-MotifVideo1.9B] WARNING: Failed to patch model_detection: {e}")

try:
    from .nodes.loader import MotifTextEncoderLoader
    from .nodes.text_encode import MotifTextEncode
    from .nodes.latent import EmptyMotifLatent
    from .nodes.vae_loader import MotifVAELoader
    from .nodes.image_encode import MotifImageEncode

    NODE_CLASS_MAPPINGS = {
        "MotifTextEncoderLoader": MotifTextEncoderLoader,
        "MotifTextEncode": MotifTextEncode,
        "EmptyMotifLatent": EmptyMotifLatent,
        "MotifVAELoader": MotifVAELoader,
        "MotifImageEncode": MotifImageEncode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "MotifTextEncoderLoader": "Load MotifVideo Text Encoder",
        "MotifTextEncode": "MotifVideo Text Encode",
        "EmptyMotifLatent": "Empty MotifVideo Latent",
        "MotifVAELoader": "Load MotifVideo VAE",
        "MotifImageEncode": "MotifVideo Image Encode",
    }

except Exception as e:
    print(f"[ComfyUI-MotifVideo1.9B] ERROR: Failed to load nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
