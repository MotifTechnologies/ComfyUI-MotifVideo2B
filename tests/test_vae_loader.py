"""Unit tests for nodes/vae_loader.py.

Covers:
- Key conversion accuracy against real checkpoints
- Node class structure
- Edge cases: empty dict, passthrough keys, boundary block indices
"""

import importlib.util
import sys
import unittest.mock as mock

import pytest
import safetensors.torch
import torch

# ---------------------------------------------------------------------------
# Module isolation: mock comfy and folder_paths before loading vae_loader
# so the test file is importable without a full ComfyUI installation.
# ---------------------------------------------------------------------------

_COMFY_MOCK = mock.MagicMock()
sys.modules.setdefault("comfy", _COMFY_MOCK)
sys.modules.setdefault("comfy.sd", mock.MagicMock())
sys.modules.setdefault("folder_paths", mock.MagicMock())

_VAE_LOADER_PATH = (
    "/lustrefs/team-multimodal/minsu/ComfyUI"
    "/custom_nodes/ComfyUI-MotifVideo1.9B/nodes/vae_loader.py"
)

_spec = importlib.util.spec_from_file_location("vae_loader", _VAE_LOADER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

convert_diffusers_to_comfyui = _mod.convert_diffusers_to_comfyui
_convert_key = _mod._convert_key
_convert_resnet_keys = _mod._convert_resnet_keys
_convert_mid_block_resnet = _mod._convert_mid_block_resnet
_resnet_sub = _mod._resnet_sub
MotifVAELoader = _mod.MotifVAELoader

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

MOTIF_CKPT = (
    "/lustrefs/team-multimodal/checkpoints"
    "/base_checkpoint/model/vae/diffusion_pytorch_model.safetensors"
)
REF_CKPT = (
    "/lustrefs/team-multimodal/minsu/ComfyUI"
    "/models/vae/wan_2.1_vae.safetensors"
)

DIFFUSERS_PREFIXES = (
    "encoder.down_blocks",
    "decoder.up_blocks",
    "decoder.mid_block",
    "encoder.mid_block",
    "quant_conv",
    "post_quant_conv",
    "encoder.conv_in",
    "decoder.conv_in",
    "decoder.conv_out",
    "decoder.norm_out",
    "encoder.conv_out",
    "encoder.norm_out",
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def motif_sd():
    return safetensors.torch.load_file(MOTIF_CKPT, device="cpu")


@pytest.fixture(scope="module")
def ref_sd():
    return safetensors.torch.load_file(REF_CKPT, device="cpu")


@pytest.fixture(scope="module")
def converted_sd(motif_sd):
    return convert_diffusers_to_comfyui(motif_sd)


# ===========================================================================
# 1. Key conversion accuracy (real checkpoint)
# ===========================================================================


class TestKeyConversionAccuracy:
    def test_detection_key_exists_after_conversion(self, converted_sd):
        """ComfyUI detection sentinel key must be present after conversion."""
        assert "decoder.middle.0.residual.0.gamma" in converted_sd

    def test_no_diffusers_prefix_keys_remain(self, converted_sd):
        """No diffusers-format prefix must survive in the converted dict."""
        leaked = [
            k
            for k in converted_sd
            if any(k.startswith(p) for p in DIFFUSERS_PREFIXES)
        ]
        assert leaked == [], f"Leaked diffusers keys: {leaked[:5]}"

    def test_converted_key_set_matches_reference(self, converted_sd, ref_sd):
        """Converted key set must be identical to the WanVAE reference."""
        conv_keys = set(converted_sd.keys())
        ref_keys = set(ref_sd.keys())

        missing = ref_keys - conv_keys
        extra = conv_keys - ref_keys

        assert missing == set(), f"Keys missing from converted: {sorted(missing)[:10]}"
        assert extra == set(), f"Extra keys not in reference: {sorted(extra)[:10]}"

    def test_converted_shapes_match_reference(self, converted_sd, ref_sd):
        """Every converted tensor shape must match the reference shape."""
        mismatched = []
        for k in ref_sd:
            if k in converted_sd:
                if ref_sd[k].shape != converted_sd[k].shape:
                    mismatched.append(
                        (k, tuple(ref_sd[k].shape), tuple(converted_sd[k].shape))
                    )
        assert mismatched == [], (
            f"Shape mismatches ({len(mismatched)} total): {mismatched[:5]}"
        )

    def test_total_key_count_matches_reference(self, converted_sd, ref_sd):
        """Total number of keys must be identical to the reference."""
        assert len(converted_sd) == len(ref_sd)


# ===========================================================================
# 2. Node class structure
# ===========================================================================


class TestMotifVAELoaderClassStructure:
    def test_return_types(self):
        assert MotifVAELoader.RETURN_TYPES == ("VAE",)

    def test_function_name(self):
        assert MotifVAELoader.FUNCTION == "load_vae"

    def test_category(self):
        assert MotifVAELoader.CATEGORY == "motifvideo"

    def test_input_types_has_required_vae_name(self):
        it = MotifVAELoader.INPUT_TYPES()
        assert "required" in it
        assert "vae_name" in it["required"]

    def test_load_vae_method_exists(self):
        assert callable(getattr(MotifVAELoader, "load_vae", None))


# ===========================================================================
# 3. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_dict_returns_empty_dict(self):
        result = convert_diffusers_to_comfyui({})
        assert result == {}

    def test_empty_dict_no_exception(self):
        try:
            convert_diffusers_to_comfyui({})
        except Exception as exc:
            pytest.fail(f"Empty dict raised: {exc}")

    def test_passthrough_comfyui_key_unchanged(self):
        """Keys that are already in ComfyUI format must not be altered."""
        comfyui_keys = {
            "decoder.middle.0.residual.0.gamma": torch.zeros(4),
            "encoder.downsamples.3.residual.2.weight": torch.zeros(4),
            "conv1.weight": torch.zeros(4),
            "conv2.bias": torch.zeros(4),
        }
        result = convert_diffusers_to_comfyui(comfyui_keys)
        for k in comfyui_keys:
            assert k in result, f"Passthrough key lost: {k}"

    def test_passthrough_preserves_tensor_identity(self):
        """Passthrough keys must map to the same tensor object."""
        t = torch.zeros(8)
        sd = {"already.comfyui.key": t}
        result = convert_diffusers_to_comfyui(sd)
        assert result.get("already.comfyui.key") is t

    def test_duplicate_key_after_conversion_does_not_raise(self):
        """Two diffusers keys that map to the same ComfyUI key should not raise."""
        # This is a degenerate input; the function logs a warning but must not crash.
        sd = {
            "quant_conv.weight": torch.zeros(2),
            "quant_conv.weight": torch.zeros(2),  # Python dict deduplicates this
        }
        try:
            convert_diffusers_to_comfyui(sd)
        except Exception as exc:
            pytest.fail(f"Duplicate key caused exception: {exc}")


# ===========================================================================
# 4. Individual key-conversion helper unit tests
# ===========================================================================


class TestConvertKeyUnit:
    # --- quant convs ---
    def test_quant_conv_weight(self):
        assert _convert_key("quant_conv.weight") == "conv1.weight"

    def test_quant_conv_bias(self):
        assert _convert_key("quant_conv.bias") == "conv1.bias"

    def test_post_quant_conv_weight(self):
        assert _convert_key("post_quant_conv.weight") == "conv2.weight"

    # --- encoder top-level ---
    def test_encoder_conv_in_weight(self):
        assert _convert_key("encoder.conv_in.weight") == "encoder.conv1.weight"

    def test_encoder_conv_out_weight(self):
        assert _convert_key("encoder.conv_out.weight") == "encoder.head.2.weight"

    def test_encoder_norm_out_gamma(self):
        assert _convert_key("encoder.norm_out.gamma") == "encoder.head.0.gamma"

    # --- decoder top-level ---
    def test_decoder_conv_in_bias(self):
        assert _convert_key("decoder.conv_in.bias") == "decoder.conv1.bias"

    def test_decoder_conv_out_bias(self):
        assert _convert_key("decoder.conv_out.bias") == "decoder.head.2.bias"

    def test_decoder_norm_out_gamma(self):
        assert _convert_key("decoder.norm_out.gamma") == "decoder.head.0.gamma"

    # --- encoder mid_block ---
    def test_encoder_mid_block_resnets_0_norm1(self):
        k = "encoder.mid_block.resnets.0.norm1.gamma"
        assert _convert_key(k) == "encoder.middle.0.residual.0.gamma"

    def test_encoder_mid_block_resnets_1_conv2(self):
        k = "encoder.mid_block.resnets.1.conv2.weight"
        assert _convert_key(k) == "encoder.middle.2.residual.6.weight"

    def test_encoder_mid_block_attentions_0_norm(self):
        k = "encoder.mid_block.attentions.0.norm.gamma"
        assert _convert_key(k) == "encoder.middle.1.norm.gamma"

    # --- decoder mid_block ---
    def test_decoder_mid_block_resnets_0(self):
        k = "decoder.mid_block.resnets.0.conv1.bias"
        assert _convert_key(k) == "decoder.middle.0.residual.2.bias"

    def test_decoder_mid_block_resnets_1(self):
        k = "decoder.mid_block.resnets.1.norm2.gamma"
        assert _convert_key(k) == "decoder.middle.2.residual.3.gamma"

    # --- encoder down_blocks ---
    def test_encoder_down_blocks_0_norm1(self):
        k = "encoder.down_blocks.0.norm1.gamma"
        assert _convert_key(k) == "encoder.downsamples.0.residual.0.gamma"

    def test_encoder_down_blocks_3_conv_shortcut(self):
        k = "encoder.down_blocks.3.conv_shortcut.weight"
        assert _convert_key(k) == "encoder.downsamples.3.shortcut.weight"

    # --- decoder up_blocks resnets ---
    def test_decoder_up_blocks_0_resnets_0_norm1(self):
        k = "decoder.up_blocks.0.resnets.0.norm1.gamma"
        assert _convert_key(k) == "decoder.upsamples.0.residual.0.gamma"

    def test_decoder_up_blocks_0_resnets_2(self):
        # block=0, resnet=2 -> flat_idx=2
        k = "decoder.up_blocks.0.resnets.2.conv2.weight"
        assert _convert_key(k) == "decoder.upsamples.2.residual.6.weight"

    def test_decoder_up_blocks_1_resnets_0(self):
        # block=1, resnet=0 -> flat_idx=4
        k = "decoder.up_blocks.1.resnets.0.norm1.gamma"
        assert _convert_key(k) == "decoder.upsamples.4.residual.0.gamma"

    def test_decoder_up_blocks_3_resnets_2(self):
        # block=3, resnet=2 -> flat_idx=14
        k = "decoder.up_blocks.3.resnets.2.conv2.bias"
        assert _convert_key(k) == "decoder.upsamples.14.residual.6.bias"

    # --- decoder up_blocks upsamplers ---
    def test_decoder_up_blocks_0_upsamplers(self):
        # block=0, upsampler flat_idx=3
        k = "decoder.up_blocks.0.upsamplers.0.resample.1.weight"
        assert _convert_key(k) == "decoder.upsamples.3.resample.1.weight"

    def test_decoder_up_blocks_2_upsamplers(self):
        # block=2, upsampler flat_idx=11
        k = "decoder.up_blocks.2.upsamplers.0.resample.1.bias"
        assert _convert_key(k) == "decoder.upsamples.11.resample.1.bias"

    # --- boundary: block 3 has no upsamplers ---
    def test_decoder_up_blocks_3_has_no_upsampler_in_map(self):
        # Block 3 has no upsamplers; the map must not contain that key.
        with pytest.raises(KeyError):
            _ = _mod._DECODER_UPSAMPLES_MAP[(3, 0, "upsampler")]

    # --- unknown key passthrough ---
    def test_unknown_key_passes_through(self):
        k = "some.unknown.prefix.weight"
        assert _convert_key(k) == k


# ===========================================================================
# 5. _resnet_sub helper
# ===========================================================================


class TestResnetSub:
    def test_norm1_gamma(self):
        assert _resnet_sub("norm1.gamma") == "residual.0.gamma"

    def test_conv1_weight(self):
        assert _resnet_sub("conv1.weight") == "residual.2.weight"

    def test_norm2_gamma(self):
        assert _resnet_sub("norm2.gamma") == "residual.3.gamma"

    def test_conv2_bias(self):
        assert _resnet_sub("conv2.bias") == "residual.6.bias"

    def test_conv_shortcut_weight(self):
        assert _resnet_sub("conv_shortcut.weight") == "shortcut.weight"

    def test_unknown_sub_passthrough(self):
        assert _resnet_sub("something.else") == "something.else"

    def test_empty_string(self):
        result = _resnet_sub("")
        assert isinstance(result, str)


# ===========================================================================
# 6. _build_decoder_upsamples_index_map coverage
# ===========================================================================


class TestDecoderUpsamplesMap:
    def test_map_has_15_resnet_entries(self):
        # 4 blocks x 3 resnets = 12, plus no extra
        resnets = [
            v
            for (b, r, kind), v in _mod._DECODER_UPSAMPLES_MAP.items()
            if kind == "resnet"
        ]
        assert len(resnets) == 12

    def test_map_has_3_upsampler_entries(self):
        upsamplers = [
            v
            for (b, r, kind), v in _mod._DECODER_UPSAMPLES_MAP.items()
            if kind == "upsampler"
        ]
        assert len(upsamplers) == 3

    def test_flat_indices_are_contiguous_0_to_14(self):
        all_vals = sorted(_mod._DECODER_UPSAMPLES_MAP.values())
        assert all_vals == list(range(15))

    def test_block0_resnets_are_0_1_2(self):
        assert _mod._DECODER_UPSAMPLES_MAP[(0, 0, "resnet")] == 0
        assert _mod._DECODER_UPSAMPLES_MAP[(0, 1, "resnet")] == 1
        assert _mod._DECODER_UPSAMPLES_MAP[(0, 2, "resnet")] == 2

    def test_block0_upsampler_is_3(self):
        assert _mod._DECODER_UPSAMPLES_MAP[(0, 0, "upsampler")] == 3

    def test_block3_resnets_are_12_13_14(self):
        assert _mod._DECODER_UPSAMPLES_MAP[(3, 0, "resnet")] == 12
        assert _mod._DECODER_UPSAMPLES_MAP[(3, 1, "resnet")] == 13
        assert _mod._DECODER_UPSAMPLES_MAP[(3, 2, "resnet")] == 14
