"""MotifVideo VAE loader node.

Converts diffusers-format MotifVideo VAE (AutoencoderKLWan) state dict
to ComfyUI WanVAE key naming, then loads via comfy.sd.VAE.
"""

import logging

import safetensors.torch
import folder_paths
import comfy.sd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key-conversion helpers
# ---------------------------------------------------------------------------

def _convert_resnet_keys(k: str, prefix_in: str, prefix_out: str) -> str:
    """Convert a single resnet block key from diffusers to ComfyUI format.

    Diffusers resnet sub-keys:
        .norm1.gamma  -> .residual.0.gamma
        .conv1.*      -> .residual.2.*
        .norm2.gamma  -> .residual.3.gamma
        .conv2.*      -> .residual.6.*
        .conv_shortcut.* -> .shortcut.*

    prefix_in  : e.g. "encoder.down_blocks.3"
    prefix_out : e.g. "encoder.downsamples.3"
    """
    suffix = k[len(prefix_in):]  # e.g. ".norm1.gamma"

    replacements = [
        (".norm1.gamma", ".residual.0.gamma"),
        (".conv1.",       ".residual.2."),
        (".norm2.gamma",  ".residual.3.gamma"),
        (".conv2.",       ".residual.6."),
        (".conv_shortcut.", ".shortcut."),
    ]
    for old, new in replacements:
        if suffix.startswith(old):
            suffix = new + suffix[len(old):]
            break

    return prefix_out + suffix


def _convert_mid_block_resnet(k: str, section: str) -> str:
    """Convert encoder/decoder mid_block.resnets.N keys.

    mid_block.resnets.0 -> middle.0.residual.*
    mid_block.attentions.0 -> middle.1.*
    mid_block.resnets.1 -> middle.2.residual.*

    section: "encoder" or "decoder"
    """
    rest = k[len(f"{section}.mid_block."):]  # e.g. "resnets.0.norm1.gamma"

    if rest.startswith("resnets.0."):
        sub = rest[len("resnets.0."):]
        new_k = f"{section}.middle.0." + _resnet_sub(sub)
    elif rest.startswith("attentions.0."):
        sub = rest[len("attentions.0."):]
        new_k = f"{section}.middle.1.{sub}"
    elif rest.startswith("resnets.1."):
        sub = rest[len("resnets.1."):]
        new_k = f"{section}.middle.2." + _resnet_sub(sub)
    else:
        new_k = k  # fallback — no conversion
    return new_k


def _resnet_sub(sub: str) -> str:
    """Convert resnet sub-key suffix (without leading dot) to ComfyUI residual format."""
    if sub.startswith("norm1.gamma"):
        return "residual.0.gamma"
    if sub.startswith("conv1."):
        return "residual.2." + sub[len("conv1."):]
    if sub.startswith("norm2.gamma"):
        return "residual.3.gamma"
    if sub.startswith("conv2."):
        return "residual.6." + sub[len("conv2."):]
    if sub.startswith("conv_shortcut."):
        return "shortcut." + sub[len("conv_shortcut."):]
    return sub  # passthrough (e.g. attention keys already flat)


def _build_decoder_upsamples_index_map() -> dict:
    """Return mapping: (block_idx, resnet_or_up_idx, kind) -> flat upsamples index.

    Decoder up_blocks layout (diffusers):
        up_blocks.0: resnets 0,1,2  + upsamplers.0  (has time_conv)
        up_blocks.1: resnets 0,1,2  + upsamplers.0  (has time_conv)
        up_blocks.2: resnets 0,1,2  + upsamplers.0  (resample only)
        up_blocks.3: resnets 0,1,2  (no upsamplers)

    ComfyUI flat upsamples index:
        0,1,2   -> up_blocks.0.resnets.0,1,2
        3       -> up_blocks.0.upsamplers.0
        4,5,6   -> up_blocks.1.resnets.0,1,2
        7       -> up_blocks.1.upsamplers.0
        8,9,10  -> up_blocks.2.resnets.0,1,2
        11      -> up_blocks.2.upsamplers.0
        12,13,14 -> up_blocks.3.resnets.0,1,2
    """
    mapping = {}
    flat = 0
    for block in range(4):
        for resnet in range(3):
            mapping[(block, resnet, "resnet")] = flat
            flat += 1
        if block < 3:  # blocks 0,1,2 have upsamplers
            mapping[(block, 0, "upsampler")] = flat
            flat += 1
    return mapping


_DECODER_UPSAMPLES_MAP = _build_decoder_upsamples_index_map()


def convert_diffusers_to_comfyui(state_dict: dict) -> dict:
    """Convert MotifVideo diffusers VAE state dict to ComfyUI WanVAE format.

    Handles:
    - quant_conv / post_quant_conv
    - encoder.conv_in / encoder.conv_out / encoder.norm_out
    - decoder.conv_in / decoder.conv_out / decoder.norm_out
    - encoder.mid_block / decoder.mid_block
    - encoder.down_blocks.N  (already flat in diffusers MotifVideo format)
    - decoder.up_blocks.N.resnets.M / decoder.up_blocks.N.upsamplers.0

    Returns a new dict with ComfyUI key names.
    """
    new_sd = {}

    for k, v in state_dict.items():
        new_k = _convert_key(k)
        if new_k in new_sd:
            logger.warning("Key collision after conversion: %s -> %s", k, new_k)
        new_sd[new_k] = v

    return new_sd


def _convert_key(k: str) -> str:  # noqa: C901 — intentionally exhaustive
    # --- quant convs ---
    if k.startswith("quant_conv."):
        return "conv1." + k[len("quant_conv."):]
    if k.startswith("post_quant_conv."):
        return "conv2." + k[len("post_quant_conv."):]

    # --- encoder top-level convs ---
    if k.startswith("encoder.conv_in."):
        return "encoder.conv1." + k[len("encoder.conv_in."):]
    if k.startswith("encoder.conv_out."):
        return "encoder.head.2." + k[len("encoder.conv_out."):]
    if k.startswith("encoder.norm_out."):
        return "encoder.head.0." + k[len("encoder.norm_out."):]

    # --- decoder top-level convs ---
    if k.startswith("decoder.conv_in."):
        return "decoder.conv1." + k[len("decoder.conv_in."):]
    if k.startswith("decoder.conv_out."):
        return "decoder.head.2." + k[len("decoder.conv_out."):]
    if k.startswith("decoder.norm_out."):
        return "decoder.head.0." + k[len("decoder.norm_out."):]

    # --- encoder / decoder mid_block ---
    if k.startswith("encoder.mid_block."):
        return _convert_mid_block_resnet(k, "encoder")
    if k.startswith("decoder.mid_block."):
        return _convert_mid_block_resnet(k, "decoder")

    # --- encoder.down_blocks.N (already flat indices in diffusers MotifVideo format) ---
    if k.startswith("encoder.down_blocks."):
        # e.g. encoder.down_blocks.3.conv1.weight
        # Maps directly to encoder.downsamples.3.*  with resnet sub-key conversion
        rest = k[len("encoder.down_blocks."):]  # "3.conv1.weight"
        dot = rest.index(".")
        idx = rest[:dot]
        sub = rest[dot + 1:]  # "conv1.weight"
        prefix_in = f"encoder.down_blocks.{idx}"
        prefix_out = f"encoder.downsamples.{idx}"
        return _convert_resnet_keys(k, prefix_in, prefix_out)

    # --- decoder.up_blocks.B.resnets.R / upsamplers.0 ---
    if k.startswith("decoder.up_blocks."):
        rest = k[len("decoder.up_blocks."):]  # "0.resnets.1.conv1.weight"
        parts = rest.split(".")
        block = int(parts[0])

        if parts[1] == "resnets":
            resnet_idx = int(parts[2])
            flat_idx = _DECODER_UPSAMPLES_MAP[(block, resnet_idx, "resnet")]
            prefix_in = f"decoder.up_blocks.{block}.resnets.{resnet_idx}"
            prefix_out = f"decoder.upsamples.{flat_idx}"
            return _convert_resnet_keys(k, prefix_in, prefix_out)

        if parts[1] == "upsamplers":
            flat_idx = _DECODER_UPSAMPLES_MAP[(block, 0, "upsampler")]
            suffix = ".".join(parts[3:])  # after "upsamplers.0."
            return f"decoder.upsamples.{flat_idx}.{suffix}"

    # Passthrough — key is already in ComfyUI format or unknown
    return k


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class MotifVAELoader:
    """Load MotifVideo VAE (diffusers format) and expose as ComfyUI VAE object.

    Select the VAE safetensors file from the models/vae/ dropdown.
    Key conversion from diffusers AutoencoderKLWan format to ComfyUI WanVAE
    format is applied automatically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "motifvideo"

    DESCRIPTION = (
        "Load MotifVideo VAE checkpoint (diffusers AutoencoderKLWan format).\n"
        "Select the .safetensors file from models/vae/.\n"
        "Keys are automatically converted to ComfyUI WanVAE format."
    )

    def load_vae(self, vae_name: str):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        logger.info("Loading MotifVideo VAE from %s", vae_path)

        sd = safetensors.torch.load_file(vae_path)

        # Detect if conversion is needed (diffusers format has quant_conv)
        if "quant_conv.weight" in sd or "encoder.conv_in.weight" in sd:
            logger.info("Detected diffusers format — applying key conversion")
            sd = convert_diffusers_to_comfyui(sd)
        else:
            logger.info("Keys appear to be already in ComfyUI format — skipping conversion")

        vae = comfy.sd.VAE(sd=sd)
        return (vae,)
