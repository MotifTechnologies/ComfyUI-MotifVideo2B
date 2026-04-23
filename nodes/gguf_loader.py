"""GGUF loader adapted from ComfyUI-GGUF (c) City96, licensed under Apache-2.0.

Original source: https://github.com/city96/ComfyUI-GGUF
Modifications (Motif Technologies, 2026):
- Added "motif_video" to IMG_ARCH_LIST whitelist.
- Renamed gguf_sd_loader -> motif_gguf_sd_loader.
- Renamed UnetLoaderGGUF -> MotifVideoUnetLoaderGGUF (coexists with upstream).
- GGMLOps / GGUFModelPatcher loaded dynamically from ComfyUI-GGUF at runtime
  (not duplicated here; maintenance stays with upstream).
"""
# SPDX-License-Identifier: Apache-2.0
import warnings
import logging
import torch
import gguf
import os
import inspect
import importlib.util
import importlib.machinery

import folder_paths

# ---------------------------------------------------------------------------
# Architecture whitelist — original 12 entries + "motif_video"
# ---------------------------------------------------------------------------
IMG_ARCH_LIST = {
    "flux", "sd1", "sdxl", "sd3", "aura", "hidream",
    "cosmos", "ltxv", "hyvid", "wan", "lumina2", "qwen_image",
    "motif_video",
}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3"}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}

# ---------------------------------------------------------------------------
# Helper functions — copied verbatim from ComfyUI-GGUF/loader.py
# (c) City96 || Apache-2.0
# ---------------------------------------------------------------------------

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")


def get_gguf_metadata(reader):
    """Extract all simple metadata fields like safetensors"""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:  # Simple scalar fields only
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except Exception:
            continue
    return metadata

# ---------------------------------------------------------------------------
# motif_gguf_sd_loader — gguf_sd_loader with expanded IMG_ARCH_LIST
# Copied from ComfyUI-GGUF/loader.py:70-163, (c) City96 || Apache-2.0
# Modification: uses module-level IMG_ARCH_LIST (includes "motif_video").
#               Inlined GGMLTensor import via _get_ggml_tensor() to avoid
#               hard dependency at import time.
# ---------------------------------------------------------------------------

def _get_ggml_tensor_cls():
    """Return GGMLTensor class from ComfyUI-GGUF/dequant.py (lazy)."""
    dequant_mod = _load_comfyui_gguf_module("dequant.py")
    return dequant_mod.GGMLTensor


def _get_is_quantized():
    dequant_mod = _load_comfyui_gguf_module("dequant.py")
    return dequant_mod.is_quantized


def _get_dequantize_tensor():
    dequant_mod = _load_comfyui_gguf_module("dequant.py")
    return dequant_mod.dequantize_tensor


def motif_gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False):
    """
    Read state dict as fake tensors.
    Identical to ComfyUI-GGUF gguf_sd_loader except IMG_ARCH_LIST includes
    "motif_video" (defined at module level above).
    """
    GGMLTensor = _get_ggml_tensor_cls()
    is_quantized = _get_is_quantized()
    dequantize_tensor = _get_dequantize_tensor()

    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    if arch_str in [None, "pig", "cow"]:
        # sd.cpp legacy compat path removed intentionally: MotifVideoUnetLoaderGGUF
        # targets motif_video GGUFs produced via diffusers, which always set
        # general.architecture. Users who need sd.cpp legacy compat should use
        # upstream ComfyUI-GGUF's UnetLoaderGGUF directly.
        raise ValueError(
            f"This GGUF has missing/legacy architecture metadata (arch={arch_str!r}). "
            "MotifVideoUnetLoaderGGUF does not support sd.cpp legacy compat; "
            f"use ComfyUI-GGUF's UnetLoaderGGUF instead. ({path})"
        )
    elif arch_str not in TXT_ARCH_LIST and is_text_model:
        if type_str not in VIS_TYPE_LIST:
            raise ValueError(f"Unexpected text model architecture type in GGUF file: {arch_str!r}")
    elif arch_str not in IMG_ARCH_LIST and not is_text_model:
        raise ValueError(f"Unexpected architecture type in GGUF file: {arch_str!r}")

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        # add to state dict
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # 1D tensors shouldn't be quantized, this is a fix for BF16
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader),
    }
    return (state_dict, extra)

# ---------------------------------------------------------------------------
# Dynamic import helpers for GGMLOps and GGUFModelPatcher
# (not duplicated here; maintenance stays with upstream ComfyUI-GGUF)
# ---------------------------------------------------------------------------

_VENDOR_PKG_NAME = "_motifvideo_gguf_vendor"


def _ensure_comfyui_gguf_package():
    """Register ComfyUI-GGUF as a namespace package in sys.modules so its
    submodules (ops, loader, nodes, dequant) can resolve their sibling
    relative imports (e.g. `from .ops import ...`). Raises ImportError if
    the ComfyUI-GGUF directory is not found.
    """
    import sys
    if _VENDOR_PKG_NAME in sys.modules:
        return sys.modules[_VENDOR_PKG_NAME]
    custom_nodes_root = folder_paths.folder_names_and_paths["custom_nodes"][0][0]
    pkg_path = os.path.join(custom_nodes_root, "ComfyUI-GGUF")
    if not os.path.isdir(pkg_path):
        raise ImportError(
            f"ComfyUI-GGUF not found at {pkg_path}. "
            "MotifVideoUnetLoaderGGUF requires ComfyUI-GGUF to be installed alongside this node pack. "
            "Install from: https://github.com/city96/ComfyUI-GGUF"
        )
    # Create an empty namespace package whose __path__ points at the real directory.
    # Python resolves `_motifvideo_gguf_vendor.<submodule>` by searching __path__.
    pkg_spec = importlib.machinery.ModuleSpec(
        _VENDOR_PKG_NAME, loader=None, is_package=True,
    )
    pkg_spec.submodule_search_locations = [pkg_path]
    pkg = importlib.util.module_from_spec(pkg_spec)
    pkg.__path__ = [pkg_path]
    sys.modules[_VENDOR_PKG_NAME] = pkg
    return pkg


def _load_comfyui_gguf_module(module_filename):
    """Load ComfyUI-GGUF/<module_filename> as a submodule of the vendor
    namespace package so sibling relative imports resolve correctly. Raises
    ImportError if the file is absent.
    """
    _ensure_comfyui_gguf_package()
    custom_nodes_root = folder_paths.folder_names_and_paths["custom_nodes"][0][0]
    path = os.path.join(custom_nodes_root, "ComfyUI-GGUF", module_filename)
    if not os.path.isfile(path):
        raise ImportError(
            f"ComfyUI-GGUF module not found at {path}. "
            "MotifVideoUnetLoaderGGUF requires ComfyUI-GGUF to be installed alongside this node pack. "
            "Install from: https://github.com/city96/ComfyUI-GGUF"
        )
    submodule_name = f"{_VENDOR_PKG_NAME}.{module_filename.replace('.py', '')}"
    import sys
    if submodule_name in sys.modules:
        return sys.modules[submodule_name]
    spec = importlib.util.spec_from_file_location(submodule_name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before executing so relative imports within the module resolve.
    sys.modules[submodule_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(submodule_name, None)
        raise
    return module


def _get_ggml_ops():
    """Lazy-load GGMLOps from ComfyUI-GGUF/ops.py."""
    return _load_comfyui_gguf_module("ops.py").GGMLOps


def _get_gguf_model_patcher():
    """Lazy-load GGUFModelPatcher from ComfyUI-GGUF/nodes.py."""
    return _load_comfyui_gguf_module("nodes.py").GGUFModelPatcher

# ---------------------------------------------------------------------------
# MotifVideoUnetLoaderGGUF — adapted from ComfyUI-GGUF UnetLoaderGGUF
# Original: ComfyUI-GGUF/nodes.py:135-184, (c) City96 || Apache-2.0
# Changes: class/title/category rename, gguf_sd_loader -> motif_gguf_sd_loader,
#          GGMLOps/GGUFModelPatcher loaded via lazy helpers above.
# ---------------------------------------------------------------------------

class MotifVideoUnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        # The "unet_gguf" folder key is registered by ComfyUI-GGUF at its
        # import time. If upstream is missing or failed to import, schema
        # generation would otherwise raise KeyError("unet_gguf"). Surface
        # a clear warning + empty dropdown instead.
        try:
            unet_names = list(folder_paths.get_filename_list("unet_gguf"))
        except KeyError:
            logging.warning(
                "MotifVideoUnetLoaderGGUF: 'unet_gguf' folder key not registered. "
                "ComfyUI-GGUF is required alongside this node pack. "
                "Install from: https://github.com/city96/ComfyUI-GGUF"
            )
            unet_names = []
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "MotifVideo/loaders"
    TITLE = "Motif Video Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        GGMLOps = _get_ggml_ops()
        GGUFModelPatcher = _get_gguf_model_patcher()

        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        import comfy.sd
        import comfy.utils

        # init model
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd, extra = motif_gguf_sd_loader(unet_path)

        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = extra.get("metadata", {})

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, **kwargs,
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)
