import sys

# Register MotifVideo19B model config with ComfyUI at startup.
# config.py is a placeholder; replace with full implementation in checklist item 3.
try:
    import comfy.supported_models
    from .config import MotifVideo19B
    comfy.supported_models.models.append(MotifVideo19B)
except Exception as e:
    print(f"[ComfyUI-MotifVideo1.9B] WARNING: Failed to register model config: {e}")

try:
    from .nodes.loader import MotifVideoModelLoader
    from .nodes.text_encode import MotifTextEncode
    from .nodes.latent import EmptyMotifLatent

    NODE_CLASS_MAPPINGS = {
        "MotifVideoModelLoader": MotifVideoModelLoader,
        "MotifTextEncode": MotifTextEncode,
        "EmptyMotifLatent": EmptyMotifLatent,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "MotifVideoModelLoader": "Load MotifVideo 1.9B Model",
        "MotifTextEncode": "MotifVideo Text Encode",
        "EmptyMotifLatent": "Empty MotifVideo Latent",
    }

except Exception as e:
    print(f"[ComfyUI-MotifVideo1.9B] ERROR: Failed to load nodes: {e}", file=sys.stderr)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
