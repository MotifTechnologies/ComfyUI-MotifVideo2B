# TODO: Full implementation in checklist item 6.
# Input:  CLIP, text (prompt), negative_prompt
# Output: CONDITIONING (positive), CONDITIONING (negative)


class MotifTextEncode:
    """Placeholder — full implementation in checklist item 6."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "motifvideo"

    def encode(self, clip, text, negative_prompt):
        raise NotImplementedError("MotifTextEncode: not yet implemented (checklist item 6)")
