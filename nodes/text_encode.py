"""MotifVideo text encoding node.

Convenience node that encodes both positive and negative prompts in one step.
Also compatible with standard CLIPTextEncode node (use that if you prefer
separate positive/negative encoding).
"""


class MotifTextEncode:
    """Encode positive and negative prompts for MotifVideo in one node.

    Equivalent to two CLIPTextEncode nodes but combined for convenience.
    Output CONDITIONING is compatible with KSampler positive/negative inputs.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry, low quality, distorted, artifacts",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "motifvideo"

    def encode(self, clip, text, negative_prompt):
        # encode_from_tokens_scheduled returns CONDITIONING format directly.
        #
        # Tokenizer padding mask propagation (P0.4 R2 fix):
        # MotifVideoT5Gemma2Model.encode_token_weights() constructs the
        # attention_mask (0 for pad tokens, 1 for real tokens) and returns it
        # as an extra dict {"attention_mask": mask} in the 3-tuple result.
        # ComfyUI's ClipTokenWeightEncoder plumbing and convert_cond() then place
        # it into pooled_dict["attention_mask"], which MotifVideoModel.extra_conds()
        # picks up via kwargs.get("attention_mask").  Without this, extra_conds()
        # falls back to an all-ones mask that treats padding tokens as real tokens
        # in cross-attention — a training/inference distribution mismatch.
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        return (positive, negative)
