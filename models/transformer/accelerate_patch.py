# Training-only FSDP2 patches. No-op in inference.
#
# In motif-core, this module applies FSDP2 activation-checkpointing patches
# required for distributed training. Since ComfyUI-MotifVideo runs inference
# only, this is a no-op stub that preserves the import/call interface.


def apply_fsdp_patches() -> None:
    """No-op stub. FSDP2 patches are only needed during training."""
    pass
